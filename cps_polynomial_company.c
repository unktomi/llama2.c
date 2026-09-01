#define _POSIX_C_SOURCE 200809L

/*
 * Execute one finite polynomial company without flattening it into a score.
 *
 * The input context table supplies four number corners (x, A, C, AC) as
 * already-tokenized causal prefixes.  Each prefix is inserted into one shared
 * trie.  Every fixed outer constructor d is then attached to its prefix leaf.
 * One llama_company_evaluate call therefore produces
 *
 *     q_D(x)                         outer constructor codata
 *     q_E(d(x)) for every d in D     shape-indexed hole codata
 *
 * for every context and number corner.  The result is the structured
 * container G(H), not a flat score over pairs (d,e).  No probability,
 * sequence likelihood, norm, or scalar completion reward is constructed.
 */

#include "llama_company.h"

#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    FEATURE_CORNER_COUNT = 4
};

static const char *FEATURE_CORNER_NAMES[FEATURE_CORNER_COUNT] = {
    "x", "A", "C", "AC"
};

typedef struct {
    int *values;
    int count;
} IntVector;

typedef struct {
    char *key;
    IntVector corners[FEATURE_CORNER_COUNT];
    int leaves[FEATURE_CORNER_COUNT];
} ContextRecord;

typedef struct {
    ContextRecord *records;
    int count;
    int capacity;
} ContextTable;

typedef struct {
    int *tokens;
    int count;
    int capacity;
} TokenFamily;

typedef struct {
    int token;
    int position;
    int parent;
    int first_child;
    int next_sibling;
} CompanyNode;

typedef struct {
    CompanyNode *nodes;
    int count;
    int capacity;
    int first_root;
} CompanyTrie;

typedef struct {
    const char *trace_path;
    int reference_token;
} Options;

static void fail(const char *message) {
    fprintf(stderr, "cps_polynomial_company: %s\n", message);
    exit(EXIT_FAILURE);
}

static void *checked_calloc(size_t count, size_t width) {
    if (width != 0 && count > SIZE_MAX / width) fail("allocation overflow");
    if (count == 0) count = 1;
    void *memory = calloc(count, width);
    if (memory == NULL) fail("allocation failed");
    return memory;
}

static void *checked_realloc(void *memory, size_t count, size_t width) {
    if (width != 0 && count > SIZE_MAX / width) fail("allocation overflow");
    if (count == 0) count = 1;
    void *resized = realloc(memory, count * width);
    if (resized == NULL) fail("allocation failed");
    return resized;
}

static char *checked_strdup(const char *text) {
    char *copy = strdup(text);
    if (copy == NULL) fail("allocation failed");
    return copy;
}

static int parse_integer(const char *text, const char *description) {
    errno = 0;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value < 0 || value > INT_MAX) {
        fprintf(stderr, "invalid %s: %s\n", description, text);
        exit(EXIT_FAILURE);
    }
    return (int)value;
}

static void strip_line_ending(char *line) {
    size_t length = strlen(line);
    while (length > 0 &&
           (line[length - 1] == '\n' || line[length - 1] == '\r')) {
        line[--length] = '\0';
    }
}

static int split_tabs(char *line, char **fields, int maximum_fields) {
    int count = 0;
    char *cursor = line;
    while (count < maximum_fields) {
        fields[count++] = cursor;
        char *tab = strchr(cursor, '\t');
        if (tab == NULL) break;
        *tab = '\0';
        cursor = tab + 1;
    }
    return count;
}

static IntVector parse_csv_tokens(const char *text, int vocab_size) {
    if (text == NULL || *text == '\0') fail("empty token sequence");
    char *copy = checked_strdup(text);
    int capacity = 16;
    int count = 0;
    int *values = checked_calloc((size_t)capacity, sizeof(*values));
    char *save = NULL;
    for (char *field = strtok_r(copy, ",", &save);
         field != NULL;
         field = strtok_r(NULL, ",", &save)) {
        int token = parse_integer(field, "context token");
        if (token >= vocab_size) fail("context token is outside vocabulary");
        if (count == capacity) {
            if (capacity > INT_MAX / 2) fail("token sequence is too large");
            capacity *= 2;
            values = checked_realloc(
                values,
                (size_t)capacity,
                sizeof(*values)
            );
        }
        values[count++] = token;
    }
    free(copy);
    if (count == 0) fail("empty token sequence");
    return (IntVector){.values = values, .count = count};
}

static TokenFamily read_token_family(const char *path, int vocab_size) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) fail("could not open constructor family");
    TokenFamily family = {0};
    char *line = NULL;
    size_t line_capacity = 0;
    while (getline(&line, &line_capacity, file) >= 0) {
        strip_line_ending(line);
        char *cursor = line;
        while (*cursor == ' ' || *cursor == '\t') cursor++;
        if (*cursor == '\0' || *cursor == '#') continue;
        char *tab = strchr(cursor, '\t');
        if (tab != NULL) *tab = '\0';
        int token = parse_integer(cursor, "constructor token");
        if (token >= vocab_size) fail("constructor token is outside vocabulary");
        for (int index = 0; index < family.count; index++) {
            if (family.tokens[index] == token) fail("duplicate constructor token");
        }
        if (family.count == family.capacity) {
            int capacity = family.capacity == 0 ? 16 : family.capacity * 2;
            if (capacity < family.capacity) fail("constructor family is too large");
            family.tokens = checked_realloc(
                family.tokens,
                (size_t)capacity,
                sizeof(*family.tokens)
            );
            family.capacity = capacity;
        }
        family.tokens[family.count++] = token;
    }
    free(line);
    if (fclose(file) != 0) fail("could not close constructor family");
    if (family.count == 0) fail("constructor family is empty");
    return family;
}

static ContextTable read_context_table(
    const char *path,
    int vocab_size,
    int sequence_length
) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) fail("could not open context table");
    ContextTable table = {0};
    char *line = NULL;
    size_t line_capacity = 0;
    while (getline(&line, &line_capacity, file) >= 0) {
        strip_line_ending(line);
        if (*line == '\0' || *line == '#') continue;
        char *fields[1 + FEATURE_CORNER_COUNT];
        int field_count = split_tabs(
            line,
            fields,
            1 + FEATURE_CORNER_COUNT
        );
        if (field_count != 1 + FEATURE_CORNER_COUNT) {
            fail("context table row must contain key and four token sequences");
        }
        if (table.count == table.capacity) {
            int capacity = table.capacity == 0 ? 64 : table.capacity * 2;
            if (capacity < table.capacity) fail("context table is too large");
            table.records = checked_realloc(
                table.records,
                (size_t)capacity,
                sizeof(*table.records)
            );
            memset(
                table.records + table.capacity,
                0,
                (size_t)(capacity - table.capacity) * sizeof(*table.records)
            );
            table.capacity = capacity;
        }
        ContextRecord *record = &table.records[table.count++];
        record->key = checked_strdup(fields[0]);
        for (int previous = 0; previous + 1 < table.count; previous++) {
            if (strcmp(table.records[previous].key, record->key) == 0) {
                fail("duplicate context key");
            }
        }
        for (int corner = 0; corner < FEATURE_CORNER_COUNT; corner++) {
            record->corners[corner] = parse_csv_tokens(
                fields[corner + 1],
                vocab_size
            );
            if (record->corners[corner].count >= sequence_length) {
                fail("context leaves no room for a constructor");
            }
        }
    }
    free(line);
    if (fclose(file) != 0) fail("could not close context table");
    if (table.count == 0) fail("context table is empty");
    return table;
}

static void ensure_trie_capacity(CompanyTrie *trie) {
    if (trie->count < trie->capacity) return;
    int capacity = trie->capacity == 0 ? 4096 : trie->capacity * 2;
    if (capacity < trie->capacity) fail("causal company is too large");
    trie->nodes = checked_realloc(
        trie->nodes,
        (size_t)capacity,
        sizeof(*trie->nodes)
    );
    trie->capacity = capacity;
}

static int trie_child(
    CompanyTrie *trie,
    int parent,
    int token,
    int position
) {
    int child = parent < 0
        ? trie->first_root
        : trie->nodes[parent].first_child;
    for (int current = child; current >= 0;
         current = trie->nodes[current].next_sibling) {
        CompanyNode *node = &trie->nodes[current];
        if (node->token == token) {
            if (node->position != position || node->parent != parent) {
                fail("shared prefix has inconsistent type");
            }
            return current;
        }
    }
    ensure_trie_capacity(trie);
    int index = trie->count++;
    CompanyNode *node = &trie->nodes[index];
    *node = (CompanyNode){
        .token = token,
        .position = position,
        .parent = parent,
        .first_child = -1,
        .next_sibling = parent < 0
            ? trie->first_root
            : trie->nodes[parent].first_child,
    };
    if (parent < 0) {
        trie->first_root = index;
    } else {
        trie->nodes[parent].first_child = index;
    }
    return index;
}

static int trie_sequence(CompanyTrie *trie, const IntVector *sequence) {
    int parent = -1;
    for (int position = 0; position < sequence->count; position++) {
        parent = trie_child(trie, parent, sequence->values[position], position);
    }
    return parent;
}

static void fprint_json_string(FILE *file, const char *text) {
    fputc('"', file);
    for (const unsigned char *cursor = (const unsigned char *)text;
         *cursor != '\0'; cursor++) {
        unsigned char value = *cursor;
        if (value == '"' || value == '\\') {
            fputc('\\', file);
            fputc(value, file);
        } else if (value == '\n') {
            fputs("\\n", file);
        } else if (value == '\r') {
            fputs("\\r", file);
        } else if (value == '\t') {
            fputs("\\t", file);
        } else if (value < 0x20) {
            fprintf(file, "\\u%04x", value);
        } else {
            fputc(value, file);
        }
    }
    fputc('"', file);
}

static void write_token_array(FILE *file, const TokenFamily *family) {
    fputc('[', file);
    for (int index = 0; index < family->count; index++) {
        if (index != 0) fputc(',', file);
        fprintf(file, "%d", family->tokens[index]);
    }
    fputc(']', file);
}

static void write_contrasts(
    FILE *file,
    const LlamaCompanyResult *result,
    int row,
    const TokenFamily *family,
    int reference_token
) {
    if (row < 0 || row >= result->row_count) fail("observation row is invalid");
    const float *logits = result->logits +
        (size_t)row * result->vocab_size;
    float reference = logits[reference_token];
    fputc('[', file);
    for (int index = 0; index < family->count; index++) {
        if (index != 0) fputc(',', file);
        fprintf(file, "%.9g", logits[family->tokens[index]] - reference);
    }
    fputc(']', file);
}

static Options parse_options(int argc, char **argv) {
    if (argc < 8) {
        fprintf(
            stderr,
            "usage: %s CHECKPOINT TOKENIZER CONTEXT_TSV OUTER_TSV HOLE_TSV "
            "--trace PATH [--reference-token ID]\n",
            argv[0]
        );
        exit(EXIT_FAILURE);
    }
    Options options = {.reference_token = 1};
    for (int index = 6; index < argc;) {
        if (strcmp(argv[index], "--trace") == 0 && index + 1 < argc) {
            options.trace_path = argv[index + 1];
            index += 2;
        } else if (strcmp(argv[index], "--reference-token") == 0 &&
                   index + 1 < argc) {
            options.reference_token = parse_integer(
                argv[index + 1],
                "reference token"
            );
            index += 2;
        } else {
            fail("unrecognized option");
        }
    }
    if (options.trace_path == NULL) fail("--trace is required");
    return options;
}

static void free_context_table(ContextTable *table) {
    for (int index = 0; index < table->count; index++) {
        ContextRecord *record = &table->records[index];
        free(record->key);
        for (int corner = 0; corner < FEATURE_CORNER_COUNT; corner++) {
            free(record->corners[corner].values);
        }
    }
    free(table->records);
}

int main(int argc, char **argv) {
    Options options = parse_options(argc, argv);
    AtkeyRuntime *runtime = atkey_runtime_new(argv[1], argv[2]);
    if (runtime == NULL) fail("could not initialize runtime");
    int vocab_size = atkey_vocab_size(runtime);
    int sequence_length = atkey_sequence_length(runtime);
    if (options.reference_token < 0 || options.reference_token >= vocab_size) {
        fail("reference token is outside vocabulary");
    }
    ContextTable contexts = read_context_table(
        argv[3],
        vocab_size,
        sequence_length
    );
    TokenFamily outer = read_token_family(argv[4], vocab_size);
    TokenFamily holes = read_token_family(argv[5], vocab_size);

    CompanyTrie trie = {.first_root = -1};
    for (int context = 0; context < contexts.count; context++) {
        for (int corner = 0; corner < FEATURE_CORNER_COUNT; corner++) {
            contexts.records[context].leaves[corner] = trie_sequence(
                &trie,
                &contexts.records[context].corners[corner]
            );
        }
    }
    size_t branch_count =
        (size_t)contexts.count * FEATURE_CORNER_COUNT * outer.count;
    int *branches = checked_calloc(branch_count, sizeof(*branches));
    for (int context = 0; context < contexts.count; context++) {
        for (int corner = 0; corner < FEATURE_CORNER_COUNT; corner++) {
            int parent = contexts.records[context].leaves[corner];
            int position = trie.nodes[parent].position + 1;
            for (int shape = 0; shape < outer.count; shape++) {
                size_t offset =
                    ((size_t)context * FEATURE_CORNER_COUNT + corner) *
                    outer.count + shape;
                branches[offset] = trie_child(
                    &trie,
                    parent,
                    outer.tokens[shape],
                    position
                );
            }
        }
    }

    int *tokens = checked_calloc((size_t)trie.count, sizeof(*tokens));
    int *positions = checked_calloc((size_t)trie.count, sizeof(*positions));
    int *parents = checked_calloc((size_t)trie.count, sizeof(*parents));
    for (int row = 0; row < trie.count; row++) {
        tokens[row] = trie.nodes[row].token;
        positions[row] = trie.nodes[row].position;
        parents[row] = trie.nodes[row].parent;
        if (parents[row] >= row) fail("causal company parent order is invalid");
    }
    LlamaCompanyShape shape = {
        .row_count = trie.count,
        .tokens = tokens,
        .positions = positions,
        .parents = parents,
    };

    int filler_count = atkey_filler_count(runtime);
    size_t *calls_before = checked_calloc(
        (size_t)filler_count,
        sizeof(*calls_before)
    );
    size_t *reads_before = checked_calloc(
        (size_t)filler_count,
        sizeof(*reads_before)
    );
    for (int filler = 0; filler < filler_count; filler++) {
        calls_before[filler] = atkey_filler_calls(runtime, filler);
        reads_before[filler] = atkey_filler_scalar_reads(runtime, filler);
    }
    LlamaCompanyResult result;
    if (!llama_company_evaluate(runtime, &shape, false, &result)) {
        fail("causal company evaluation failed");
    }
    uint64_t total_calls = 0;
    uint64_t maximum_calls = 0;
    uint64_t total_reads = 0;
    for (int filler = 0; filler < filler_count; filler++) {
        size_t calls = atkey_filler_calls(runtime, filler) - calls_before[filler];
        size_t reads =
            atkey_filler_scalar_reads(runtime, filler) - reads_before[filler];
        total_calls += calls;
        total_reads += reads;
        if (calls > maximum_calls) maximum_calls = calls;
    }
    if (maximum_calls != 1 || total_calls != (uint64_t)filler_count) {
        fail("learned filler was not applied exactly once to the company family");
    }

    FILE *trace = fopen(options.trace_path, "wb");
    if (trace == NULL) fail("could not create trace");
    fprintf(
        trace,
        "{\"kind\":\"polynomial_company_meta\",\"schema_version\":1,"
        "\"semantics\":\"structured_polynomial_company_G_of_H\","
        "\"contexts\":%d,\"feature_corners\":4,\"company_rows\":%d,"
        "\"outer_width\":%d,\"hole_width\":%d,"
        "\"reference_token\":%d,\"outer_tokens\":",
        contexts.count,
        trie.count,
        outer.count,
        holes.count,
        options.reference_token
    );
    write_token_array(trace, &outer);
    fputs(",\"hole_tokens\":", trace);
    write_token_array(trace, &holes);
    fprintf(
        trace,
        ",\"learned_fillers\":%d,\"family_filler_calls\":%llu,"
        "\"maximum_calls_per_filler\":%llu,"
        "\"family_scalar_reads\":%llu,"
        "\"probabilities_used\":false,\"scalar_reward_used\":false,"
        "\"pair_scores_flattened\":false}\n",
        filler_count,
        (unsigned long long)total_calls,
        (unsigned long long)maximum_calls,
        (unsigned long long)total_reads
    );
    fflush(trace);

    for (int context = 0; context < contexts.count; context++) {
        ContextRecord *record = &contexts.records[context];
        for (int corner = 0; corner < FEATURE_CORNER_COUNT; corner++) {
            int outer_row = record->leaves[corner];
            fputs("{\"kind\":\"polynomial_company_outer\",\"case\":", trace);
            fprint_json_string(trace, record->key);
            fputs(",\"corner\":", trace);
            fprint_json_string(trace, FEATURE_CORNER_NAMES[corner]);
            fprintf(trace, ",\"row\":%d,\"contrasts\":", outer_row);
            write_contrasts(
                trace,
                &result,
                outer_row,
                &outer,
                options.reference_token
            );
            fputs("}\n", trace);
            fflush(trace);

            for (int outer_shape = 0; outer_shape < outer.count; outer_shape++) {
                size_t offset =
                    ((size_t)context * FEATURE_CORNER_COUNT + corner) *
                    outer.count + outer_shape;
                int hole_row = branches[offset];
                fputs("{\"kind\":\"polynomial_company_hole\",\"case\":", trace);
                fprint_json_string(trace, record->key);
                fputs(",\"corner\":", trace);
                fprint_json_string(trace, FEATURE_CORNER_NAMES[corner]);
                fprintf(
                    trace,
                    ",\"outer_token\":%d,\"row\":%d,\"contrasts\":",
                    outer.tokens[outer_shape],
                    hole_row
                );
                write_contrasts(
                    trace,
                    &result,
                    hole_row,
                    &holes,
                    options.reference_token
                );
                fputs("}\n", trace);
                fflush(trace);
            }
        }
    }
    fprintf(
        trace,
        "{\"kind\":\"polynomial_company_check\","
        "\"contexts\":%d,\"outer_observations\":%d,"
        "\"shape_indexed_hole_observations\":%d,"
        "\"maximum_calls_per_filler\":%llu}\n",
        contexts.count,
        contexts.count * FEATURE_CORNER_COUNT,
        contexts.count * FEATURE_CORNER_COUNT * outer.count,
        (unsigned long long)maximum_calls
    );
    fflush(trace);
    if (fclose(trace) != 0) fail("could not close trace");

    printf(
        "polynomial_company contexts=%d rows=%d outer=%d holes=%d "
        "fillers=%d max_calls=%llu scalar_reads=%llu\n",
        contexts.count,
        trie.count,
        outer.count,
        holes.count,
        filler_count,
        (unsigned long long)maximum_calls,
        (unsigned long long)total_reads
    );

    llama_company_result_free(&result);
    free(reads_before);
    free(calls_before);
    free(parents);
    free(positions);
    free(tokens);
    free(branches);
    free(trie.nodes);
    free(holes.tokens);
    free(outer.tokens);
    free_context_table(&contexts);
    atkey_runtime_free(runtime);
    return EXIT_SUCCESS;
}
