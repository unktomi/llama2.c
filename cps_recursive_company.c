#define _POSIX_C_SOURCE 200809L

/*
 * Compile a finite dependent constructor tree into one causal company.
 *
 * A root table supplies already-tokenized causal prefixes. Repeated
 * --family arguments supply the finite constructor family demanded at each
 * recursive depth. The family at depth d is observed at the prefix selected
 * by the preceding d constructors, and every member is retained as a child.
 * A separate terminal family observes every complete retained branch.
 *
 * The whole tree is built before llama_company_evaluate. Consequently every
 * learned filler is applied once to the complete occurrence family. The
 * result remains a tree of token-indexed contrast vectors. This executable
 * intentionally supplies no path score, probability fold, whole-completion
 * argmax, or hidden local-greedy terminalization.
 */

#include "llama_company.h"

#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int *values;
    int count;
} IntVector;

typedef struct {
    int *tokens;
    int count;
    int capacity;
    char *path;
} TokenFamily;

typedef struct {
    TokenFamily *values;
    int count;
    int capacity;
} FamilyVector;

typedef struct {
    char *key;
    IntVector prefix;
    int leaf;
} RootRecord;

typedef struct {
    RootRecord *values;
    int count;
    int capacity;
} RootTable;

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

/* A demand observes family[depth] at row. Its incoming token is the edge from
 * parent_demand; root demands have neither. */
typedef struct {
    int root;
    int depth;
    int row;
    int parent_demand;
    int incoming_token;
} DemandRecord;

typedef struct {
    DemandRecord *values;
    int count;
    int capacity;
} DemandTable;

typedef struct {
    int root;
    int row;
    int parent_demand;
    int incoming_token;
} LeafRecord;

typedef struct {
    LeafRecord *values;
    int count;
    int capacity;
} LeafTable;

typedef struct {
    const char *trace_path;
    const char *terminal_family_path;
    const char *metal_library;
    int reference_token;
    bool use_metal;
    FamilyVector families;
} Options;

static _Noreturn void fail(const char *message) {
    fprintf(stderr, "cps_recursive_company: %s\n", message);
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
        int token = parse_integer(field, "token");
        if (token >= vocab_size) fail("token is outside vocabulary");
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
    TokenFamily family = {.path = checked_strdup(path)};
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
            if (family.tokens[index] == token) {
                fail("duplicate constructor token");
            }
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

static void family_vector_append(FamilyVector *families, TokenFamily family) {
    if (families->count == families->capacity) {
        int capacity = families->capacity == 0 ? 4 : families->capacity * 2;
        if (capacity < families->capacity) fail("too many recursive families");
        families->values = checked_realloc(
            families->values,
            (size_t)capacity,
            sizeof(*families->values)
        );
        families->capacity = capacity;
    }
    families->values[families->count++] = family;
}

static RootTable read_roots(
    const char *path,
    int vocab_size,
    int sequence_length,
    int recursive_depth
) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) fail("could not open root table");
    RootTable table = {0};
    char *line = NULL;
    size_t line_capacity = 0;
    while (getline(&line, &line_capacity, file) >= 0) {
        strip_line_ending(line);
        if (*line == '\0' || *line == '#') continue;
        char *tab = strchr(line, '\t');
        if (tab == NULL || tab == line || tab[1] == '\0') {
            fail("root row must contain key and comma-separated tokens");
        }
        *tab = '\0';
        if (table.count == table.capacity) {
            int capacity = table.capacity == 0 ? 64 : table.capacity * 2;
            if (capacity < table.capacity) fail("root table is too large");
            table.values = checked_realloc(
                table.values,
                (size_t)capacity,
                sizeof(*table.values)
            );
            memset(
                table.values + table.capacity,
                0,
                (size_t)(capacity - table.capacity) * sizeof(*table.values)
            );
            table.capacity = capacity;
        }
        RootRecord *record = &table.values[table.count++];
        record->key = checked_strdup(line);
        for (int previous = 0; previous + 1 < table.count; previous++) {
            if (strcmp(table.values[previous].key, record->key) == 0) {
                fail("duplicate root key");
            }
        }
        record->prefix = parse_csv_tokens(tab + 1, vocab_size);
        if (record->prefix.count + recursive_depth > sequence_length) {
            fail("root and recursive term exceed model context");
        }
    }
    free(line);
    if (fclose(file) != 0) fail("could not close root table");
    if (table.count == 0) fail("root table is empty");
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
    trie->nodes[index] = (CompanyNode){
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

static int demand_append(DemandTable *table, DemandRecord record) {
    if (table->count == table->capacity) {
        int capacity = table->capacity == 0 ? 1024 : table->capacity * 2;
        if (capacity < table->capacity) fail("demand table is too large");
        table->values = checked_realloc(
            table->values,
            (size_t)capacity,
            sizeof(*table->values)
        );
        table->capacity = capacity;
    }
    int index = table->count++;
    table->values[index] = record;
    return index;
}

static void leaf_append(LeafTable *table, LeafRecord record) {
    if (table->count == table->capacity) {
        int capacity = table->capacity == 0 ? 1024 : table->capacity * 2;
        if (capacity < table->capacity) fail("leaf table is too large");
        table->values = checked_realloc(
            table->values,
            (size_t)capacity,
            sizeof(*table->values)
        );
        table->capacity = capacity;
    }
    table->values[table->count++] = record;
}

static void attach_recursive_company(
    CompanyTrie *trie,
    const FamilyVector *families,
    DemandTable *demands,
    LeafTable *leaves,
    int root,
    int row,
    int depth,
    int parent_demand,
    int incoming_token
) {
    int demand = demand_append(
        demands,
        (DemandRecord){
            .root = root,
            .depth = depth,
            .row = row,
            .parent_demand = parent_demand,
            .incoming_token = incoming_token,
        }
    );
    const TokenFamily *family = &families->values[depth];
    int position = trie->nodes[row].position + 1;
    for (int index = 0; index < family->count; index++) {
        int token = family->tokens[index];
        int child = trie_child(trie, row, token, position);
        if (depth + 1 == families->count) {
            leaf_append(
                leaves,
                (LeafRecord){
                    .root = root,
                    .row = child,
                    .parent_demand = demand,
                    .incoming_token = token,
                }
            );
        } else {
            attach_recursive_company(
                trie,
                families,
                demands,
                leaves,
                root,
                child,
                depth + 1,
                demand,
                token
            );
        }
    }
}

static void json_fragment(FILE *file, const char *text) {
    if (text == NULL) return;
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
        } else if (value < 0x20 || value >= 0x80) {
            fprintf(file, "\\u%04x", value);
        } else {
            fputc(value, file);
        }
    }
}

static void json_string(FILE *file, const char *text) {
    fputc('"', file);
    json_fragment(file, text);
    fputc('"', file);
}

static void write_int_array(FILE *file, const int *values, int count) {
    fputc('[', file);
    for (int index = 0; index < count; index++) {
        if (index != 0) fputc(',', file);
        fprintf(file, "%d", values[index]);
    }
    fputc(']', file);
}

static void fill_demand_path(
    const DemandTable *demands,
    int demand,
    int *path,
    int depth
) {
    for (int index = 0; index < depth; index++) path[index] = -1;
    for (int current = demand; current >= 0;) {
        const DemandRecord *record = &demands->values[current];
        if (record->depth > 0) {
            path[record->depth - 1] = record->incoming_token;
        }
        current = record->parent_demand;
    }
    for (int index = 0; index < depth; index++) {
        if (path[index] < 0) fail("recursive demand lost its constructor path");
    }
}

static void write_decoded_company(
    FILE *file,
    AtkeyRuntime *runtime,
    const RootRecord *root,
    const int *path,
    int path_count
) {
    fputc('"', file);
    int previous = 0;
    for (int index = 0; index < root->prefix.count; index++) {
        int token = root->prefix.values[index];
        json_fragment(file, atkey_decode(runtime, previous, token));
        previous = token;
    }
    for (int index = 0; index < path_count; index++) {
        int token = path[index];
        json_fragment(file, atkey_decode(runtime, previous, token));
        previous = token;
    }
    fputc('"', file);
}

static void write_candidate_codata(
    FILE *file,
    AtkeyRuntime *runtime,
    const LlamaCompanyResult *result,
    int row,
    const TokenFamily *family,
    int reference_token,
    int previous_token
) {
    if (row < 0 || row >= result->row_count) fail("observation row is invalid");
    const float *logits = result->logits + (size_t)row * result->vocab_size;
    float reference = logits[reference_token];
    fputc('[', file);
    for (int index = 0; index < family->count; index++) {
        if (index != 0) fputc(',', file);
        int token = family->tokens[index];
        fprintf(file, "{\"token\":%d,\"piece\":", token);
        json_string(file, atkey_decode(runtime, previous_token, token));
        fprintf(file, ",\"contrast\":%.9g}", logits[token] - reference);
    }
    fputc(']', file);
}

static Options parse_options(int argc, char **argv) {
    if (argc < 10) {
        fprintf(
            stderr,
            "usage: %s CHECKPOINT TOKENIZER ROOTS_TSV "
            "--family PATH [--family PATH ...] --terminal-family PATH "
            "--trace PATH [--reference-token ID] "
            "[--metal [--metal-library PATH] | --cpu]\n",
            argv[0]
        );
        exit(EXIT_FAILURE);
    }
    Options options = {
        .reference_token = 1,
        .metal_library = "metal_kernels.metallib",
    };
    for (int index = 4; index < argc;) {
        if (strcmp(argv[index], "--family") == 0 && index + 1 < argc) {
            TokenFamily placeholder = {.path = checked_strdup(argv[index + 1])};
            family_vector_append(&options.families, placeholder);
            index += 2;
        } else if (strcmp(argv[index], "--terminal-family") == 0 &&
                   index + 1 < argc) {
            options.terminal_family_path = argv[index + 1];
            index += 2;
        } else if (strcmp(argv[index], "--trace") == 0 && index + 1 < argc) {
            options.trace_path = argv[index + 1];
            index += 2;
        } else if (strcmp(argv[index], "--reference-token") == 0 &&
                   index + 1 < argc) {
            options.reference_token = parse_integer(
                argv[index + 1],
                "reference token"
            );
            index += 2;
        } else if (strcmp(argv[index], "--metal") == 0) {
            options.use_metal = true;
            index++;
        } else if (strcmp(argv[index], "--cpu") == 0) {
            options.use_metal = false;
            index++;
        } else if (strcmp(argv[index], "--metal-library") == 0 &&
                   index + 1 < argc) {
            options.metal_library = argv[index + 1];
            index += 2;
        } else {
            fail("unrecognized option");
        }
    }
    if (options.families.count == 0) fail("at least one --family is required");
    if (options.terminal_family_path == NULL) fail("--terminal-family is required");
    if (options.trace_path == NULL) fail("--trace is required");
    return options;
}

static void free_family(TokenFamily *family) {
    free(family->path);
    free(family->tokens);
}

int main(int argc, char **argv) {
    Options options = parse_options(argc, argv);
    AtkeyRuntime *runtime = atkey_runtime_new(argv[1], argv[2]);
    if (runtime == NULL) fail("could not initialize runtime");
    if (options.use_metal &&
        !atkey_enable_metal(runtime, options.metal_library)) {
        fail("could not initialize Metal backend");
    }
    int vocab_size = atkey_vocab_size(runtime);
    int sequence_length = atkey_sequence_length(runtime);
    if (options.reference_token < 0 || options.reference_token >= vocab_size) {
        fail("reference token is outside vocabulary");
    }

    for (int index = 0; index < options.families.count; index++) {
        char *path = options.families.values[index].path;
        TokenFamily family = read_token_family(path, vocab_size);
        free(path);
        options.families.values[index] = family;
    }
    TokenFamily terminal = read_token_family(
        options.terminal_family_path,
        vocab_size
    );
    RootTable roots = read_roots(
        argv[3],
        vocab_size,
        sequence_length,
        options.families.count
    );

    CompanyTrie trie = {.first_root = -1};
    for (int root = 0; root < roots.count; root++) {
        roots.values[root].leaf = trie_sequence(
            &trie,
            &roots.values[root].prefix
        );
    }
    DemandTable demands = {0};
    LeafTable leaves = {0};
    for (int root = 0; root < roots.count; root++) {
        attach_recursive_company(
            &trie,
            &options.families,
            &demands,
            &leaves,
            root,
            roots.values[root].leaf,
            0,
            -1,
            -1
        );
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
        fail("learned filler was not applied exactly once to the recursive company");
    }

    FILE *trace = fopen(options.trace_path, "wb");
    if (trace == NULL) fail("could not create trace");
    fprintf(
        trace,
        "{\"kind\":\"recursive_company_meta\",\"schema_version\":1,"
        "\"semantics\":\"dependent_polynomial_company_tree\","
        "\"roots\":%d,\"depth\":%d,\"company_rows\":%d,"
        "\"demand_nodes\":%d,\"complete_branches\":%d,"
        "\"reference_token\":%d,\"family_widths\":[",
        roots.count,
        options.families.count,
        trie.count,
        demands.count,
        leaves.count,
        options.reference_token
    );
    for (int depth = 0; depth < options.families.count; depth++) {
        if (depth != 0) fputc(',', trace);
        fprintf(trace, "%d", options.families.values[depth].count);
    }
    fprintf(
        trace,
        "],\"terminal_width\":%d,\"learned_fillers\":%d,"
        "\"family_filler_calls\":%llu,\"maximum_calls_per_filler\":%llu,"
        "\"family_scalar_reads\":%llu,\"backend\":",
        terminal.count,
        filler_count,
        (unsigned long long)total_calls,
        (unsigned long long)maximum_calls,
        (unsigned long long)total_reads
    );
    json_string(trace, atkey_backend_name(runtime));
    fputs(
        ",\"probabilities_used\":false,\"scalar_reward_used\":false,"
        "\"whole_completion_argmax_used\":false,"
        "\"complete_paths_flattened\":false}\n",
        trace
    );
    fflush(trace);

    int *path = checked_calloc(
        (size_t)options.families.count,
        sizeof(*path)
    );
    for (int index = 0; index < demands.count; index++) {
        const DemandRecord *demand = &demands.values[index];
        const RootRecord *root = &roots.values[demand->root];
        fill_demand_path(&demands, index, path, demand->depth);
        fputs("{\"kind\":\"recursive_company_demand\",\"root\":", trace);
        json_string(trace, root->key);
        fprintf(
            trace,
            ",\"depth\":%d,\"row\":%d,\"path_tokens\":",
            demand->depth,
            demand->row
        );
        write_int_array(trace, path, demand->depth);
        fputs(",\"text\":", trace);
        write_decoded_company(trace, runtime, root, path, demand->depth);
        fputs(",\"candidates\":", trace);
        int previous = demand->depth == 0
            ? root->prefix.values[root->prefix.count - 1]
            : path[demand->depth - 1];
        write_candidate_codata(
            trace,
            runtime,
            &result,
            demand->row,
            &options.families.values[demand->depth],
            options.reference_token,
            previous
        );
        fputs("}\n", trace);
        fflush(trace);
    }

    for (int index = 0; index < leaves.count; index++) {
        const LeafRecord *leaf = &leaves.values[index];
        const RootRecord *root = &roots.values[leaf->root];
        const DemandRecord *parent = &demands.values[leaf->parent_demand];
        fill_demand_path(
            &demands,
            leaf->parent_demand,
            path,
            parent->depth
        );
        path[parent->depth] = leaf->incoming_token;
        fputs("{\"kind\":\"recursive_company_terminal\",\"root\":", trace);
        json_string(trace, root->key);
        fprintf(trace, ",\"row\":%d,\"path_tokens\":", leaf->row);
        write_int_array(trace, path, options.families.count);
        fputs(",\"text\":", trace);
        write_decoded_company(
            trace,
            runtime,
            root,
            path,
            options.families.count
        );
        fputs(",\"terminal_candidates\":", trace);
        write_candidate_codata(
            trace,
            runtime,
            &result,
            leaf->row,
            &terminal,
            options.reference_token,
            leaf->incoming_token
        );
        fputs("}\n", trace);
        fflush(trace);
    }
    fprintf(
        trace,
        "{\"kind\":\"recursive_company_check\",\"roots\":%d,"
        "\"depth\":%d,\"demand_nodes\":%d,\"complete_branches\":%d,"
        "\"maximum_calls_per_filler\":%llu}\n",
        roots.count,
        options.families.count,
        demands.count,
        leaves.count,
        (unsigned long long)maximum_calls
    );
    fflush(trace);
    if (fclose(trace) != 0) fail("could not close trace");

    printf(
        "recursive_company roots=%d depth=%d rows=%d demands=%d leaves=%d "
        "fillers=%d max_calls=%llu scalar_reads=%llu backend=%s\n",
        roots.count,
        options.families.count,
        trie.count,
        demands.count,
        leaves.count,
        filler_count,
        (unsigned long long)maximum_calls,
        (unsigned long long)total_reads,
        atkey_backend_name(runtime)
    );

    free(path);
    llama_company_result_free(&result);
    free(reads_before);
    free(calls_before);
    free(parents);
    free(positions);
    free(tokens);
    free(leaves.values);
    free(demands.values);
    free(trie.nodes);
    for (int root = 0; root < roots.count; root++) {
        free(roots.values[root].key);
        free(roots.values[root].prefix.values);
    }
    free(roots.values);
    free_family(&terminal);
    for (int index = 0; index < options.families.count; index++) {
        free_family(&options.families.values[index]);
    }
    free(options.families.values);
    atkey_runtime_free(runtime);
    return 0;
}
