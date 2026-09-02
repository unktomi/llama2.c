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
 * result remains a tree of token-indexed contrast vectors. Once the full tree
 * exists, a memoized dependent selection product compares each candidate x
 * through the x coordinate of its candidate-specific terminal-frontier
 * codata. No path score, probability fold, whole-completion argmax, or hidden
 * local-greedy terminalization is used. Every ballot and witness is flushed.
 */

#include "llama_company.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
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

static int trie_find_child(
    const CompanyTrie *trie,
    int parent,
    int token
) {
    if (parent < 0 || parent >= trie->count) return -1;
    for (int child = trie->nodes[parent].first_child;
         child >= 0;
         child = trie->nodes[child].next_sibling) {
        if (trie->nodes[child].token == token) return child;
    }
    return -1;
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
    int row_count,
    int vocab_size,
    const float *all_logits,
    int row,
    const TokenFamily *family,
    int reference_token,
    int previous_token
) {
    if (row < 0 || row >= row_count) fail("observation row is invalid");
    const float *logits = all_logits + (size_t)row * vocab_size;
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

/* A complete observation is built by continuation composition. A terminal
 * codata observation is passed to the innermost continuation. Each parent
 * continuation prepends exactly the edge observation it owns and passes the
 * enlarged value outward. No edge observation is reduced to a path score. */
typedef struct EdgeObservation EdgeObservation;
struct EdgeObservation {
    int depth;
    int row;
    int token;
    float contrast;
    const EdgeObservation *tail;
};

typedef struct {
    int root;
    int terminal_row;
    const EdgeObservation *edges;
} ComposedObservation;

typedef void (*ComposedContinuationApply)(
    void *environment,
    const ComposedObservation *observation
);

typedef struct {
    ComposedContinuationApply apply;
    void *environment;
} ComposedContinuation;

typedef struct {
    EdgeObservation edge;
    ComposedContinuation next;
    uint64_t *composition_steps;
} ObservationBind;

/* A memo entry is the deforested finite selection product at one dependent
 * constructor demand.  The structured outcome R is represented by the row
 * containing its complete terminal-frontier codata.  At site s and candidate
 * x, the local selector observes only the matching coordinate q_R[x]. */
typedef struct {
    bool computed;
    int selected_token;
    int selected_child_demand;
    int selected_leaf;
    int terminal_row;
    float diagonal_contrast;
    int exact_ties;
} SelectionMemo;

static void prepend_edge_observation(
    void *raw_environment,
    const ComposedObservation *suffix
) {
    ObservationBind *environment = raw_environment;
    EdgeObservation edge = environment->edge;
    edge.tail = suffix->edges;
    ComposedObservation composed = *suffix;
    composed.edges = &edge;
    (*environment->composition_steps)++;
    environment->next.apply(environment->next.environment, &composed);
}

typedef struct {
    AtkeyRuntime *runtime;
    const RootTable *roots;
    const CompanyTrie *trie;
    const DemandTable *demands;
    const LeafTable *leaves;
    const FamilyVector *families;
    const TokenFamily *terminal;
    const int *root_demands;
    const size_t *calls_before;
    const size_t *reads_before;
    int filler_count;
    int reference_token;
    FILE *trace;
    int row_count;
    int vocab_size;
    const float *logits;
    uint64_t root_observer_runs;
    uint64_t composition_steps;
    uint64_t composed_observations;
    SelectionMemo *selection_memo;
    uint64_t selection_nodes;
    uint64_t selection_candidate_evaluations;
    uint64_t selection_root_mates;
    uint64_t selection_exact_tie_nodes;
    uint64_t total_calls;
    uint64_t maximum_calls;
    uint64_t total_reads;
} RecursiveObservationTerm;

typedef struct {
    int token;
    int child_demand;
    int leaf;
    int terminal_row;
    float diagonal_contrast;
} SelectionCandidate;

static int child_demand_index(
    const RecursiveObservationTerm *term,
    int parent_demand,
    int token
) {
    for (int index = 0; index < term->demands->count; index++) {
        const DemandRecord *candidate = &term->demands->values[index];
        if (candidate->parent_demand == parent_demand &&
            candidate->incoming_token == token) {
            return index;
        }
    }
    return -1;
}

static int child_leaf_index(
    const RecursiveObservationTerm *term,
    int parent_demand,
    int token
) {
    for (int index = 0; index < term->leaves->count; index++) {
        const LeafRecord *candidate = &term->leaves->values[index];
        if (candidate->parent_demand == parent_demand &&
            candidate->incoming_token == token) {
            return index;
        }
    }
    return -1;
}

static int selected_continuation_length(
    const RecursiveObservationTerm *term,
    int demand_index
) {
    if (demand_index < 0 || demand_index >= term->demands->count ||
        !term->selection_memo[demand_index].computed) {
        fail("selection continuation is not memoized");
    }
    return term->families->count - term->demands->values[demand_index].depth;
}

static void fill_selected_continuation(
    const RecursiveObservationTerm *term,
    int demand_index,
    int *tokens,
    int token_count
) {
    int output = 0;
    int current = demand_index;
    while (current >= 0) {
        if (current >= term->demands->count ||
            !term->selection_memo[current].computed || output >= token_count) {
            fail("selected continuation is malformed");
        }
        const SelectionMemo *selection = &term->selection_memo[current];
        tokens[output++] = selection->selected_token;
        current = selection->selected_child_demand;
    }
    if (output != token_count) fail("selected continuation length differs");
}

static void write_selection_candidate_path(
    FILE *file,
    const RecursiveObservationTerm *term,
    const SelectionCandidate *candidate
) {
    fputc('[', file);
    fprintf(file, "%d", candidate->token);
    if (candidate->child_demand >= 0) {
        int count = selected_continuation_length(term, candidate->child_demand);
        int *suffix = checked_calloc((size_t)count, sizeof(*suffix));
        fill_selected_continuation(
            term,
            candidate->child_demand,
            suffix,
            count
        );
        for (int index = 0; index < count; index++) {
            fprintf(file, ",%d", suffix[index]);
        }
        free(suffix);
    }
    fputc(']', file);
}

/* Escardo's dependent product, specialized to the retained finite company.
 *
 *   b(x) = select(child_x)
 *   r_x  = p(x, b(x))
 *   eps_s(k) = argmax_x q_{r_x}[x]
 *
 * The continuation result r_x remains the complete codata row.  No edge
 * logit, path probability, or score from another selection site is added.
 * Exact float ties remain visible in the trace; the smallest token id is only
 * a deterministic representative of that retained argmax set. */
static const SelectionMemo *select_dependent_demand(
    RecursiveObservationTerm *term,
    int demand_index
) {
    if (demand_index < 0 || demand_index >= term->demands->count) {
        fail("selection received an invalid demand");
    }
    SelectionMemo *memo = &term->selection_memo[demand_index];
    if (memo->computed) return memo;

    const DemandRecord *demand = &term->demands->values[demand_index];
    const TokenFamily *family = &term->families->values[demand->depth];
    SelectionCandidate *candidates = checked_calloc(
        (size_t)family->count,
        sizeof(*candidates)
    );
    int best = -1;
    for (int index = 0; index < family->count; index++) {
        int token = family->tokens[index];
        int child_demand = -1;
        int leaf = -1;
        int terminal_row = -1;
        if (demand->depth + 1 == term->families->count) {
            leaf = child_leaf_index(term, demand_index, token);
            if (leaf < 0 || leaf >= term->leaves->count) {
                fail("selection lost a terminal outcome");
            }
            terminal_row = term->leaves->values[leaf].row;
        } else {
            child_demand = child_demand_index(term, demand_index, token);
            if (child_demand < 0) {
                fail("selection lost a dependent continuation");
            }
            const SelectionMemo *suffix = select_dependent_demand(
                term,
                child_demand
            );
            terminal_row = suffix->terminal_row;
        }
        if (terminal_row < 0 || terminal_row >= term->row_count) {
            fail("selection produced an invalid terminal outcome row");
        }
        const float *outcome = term->logits +
            (size_t)terminal_row * term->vocab_size;
        float diagonal = outcome[token] - outcome[term->reference_token];
        if (!isfinite(diagonal)) fail("selection produced nonfinite dislike");
        candidates[index] = (SelectionCandidate){
            .token = token,
            .child_demand = child_demand,
            .leaf = leaf,
            .terminal_row = terminal_row,
            .diagonal_contrast = diagonal,
        };
        term->selection_candidate_evaluations++;
        if (best < 0 || diagonal > candidates[best].diagonal_contrast ||
            (diagonal == candidates[best].diagonal_contrast &&
             token < candidates[best].token)) {
            best = index;
        }
    }
    if (best < 0) fail("selection demand has no retained constructors");
    int exact_ties = 0;
    for (int index = 0; index < family->count; index++) {
        if (candidates[index].diagonal_contrast ==
            candidates[best].diagonal_contrast) {
            exact_ties++;
        }
    }
    *memo = (SelectionMemo){
        .computed = true,
        .selected_token = candidates[best].token,
        .selected_child_demand = candidates[best].child_demand,
        .selected_leaf = candidates[best].leaf,
        .terminal_row = candidates[best].terminal_row,
        .diagonal_contrast = candidates[best].diagonal_contrast,
        .exact_ties = exact_ties,
    };
    term->selection_nodes++;
    if (exact_ties > 1) term->selection_exact_tie_nodes++;

    const RootRecord *root = &term->roots->values[demand->root];
    int *prefix = checked_calloc(
        (size_t)term->families->count,
        sizeof(*prefix)
    );
    fill_demand_path(term->demands, demand_index, prefix, demand->depth);
    fputs("{\"kind\":\"recursive_company_selection\",\"root\":", term->trace);
    json_string(term->trace, root->key);
    fprintf(term->trace, ",\"depth\":%d,\"path_tokens\":", demand->depth);
    write_int_array(term->trace, prefix, demand->depth);
    fputs(",\"text\":", term->trace);
    write_decoded_company(
        term->trace,
        term->runtime,
        root,
        prefix,
        demand->depth
    );
    fputs(",\"observer\":\"full_company_diagonal\",\"candidates\":[", term->trace);
    int previous = demand->depth == 0
        ? root->prefix.values[root->prefix.count - 1]
        : prefix[demand->depth - 1];
    for (int index = 0; index < family->count; index++) {
        if (index != 0) fputc(',', term->trace);
        const SelectionCandidate *candidate = &candidates[index];
        fprintf(term->trace, "{\"token\":%d,\"piece\":", candidate->token);
        json_string(
            term->trace,
            atkey_decode(term->runtime, previous, candidate->token)
        );
        fprintf(
            term->trace,
            ",\"terminal_row\":%d,\"diagonal_contrast\":%.9g,"
            "\"selected\":%s,\"continuation_tokens\":",
            candidate->terminal_row,
            candidate->diagonal_contrast,
            index == best ? "true" : "false"
        );
        write_selection_candidate_path(term->trace, term, candidate);
        fputc('}', term->trace);
    }
    fprintf(
        term->trace,
        "],\"selected_token\":%d,\"selected_terminal_row\":%d,"
        "\"selected_diagonal_contrast\":%.9g,\"exact_argmax_size\":%d}\n",
        memo->selected_token,
        memo->terminal_row,
        memo->diagonal_contrast,
        memo->exact_ties
    );
    fflush(term->trace);
    free(prefix);
    free(candidates);
    return memo;
}

static void write_selected_root(
    RecursiveObservationTerm *term,
    int root_index,
    int demand_index
) {
    const SelectionMemo *selection = select_dependent_demand(term, demand_index);
    int depth = selected_continuation_length(term, demand_index);
    int *path = checked_calloc((size_t)depth, sizeof(*path));
    fill_selected_continuation(term, demand_index, path, depth);
    const RootRecord *root = &term->roots->values[root_index];
    fputs(
        "{\"kind\":\"recursive_company_selected_completion\",\"root\":",
        term->trace
    );
    json_string(term->trace, root->key);
    fputs(",\"path_tokens\":", term->trace);
    write_int_array(term->trace, path, depth);
    fputs(",\"text\":", term->trace);
    write_decoded_company(term->trace, term->runtime, root, path, depth);
    fprintf(
        term->trace,
        ",\"terminal_row\":%d,\"selection_semantics\":"
        "\"escardo_dependent_product_full_company_diagonal\","
        "\"position_ballots\":[",
        selection->terminal_row
    );
    const float *outcome = term->logits +
        (size_t)selection->terminal_row * term->vocab_size;
    for (int index = 0; index < depth; index++) {
        if (index != 0) fputc(',', term->trace);
        float diagonal = outcome[path[index]] - outcome[term->reference_token];
        fprintf(
            term->trace,
            "{\"depth\":%d,\"token\":%d,\"diagonal_contrast\":%.9g}",
            index,
            path[index],
            diagonal
        );
    }
    fprintf(term->trace, "]}\n");
    fflush(term->trace);
    term->selection_root_mates++;
    free(path);
}

static void write_composed_observation(
    void *raw_environment,
    const ComposedObservation *observation
) {
    RecursiveObservationTerm *term = raw_environment;
    int depth = term->families->count;
    int *path = checked_calloc((size_t)depth, sizeof(*path));
    float *contrasts = checked_calloc((size_t)depth, sizeof(*contrasts));
    int *rows = checked_calloc((size_t)depth, sizeof(*rows));
    for (int index = 0; index < depth; index++) {
        path[index] = -1;
        rows[index] = -1;
    }
    for (const EdgeObservation *edge = observation->edges;
         edge != NULL;
         edge = edge->tail) {
        if (edge->depth < 0 || edge->depth >= depth ||
            path[edge->depth] != -1) {
            fail("composed observation has an invalid edge spine");
        }
        path[edge->depth] = edge->token;
        contrasts[edge->depth] = edge->contrast;
        rows[edge->depth] = edge->row;
    }
    for (int index = 0; index < depth; index++) {
        if (path[index] < 0 || rows[index] < 0) {
            fail("composed observation lost an edge");
        }
    }

    const RootRecord *root = &term->roots->values[observation->root];
    fputs(
        "{\"kind\":\"recursive_company_composed_observation\",\"root\":",
        term->trace
    );
    json_string(term->trace, root->key);
    fputs(",\"path_tokens\":", term->trace);
    write_int_array(term->trace, path, depth);
    fputs(",\"text\":", term->trace);
    write_decoded_company(
        term->trace,
        term->runtime,
        root,
        path,
        depth
    );
    fputs(",\"edge_observations\":[", term->trace);
    int previous = root->prefix.values[root->prefix.count - 1];
    for (int index = 0; index < depth; index++) {
        if (index != 0) fputc(',', term->trace);
        fprintf(
            term->trace,
            "{\"depth\":%d,\"row\":%d,\"token\":%d,\"piece\":",
            index,
            rows[index],
            path[index]
        );
        json_string(
            term->trace,
            atkey_decode(term->runtime, previous, path[index])
        );
        fprintf(term->trace, ",\"contrast\":%.9g}", contrasts[index]);
        previous = path[index];
    }
    fprintf(
        term->trace,
        "],\"terminal_row\":%d,\"terminal_candidates\":",
        observation->terminal_row
    );
    write_candidate_codata(
        term->trace,
        term->runtime,
        term->row_count,
        term->vocab_size,
        term->logits,
        observation->terminal_row,
        term->terminal,
        term->reference_token,
        previous
    );
    fputs("}\n", term->trace);
    fflush(term->trace);
    term->composed_observations++;
    free(rows);
    free(contrasts);
    free(path);
}

static void observe_demand(
    RecursiveObservationTerm *term,
    int demand_index,
    ComposedContinuation continuation
) {
    if (demand_index < 0 || demand_index >= term->demands->count) {
        fail("composed observation received an invalid demand");
    }
    const DemandRecord *demand = &term->demands->values[demand_index];
    const TokenFamily *family = &term->families->values[demand->depth];
    const float *row_logits = term->logits +
        (size_t)demand->row * term->vocab_size;
    float reference = row_logits[term->reference_token];
    for (int index = 0; index < family->count; index++) {
        int token = family->tokens[index];
        int child_row = trie_find_child(term->trie, demand->row, token);
        if (child_row < 0) fail("composed observation lost a constructor edge");
        ObservationBind bind = {
            .edge = {
                .depth = demand->depth,
                .row = demand->row,
                .token = token,
                .contrast = row_logits[token] - reference,
            },
            .next = continuation,
            .composition_steps = &term->composition_steps,
        };
        ComposedContinuation child_continuation = {
            .apply = prepend_edge_observation,
            .environment = &bind,
        };
        if (demand->depth + 1 == term->families->count) {
            int leaf_index = child_leaf_index(term, demand_index, token);
            if (leaf_index < 0 || leaf_index >= term->leaves->count) {
                fail("composed observation lost its terminal codata");
            }
            const LeafRecord *leaf = &term->leaves->values[leaf_index];
            if (leaf->row != child_row) {
                fail("composed terminal edge has inconsistent causal row");
            }
            ComposedObservation terminal = {
                .root = leaf->root,
                .terminal_row = leaf->row,
            };
            child_continuation.apply(
                child_continuation.environment,
                &terminal
            );
        } else {
            int child_demand = child_demand_index(term, demand_index, token);
            if (child_demand < 0) {
                fail("composed observation lost its child continuation");
            }
            if (term->demands->values[child_demand].row != child_row) {
                fail("composed child edge has inconsistent causal row");
            }
            observe_demand(term, child_demand, child_continuation);
        }
    }
}

static bool observe_recursive_company(
    void *raw_environment,
    int row_count,
    int vocab_size,
    const float *logits
) {
    RecursiveObservationTerm *term = raw_environment;
    if (term->root_observer_runs != 0 || row_count != term->trie->count ||
        vocab_size != atkey_vocab_size(term->runtime) || logits == NULL) {
        return false;
    }
    term->root_observer_runs++;
    term->row_count = row_count;
    term->vocab_size = vocab_size;
    term->logits = logits;

    for (int filler = 0; filler < term->filler_count; filler++) {
        size_t calls = atkey_filler_calls(term->runtime, filler) -
            term->calls_before[filler];
        size_t reads = atkey_filler_scalar_reads(term->runtime, filler) -
            term->reads_before[filler];
        term->total_calls += calls;
        term->total_reads += reads;
        if (calls > term->maximum_calls) term->maximum_calls = calls;
    }
    if (term->maximum_calls != 1 ||
        term->total_calls != (uint64_t)term->filler_count) {
        return false;
    }

    fprintf(
        term->trace,
        "{\"kind\":\"recursive_company_meta\",\"schema_version\":1,"
        "\"semantics\":\"dependent_polynomial_company_tree\","
        "\"observation_semantics\":\"continuation_composed_codata\","
        "\"roots\":%d,\"depth\":%d,\"company_rows\":%d,"
        "\"demand_nodes\":%d,\"complete_branches\":%d,"
        "\"reference_token\":%d,\"family_widths\":[",
        term->roots->count,
        term->families->count,
        term->trie->count,
        term->demands->count,
        term->leaves->count,
        term->reference_token
    );
    for (int depth = 0; depth < term->families->count; depth++) {
        if (depth != 0) fputc(',', term->trace);
        fprintf(term->trace, "%d", term->families->values[depth].count);
    }
    fprintf(
        term->trace,
        "],\"terminal_width\":%d,\"learned_fillers\":%d,"
        "\"family_filler_calls\":%llu,\"maximum_calls_per_filler\":%llu,"
        "\"family_scalar_reads\":%llu,\"backend\":",
        term->terminal->count,
        term->filler_count,
        (unsigned long long)term->total_calls,
        (unsigned long long)term->maximum_calls,
        (unsigned long long)term->total_reads
    );
    json_string(term->trace, atkey_backend_name(term->runtime));
    fputs(
        ",\"codata_constructed_before_observation\":true,"
        "\"root_observer_runs\":1,\"observations_composed\":true,"
        "\"probabilities_used\":false,\"scalar_reward_used\":false,"
        "\"whole_completion_argmax_used\":false,"
        "\"complete_paths_flattened\":false,"
        "\"completion_selected\":true,"
        "\"selection_semantics\":"
        "\"escardo_dependent_product_full_company_diagonal\","
        "\"local_edge_logits_terminalized\":false,"
        "\"path_likelihoods_summed\":false}\n",
        term->trace
    );
    fflush(term->trace);

    /* Retain the individual edge records as an audit of the operands. They
     * are emitted inside the same root observation, not by separately
     * forcing the codata. */
    int *path = checked_calloc(
        (size_t)term->families->count,
        sizeof(*path)
    );
    for (int index = 0; index < term->demands->count; index++) {
        const DemandRecord *demand = &term->demands->values[index];
        const RootRecord *root = &term->roots->values[demand->root];
        fill_demand_path(term->demands, index, path, demand->depth);
        fputs("{\"kind\":\"recursive_company_demand\",\"root\":", term->trace);
        json_string(term->trace, root->key);
        fprintf(
            term->trace,
            ",\"depth\":%d,\"row\":%d,\"path_tokens\":",
            demand->depth,
            demand->row
        );
        write_int_array(term->trace, path, demand->depth);
        fputs(",\"text\":", term->trace);
        write_decoded_company(
            term->trace,
            term->runtime,
            root,
            path,
            demand->depth
        );
        fputs(",\"candidates\":", term->trace);
        int previous = demand->depth == 0
            ? root->prefix.values[root->prefix.count - 1]
            : path[demand->depth - 1];
        write_candidate_codata(
            term->trace,
            term->runtime,
            row_count,
            vocab_size,
            logits,
            demand->row,
            &term->families->values[demand->depth],
            term->reference_token,
            previous
        );
        fputs("}\n", term->trace);
        fflush(term->trace);
    }

    for (int index = 0; index < term->leaves->count; index++) {
        const LeafRecord *leaf = &term->leaves->values[index];
        const RootRecord *root = &term->roots->values[leaf->root];
        const DemandRecord *parent =
            &term->demands->values[leaf->parent_demand];
        fill_demand_path(
            term->demands,
            leaf->parent_demand,
            path,
            parent->depth
        );
        path[parent->depth] = leaf->incoming_token;
        fputs("{\"kind\":\"recursive_company_terminal\",\"root\":", term->trace);
        json_string(term->trace, root->key);
        fprintf(term->trace, ",\"row\":%d,\"path_tokens\":", leaf->row);
        write_int_array(term->trace, path, term->families->count);
        fputs(",\"text\":", term->trace);
        write_decoded_company(
            term->trace,
            term->runtime,
            root,
            path,
            term->families->count
        );
        fputs(",\"terminal_candidates\":", term->trace);
        write_candidate_codata(
            term->trace,
            term->runtime,
            row_count,
            vocab_size,
            logits,
            leaf->row,
            term->terminal,
            term->reference_token,
            leaf->incoming_token
        );
        fputs("}\n", term->trace);
        fflush(term->trace);
    }
    free(path);

    ComposedContinuation root_continuation = {
        .apply = write_composed_observation,
        .environment = term,
    };
    for (int root = 0; root < term->roots->count; root++) {
        observe_demand(
            term,
            term->root_demands[root],
            root_continuation
        );
    }
    uint64_t expected_steps = (uint64_t)term->leaves->count *
        (uint64_t)term->families->count;
    if (term->composed_observations != (uint64_t)term->leaves->count ||
        term->composition_steps != expected_steps) {
        return false;
    }
    term->selection_memo = checked_calloc(
        (size_t)term->demands->count,
        sizeof(*term->selection_memo)
    );
    for (int root = 0; root < term->roots->count; root++) {
        write_selected_root(term, root, term->root_demands[root]);
    }
    uint64_t expected_selection_candidates = 0;
    for (int index = 0; index < term->demands->count; index++) {
        int depth = term->demands->values[index].depth;
        expected_selection_candidates +=
            (uint64_t)term->families->values[depth].count;
    }
    if (term->selection_nodes != (uint64_t)term->demands->count ||
        term->selection_candidate_evaluations != expected_selection_candidates ||
        term->selection_root_mates != (uint64_t)term->roots->count) {
        free(term->selection_memo);
        term->selection_memo = NULL;
        return false;
    }
    fprintf(
        term->trace,
        "{\"kind\":\"recursive_company_check\",\"roots\":%d,"
        "\"depth\":%d,\"demand_nodes\":%d,\"complete_branches\":%d,"
        "\"maximum_calls_per_filler\":%llu,\"root_observer_runs\":%llu,"
        "\"composed_observations\":%llu,\"composition_steps\":%llu,"
        "\"selection_nodes\":%llu,\"selection_candidate_evaluations\":%llu,"
        "\"selection_root_mates\":%llu,\"selection_exact_tie_nodes\":%llu}\n",
        term->roots->count,
        term->families->count,
        term->demands->count,
        term->leaves->count,
        (unsigned long long)term->maximum_calls,
        (unsigned long long)term->root_observer_runs,
        (unsigned long long)term->composed_observations,
        (unsigned long long)term->composition_steps,
        (unsigned long long)term->selection_nodes,
        (unsigned long long)term->selection_candidate_evaluations,
        (unsigned long long)term->selection_root_mates,
        (unsigned long long)term->selection_exact_tie_nodes
    );
    fflush(term->trace);
    free(term->selection_memo);
    term->selection_memo = NULL;
    return true;
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
    FILE *trace = fopen(options.trace_path, "wb");
    if (trace == NULL) fail("could not create trace");

    int *root_demands = checked_calloc(
        (size_t)roots.count,
        sizeof(*root_demands)
    );
    for (int root = 0; root < roots.count; root++) root_demands[root] = -1;
    for (int index = 0; index < demands.count; index++) {
        const DemandRecord *demand = &demands.values[index];
        if (demand->depth != 0) continue;
        if (demand->root < 0 || demand->root >= roots.count ||
            root_demands[demand->root] != -1) {
            fail("recursive company has an invalid root demand");
        }
        root_demands[demand->root] = index;
    }
    for (int root = 0; root < roots.count; root++) {
        if (root_demands[root] < 0) fail("recursive company lost a root demand");
    }

    LlamaCompanyCodata codata;
    if (!llama_company_codata_construct(runtime, &shape, false, &codata)) {
        fail("causal company codata construction failed");
    }
    RecursiveObservationTerm observation = {
        .runtime = runtime,
        .roots = &roots,
        .trie = &trie,
        .demands = &demands,
        .leaves = &leaves,
        .families = &options.families,
        .terminal = &terminal,
        .root_demands = root_demands,
        .calls_before = calls_before,
        .reads_before = reads_before,
        .filler_count = filler_count,
        .reference_token = options.reference_token,
        .trace = trace,
    };
    if (!llama_company_codata_observe(
            &codata,
            observe_recursive_company,
            &observation
        )) {
        fail("composed root observation failed");
    }
    if (fclose(trace) != 0) fail("could not close trace");

    printf(
        "recursive_company roots=%d depth=%d rows=%d demands=%d leaves=%d "
        "fillers=%d max_calls=%llu scalar_reads=%llu root_runs=%llu "
        "composed=%llu backend=%s\n",
        roots.count,
        options.families.count,
        trie.count,
        demands.count,
        leaves.count,
        filler_count,
        (unsigned long long)observation.maximum_calls,
        (unsigned long long)observation.total_reads,
        (unsigned long long)observation.root_observer_runs,
        (unsigned long long)observation.composed_observations,
        atkey_backend_name(runtime)
    );

    llama_company_codata_free(&codata);
    free(root_demands);
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
