/*
 * DO NOT USE AS THE ESCARDO RUNTIME.
 *
 * This quarantined attempt resumes previously reached Selects, but completes
 * every newly introduced prefix with the fixed hidden-feedback default before
 * observing that prefix.  Its sampled subtree therefore couples a useful new
 * earlier token to an unrelated neutral suffix.  The measured six-token,
 * four-sweep tree contained 84 leaves, but all inherited malformed tails; the
 * selected `Lily was sick a a sick sick.` is evidence of this scheduler bug,
 * not a successful Firthian selection.
 *
 * Resumable selection-product interpreter over llama2.c hidden feedback.
 *
 * This file deliberately does not construct complete candidate paths before
 * observation. A local Select demands one argument, its nested continuation
 * eventually suspends at a complete company observation, and the frozen model
 * evaluates all suspensions reached in that round as one family. The returned
 * prefix-owned covectors resume the same memoized Selects and provide their
 * next demanded arguments. Only the last round terminalizes the root witness.
 *
 * Numerical transformer code is inherited from the standalone hidden-feedback
 * reference. Its CLI is excluded; this file owns only the higher-order control
 * path and delegates candidate-company evaluation to llama_company.c.
 */

#define TESTING
#include "run_hidden_feedback.c"
#undef TESTING

#include "llama_company.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>

typedef struct {
    int token;
    int fixed_rank;
    float fixed_logit;
    double fixed_log_probability;
} SelectCandidate;

typedef struct {
    int position;
    int count;
    SelectCandidate *candidates;
    int *by_token;
} SelectFrame;

typedef struct {
    int count;
    double *coordinates;
    int *leximin_positions;
    double *leximin_coordinates;
    int *context_nodes;
    int *company_nodes;
    int *company_tokens;
    double *logits;
    double *log_partitions;
} SelectOutcome;

typedef enum {
    DEMAND_SELECT_UNIT,
    DEMAND_INITIAL_HIDDEN_FEEDBACK,
    DEMAND_RESUMED_CONTEXT_COVECTOR,
} DemandSource;

typedef enum {
    SELECT_UNFORCED,
    SELECT_FORCING,
    SELECT_FORCED,
} SelectState;

typedef struct {
    int parent;
    int position;
    int candidate_index;
    int demand_rank;
    DemandSource demand_source;
    int *children;
    int child_count;
    int child_capacity;
    unsigned long long reachability;
    float *context_logits;
    int context_logits_ready;
    double log_partition;
    SelectOutcome *outcome;
    SelectState state;
    int selected_child;
    int selected_leaf;
} SelectNode;

typedef struct {
    SelectNode *nodes;
    int count;
    int capacity;
    unsigned long long leaf_count;
} SelectTerm;

typedef struct {
    unsigned long long continuation_demands;
    unsigned long long company_batches;
    unsigned long long company_rows;
    unsigned long long family_filler_calls;
    unsigned long long family_scalar_reads;
    unsigned long long maximum_calls_per_filler;
    unsigned long long strength_nodes;
    unsigned long long candidate_ratings;
    unsigned long long structured_outcomes;
    unsigned long long root_terminalizations;
    unsigned long long strength_filler_calls;
    unsigned long long strength_scalar_reads;
    unsigned long long company_nanoseconds;
    unsigned long long strength_nanoseconds;
} SelectCounters;

typedef struct {
    SelectFrame *frames;
    int frame_count;
    int prompt_count;
    int vocab_size;
    int maximum_demands_per_select;
    unsigned long long leaf_safety_limit;
    unsigned long long tie_seed;
    Tokenizer *tokenizer;
    AtkeyRuntime *company_runtime;
    FILE *trace;
    SelectTerm term;
    SelectCounters counters;
    size_t *filler_calls_by_id;
} SelectRuntime;

typedef struct {
    double coordinate;
    int position;
} OrderedCoordinate;

typedef struct {
    int candidate_index;
    int demand_rank;
    DemandSource source;
} CandidateDemand;

static void select_fail(const char *message) {
    fprintf(stderr, "resumable Select: %s\n", message);
    exit(EXIT_FAILURE);
}

static void *select_calloc(size_t count, size_t width) {
    if (width != 0 && count > SIZE_MAX / width) {
        select_fail("allocation size overflow");
    }
    void *memory = calloc(count, width);
    if (memory == NULL) select_fail("allocation failed");
    return memory;
}

static unsigned long long monotonic_nanoseconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        select_fail("could not read monotonic clock");
    }
    return (unsigned long long)value.tv_sec * 1000000000ULL +
        (unsigned long long)value.tv_nsec;
}

static uint64_t mix_u64(uint64_t value) {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31;
    return value;
}

static const char *demand_source_name(DemandSource source) {
    switch (source) {
        case DEMAND_SELECT_UNIT: return "select_unit";
        case DEMAND_INITIAL_HIDDEN_FEEDBACK:
            return "initial_hidden_feedback_demand";
        case DEMAND_RESUMED_CONTEXT_COVECTOR:
            return "resumed_callback_context_covector";
    }
    return "invalid";
}

static void json_string(FILE *stream, const char *value) {
    fputc('"', stream);
    if (value != NULL) {
        for (const unsigned char *cursor = (const unsigned char *)value;
             *cursor != '\0'; cursor++) {
            unsigned char byte = *cursor;
            if (byte == '"') fputs("\\\"", stream);
            else if (byte == '\\') fputs("\\\\", stream);
            else if (byte == '\n') fputs("\\n", stream);
            else if (byte == '\r') fputs("\\r", stream);
            else if (byte == '\t') fputs("\\t", stream);
            else if (byte < 0x20 || byte >= 0x80) {
                fprintf(stream, "\\u%04x", byte);
            } else {
                fputc(byte, stream);
            }
        }
    }
    fputc('"', stream);
}

static void json_piece(FILE *stream, Tokenizer *tokenizer, int token) {
    char *piece = decode(tokenizer, 0, token);
    if (piece != NULL && piece[0] != '\0' && piece[1] == '\0' &&
        (unsigned char)piece[0] >= 0x80) {
        fprintf(stream, "\"\\u%04x\"", (unsigned char)piece[0]);
    } else {
        json_string(stream, piece);
    }
}

static double log_partition(const float *logits, int count) {
    double maximum = -INFINITY;
    for (int index = 0; index < count; index++) {
        if ((double)logits[index] > maximum) maximum = logits[index];
    }
    double total = 0.0;
    for (int index = 0; index < count; index++) {
        total += exp((double)logits[index] - maximum);
    }
    if (!isfinite(maximum) || !(total > 0.0) || !isfinite(total)) {
        select_fail("invalid logit covector");
    }
    return maximum + log(total);
}

static int candidate_compare(const void *left_value, const void *right_value) {
    const SelectCandidate *left = left_value;
    const SelectCandidate *right = right_value;
    if (left->fixed_logit > right->fixed_logit) return -1;
    if (left->fixed_logit < right->fixed_logit) return 1;
    if (left->token < right->token) return -1;
    if (left->token > right->token) return 1;
    return 0;
}

static int ordered_coordinate_compare(
    const void *left_value,
    const void *right_value
) {
    const OrderedCoordinate *left = left_value;
    const OrderedCoordinate *right = right_value;
    if (left->coordinate < right->coordinate) return -1;
    if (left->coordinate > right->coordinate) return 1;
    if (left->position < right->position) return -1;
    if (left->position > right->position) return 1;
    return 0;
}

static int term_add_node(
    SelectTerm *term,
    int parent,
    int position,
    int candidate_index,
    int demand_rank,
    DemandSource source
) {
    if (term->count == term->capacity) {
        int capacity = term->capacity == 0 ? 1024 : term->capacity * 2;
        if (capacity <= term->capacity) select_fail("term capacity overflow");
        SelectNode *nodes = realloc(
            term->nodes,
            (size_t)capacity * sizeof(*nodes)
        );
        if (nodes == NULL) select_fail("could not grow selection term");
        term->nodes = nodes;
        term->capacity = capacity;
    }
    int index = term->count++;
    term->nodes[index] = (SelectNode){
        .parent = parent,
        .position = position,
        .candidate_index = candidate_index,
        .demand_rank = demand_rank,
        .demand_source = source,
        .selected_child = -1,
        .selected_leaf = -1,
    };
    return index;
}

static int term_append_child(
    SelectTerm *term,
    int parent,
    int position,
    CandidateDemand demand
) {
    int child = term_add_node(
        term,
        parent,
        position,
        demand.candidate_index,
        demand.demand_rank,
        demand.source
    );
    SelectNode *node = &term->nodes[parent];
    if (node->child_count == node->child_capacity) {
        int capacity = node->child_capacity == 0 ? 4 : node->child_capacity * 2;
        if (capacity <= node->child_capacity) {
            select_fail("child capacity overflow");
        }
        int *children = realloc(
            node->children,
            (size_t)capacity * sizeof(*children)
        );
        if (children == NULL) select_fail("could not grow Select arguments");
        node->children = children;
        node->child_capacity = capacity;
    }
    node->children[node->child_count++] = child;
    return child;
}

static int term_child_has_token(
    const SelectRuntime *runtime,
    int parent,
    int position,
    int token
) {
    const SelectNode *node = &runtime->term.nodes[parent];
    const SelectFrame *frame = &runtime->frames[position];
    for (int ordinal = 0; ordinal < node->child_count; ordinal++) {
        const SelectNode *child =
            &runtime->term.nodes[node->children[ordinal]];
        if (frame->candidates[child->candidate_index].token == token) return 1;
    }
    return 0;
}

static int context_rank(
    const float *logits,
    int vocab_size,
    int token,
    uint64_t tie_seed
) {
    int rank = 1;
    uint64_t token_tie = mix_u64(tie_seed ^ (uint64_t)(unsigned)token);
    for (int other = 0; other < vocab_size; other++) {
        if (logits[other] > logits[token]) rank++;
        else if (logits[other] == logits[token] && other != token &&
                 mix_u64(tie_seed ^ (uint64_t)(unsigned)other) < token_tie) {
            rank++;
        }
    }
    return rank;
}

static CandidateDemand next_demand(
    SelectRuntime *runtime,
    int parent,
    int position
) {
    SelectFrame *frame = &runtime->frames[position];
    if (frame->count == 1) {
        return (CandidateDemand){0, 1, DEMAND_SELECT_UNIT};
    }
    SelectNode *context = &runtime->term.nodes[parent];
    if (!context->context_logits_ready) {
        for (int index = 0; index < frame->count; index++) {
            int token = frame->candidates[index].token;
            if (!term_child_has_token(runtime, parent, position, token)) {
                return (CandidateDemand){
                    index,
                    frame->candidates[index].fixed_rank,
                    DEMAND_INITIAL_HIDDEN_FEEDBACK,
                };
            }
        }
    } else {
        int selected = -1;
        float selected_logit = -INFINITY;
        uint64_t selected_tie = UINT64_MAX;
        for (int token = 0; token < runtime->vocab_size; token++) {
            if (term_child_has_token(runtime, parent, position, token)) continue;
            float logit = context->context_logits[token];
            uint64_t tie = mix_u64(runtime->tie_seed ^ (uint64_t)(unsigned)token);
            if (selected < 0 || logit > selected_logit ||
                (logit == selected_logit && tie < selected_tie)) {
                selected = token;
                selected_logit = logit;
                selected_tie = tie;
            }
        }
        if (selected >= 0) {
            int candidate_index = frame->by_token[selected];
            if (candidate_index < 0 || candidate_index >= frame->count) {
                select_fail("context covector escaped its token frame");
            }
            return (CandidateDemand){
                candidate_index,
                context_rank(
                    context->context_logits,
                    runtime->vocab_size,
                    selected,
                    runtime->tie_seed
                ),
                DEMAND_RESUMED_CONTEXT_COVECTOR,
            };
        }
    }
    select_fail("Select exhausted its vocabulary");
    return (CandidateDemand){0};
}

static void trace_path(
    FILE *stream,
    const SelectRuntime *runtime,
    int leaf
) {
    int *nodes = select_calloc(
        (size_t)runtime->frame_count,
        sizeof(*nodes)
    );
    for (int node = leaf; node != 0; node = runtime->term.nodes[node].parent) {
        int position = runtime->term.nodes[node].position;
        if (position < 0 || position >= runtime->frame_count) {
            select_fail("path contains an invalid position");
        }
        nodes[position] = node;
    }
    fputc('"', stream);
    int previous = -1;
    for (int position = 0; position < runtime->frame_count; position++) {
        int node = nodes[position];
        if (node <= 0) select_fail("path lost a token occurrence");
        const SelectNode *entry = &runtime->term.nodes[node];
        const SelectFrame *frame = &runtime->frames[position];
        int token = frame->candidates[entry->candidate_index].token;
        char *piece = decode(runtime->tokenizer, previous < 0 ? 0 : previous, token);
        if (piece != NULL) {
            for (const unsigned char *cursor = (const unsigned char *)piece;
                 *cursor != '\0'; cursor++) {
                unsigned char byte = *cursor;
                if (byte == '"') fputs("\\\"", stream);
                else if (byte == '\\') fputs("\\\\", stream);
                else if (byte == '\n') fputs("\\n", stream);
                else if (byte == '\r') fputs("\\r", stream);
                else if (byte == '\t') fputs("\\t", stream);
                else if (byte < 0x20 || byte >= 0x80) {
                    fprintf(stream, "\\u%04x", byte);
                } else fputc(byte, stream);
            }
        }
        previous = token;
    }
    fputc('"', stream);
    free(nodes);
}

static void trace_demand(
    SelectRuntime *runtime,
    int round,
    int parent,
    int child
) {
    if (runtime->trace == NULL) return;
    SelectNode *entry = &runtime->term.nodes[child];
    SelectCandidate *candidate =
        &runtime->frames[entry->position].candidates[entry->candidate_index];
    fprintf(
        runtime->trace,
        "{\"event\":\"continuation_demand\",\"round\":%d,"
        "\"context_node\":%d,\"node\":%d,\"position\":%d,"
        "\"token\":%d,\"demand_rank\":%d,\"fixed_tape_rank\":%d,"
        "\"source\":\"%s\",\"piece\":",
        round,
        parent,
        child,
        entry->position,
        candidate->token,
        entry->demand_rank,
        candidate->fixed_rank,
        demand_source_name(entry->demand_source)
    );
    json_piece(runtime->trace, runtime->tokenizer, candidate->token);
    fputs("}\n", runtime->trace);
    fflush(runtime->trace);
}

static void resume_round(
    SelectRuntime *runtime,
    int parent,
    int position,
    int round
) {
    if (position == runtime->frame_count) return;
    SelectFrame *frame = &runtime->frames[position];
    int limit = frame->count == 1 ? 1 : runtime->maximum_demands_per_select;
    if (limit <= 0 || limit > frame->count) limit = frame->count;
    if (runtime->term.nodes[parent].child_count < limit) {
        CandidateDemand demand = next_demand(runtime, parent, position);
        int child = term_append_child(
            &runtime->term,
            parent,
            position,
            demand
        );
        runtime->counters.continuation_demands++;
        if (position == runtime->frame_count - 1) runtime->term.leaf_count++;
        trace_demand(runtime, round, parent, child);
    }

    int child_count = runtime->term.nodes[parent].child_count;
    int *children = select_calloc((size_t)child_count, sizeof(*children));
    memcpy(
        children,
        runtime->term.nodes[parent].children,
        (size_t)child_count * sizeof(*children)
    );
    for (int ordinal = 0; ordinal < child_count; ordinal++) {
        resume_round(runtime, children[ordinal], position + 1, round);
    }
    free(children);
}

static unsigned long long update_reachability(SelectTerm *term, int node_index) {
    SelectNode *node = &term->nodes[node_index];
    if (node->child_count == 0) {
        node->reachability = node_index == 0 ? 0 : 1;
        return node->reachability;
    }
    unsigned long long total = 0;
    for (int ordinal = 0; ordinal < node->child_count; ordinal++) {
        unsigned long long child = update_reachability(
            term,
            node->children[ordinal]
        );
        if (ULLONG_MAX - total < child) select_fail("reachability overflow");
        total += child;
    }
    node->reachability = total;
    return total;
}

static void outcome_free(SelectOutcome *outcome) {
    if (outcome == NULL) return;
    free(outcome->log_partitions);
    free(outcome->logits);
    free(outcome->company_tokens);
    free(outcome->company_nodes);
    free(outcome->context_nodes);
    free(outcome->leximin_coordinates);
    free(outcome->leximin_positions);
    free(outcome->coordinates);
    free(outcome);
}

static void clear_round_outcomes(SelectRuntime *runtime) {
    for (int index = 0; index < runtime->term.count; index++) {
        SelectNode *node = &runtime->term.nodes[index];
        outcome_free(node->outcome);
        node->outcome = NULL;
        node->state = SELECT_UNFORCED;
        node->selected_child = -1;
        node->selected_leaf = -1;
    }
}

static void path_nodes(
    const SelectRuntime *runtime,
    int leaf,
    int *nodes
) {
    for (int position = 0; position < runtime->frame_count; position++) {
        nodes[position] = -1;
    }
    for (int node = leaf; node != 0; node = runtime->term.nodes[node].parent) {
        int position = runtime->term.nodes[node].position;
        if (position < 0 || position >= runtime->frame_count) {
            select_fail("leaf path contains an invalid position");
        }
        nodes[position] = node;
    }
    for (int position = 0; position < runtime->frame_count; position++) {
        if (nodes[position] <= 0) select_fail("leaf path is incomplete");
    }
}

static SelectOutcome *observe_leaf(
    SelectRuntime *runtime,
    int leaf,
    int *nodes
) {
    path_nodes(runtime, leaf, nodes);
    SelectOutcome *outcome = select_calloc(1, sizeof(*outcome));
    int count = runtime->frame_count;
    outcome->count = count;
    outcome->coordinates = select_calloc((size_t)count, sizeof(double));
    outcome->leximin_positions = select_calloc((size_t)count, sizeof(int));
    outcome->leximin_coordinates = select_calloc((size_t)count, sizeof(double));
    outcome->context_nodes = select_calloc((size_t)count, sizeof(int));
    outcome->company_nodes = select_calloc((size_t)count, sizeof(int));
    outcome->company_tokens = select_calloc((size_t)count, sizeof(int));
    outcome->logits = select_calloc((size_t)count, sizeof(double));
    outcome->log_partitions = select_calloc((size_t)count, sizeof(double));

    OrderedCoordinate *ordered = select_calloc(
        (size_t)count,
        sizeof(*ordered)
    );
    for (int position = 0; position < count; position++) {
        int company_node = nodes[position];
        SelectNode *company = &runtime->term.nodes[company_node];
        SelectFrame *frame = &runtime->frames[position];
        int token = frame->candidates[company->candidate_index].token;
        int context_node = company->parent;
        outcome->context_nodes[position] = context_node;
        outcome->company_nodes[position] = company_node;
        outcome->company_tokens[position] = token;
        if (context_node == 0) {
            outcome->coordinates[position] = 0.0;
        } else {
            SelectNode *context = &runtime->term.nodes[context_node];
            if (!context->context_logits_ready ||
                !isfinite(context->log_partition)) {
                select_fail("company outcome lacks its incoming context");
            }
            double logit = context->context_logits[token];
            double coordinate = logit - context->log_partition;
            if (!isfinite(coordinate)) select_fail("non-finite company opinion");
            outcome->logits[position] = logit;
            outcome->log_partitions[position] = context->log_partition;
            outcome->coordinates[position] = coordinate;
        }
        ordered[position] = (OrderedCoordinate){
            outcome->coordinates[position],
            position,
        };
    }
    qsort(
        ordered,
        (size_t)count,
        sizeof(*ordered),
        ordered_coordinate_compare
    );
    for (int rank = 0; rank < count; rank++) {
        outcome->leximin_positions[rank] = ordered[rank].position;
        outcome->leximin_coordinates[rank] = ordered[rank].coordinate;
    }
    free(ordered);
    runtime->term.nodes[leaf].outcome = outcome;
    return outcome;
}

static int outcome_compare(
    const SelectOutcome *left,
    const SelectOutcome *right
) {
    if (left == NULL || right == NULL || left->count != right->count) {
        select_fail("cannot compare incompatible outcomes");
    }
    for (int rank = 0; rank < left->count; rank++) {
        if (left->leximin_coordinates[rank] >
            right->leximin_coordinates[rank]) return 1;
        if (left->leximin_coordinates[rank] <
            right->leximin_coordinates[rank]) return -1;
    }
    return 0;
}

static void trace_outcome(
    SelectRuntime *runtime,
    int round,
    int leaf,
    const SelectOutcome *outcome
) {
    if (runtime->trace == NULL) return;
    FILE *stream = runtime->trace;
    fprintf(
        stream,
        "{\"event\":\"company_outcome\",\"round\":%d,"
        "\"leaf_node\":%d,\"coordinates\":[",
        round,
        leaf
    );
    for (int position = 0; position < outcome->count; position++) {
        if (position != 0) fputc(',', stream);
        fprintf(stream, "%.17g", outcome->coordinates[position]);
    }
    fputs("],\"leximin\":[", stream);
    for (int rank = 0; rank < outcome->count; rank++) {
        if (rank != 0) fputc(',', stream);
        fprintf(
            stream,
            "{\"rank\":%d,\"position\":%d,\"opinion\":%.17g}",
            rank,
            outcome->leximin_positions[rank],
            outcome->leximin_coordinates[rank]
        );
    }
    fputs("],\"text\":", stream);
    trace_path(stream, runtime, leaf);
    fputs("}\n", stream);
    fflush(stream);
}

static void trace_choice(
    SelectRuntime *runtime,
    const char *event,
    int round,
    int parent,
    int child,
    int selected_leaf,
    const SelectOutcome *outcome
) {
    if (runtime->trace == NULL) return;
    SelectNode *entry = &runtime->term.nodes[child];
    SelectCandidate *candidate =
        &runtime->frames[entry->position].candidates[entry->candidate_index];
    fprintf(
        runtime->trace,
        "{\"event\":\"%s\",\"round\":%d,\"frame\":%d,"
        "\"position\":%d,\"token\":%d,\"demand_rank\":%d,"
        "\"fixed_tape_rank\":%d,\"observer_leaf\":%d,"
        "\"worst_opinion\":%.17g,\"text\":",
        event,
        round,
        parent,
        entry->position,
        candidate->token,
        entry->demand_rank,
        candidate->fixed_rank,
        selected_leaf,
        outcome->leximin_coordinates[0]
    );
    trace_path(runtime->trace, runtime, selected_leaf);
    fputs("}\n", runtime->trace);
    fflush(runtime->trace);
}

static int force_select(SelectRuntime *runtime, int node_index, int round) {
    SelectNode *node = &runtime->term.nodes[node_index];
    if (node->state == SELECT_FORCED) return node->selected_leaf;
    if (node->state == SELECT_FORCING) select_fail("recursive selection cycle");
    node->state = SELECT_FORCING;
    if (node->child_count == 0) {
        if (node->outcome == NULL) select_fail("unobserved terminal continuation");
        node->selected_leaf = node_index;
        node->state = SELECT_FORCED;
        return node_index;
    }

    runtime->counters.strength_nodes++;
    int best_child = -1;
    SelectOutcome *best_outcome = NULL;
    for (int ordinal = 0; ordinal < node->child_count; ordinal++) {
        int child = node->children[ordinal];
        int leaf = force_select(runtime, child, round);
        SelectOutcome *outcome = runtime->term.nodes[leaf].outcome;
        if (outcome == NULL) select_fail("continuation returned no outcome");
        runtime->counters.candidate_ratings++;
        trace_choice(
            runtime,
            "candidate_rated",
            round,
            node_index,
            child,
            leaf,
            outcome
        );
        int order = best_outcome == NULL
            ? 1
            : outcome_compare(outcome, best_outcome);
        SelectNode *candidate = &runtime->term.nodes[child];
        SelectNode *selected = best_child < 0
            ? NULL
            : &runtime->term.nodes[best_child];
        if (best_child < 0 || order > 0 ||
            (order == 0 && candidate->demand_rank < selected->demand_rank)) {
            best_child = child;
            best_outcome = outcome;
        }
    }
    node = &runtime->term.nodes[node_index];
    node->selected_child = best_child;
    node->selected_leaf = runtime->term.nodes[best_child].selected_leaf;
    node->state = SELECT_FORCED;
    trace_choice(
        runtime,
        "select",
        round,
        node_index,
        best_child,
        node->selected_leaf,
        best_outcome
    );
    return node->selected_leaf;
}

static void update_context_covectors(
    SelectRuntime *runtime,
    const LlamaCompanyResult *result
) {
    for (int index = 1; index < runtime->term.count; index++) {
        SelectNode *node = &runtime->term.nodes[index];
        const float *logits = result->logits +
            (size_t)(index - 1) * result->vocab_size;
        node->log_partition = log_partition(logits, result->vocab_size);
        if (node->position >= runtime->frame_count - 1) continue;
        if (node->context_logits == NULL) {
            node->context_logits = select_calloc(
                (size_t)runtime->vocab_size,
                sizeof(*node->context_logits)
            );
        }
        memcpy(
            node->context_logits,
            logits,
            (size_t)runtime->vocab_size * sizeof(*node->context_logits)
        );
        node->context_logits_ready = 1;
    }
}

static int observe_round(SelectRuntime *runtime, int round) {
    clear_round_outcomes(runtime);
    int row_count = runtime->term.count - 1;
    if (row_count <= 0 || runtime->term.leaf_count == 0) {
        select_fail("resumed term has no complete observation");
    }
    int *tokens = select_calloc((size_t)row_count, sizeof(*tokens));
    int *positions = select_calloc((size_t)row_count, sizeof(*positions));
    int *parents = select_calloc((size_t)row_count, sizeof(*parents));
    for (int index = 1; index < runtime->term.count; index++) {
        SelectNode *node = &runtime->term.nodes[index];
        SelectFrame *frame = &runtime->frames[node->position];
        int row = index - 1;
        tokens[row] = frame->candidates[node->candidate_index].token;
        positions[row] = node->position;
        parents[row] = node->parent == 0 ? -1 : node->parent - 1;
    }
    LlamaCompanyShape shape = {
        .row_count = row_count,
        .tokens = tokens,
        .positions = positions,
        .parents = parents,
    };

    int filler_count = atkey_filler_count(runtime->company_runtime);
    size_t *calls_before = select_calloc(
        (size_t)filler_count,
        sizeof(*calls_before)
    );
    size_t *reads_before = select_calloc(
        (size_t)filler_count,
        sizeof(*reads_before)
    );
    for (int filler = 0; filler < filler_count; filler++) {
        calls_before[filler] = atkey_filler_calls(runtime->company_runtime, filler);
        reads_before[filler] =
            atkey_filler_scalar_reads(runtime->company_runtime, filler);
    }

    LlamaCompanyResult result;
    unsigned long long started = monotonic_nanoseconds();
    if (!llama_company_evaluate(
            runtime->company_runtime,
            &shape,
            false,
            &result
        )) {
        select_fail("company batch failed");
    }
    unsigned long long elapsed = monotonic_nanoseconds() - started;
    runtime->counters.company_nanoseconds += elapsed;
    runtime->counters.company_batches++;
    runtime->counters.company_rows += (unsigned long long)row_count;

    unsigned long long batch_calls = 0;
    unsigned long long batch_reads = 0;
    for (int filler = 0; filler < filler_count; filler++) {
        size_t calls = atkey_filler_calls(runtime->company_runtime, filler) -
            calls_before[filler];
        size_t reads = atkey_filler_scalar_reads(runtime->company_runtime, filler) -
            reads_before[filler];
        runtime->filler_calls_by_id[filler] += calls;
        runtime->counters.family_filler_calls += calls;
        runtime->counters.family_scalar_reads += reads;
        batch_calls += calls;
        batch_reads += reads;
        if (runtime->filler_calls_by_id[filler] >
            runtime->counters.maximum_calls_per_filler) {
            runtime->counters.maximum_calls_per_filler =
                runtime->filler_calls_by_id[filler];
        }
    }
    if (runtime->trace != NULL) {
        fprintf(
            runtime->trace,
            "{\"event\":\"company_run\",\"round\":%d,\"rows\":%d,"
            "\"batch_filler_calls\":%llu,\"batch_scalar_reads\":%llu,"
            "\"cumulative_maximum_calls_per_filler\":%llu,"
            "\"model_ms\":%.9g}\n",
            round,
            row_count,
            batch_calls,
            batch_reads,
            runtime->counters.maximum_calls_per_filler,
            (double)elapsed / 1000000.0
        );
        fflush(runtime->trace);
    }

    update_context_covectors(runtime, &result);
    int *nodes = select_calloc(
        (size_t)runtime->frame_count,
        sizeof(*nodes)
    );
    for (int index = 1; index < runtime->term.count; index++) {
        if (runtime->term.nodes[index].child_count != 0) continue;
        SelectOutcome *outcome = observe_leaf(runtime, index, nodes);
        runtime->counters.structured_outcomes++;
        trace_outcome(runtime, round, index, outcome);
    }
    free(nodes);

    for (int filler = 0; filler < filler_count; filler++) {
        calls_before[filler] = atkey_filler_calls(runtime->company_runtime, filler);
        reads_before[filler] =
            atkey_filler_scalar_reads(runtime->company_runtime, filler);
    }
    unsigned long long strength_started = monotonic_nanoseconds();
    int selected_leaf = force_select(runtime, 0, round);
    runtime->counters.strength_nanoseconds +=
        monotonic_nanoseconds() - strength_started;
    for (int filler = 0; filler < filler_count; filler++) {
        runtime->counters.strength_filler_calls +=
            atkey_filler_calls(runtime->company_runtime, filler) -
            calls_before[filler];
        runtime->counters.strength_scalar_reads +=
            atkey_filler_scalar_reads(runtime->company_runtime, filler) -
            reads_before[filler];
    }
    if (runtime->counters.strength_filler_calls != 0 ||
        runtime->counters.strength_scalar_reads != 0) {
        select_fail("strength invoked a learned filler");
    }

    if (runtime->trace != NULL) {
        SelectOutcome *outcome = runtime->term.nodes[selected_leaf].outcome;
        fprintf(
            runtime->trace,
            "{\"event\":\"round_selected\",\"round\":%d,"
            "\"selected_leaf\":%d,\"worst_opinion\":%.17g,\"text\":",
            round,
            selected_leaf,
            outcome->leximin_coordinates[0]
        );
        trace_path(runtime->trace, runtime, selected_leaf);
        fputs("}\n", runtime->trace);
        fflush(runtime->trace);
    }

    llama_company_result_free(&result);
    free(reads_before);
    free(calls_before);
    free(parents);
    free(positions);
    free(tokens);
    return selected_leaf;
}

static void trace_term_round(SelectRuntime *runtime, int round) {
    if (runtime->trace == NULL) return;
    fprintf(
        runtime->trace,
        "{\"event\":\"selection_term_resumed\",\"round\":%d,"
        "\"rows\":%d,\"leaves\":%llu,\"root_reachability\":%llu,"
        "\"maximum_demands_per_select\":%d,"
        "\"paths_prebuilt_before_observation\":false}\n",
        round,
        runtime->term.count - 1,
        runtime->term.leaf_count,
        runtime->term.nodes[0].reachability,
        runtime->maximum_demands_per_select
    );
    fflush(runtime->trace);
}

static void free_frames(SelectFrame *frames, int count) {
    if (frames == NULL) return;
    for (int position = 0; position < count; position++) {
        free(frames[position].by_token);
        free(frames[position].candidates);
    }
    free(frames);
}

static void free_term(SelectTerm *term) {
    for (int index = 0; index < term->count; index++) {
        free(term->nodes[index].children);
        free(term->nodes[index].context_logits);
        outcome_free(term->nodes[index].outcome);
    }
    free(term->nodes);
    memset(term, 0, sizeof(*term));
}

static SelectFrame *build_frames(
    Transformer *transformer,
    const int *prompt_tokens,
    int prompt_count,
    int completion_count
) {
    Config *config = &transformer->config;
    int frame_count = prompt_count + completion_count;
    SelectFrame *frames = select_calloc((size_t)frame_count, sizeof(*frames));
    float *hidden = select_calloc(
        (size_t)(completion_count + 1) * config->dim,
        sizeof(*hidden)
    );
    for (int position = 0; position < prompt_count; position++) {
        float *value = forward_token_hidden(
            transformer,
            prompt_tokens[position],
            position
        );
        if (position == prompt_count - 1) {
            memcpy(hidden, value, (size_t)config->dim * sizeof(*hidden));
        }
        frames[position] = (SelectFrame){
            .position = position,
            .count = 1,
            .candidates = select_calloc(1, sizeof(SelectCandidate)),
        };
        frames[position].candidates[0] = (SelectCandidate){
            .token = prompt_tokens[position],
            .fixed_rank = 1,
        };
    }
    for (int index = 1; index <= completion_count; index++) {
        int position = prompt_count + index - 1;
        float *value = forward_feedback_hidden(transformer, position);
        memcpy(
            hidden + (size_t)index * config->dim,
            value,
            (size_t)config->dim * sizeof(*hidden)
        );
    }

    for (int output = 0; output < completion_count; output++) {
        int position = prompt_count + output;
        SelectFrame *frame = &frames[position];
        frame->position = position;
        frame->count = config->vocab_size;
        frame->candidates = select_calloc(
            (size_t)config->vocab_size,
            sizeof(*frame->candidates)
        );
        frame->by_token = select_calloc(
            (size_t)config->vocab_size,
            sizeof(*frame->by_token)
        );
        matmul(
            transformer->state.logits,
            hidden + (size_t)output * config->dim,
            transformer->weights.wcls,
            config->dim,
            config->vocab_size
        );
        double partition = log_partition(
            transformer->state.logits,
            config->vocab_size
        );
        for (int token = 0; token < config->vocab_size; token++) {
            frame->candidates[token] = (SelectCandidate){
                .token = token,
                .fixed_logit = transformer->state.logits[token],
                .fixed_log_probability =
                    (double)transformer->state.logits[token] - partition,
            };
        }
        qsort(
            frame->candidates,
            (size_t)frame->count,
            sizeof(*frame->candidates),
            candidate_compare
        );
        for (int rank = 0; rank < frame->count; rank++) {
            frame->candidates[rank].fixed_rank = rank + 1;
            frame->by_token[frame->candidates[rank].token] = rank;
        }
    }
    free(hidden);
    return frames;
}

static int *selected_token_path(
    const SelectRuntime *runtime,
    int selected_leaf
) {
    int *tokens = select_calloc(
        (size_t)runtime->frame_count,
        sizeof(*tokens)
    );
    for (int node = selected_leaf; node != 0;
         node = runtime->term.nodes[node].parent) {
        SelectNode *entry = &runtime->term.nodes[node];
        SelectFrame *frame = &runtime->frames[entry->position];
        tokens[entry->position] =
            frame->candidates[entry->candidate_index].token;
    }
    return tokens;
}

static void print_selected_path(
    SelectRuntime *runtime,
    int selected_leaf
) {
    int *tokens = selected_token_path(runtime, selected_leaf);
    int previous = tokens[0];
    for (int position = 1; position < runtime->frame_count; position++) {
        int token = tokens[position];
        if (token == 1) break;
        safe_printf(decode(runtime->tokenizer, previous, token));
        previous = token;
    }
    printf("\n");
    fflush(stdout);
    free(tokens);
}

static void run_resumable_selection(
    Transformer *transformer,
    AtkeyRuntime *company_runtime,
    Tokenizer *tokenizer,
    const int *prompt_tokens,
    int prompt_count,
    int completion_count,
    int maximum_demands_per_select,
    unsigned long long leaf_safety_limit,
    unsigned long long tie_seed,
    FILE *trace
) {
    SelectRuntime runtime = {
        .frame_count = prompt_count + completion_count,
        .prompt_count = prompt_count,
        .vocab_size = transformer->config.vocab_size,
        .maximum_demands_per_select = maximum_demands_per_select,
        .leaf_safety_limit = leaf_safety_limit,
        .tie_seed = tie_seed,
        .tokenizer = tokenizer,
        .company_runtime = company_runtime,
        .trace = trace,
    };
    runtime.frames = build_frames(
        transformer,
        prompt_tokens,
        prompt_count,
        completion_count
    );
    int filler_count = atkey_filler_count(company_runtime);
    runtime.filler_calls_by_id = select_calloc(
        (size_t)filler_count,
        sizeof(*runtime.filler_calls_by_id)
    );
    int root = term_add_node(
        &runtime.term,
        -1,
        -1,
        -1,
        0,
        DEMAND_SELECT_UNIT
    );
    if (root != 0) select_fail("term lost its synthetic root");

    if (trace != NULL) {
        fprintf(
            trace,
            "{\"event\":\"run\",\"mode\":"
            "\"resumable_hidden_feedback_select\","
            "\"prompt_positions\":%d,\"completion_positions\":%d,"
            "\"vocabulary_size\":%d,\"maximum_demands_per_select\":%d,"
            "\"leaf_safety_limit\":%llu,"
            "\"paths_prebuilt_before_observation\":false}\n",
            prompt_count,
            completion_count,
            runtime.vocab_size,
            maximum_demands_per_select,
            leaf_safety_limit
        );
        fflush(trace);
    }

    int selected_leaf = -1;
    for (int round = 1; round <= maximum_demands_per_select; round++) {
        resume_round(&runtime, 0, 0, round);
        update_reachability(&runtime.term, 0);
        if (leaf_safety_limit != ULLONG_MAX &&
            runtime.term.leaf_count > leaf_safety_limit) {
            fprintf(
                stderr,
                "resumed round %d requires %llu leaves; -b %llu is only "
                "a safety limit and cannot reshape the term\n",
                round,
                runtime.term.leaf_count,
                leaf_safety_limit
            );
            exit(EXIT_FAILURE);
        }
        trace_term_round(&runtime, round);
        selected_leaf = observe_round(&runtime, round);
    }
    if (selected_leaf <= 0) select_fail("root produced no witness");
    SelectOutcome *outcome = runtime.term.nodes[selected_leaf].outcome;
    runtime.counters.root_terminalizations++;
    if (trace != NULL) {
        fprintf(
            trace,
            "{\"event\":\"root_terminalized\",\"selected_leaf\":%d,"
            "\"order\":\"leximin_complete_company_outcome\","
            "\"worst_opinion\":%.17g,\"coordinates\":[",
            selected_leaf,
            outcome->leximin_coordinates[0]
        );
        for (int position = 0; position < outcome->count; position++) {
            if (position != 0) fputc(',', trace);
            fprintf(trace, "%.17g", outcome->coordinates[position]);
        }
        fputs("],\"text\":", trace);
        trace_path(trace, &runtime, selected_leaf);
        fputs("}\n", trace);
        fflush(trace);
    }

    print_selected_path(&runtime, selected_leaf);
    fprintf(
        stderr,
        "mode: resumable_hidden_feedback_select\n"
        "prompt_positions: %d\n"
        "completion_positions: %d\n"
        "maximum_demands_per_select: %d\n"
        "continuation_demands: %llu\n"
        "final_unique_leaves: %llu\n"
        "company_batches: %llu\n"
        "cumulative_company_rows: %llu\n"
        "family_filler_calls: %llu\n"
        "maximum_calls_per_filler: %llu\n"
        "family_scalar_reads: %llu\n"
        "strength_nodes: %llu\n"
        "strength_candidate_ratings: %llu\n"
        "structured_outcomes: %llu\n"
        "root_terminalizations: %llu\n"
        "strength_model_filler_calls: %llu\n"
        "strength_model_scalar_reads: %llu\n"
        "company_model_ms: %.3f\n"
        "pure_strength_ms: %.3f\n"
        "selected_terminal_diagnostic: %.17g\n",
        prompt_count,
        completion_count,
        maximum_demands_per_select,
        runtime.counters.continuation_demands,
        runtime.term.leaf_count,
        runtime.counters.company_batches,
        runtime.counters.company_rows,
        runtime.counters.family_filler_calls,
        runtime.counters.maximum_calls_per_filler,
        runtime.counters.family_scalar_reads,
        runtime.counters.strength_nodes,
        runtime.counters.candidate_ratings,
        runtime.counters.structured_outcomes,
        runtime.counters.root_terminalizations,
        runtime.counters.strength_filler_calls,
        runtime.counters.strength_scalar_reads,
        (double)runtime.counters.company_nanoseconds / 1000000.0,
        (double)runtime.counters.strength_nanoseconds / 1000000.0,
        outcome->leximin_coordinates[0]
    );

    if (trace != NULL) {
        fprintf(
            trace,
            "{\"event\":\"run_end\",\"continuation_demands\":%llu,"
            "\"final_unique_leaves\":%llu,\"company_batches\":%llu,"
            "\"cumulative_company_rows\":%llu,"
            "\"family_filler_calls\":%llu,"
            "\"maximum_calls_per_filler\":%llu,"
            "\"strength_model_filler_calls\":%llu,"
            "\"strength_model_scalar_reads\":%llu,"
            "\"company_model_ms\":%.9g,\"pure_strength_ms\":%.9g}\n",
            runtime.counters.continuation_demands,
            runtime.term.leaf_count,
            runtime.counters.company_batches,
            runtime.counters.company_rows,
            runtime.counters.family_filler_calls,
            runtime.counters.maximum_calls_per_filler,
            runtime.counters.strength_filler_calls,
            runtime.counters.strength_scalar_reads,
            (double)runtime.counters.company_nanoseconds / 1000000.0,
            (double)runtime.counters.strength_nanoseconds / 1000000.0
        );
        fflush(trace);
    }

    free(runtime.filler_calls_by_id);
    free_term(&runtime.term);
    free_frames(runtime.frames, runtime.frame_count);
}

static void usage(void) {
    fprintf(stderr, "Usage: run_hidden_feedback_select CHECKPOINT [options]\n");
    fprintf(stderr, "  -i <text> prompt\n");
    fprintf(stderr, "  -l <int>  completion positions, default 6\n");
    fprintf(stderr, "  -k <int>  continuation demands per Select, default 4\n");
    fprintf(stderr, "  -b <int>  optional leaf safety limit; never truncates\n");
    fprintf(stderr, "  -s <int>  deterministic tie seed, default 42\n");
    fprintf(stderr, "  -z <path> tokenizer, default tokenizer.bin\n");
    fprintf(stderr, "  -o <path> flushed JSONL trace\n");
    exit(EXIT_FAILURE);
}

static unsigned long long parse_positive_u64(const char *value) {
    errno = 0;
    char *end = NULL;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed == 0) usage();
    return parsed;
}

int main(int argc, char **argv) {
    if (argc < 2) usage();
    char *checkpoint_path = argv[1];
    char *tokenizer_path = "tokenizer.bin";
    char *prompt = "";
    char *trace_path = NULL;
    int completion_count = 6;
    int maximum_demands_per_select = 4;
    unsigned long long leaf_safety_limit = ULLONG_MAX;
    unsigned long long tie_seed = 42;

    for (int argument = 2; argument < argc; argument += 2) {
        if (argument + 1 >= argc || argv[argument][0] != '-' ||
            strlen(argv[argument]) != 2) usage();
        char option = argv[argument][1];
        char *value = argv[argument + 1];
        if (option == 'i') prompt = value;
        else if (option == 'z') tokenizer_path = value;
        else if (option == 'o') trace_path = value;
        else if (option == 'l') {
            unsigned long long parsed = parse_positive_u64(value);
            if (parsed > INT_MAX) usage();
            completion_count = (int)parsed;
        } else if (option == 'k') {
            unsigned long long parsed = parse_positive_u64(value);
            if (parsed > INT_MAX) usage();
            maximum_demands_per_select = (int)parsed;
        } else if (option == 'b') {
            leaf_safety_limit = parse_positive_u64(value);
        } else if (option == 's') {
            tie_seed = parse_positive_u64(value);
        } else usage();
    }

    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
    size_t token_capacity = strlen(prompt) + 3;
    if (token_capacity > (size_t)INT_MAX) select_fail("prompt is too large");
    int *prompt_tokens = select_calloc(token_capacity, sizeof(*prompt_tokens));
    int prompt_count = 0;
    encode(&tokenizer, prompt, 1, 0, prompt_tokens, &prompt_count);
    if (prompt_count <= 0) select_fail("prompt encoded to no tokens");
    if (prompt_count > transformer.config.seq_len - completion_count) {
        select_fail("prompt plus completion exceeds model context");
    }

    AtkeyRuntime *company_runtime = atkey_runtime_new(
        checkpoint_path,
        tokenizer_path
    );
    if (company_runtime == NULL) select_fail("could not create company runtime");
    FILE *trace = NULL;
    if (trace_path != NULL) {
        trace = fopen(trace_path, "w");
        if (trace == NULL) select_fail("could not open trace");
    }

    run_resumable_selection(
        &transformer,
        company_runtime,
        &tokenizer,
        prompt_tokens,
        prompt_count,
        completion_count,
        maximum_demands_per_select,
        leaf_safety_limit,
        tie_seed,
        trace
    );

    if (trace != NULL) fclose(trace);
    atkey_runtime_free(company_runtime);
    free(prompt_tokens);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
