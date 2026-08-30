#define _POSIX_C_SOURCE 200809L

/*
 * REJECTED: this program used Q(tokens)'s final hidden state to decode the
 * token after a hypothetical completion, then fed that token back into an
 * earlier completion hole.  That is not Escardo's logit-carrier selection
 * product and produced the demonstrably invalid "Lily was ot" result.
 * It is retained only as a regression warning and must not be built or run.
 */

#include "atkey_term_c.h"
#include "candidate_ledger.h"
#include "escardo_model.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Program Program;
typedef struct TokenExpr TokenExpr;
typedef struct ModelNode ModelNode;
typedef struct DecodeNode DecodeNode;
typedef struct ProductFrame ProductFrame;
typedef struct ChiOccurrence ChiOccurrence;

struct TokenExpr {
    uint64_t id;
    bool resolved;
    int token;
    DecodeNode *producer;
};

typedef struct {
    ModelNode *model;
    int predictor_position;
} HiddenRef;

struct ModelNode {
    uint64_t id;
    TokenExpr **completion;
    uint64_t multiplicity;
    bool seed;
    bool resolved;
    bool scored;
    float *final_hidden;
    float *predictor_hidden;
    double log_probability_sum;
    double mean_log_probability;
};

struct DecodeNode {
    uint64_t id;
    uint64_t frame_id;
    int selector_position;
    const char *role;
    HiddenRef input;
    TokenExpr *output;
    bool resolved;
    uint64_t selected_candidate_id;
};

typedef struct {
    int count;
    TokenExpr **items;
} SequenceExpr;

typedef HiddenRef (*SequenceObserverApply)(
    void *environment,
    const SequenceExpr *sequence
);

typedef struct {
    SequenceObserverApply apply;
    void *environment;
} SequenceObserver;

typedef HiddenRef (*TokenContinuationApply)(
    void *environment,
    TokenExpr *token
);

typedef struct {
    TokenContinuationApply apply;
    void *environment;
} TokenContinuation;

typedef struct TailMemo TailMemo;
struct TailMemo {
    TokenExpr *head;
    SequenceExpr tail;
    TailMemo *next;
};

struct ProductFrame {
    Program *program;
    uint64_t id;
    uint64_t parent_id;
    int position;
    SequenceObserver observer;
    TailMemo *tails;
    TokenExpr *selected_head;
    SequenceExpr selected_tail;
};

struct ChiOccurrence {
    uint64_t frame_id;
    int position;
    DecodeNode *initial;
    ModelNode *feedback_model;
    DecodeNode *feedback;
};

typedef struct {
    const char *checkpoint;
    const char *tokenizer;
    const char *prompt;
    const char *ledger_path;
    int horizon;
    int top_k;
    int seed_token;
    double temperature;
    uint64_t sample_seed;
    bool durable_ledger;
} Options;

struct Program {
    Options options;
    AtkeyRuntime *runtime;
    EscardoModel *model;
    CandidateLedger ledger;
    int *prompt;
    int prompt_count;
    ModelNode *seed_model;
    TokenExpr **constant_tokens;
    size_t constant_tokens_count;
    size_t constant_tokens_capacity;
    TokenExpr **tokens;
    size_t tokens_count;
    size_t tokens_capacity;
    ModelNode **model_nodes;
    size_t model_nodes_count;
    size_t model_nodes_capacity;
    DecodeNode **decode_nodes;
    size_t decode_nodes_count;
    size_t decode_nodes_capacity;
    ProductFrame **frames;
    size_t frames_count;
    size_t frames_capacity;
    ChiOccurrence **chi_occurrences;
    size_t chi_occurrences_count;
    size_t chi_occurrences_capacity;
    void **owned;
    size_t owned_count;
    size_t owned_capacity;
    uint64_t next_token_id;
    uint64_t next_model_id;
    uint64_t next_decode_id;
    uint64_t next_frame_id;
    uint64_t next_candidate_id;
    uint64_t model_frontiers;
    uint64_t decoder_frontiers;
};

static void fail(const char *message) {
    fprintf(stderr, "escardo infer: %s\n", message);
    exit(EXIT_FAILURE);
}

static void *xcalloc(size_t count, size_t width) {
    if (width != 0 && count > SIZE_MAX / width) fail("allocation overflow");
    void *memory = calloc(count, width);
    if (memory == NULL) fail("allocation failed");
    return memory;
}

static void *owned(Program *program, size_t count, size_t width) {
    void *memory = xcalloc(count, width);
    if (program->owned_count == program->owned_capacity) {
        size_t capacity =
            program->owned_capacity == 0 ? 256 : program->owned_capacity * 2;
        void **items = realloc(program->owned, capacity * sizeof(*items));
        if (items == NULL) fail("owned allocation table failed");
        program->owned = items;
        program->owned_capacity = capacity;
    }
    program->owned[program->owned_count++] = memory;
    return memory;
}

#define APPEND_POINTER(program, member, value) do { \
    if ((program)->member##_count == (program)->member##_capacity) { \
        size_t next_capacity = (program)->member##_capacity == 0 ? \
            64 : (program)->member##_capacity * 2; \
        void *next_items = realloc( \
            (program)->member, \
            next_capacity * sizeof(*(program)->member) \
        ); \
        if (next_items == NULL) fail("node table allocation failed"); \
        (program)->member = next_items; \
        (program)->member##_capacity = next_capacity; \
    } \
    (program)->member[(program)->member##_count++] = (value); \
} while (0)

static CandidateLedgerEvent ledger_event(const char *kind) {
    return (CandidateLedgerEvent){
        .kind = kind,
        .frame_id = CANDIDATE_LEDGER_NONE_U64,
        .parent_frame_id = CANDIDATE_LEDGER_NONE_U64,
        .demand_id = CANDIDATE_LEDGER_NONE_U64,
        .candidate_id = CANDIDATE_LEDGER_NONE_U64,
        .source_candidate_id = CANDIDATE_LEDGER_NONE_U64,
        .multiplicity = CANDIDATE_LEDGER_NONE_U64,
        .depth = CANDIDATE_LEDGER_NONE_INT,
        .position = CANDIDATE_LEDGER_NONE_INT,
        .rank = CANDIDATE_LEDGER_NONE_INT,
        .token_id = CANDIDATE_LEDGER_NONE_INT,
        .local_logit = NAN,
        .local_log_probability = NAN,
        .observer_score = NAN,
        .backed_score = NAN,
        .aggregate_before = NAN,
        .aggregate_after = NAN,
    };
}

static int parse_int(const char *text, const char *flag) {
    errno = 0;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value < INT32_MIN || value > INT32_MAX) {
        fprintf(stderr, "%s expects an integer\n", flag);
        exit(EXIT_FAILURE);
    }
    return (int)value;
}

static uint64_t parse_u64(const char *text, const char *flag) {
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        fprintf(stderr, "%s expects an unsigned integer\n", flag);
        exit(EXIT_FAILURE);
    }
    return (uint64_t)value;
}

static double parse_double(const char *text, const char *flag) {
    errno = 0;
    char *end = NULL;
    double value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value)) {
        fprintf(stderr, "%s expects a finite number\n", flag);
        exit(EXIT_FAILURE);
    }
    return value;
}

static Options parse_options(int argc, char **argv) {
    if (argc < 2) fail("usage: run_escardo_term checkpoint [options]");
    Options options = {
        .checkpoint = argv[1],
        .tokenizer = "tokenizer.bin",
        .prompt = "Once upon a time",
        .ledger_path = "candidates.jsonl",
        .horizon = 3,
        .top_k = 4,
        .seed_token = 0,
        .temperature = 0.0,
        .sample_seed = 1,
        .durable_ledger = false,
    };
    for (int index = 2; index < argc; index++) {
        const char *flag = argv[index];
        if (index + 1 >= argc) fail("incomplete option");
        const char *value = argv[++index];
        if (strcmp(flag, "-z") == 0) {
            options.tokenizer = value;
        } else if (strcmp(flag, "-i") == 0) {
            options.prompt = value;
        } else if (strcmp(flag, "-n") == 0) {
            options.horizon = parse_int(value, flag);
        } else if (strcmp(flag, "-k") == 0) {
            options.top_k = parse_int(value, flag);
        } else if (strcmp(flag, "-t") == 0) {
            options.temperature = parse_double(value, flag);
        } else if (strcmp(flag, "-s") == 0) {
            options.sample_seed = parse_u64(value, flag);
        } else if (strcmp(flag, "--seed-token") == 0) {
            options.seed_token = parse_int(value, flag);
        } else if (strcmp(flag, "--ledger") == 0) {
            options.ledger_path = value;
        } else if (strcmp(flag, "--durable-ledger") == 0) {
            int parsed = parse_int(value, flag);
            if (parsed != 0 && parsed != 1) {
                fail("--durable-ledger expects 0 or 1");
            }
            options.durable_ledger = parsed != 0;
        } else {
            fprintf(stderr, "unknown option: %s\n", flag);
            exit(EXIT_FAILURE);
        }
    }
    if (options.horizon <= 0) fail("horizon must be positive");
    if (options.top_k <= 0) fail("top-k must be positive");
    if (options.temperature < 0.0) fail("temperature must be nonnegative");
    return options;
}

static TokenExpr *new_token(Program *program, bool resolved, int token) {
    TokenExpr *expression = owned(program, 1, sizeof(*expression));
    expression->id = program->next_token_id++;
    expression->resolved = resolved;
    expression->token = token;
    APPEND_POINTER(program, tokens, expression);
    return expression;
}

static TokenExpr *constant_token(Program *program, int token) {
    for (size_t index = 0; index < program->constant_tokens_count; index++) {
        if (program->constant_tokens[index]->token == token) {
            return program->constant_tokens[index];
        }
    }
    TokenExpr *expression = new_token(program, true, token);
    APPEND_POINTER(program, constant_tokens, expression);
    return expression;
}

static bool same_completion(
    const Program *program,
    const ModelNode *node,
    const SequenceExpr *sequence
) {
    if (node->seed || sequence->count != program->options.horizon) return false;
    for (int index = 0; index < sequence->count; index++) {
        if (node->completion[index] != sequence->items[index]) return false;
    }
    return true;
}

static ModelNode *new_model_node(
    Program *program,
    const SequenceExpr *sequence,
    bool seed
) {
    ModelNode *node = owned(program, 1, sizeof(*node));
    node->id = program->next_model_id++;
    node->seed = seed;
    node->multiplicity = 1;
    node->completion = owned(
        program,
        (size_t)program->options.horizon,
        sizeof(*node->completion)
    );
    memcpy(
        node->completion,
        sequence->items,
        (size_t)program->options.horizon * sizeof(*node->completion)
    );
    APPEND_POINTER(program, model_nodes, node);
    return node;
}

static ModelNode *whole_context_demand(
    Program *program,
    const SequenceExpr *sequence
) {
    if (sequence->count != program->options.horizon) {
        fail("Q received a partial completion");
    }
    ModelNode *node = NULL;
    for (size_t index = 0; index < program->model_nodes_count; index++) {
        if (same_completion(program, program->model_nodes[index], sequence)) {
            node = program->model_nodes[index];
            node->multiplicity++;
            break;
        }
    }
    if (node == NULL) node = new_model_node(program, sequence, false);
    CandidateLedgerEvent event = ledger_event("q_demand");
    event.demand_id = node->id;
    event.multiplicity = node->multiplicity;
    candidate_ledger_write(&program->ledger, &event);
    return node;
}

static HiddenRef root_observer_apply(
    void *raw_program,
    const SequenceExpr *sequence
) {
    Program *program = raw_program;
    return (HiddenRef){
        .model = whole_context_demand(program, sequence),
        .predictor_position = CANDIDATE_LEDGER_NONE_INT,
    };
}

static DecodeNode *new_decode(
    Program *program,
    uint64_t frame_id,
    int selector_position,
    const char *role,
    HiddenRef input
) {
    DecodeNode *node = owned(program, 1, sizeof(*node));
    node->id = program->next_decode_id++;
    node->frame_id = frame_id;
    node->selector_position = selector_position;
    node->role = role;
    node->input = input;
    node->selected_candidate_id = CANDIDATE_LEDGER_NONE_U64;
    node->output = new_token(program, false, CANDIDATE_LEDGER_NONE_INT);
    node->output->producer = node;
    APPEND_POINTER(program, decode_nodes, node);

    CandidateLedgerEvent event = ledger_event("decode_demand");
    event.frame_id = frame_id;
    event.demand_id = node->id;
    event.position = selector_position;
    event.reason = role;
    candidate_ledger_write(&program->ledger, &event);
    return node;
}

static SequenceExpr prepend_sequence(
    Program *program,
    TokenExpr *head,
    const SequenceExpr *tail
) {
    SequenceExpr sequence;
    sequence.count = tail->count + 1;
    sequence.items = owned(
        program,
        (size_t)sequence.count,
        sizeof(*sequence.items)
    );
    sequence.items[0] = head;
    if (tail->count > 0) {
        memcpy(
            sequence.items + 1,
            tail->items,
            (size_t)tail->count * sizeof(*tail->items)
        );
    }
    return sequence;
}

static SequenceExpr select_suffix(
    Program *program,
    int position,
    SequenceObserver observer,
    uint64_t parent_frame_id
);

typedef struct {
    SequenceObserver parent;
    Program *program;
    TokenExpr *head;
} PrependObserverEnvironment;

static HiddenRef prepend_observer_apply(
    void *raw_environment,
    const SequenceExpr *tail
) {
    PrependObserverEnvironment *environment = raw_environment;
    SequenceExpr sequence = prepend_sequence(
        environment->program,
        environment->head,
        tail
    );
    return environment->parent.apply(
        environment->parent.environment,
        &sequence
    );
}

static SequenceExpr frame_tail(ProductFrame *frame, TokenExpr *head) {
    for (TailMemo *entry = frame->tails; entry != NULL; entry = entry->next) {
        if (entry->head == head) return entry->tail;
    }
    PrependObserverEnvironment *prepend = owned(
        frame->program,
        1,
        sizeof(*prepend)
    );
    prepend->parent = frame->observer;
    prepend->program = frame->program;
    prepend->head = head;
    SequenceObserver tail_observer = {
        .apply = prepend_observer_apply,
        .environment = prepend,
    };
    SequenceExpr tail = select_suffix(
        frame->program,
        frame->position + 1,
        tail_observer,
        frame->id
    );
    TailMemo *memo = owned(frame->program, 1, sizeof(*memo));
    memo->head = head;
    memo->tail = tail;
    memo->next = frame->tails;
    frame->tails = memo;
    return tail;
}

static HiddenRef chi_continuation_apply(
    void *raw_frame,
    TokenExpr *token
) {
    ProductFrame *frame = raw_frame;
    SequenceExpr tail = frame_tail(frame, token);
    SequenceExpr sequence = prepend_sequence(frame->program, token, &tail);
    return frame->observer.apply(frame->observer.environment, &sequence);
}

static TokenExpr *run_chi(
    ProductFrame *frame,
    TokenContinuation continuation
) {
    Program *program = frame->program;
    HiddenRef seed = {
        .model = program->seed_model,
        .predictor_position = frame->position,
    };
    DecodeNode *initial = new_decode(
        program,
        frame->id,
        frame->position,
        "chi_initial_f_of_h",
        seed
    );
    HiddenRef revised_focus = continuation.apply(
        continuation.environment,
        initial->output
    );
    DecodeNode *feedback = new_decode(
        program,
        frame->id,
        frame->position,
        "chi_feedback_f_of_k_f_h",
        revised_focus
    );
    ChiOccurrence *occurrence = owned(program, 1, sizeof(*occurrence));
    occurrence->frame_id = frame->id;
    occurrence->position = frame->position;
    occurrence->initial = initial;
    occurrence->feedback_model = revised_focus.model;
    occurrence->feedback = feedback;
    APPEND_POINTER(program, chi_occurrences, occurrence);
    return feedback->output;
}

static SequenceExpr select_suffix(
    Program *program,
    int position,
    SequenceObserver observer,
    uint64_t parent_frame_id
) {
    if (position == program->options.horizon) {
        return (SequenceExpr){.count = 0, .items = NULL};
    }
    ProductFrame *frame = owned(program, 1, sizeof(*frame));
    frame->program = program;
    frame->id = program->next_frame_id++;
    frame->parent_id = parent_frame_id;
    frame->position = position;
    frame->observer = observer;
    APPEND_POINTER(program, frames, frame);

    CandidateLedgerEvent opened = ledger_event("product_frame_open");
    opened.frame_id = frame->id;
    opened.parent_frame_id = parent_frame_id;
    opened.depth = position;
    opened.position = position;
    candidate_ledger_write(&program->ledger, &opened);

    TokenContinuation continuation = {
        .apply = chi_continuation_apply,
        .environment = frame,
    };
    TokenExpr *head = run_chi(frame, continuation);
    SequenceExpr tail = frame_tail(frame, head);
    frame->selected_head = head;
    frame->selected_tail = tail;
    SequenceExpr selected = prepend_sequence(program, head, &tail);

    CandidateLedgerEvent composed = ledger_event("product_frame_composed");
    composed.frame_id = frame->id;
    composed.parent_frame_id = parent_frame_id;
    composed.depth = position;
    composed.position = position;
    composed.demand_id = head->id;
    candidate_ledger_write(&program->ledger, &composed);
    return selected;
}

static SequenceExpr compose_selection_term(Program *program) {
    SequenceObserver root = {
        .apply = root_observer_apply,
        .environment = program,
    };
    return select_suffix(
        program,
        0,
        root,
        CANDIDATE_LEDGER_NONE_U64
    );
}

typedef struct {
    char *bytes;
    size_t length;
    size_t capacity;
} TextBuffer;

static void text_append(TextBuffer *buffer, const char *piece) {
    if (piece == NULL) return;
    size_t length = strlen(piece);
    if (buffer->length + length + 1 > buffer->capacity) {
        size_t capacity = buffer->capacity == 0 ? 128 : buffer->capacity;
        while (capacity < buffer->length + length + 1) capacity *= 2;
        char *bytes = realloc(buffer->bytes, capacity);
        if (bytes == NULL) fail("text buffer allocation failed");
        buffer->bytes = bytes;
        buffer->capacity = capacity;
    }
    memcpy(buffer->bytes + buffer->length, piece, length);
    buffer->length += length;
    buffer->bytes[buffer->length] = '\0';
}

static int resolved_model_token(const ModelNode *node, int position) {
    TokenExpr *expression = node->completion[position];
    if (!expression->resolved) fail("read unresolved model token");
    return expression->token;
}

static char *decode_token_array(
    Program *program,
    const int *tokens,
    int count
) {
    TextBuffer buffer = {0};
    int previous = program->prompt[program->prompt_count - 1];
    for (int index = 0; index < count; index++) {
        text_append(
            &buffer,
            atkey_decode(program->runtime, previous, tokens[index])
        );
        previous = tokens[index];
    }
    if (buffer.bytes == NULL) {
        buffer.bytes = xcalloc(1, 1);
    }
    return buffer.bytes;
}

static char *decode_model_completion(
    Program *program,
    const ModelNode *node
) {
    int horizon = program->options.horizon;
    int *tokens = xcalloc((size_t)horizon, sizeof(*tokens));
    for (int position = 0; position < horizon; position++) {
        tokens[position] = resolved_model_token(node, position);
    }
    char *text = decode_token_array(program, tokens, horizon);
    free(tokens);
    return text;
}

static char *decode_replacement(
    Program *program,
    const ModelNode *context,
    int position,
    int replacement
) {
    int horizon = program->options.horizon;
    int *tokens = xcalloc((size_t)horizon, sizeof(*tokens));
    for (int index = 0; index < horizon; index++) {
        tokens[index] = resolved_model_token(context, index);
    }
    tokens[position] = replacement;
    char *text = decode_token_array(program, tokens, horizon);
    free(tokens);
    return text;
}

static double log_probability(
    const float *logits,
    int vocab,
    int token
) {
    double maximum = -INFINITY;
    for (int index = 0; index < vocab; index++) {
        if ((double)logits[index] > maximum) maximum = logits[index];
    }
    double sum = 0.0;
    for (int index = 0; index < vocab; index++) {
        sum += exp((double)logits[index] - maximum);
    }
    return (double)logits[token] - (maximum + log(sum));
}

static void top_k_tokens(
    const float *logits,
    int vocab,
    int top_k,
    int *tokens
) {
    for (int rank = 0; rank < top_k; rank++) tokens[rank] = -1;
    for (int token = 0; token < vocab; token++) {
        int insert = top_k;
        for (int rank = 0; rank < top_k; rank++) {
            int incumbent = tokens[rank];
            if (incumbent < 0 || logits[token] > logits[incumbent] ||
                (logits[token] == logits[incumbent] && token < incumbent)) {
                insert = rank;
                break;
            }
        }
        if (insert == top_k) continue;
        for (int rank = top_k - 1; rank > insert; rank--) {
            tokens[rank] = tokens[rank - 1];
        }
        tokens[insert] = token;
    }
}

static uint64_t hash_hidden(
    const float *hidden,
    int dim,
    uint64_t seed
) {
    uint64_t hash = UINT64_C(1469598103934665603) ^ seed;
    for (int index = 0; index < dim; index++) {
        uint32_t bits;
        memcpy(&bits, &hidden[index], sizeof(bits));
        for (int byte = 0; byte < 4; byte++) {
            hash ^= (bits >> (byte * 8)) & 0xffU;
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

static double uniform_from_hash(uint64_t state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    uint64_t value = state * UINT64_C(0x2545F4914F6CDD1D);
    return (double)(value >> 11) * (1.0 / 9007199254740992.0);
}

static int choose_rank(
    const Program *program,
    const float *hidden,
    const float *logits,
    const int *support
) {
    if (program->options.temperature == 0.0) return 0;
    double maximum = -INFINITY;
    for (int rank = 0; rank < program->options.top_k; rank++) {
        double scaled = logits[support[rank]] / program->options.temperature;
        if (scaled > maximum) maximum = scaled;
    }
    double total = 0.0;
    for (int rank = 0; rank < program->options.top_k; rank++) {
        total += exp(
            logits[support[rank]] / program->options.temperature - maximum
        );
    }
    double draw = uniform_from_hash(
        hash_hidden(
            hidden,
            escardo_model_dim(program->model),
            program->options.sample_seed
        )
    ) * total;
    double cumulative = 0.0;
    for (int rank = 0; rank < program->options.top_k; rank++) {
        cumulative += exp(
            logits[support[rank]] / program->options.temperature - maximum
        );
        if (draw < cumulative) return rank;
    }
    return program->options.top_k - 1;
}

static bool model_ready(const Program *program, const ModelNode *node) {
    if (node->resolved) return false;
    for (int position = 0; position < program->options.horizon; position++) {
        if (!node->completion[position]->resolved) return false;
    }
    return true;
}

static int evaluate_ready_models(Program *program) {
    int ready_count = 0;
    for (size_t index = 0; index < program->model_nodes_count; index++) {
        if (model_ready(program, program->model_nodes[index])) ready_count++;
    }
    if (ready_count == 0) return 0;
    ModelNode **ready = xcalloc((size_t)ready_count, sizeof(*ready));
    int cursor = 0;
    for (size_t index = 0; index < program->model_nodes_count; index++) {
        ModelNode *node = program->model_nodes[index];
        if (model_ready(program, node)) ready[cursor++] = node;
    }
    int horizon = program->options.horizon;
    int dim = escardo_model_dim(program->model);
    int *completions = xcalloc(
        (size_t)ready_count * horizon,
        sizeof(*completions)
    );
    for (int batch = 0; batch < ready_count; batch++) {
        for (int position = 0; position < horizon; position++) {
            completions[(size_t)batch * horizon + position] =
                resolved_model_token(ready[batch], position);
        }
    }
    float *final_hidden = xcalloc(
        (size_t)ready_count * dim,
        sizeof(*final_hidden)
    );
    float *predictor_hidden = xcalloc(
        (size_t)ready_count * horizon * dim,
        sizeof(*predictor_hidden)
    );
    program->model_frontiers++;
    escardo_model_apply_whole_context(
        program->model,
        program->prompt,
        program->prompt_count,
        completions,
        ready_count,
        horizon,
        final_hidden,
        predictor_hidden
    );
    for (int batch = 0; batch < ready_count; batch++) {
        ModelNode *node = ready[batch];
        node->final_hidden = owned(program, (size_t)dim, sizeof(float));
        memcpy(
            node->final_hidden,
            final_hidden + (size_t)batch * dim,
            (size_t)dim * sizeof(float)
        );
        node->predictor_hidden = owned(
            program,
            (size_t)horizon * dim,
            sizeof(float)
        );
        memcpy(
            node->predictor_hidden,
            predictor_hidden + (size_t)batch * horizon * dim,
            (size_t)horizon * dim * sizeof(float)
        );
        node->resolved = true;
        char *completion = decode_model_completion(program, node);
        CandidateLedgerEvent event = ledger_event(
            node->seed ? "seed_context_evaluated" : "q_evaluated"
        );
        event.demand_id = node->id;
        event.multiplicity = node->multiplicity;
        event.prefix = program->options.prompt;
        event.completion = completion;
        candidate_ledger_write(&program->ledger, &event);
        free(completion);
    }
    free(predictor_hidden);
    free(final_hidden);
    free(completions);
    free(ready);
    return ready_count;
}

static const float *decode_hidden(
    const Program *program,
    const DecodeNode *node
) {
    if (!node->input.model->resolved) fail("decode forced before Q");
    if (node->input.predictor_position == CANDIDATE_LEDGER_NONE_INT) {
        return node->input.model->final_hidden;
    }
    int position = node->input.predictor_position;
    if (position < 0 || position >= program->options.horizon) {
        fail("invalid seed hidden position");
    }
    return node->input.model->predictor_hidden +
        (size_t)position * escardo_model_dim(program->model);
}

static int ready_decode_count(const Program *program) {
    int count = 0;
    for (size_t index = 0; index < program->decode_nodes_count; index++) {
        DecodeNode *node = program->decode_nodes[index];
        if (!node->resolved && node->input.model->resolved) count++;
    }
    return count;
}

static int unscored_model_count(const Program *program) {
    int count = 0;
    for (size_t index = 0; index < program->model_nodes_count; index++) {
        ModelNode *node = program->model_nodes[index];
        if (node->resolved && !node->scored) count++;
    }
    return count;
}

static int previous_context_token(
    const Program *program,
    const ModelNode *context,
    int position
) {
    if (position == 0) return program->prompt[program->prompt_count - 1];
    return resolved_model_token(context, position - 1);
}

static void score_model_node(
    Program *program,
    ModelNode *node,
    const float *logits
) {
    int horizon = program->options.horizon;
    int vocab = escardo_model_vocab(program->model);
    char *completion = decode_model_completion(program, node);
    double total = 0.0;
    for (int position = 0; position < horizon; position++) {
        int token = resolved_model_token(node, position);
        const float *position_logits = logits + (size_t)position * vocab;
        double local = log_probability(position_logits, vocab, token);
        total += local;
        CandidateLedgerEvent event = ledger_event("model_token_score");
        event.demand_id = node->id;
        event.multiplicity = node->multiplicity;
        event.depth = position;
        event.position = position;
        event.token_id = token;
        event.piece = atkey_decode(
            program->runtime,
            previous_context_token(program, node, position),
            token
        );
        event.prefix = program->options.prompt;
        event.context = completion;
        event.completion = completion;
        event.local_logit = position_logits[token];
        event.local_log_probability = local;
        candidate_ledger_write(&program->ledger, &event);
    }
    node->log_probability_sum = total;
    node->mean_log_probability = total / horizon;
    node->scored = true;
    CandidateLedgerEvent event = ledger_event("model_score");
    event.demand_id = node->id;
    event.multiplicity = node->multiplicity;
    event.prefix = program->options.prompt;
    event.context = completion;
    event.completion = completion;
    event.reason = "diagnostic_final_layer_teacher_forced_log_probability";
    event.observer_score = node->mean_log_probability;
    event.aggregate_after = node->log_probability_sum;
    candidate_ledger_write(&program->ledger, &event);
    free(completion);
}

static void resolve_decode_node(
    Program *program,
    DecodeNode *node,
    const float *hidden,
    const float *logits
) {
    int vocab = escardo_model_vocab(program->model);
    int top_k = program->options.top_k;
    int *support = xcalloc((size_t)top_k, sizeof(*support));
    top_k_tokens(logits, vocab, top_k, support);
    int selected_rank = choose_rank(program, hidden, logits, support);
    int selected_token = support[selected_rank];
    ModelNode *context_model = node->input.model;
    char *context = decode_model_completion(program, context_model);

    for (int rank = 0; rank < top_k; rank++) {
        int token = support[rank];
        char *completion = decode_replacement(
            program,
            context_model,
            node->selector_position,
            token
        );
        uint64_t candidate_id = program->next_candidate_id++;
        CandidateLedgerEvent event = ledger_event("candidate");
        event.frame_id = node->frame_id;
        event.demand_id = node->id;
        event.candidate_id = candidate_id;
        event.multiplicity = context_model->multiplicity;
        event.depth = node->selector_position;
        event.position = node->selector_position;
        event.rank = rank;
        event.token_id = token;
        event.piece = atkey_decode(
            program->runtime,
            previous_context_token(
                program,
                context_model,
                node->selector_position
            ),
            token
        );
        event.prefix = program->options.prompt;
        event.context = context;
        event.completion = completion;
        event.status = rank == selected_rank ? "selected" : "available";
        event.reason = node->role;
        event.local_logit = logits[token];
        event.local_log_probability = log_probability(logits, vocab, token);
        event.observer_score = context_model->mean_log_probability;
        event.aggregate_after = context_model->log_probability_sum;
        candidate_ledger_write(&program->ledger, &event);
        if (rank == selected_rank) {
            node->selected_candidate_id = candidate_id;
        }
        free(completion);
    }

    node->output->token = selected_token;
    node->output->resolved = true;
    node->resolved = true;
    CandidateLedgerEvent selected = ledger_event("decode_selected");
    selected.frame_id = node->frame_id;
    selected.demand_id = node->id;
    selected.candidate_id = node->selected_candidate_id;
    selected.multiplicity = context_model->multiplicity;
    selected.depth = node->selector_position;
    selected.position = node->selector_position;
    selected.rank = selected_rank;
    selected.token_id = selected_token;
    selected.piece = atkey_decode(
        program->runtime,
        previous_context_token(
            program,
            context_model,
            node->selector_position
        ),
        selected_token
    );
    selected.prefix = program->options.prompt;
    selected.context = context;
    selected.reason = node->role;
    selected.local_logit = logits[selected_token];
    selected.local_log_probability = log_probability(
        logits,
        vocab,
        selected_token
    );
    selected.observer_score = context_model->mean_log_probability;
    selected.aggregate_after = context_model->log_probability_sum;
    candidate_ledger_write(&program->ledger, &selected);
    free(context);
    free(support);
}

static int evaluate_ready_observations(Program *program) {
    int decode_count = ready_decode_count(program);
    int score_models = unscored_model_count(program);
    if (decode_count == 0 && score_models == 0) return 0;
    int horizon = program->options.horizon;
    int score_count = score_models * horizon;
    int observation_count = decode_count + score_count;
    int dim = escardo_model_dim(program->model);
    int vocab = escardo_model_vocab(program->model);

    DecodeNode **decodes = xcalloc((size_t)decode_count, sizeof(*decodes));
    ModelNode **models = xcalloc((size_t)score_models, sizeof(*models));
    int cursor = 0;
    for (size_t index = 0; index < program->decode_nodes_count; index++) {
        DecodeNode *node = program->decode_nodes[index];
        if (!node->resolved && node->input.model->resolved) {
            decodes[cursor++] = node;
        }
    }
    cursor = 0;
    for (size_t index = 0; index < program->model_nodes_count; index++) {
        ModelNode *node = program->model_nodes[index];
        if (node->resolved && !node->scored) models[cursor++] = node;
    }

    float *inputs = xcalloc(
        (size_t)observation_count * dim,
        sizeof(*inputs)
    );
    for (int index = 0; index < decode_count; index++) {
        memcpy(
            inputs + (size_t)index * dim,
            decode_hidden(program, decodes[index]),
            (size_t)dim * sizeof(*inputs)
        );
    }
    int observation = decode_count;
    for (int model_index = 0; model_index < score_models; model_index++) {
        ModelNode *node = models[model_index];
        memcpy(
            inputs + (size_t)observation * dim,
            node->predictor_hidden,
            (size_t)horizon * dim * sizeof(*inputs)
        );
        observation += horizon;
    }

    float *normalized = xcalloc(
        (size_t)observation_count * dim,
        sizeof(*normalized)
    );
    float *logits = xcalloc(
        (size_t)observation_count * vocab,
        sizeof(*logits)
    );
    program->decoder_frontiers++;
    atkey_rms_family_apply(
        program->runtime,
        atkey_final_rms_filler_id(program->runtime),
        normalized,
        inputs,
        observation_count,
        atkey_final_rms_weight(program->runtime),
        dim
    );
    atkey_matmul_family_apply(
        program->runtime,
        atkey_output_filler_id(program->runtime),
        logits,
        normalized,
        observation_count,
        atkey_output_weight(program->runtime),
        dim,
        vocab
    );

    observation = decode_count;
    for (int model_index = 0; model_index < score_models; model_index++) {
        score_model_node(
            program,
            models[model_index],
            logits + (size_t)observation * vocab
        );
        observation += horizon;
    }
    for (int index = 0; index < decode_count; index++) {
        resolve_decode_node(
            program,
            decodes[index],
            inputs + (size_t)index * dim,
            logits + (size_t)index * vocab
        );
    }

    free(logits);
    free(normalized);
    free(inputs);
    free(models);
    free(decodes);
    return observation_count;
}

static bool all_nodes_resolved(const Program *program) {
    for (size_t index = 0; index < program->model_nodes_count; index++) {
        if (!program->model_nodes[index]->resolved ||
            !program->model_nodes[index]->scored) {
            return false;
        }
    }
    for (size_t index = 0; index < program->decode_nodes_count; index++) {
        if (!program->decode_nodes[index]->resolved) return false;
    }
    return true;
}

static void run_composed_term(Program *program) {
    while (!all_nodes_resolved(program)) {
        int models = evaluate_ready_models(program);
        int observations = evaluate_ready_observations(program);
        if (models == 0 && observations == 0) {
            fail("composed term contains an unresolved cycle");
        }
    }
}

static size_t total_filler_calls(AtkeyRuntime *runtime) {
    size_t total = 0;
    int count = atkey_filler_count(runtime);
    for (int filler_id = 0; filler_id < count; filler_id++) {
        total += atkey_filler_calls(runtime, filler_id);
    }
    return total;
}

static void emit_resolved_structure(
    Program *program,
    const SequenceExpr *result
) {
    int horizon = program->options.horizon;
    int *tokens = xcalloc((size_t)horizon, sizeof(*tokens));
    for (int position = 0; position < horizon; position++) {
        if (!result->items[position]->resolved) fail("unresolved root result");
        tokens[position] = result->items[position]->token;
    }
    char *completion = decode_token_array(program, tokens, horizon);

    for (size_t index = 0; index < program->chi_occurrences_count; index++) {
        ChiOccurrence *occurrence = program->chi_occurrences[index];
        CandidateLedgerEvent event = ledger_event("chi_feedback");
        event.frame_id = occurrence->frame_id;
        event.demand_id = occurrence->feedback_model->id;
        event.candidate_id = occurrence->feedback->selected_candidate_id;
        event.source_candidate_id =
            occurrence->initial->selected_candidate_id;
        event.multiplicity = occurrence->feedback_model->multiplicity;
        event.depth = occurrence->position;
        event.position = occurrence->position;
        event.token_id = occurrence->feedback->output->token;
        event.prefix = program->options.prompt;
        event.completion = completion;
        event.reason = "f(k(f(h)))";
        event.observer_score =
            occurrence->feedback_model->mean_log_probability;
        event.aggregate_after =
            occurrence->feedback_model->log_probability_sum;
        candidate_ledger_write(&program->ledger, &event);
    }
    for (size_t index = 0; index < program->frames_count; index++) {
        ProductFrame *frame = program->frames[index];
        DecodeNode *producer = frame->selected_head->producer;
        CandidateLedgerEvent event = ledger_event("product_selected");
        event.frame_id = frame->id;
        event.parent_frame_id = frame->parent_id;
        event.demand_id = frame->selected_head->id;
        event.candidate_id = producer == NULL ?
            CANDIDATE_LEDGER_NONE_U64 : producer->selected_candidate_id;
        event.depth = frame->position;
        event.position = frame->position;
        event.token_id = frame->selected_head->token;
        event.prefix = program->options.prompt;
        event.completion = completion;
        event.reason = "selection_monad_product";
        candidate_ledger_write(&program->ledger, &event);
    }
    CandidateLedgerEvent root = ledger_event("root_selected");
    root.prefix = program->options.prompt;
    root.completion = completion;
    root.reason = "sequenceS_chi_applied_to_Q";
    candidate_ledger_write(&program->ledger, &root);
    free(completion);
    free(tokens);
}

static void print_result(
    Program *program,
    const SequenceExpr *result,
    size_t calls_before_run
) {
    printf("prompt: %s\n", program->options.prompt);
    fputs("completion: ", stdout);
    int previous = program->prompt[program->prompt_count - 1];
    for (int position = 0; position < result->count; position++) {
        int token = result->items[position]->token;
        atkey_print_piece(program->runtime, previous, token);
        previous = token;
    }
    putchar('\n');
    fputs("selected_tokens=[", stdout);
    for (int position = 0; position < result->count; position++) {
        if (position != 0) putchar(',');
        printf("%d", result->items[position]->token);
    }
    puts("]");
    printf(
        "semantic_term=sequenceS(map(chi,CoCont_H_Token))_applied_to_Q\n"
        "whole_context_type=Token^%d_to_H\n"
        "decoder_type=H_to_Token\n",
        program->options.horizon
    );
    printf(
        "product_frames=%zu chi_occurrences=%zu "
        "q_demands=%zu decode_demands=%zu\n",
        program->frames_count,
        program->chi_occurrences_count,
        program->model_nodes_count - 1,
        program->decode_nodes_count
    );
    printf(
        "model_demand_frontiers=%" PRIu64
        " decoder_demand_frontiers=%" PRIu64 "\n",
        program->model_frontiers,
        program->decoder_frontiers
    );
    printf(
        "learned_kernel_calls_before_run=%zu "
        "learned_kernel_calls_after_run=%zu\n",
        calls_before_run,
        total_filler_calls(program->runtime)
    );
    puts("learned_filler_applications:");
    int filler_count = atkey_filler_count(program->runtime);
    for (int filler_id = 0; filler_id < filler_count; filler_id++) {
        printf(
            "  filler=%d applications=%zu scalar_accesses=%zu\n",
            filler_id,
            atkey_filler_calls(program->runtime, filler_id),
            atkey_filler_scalar_reads(program->runtime, filler_id)
        );
    }
    printf("candidate_ledger=%s\n", program->options.ledger_path);
}

static void free_program(Program *program) {
    for (size_t index = 0; index < program->owned_count; index++) {
        free(program->owned[index]);
    }
    free(program->owned);
    free(program->chi_occurrences);
    free(program->frames);
    free(program->decode_nodes);
    free(program->model_nodes);
    free(program->tokens);
    free(program->constant_tokens);
    atkey_free_tokens(program->prompt);
    escardo_model_free(program->model);
    atkey_runtime_free(program->runtime);
}

int main(int argc, char **argv) {
    Options options = parse_options(argc, argv);
    Program program = {0};
    program.options = options;
    program.runtime = atkey_runtime_new(options.checkpoint, options.tokenizer);
    if (program.runtime == NULL) fail("could not load model");
    program.model = escardo_model_new(program.runtime);
    if (program.model == NULL) fail("could not construct model term");
    if (options.top_k > escardo_model_vocab(program.model)) {
        fail("top-k exceeds vocabulary");
    }
    if (options.seed_token < 0 ||
        options.seed_token >= escardo_model_vocab(program.model)) {
        fail("seed token is outside vocabulary");
    }
    program.prompt = atkey_encode(
        program.runtime,
        options.prompt,
        &program.prompt_count
    );
    if (program.prompt == NULL || program.prompt_count <= 0) {
        fail("could not tokenize prompt");
    }
    if (program.prompt_count + options.horizon >
        atkey_sequence_length(program.runtime)) {
        fail("prompt and completion exceed model context");
    }
    if (!candidate_ledger_open(
            &program.ledger,
            options.ledger_path,
            options.durable_ledger,
            "escardo_chi_sequence_product",
            options.checkpoint,
            options.tokenizer,
            options.prompt,
            options.horizon,
            options.top_k,
            options.sample_seed
        )) {
        fail("could not open candidate ledger");
    }

    SequenceExpr seed_sequence;
    seed_sequence.count = options.horizon;
    seed_sequence.items = owned(
        &program,
        (size_t)options.horizon,
        sizeof(*seed_sequence.items)
    );
    TokenExpr *seed_token = constant_token(&program, options.seed_token);
    for (int position = 0; position < options.horizon; position++) {
        seed_sequence.items[position] = seed_token;
    }
    program.seed_model = new_model_node(&program, &seed_sequence, true);
    CandidateLedgerEvent seed = ledger_event("seed_context_composed");
    seed.demand_id = program.seed_model->id;
    seed.token_id = options.seed_token;
    seed.prefix = options.prompt;
    seed.reason = "initial_CoCont_focus_only";
    candidate_ledger_write(&program.ledger, &seed);

    size_t calls_before_composition = total_filler_calls(program.runtime);
    SequenceExpr result = compose_selection_term(&program);
    size_t calls_before_run = total_filler_calls(program.runtime);
    if (calls_before_composition != 0 || calls_before_run != 0) {
        fail("learned filler evaluated while composing the term");
    }
    CandidateLedgerEvent composed = ledger_event("term_composed");
    composed.multiplicity = program.model_nodes_count - 1;
    composed.reason = "no_model_observation_before_run";
    candidate_ledger_write(&program.ledger, &composed);

    run_composed_term(&program);
    emit_resolved_structure(&program, &result);
    print_result(&program, &result, calls_before_run);
    candidate_ledger_close(&program.ledger);
    free_program(&program);
    return 0;
}
