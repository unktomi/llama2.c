/*
 * DO NOT USE AS A STRENGTH CHECK FOR THE REQUESTED INFERENCER.
 *
 * This file includes the rejected token-carrier/prefix-field evaluator and
 * therefore inherits its strict per-prefix observations and repeated learned
 * filler applications.  Its balancing experiment cannot validate the missing
 * ModelLogit-carrier term.
 */

#define main atkey_term_program_main
#include "atkey_term.c"
#undef main

typedef struct BalancedContinuation BalancedContinuation;
typedef struct BalancedResult BalancedResult;

struct BalancedContinuation {
    double (*apply)(void *environment, Prefix *end);
    void *environment;
};

struct BalancedResult {
    Prefix *end;
    double score;
};

typedef struct {
    Search *search;
    double *values;
    unsigned char *known;
    size_t capacity;
} PrefixScoreMemo;

static void *score_realloc(void *pointer, size_t size) {
    void *result = realloc(pointer, size);
    if (result == NULL) fail("strength-check allocation failed");
    return result;
}

static void score_memo_reserve(PrefixScoreMemo *memo, uint32_t id) {
    if ((size_t)id < memo->capacity) return;
    size_t capacity = memo->capacity == 0 ? 1024 : memo->capacity;
    while (capacity <= (size_t)id) capacity *= 2;
    memo->values = score_realloc(
        memo->values,
        capacity * sizeof(*memo->values)
    );
    memo->known = score_realloc(
        memo->known,
        capacity * sizeof(*memo->known)
    );
    memset(
        memo->known + memo->capacity,
        0,
        (capacity - memo->capacity) * sizeof(*memo->known)
    );
    memo->capacity = capacity;
}

static double score_prefix(PrefixScoreMemo *memo, Prefix *prefix) {
    score_memo_reserve(memo, prefix->id);
    if (memo->known[prefix->id]) return memo->values[prefix->id];
    double score;
    if (prefix->parent == NULL) {
        score = 0.0;
    } else {
        score = score_prefix(memo, prefix->parent);
        if (!prefix->parent->terminated) {
            int depth = prefix->parent->depth;
            Field *field = memo->search->model->logits[depth];
            Vec *logits = sample_field(
                memo->search->evaluator,
                field,
                prefix->parent
            );
            score += token_log_probability(logits, prefix->token);
        }
    }
    memo->known[prefix->id] = 1;
    memo->values[prefix->id] = score;
    return score;
}

static double observe_prefix_score(void *environment, Prefix *end) {
    return score_prefix(environment, end);
}

typedef struct RightEntry RightEntry;
struct RightEntry {
    Prefix *left_end;
    BalancedResult result;
    RightEntry *next;
};

typedef struct {
    Search *search;
    int right_length;
    BalancedContinuation outer;
    RightEntry *entries;
} PairEnvironment;

static BalancedResult balanced_select(
    Search *search,
    Prefix *start,
    int length,
    BalancedContinuation continuation
);

static BalancedResult select_right(
    PairEnvironment *environment,
    Prefix *left_end
) {
    for (RightEntry *entry = environment->entries;
         entry != NULL;
         entry = entry->next) {
        if (entry->left_end == left_end) return entry->result;
    }
    RightEntry *entry = arena_alloc(
        &environment->search->evaluator->run_arena,
        sizeof(*entry)
    );
    entry->left_end = left_end;
    entry->result = balanced_select(
        environment->search,
        left_end,
        environment->right_length,
        environment->outer
    );
    entry->next = environment->entries;
    environment->entries = entry;
    return entry->result;
}

static double observe_left(void *raw_environment, Prefix *left_end) {
    PairEnvironment *environment = raw_environment;
    return select_right(environment, left_end).score;
}

static BalancedResult balanced_leaf(
    Search *search,
    Prefix *start,
    BalancedContinuation continuation
) {
    if (start->depth >= search->model->horizon) fail("balanced leaf overflow");
    if (start->terminated) {
        Prefix *end = prefix_child(
            &search->evaluator->prefixes,
            start,
            SEQUENCE_DELIMITER
        );
        return (BalancedResult){end, continuation.apply(
            continuation.environment,
            end
        )};
    }
    Field *field = search->model->logits[start->depth];
    Vec *logits = sample_field(search->evaluator, field, start);
    int support[search->top_k];
    top_k_tokens(logits, search->top_k, support);
    Prefix *best = NULL;
    double best_score = -INFINITY;
    for (int rank = 0; rank < search->top_k; rank++) {
        Prefix *candidate = prefix_child(
            &search->evaluator->prefixes,
            start,
            support[rank]
        );
        double score = continuation.apply(
            continuation.environment,
            candidate
        );
        if (best == NULL || score > best_score) {
            best = candidate;
            best_score = score;
        }
    }
    return (BalancedResult){best, best_score};
}

static BalancedResult balanced_select(
    Search *search,
    Prefix *start,
    int length,
    BalancedContinuation continuation
) {
    if (length <= 0) fail("empty balanced selection span");
    if (length == 1) return balanced_leaf(search, start, continuation);
    int left_length = length / 2;
    int right_length = length - left_length;
    PairEnvironment pair = {
        .search = search,
        .right_length = right_length,
        .outer = continuation,
        .entries = NULL,
    };
    BalancedContinuation left_continuation = {observe_left, &pair};
    BalancedResult left = balanced_select(
        search,
        start,
        left_length,
        left_continuation
    );
    return select_right(&pair, left.end);
}

static void prefix_tokens(Prefix *end, int *tokens) {
    for (int index = end->depth - 1; index >= 0; index--) {
        tokens[index] = end->token;
        end = end->parent;
    }
}

static void print_tokens_line(const char *label, const int *tokens, int count) {
    fputs(label, stdout);
    print_token_array(tokens, count);
    putchar('\n');
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(
            stderr,
            "usage: %s checkpoint tokenizer horizon top-k prompt\n",
            argv[0]
        );
        return EXIT_FAILURE;
    }
    int horizon = parse_integer(argv[3], "horizon");
    int top_k = parse_integer(argv[4], "top-k");
    AtkeyRuntime *runtime = atkey_runtime_new(argv[1], argv[2]);
    if (runtime == NULL) fail("failed to load model");
    Evaluator evaluator = {0};
    evaluator.runtime = runtime;
    evaluator.config = (TermConfig){
        .dim = atkey_dim(runtime),
        .hidden_dim = atkey_hidden_dim(runtime),
        .layers = atkey_layer_count(runtime),
        .heads = atkey_head_count(runtime),
        .kv_heads = atkey_kv_head_count(runtime),
        .vocab = atkey_vocab_size(runtime),
        .sequence_length = atkey_sequence_length(runtime),
    };
    evaluator.term_arena.default_capacity = 1 << 20;
    evaluator.run_arena.default_capacity = 4 << 20;
    memo_init(&evaluator.memo);
    init_prefix_space(&evaluator.prefixes, &evaluator.run_arena);
    int prompt_count = 0;
    int *prompt = atkey_encode(runtime, argv[5], &prompt_count);
    ModelFillers fillers = build_fillers(&evaluator);
    ModelTerm model = model_fields_term(
        &evaluator,
        &fillers,
        prompt,
        prompt_count,
        horizon
    );
    AtkeyProgram program;
    compose_program(
        &program,
        &evaluator,
        &model,
        argv[5],
        prompt[prompt_count - 1],
        top_k,
        false,
        false,
        -INFINITY,
        0,
        0,
        false,
        NULL,
        NULL
    );
    Outcome *right = run_pcont(&program);
    int right_tokens[horizon];
    int right_count = collect_tokens(right, right_tokens);
    double right_score = outcome_reward(right);

    PrefixScoreMemo score_memo = {.search = &program.search};
    BalancedContinuation terminal = {observe_prefix_score, &score_memo};
    BalancedResult balanced = balanced_select(
        &program.search,
        evaluator.prefixes.root,
        horizon,
        terminal
    );
    int balanced_tokens[horizon];
    prefix_tokens(balanced.end, balanced_tokens);

    print_tokens_line("right_tokens=", right_tokens, right_count);
    printf("right_score=%.17g\n", right_score);
    print_tokens_line("balanced_tokens=", balanced_tokens, horizon);
    printf("balanced_score=%.17g\n", balanced.score);
    int first = -1;
    if (right_count != horizon) {
        first = right_count;
    } else {
        for (int index = 0; index < horizon; index++) {
            if (right_tokens[index] != balanced_tokens[index]) {
                first = index;
                break;
            }
        }
    }
    printf("first_disagreement=%d\n", first);

    free(score_memo.values);
    free(score_memo.known);
    atkey_free_tokens(prompt);
    free(evaluator.memo.entries);
    arena_free(&evaluator.run_arena);
    arena_free(&evaluator.term_arena);
    atkey_runtime_free(runtime);
    return first < 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
