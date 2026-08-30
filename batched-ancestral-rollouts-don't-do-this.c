/*
 * DO NOT USE AS SELECTION-PRODUCT INFERENCE.
 *
 * I should never have implemented or presented this as the requested
 * Escardo inference path.  It commits complete ancestral autoregressive
 * rollouts first and only backs their terminal scores through a prefix tree
 * afterward.  Batching those eager paths and retaining their multiplicity
 * does not compose Select at each prefix, so it is best-of-N AR rather than
 * recursive selection-monad strength.  It is kept only as a reminder of the
 * rejected approach.
 */

#define main escardo_strength_depth2_main
#include "exhaustive-prefix-company-don't-do-this.c"
#undef main

/*
 * Long-horizon sampled reachability with batched model evaluation.
 *
 * A rollout batch crosses each learned filler once per generated position,
 * regardless of how many rollout continuations are active in that batch.
 * Complete sampled observations are retained in a shared prefix term.  Each
 * observation is propagated immediately through every remembered predecessor,
 * including repeated visits to the same path.  At each prefix, Select retains
 * the outgoing continuation with the greatest accumulated backed reward.
 *
 * This file deliberately reports repeated per-position filler crossings.  It
 * is a scalable sampled experiment, not the complete-company one-shot claim
 * made by escardo_logit_strength.c for its bounded three-position control.
 */

enum { GAME_NO_NODE = UINT32_MAX };

typedef struct {
    uint32_t parent;
    uint32_t first_child;
    uint32_t next_sibling;
    uint32_t selected_child;
    int token;
    int depth;
    double edge_log_probability;
    bool leaf;
    double leaf_log_reward;
    uint64_t leaf_observations;
    bool backed;
    double backed_log_reward_sum;
    uint64_t backed_observation_count;
    uint64_t backed_unique_leaf_count;
} GameNode;

typedef struct {
    GameNode *nodes;
    uint32_t count;
    uint32_t capacity;
} GameTerm;

typedef struct {
    int token;
    uint32_t node;
    double backed_log_reward_sum;
    uint64_t observation_count;
    uint64_t unique_leaf_count;
} GameBranch;

typedef struct {
    int top_k;
    int horizon;
    int sample_milliseconds;
    int sample_rollouts;
    int batch_size;
    bool verify;
    bool trace_fillers;
} GameOptions;

typedef struct {
    int capacity;
    int horizon;
    int prompt_count;
    int maximum_context;
    int kv_dim;
    float *prompt_key;
    float *prompt_value;
    float *generated_key;
    float *generated_value;
    float *x;
    float *normalized;
    float *query;
    float *key;
    float *value;
    float *attended;
    float *projected;
    float *gate;
    float *up;
    float *down;
    float *final_hidden;
    float *logits;
    float *attention_scores;
} GameBatchState;

typedef struct {
    uint32_t node;
    int current_token;
    int length;
    double log_reward;
    bool active;
} GameRollout;

typedef struct {
    Transformer *transformer;
    const int *prompt_tokens;
    int prompt_count;
    GameOptions options;
    FillerLedger *ledger;
    GameTerm term;
    GameBatchState batch;
    float *root_logits;
    uint64_t attempted_rollouts;
    uint64_t unique_completions;
    uint64_t sampled_path_edges;
    uint64_t rollout_batches;
    uint64_t random_state;
    long prefill_milliseconds;
    long sampling_milliseconds;
} GameProgram;

static uint64_t game_monotonic_nanoseconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        strength_fail("could not read game sampling clock");
    }
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) +
        (uint64_t)now.tv_nsec;
}

static uint64_t game_random_next(uint64_t *state) {
    uint64_t value = *state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    return value * UINT64_C(2685821657736338717);
}

static int game_random_bounded(uint64_t *state, int count) {
    if (count <= 0) strength_fail("sampled empty game support");
    uint64_t bound = (uint64_t)count;
    uint64_t threshold = (uint64_t)(-bound) % bound;
    uint64_t value;
    do {
        value = game_random_next(state);
    } while (value < threshold);
    return (int)(value % bound);
}

static double game_random_unit(uint64_t *state) {
    return (double)(game_random_next(state) >> 11) *
        (1.0 / 9007199254740992.0);
}

static double game_log_add(double total, double reward, bool has_total) {
    if (!has_total) return reward;
    double maximum = total > reward ? total : reward;
    double minimum = total > reward ? reward : total;
    return maximum + log1p(exp(minimum - maximum));
}

static void game_term_init(GameTerm *term) {
    term->capacity = 1024;
    term->nodes = strength_calloc(
        (size_t)term->capacity,
        sizeof(*term->nodes)
    );
    term->count = 1;
    term->nodes[0] = (GameNode){
        .parent = GAME_NO_NODE,
        .first_child = GAME_NO_NODE,
        .next_sibling = GAME_NO_NODE,
        .selected_child = GAME_NO_NODE,
        .token = -1,
        .depth = 0,
    };
}

static void game_term_grow(GameTerm *term) {
    if (term->capacity > UINT32_MAX / 2) {
        strength_fail("sampled game term exceeded uint32 capacity");
    }
    uint32_t old_capacity = term->capacity;
    uint32_t new_capacity = old_capacity * 2;
    GameNode *grown = realloc(
        term->nodes,
        (size_t)new_capacity * sizeof(*grown)
    );
    if (grown == NULL) strength_fail("could not grow sampled game term");
    memset(
        grown + old_capacity,
        0,
        (size_t)(new_capacity - old_capacity) * sizeof(*grown)
    );
    term->nodes = grown;
    term->capacity = new_capacity;
}

static uint32_t game_child(
    GameTerm *term,
    uint32_t parent,
    int token,
    double edge_log_probability
) {
    if (term->nodes == NULL || term->capacity == 0 ||
        parent >= term->count) {
        strength_fail("sampled game child received an invalid prefix term");
    }
    for (uint32_t child = term->nodes[parent].first_child;
         child != GAME_NO_NODE;
         child = term->nodes[child].next_sibling) {
        if (term->nodes[child].token == token) {
            if (fabs(term->nodes[child].edge_log_probability -
                     edge_log_probability) > 1e-10) {
                strength_fail("memoized game edge changed probability");
            }
            return child;
        }
    }
    if (term->count == term->capacity) game_term_grow(term);
    uint32_t child = term->count++;
    term->nodes[child] = (GameNode){
        .parent = parent,
        .first_child = GAME_NO_NODE,
        .next_sibling = term->nodes[parent].first_child,
        .selected_child = GAME_NO_NODE,
        .token = token,
        .depth = term->nodes[parent].depth + 1,
        .edge_log_probability = edge_log_probability,
    };
    term->nodes[parent].first_child = child;
    return child;
}

static void game_term_free(GameTerm *term) {
    free(term->nodes);
    memset(term, 0, sizeof(*term));
}

static size_t game_prompt_cache_index(
    const Config *config,
    const GameBatchState *state,
    int layer,
    int position
) {
    (void)config;
    return ((size_t)layer * state->prompt_count + position) * state->kv_dim;
}

static size_t game_generated_cache_index(
    const Config *config,
    const GameBatchState *state,
    int layer,
    int rollout,
    int position
) {
    (void)config;
    return (((size_t)layer * state->capacity + rollout) * state->horizon +
            position) * state->kv_dim;
}

static GameBatchState game_batch_state_new(
    const Config *config,
    int capacity,
    int horizon,
    int prompt_count
) {
    int dim = config->dim;
    int hidden_dim = config->hidden_dim;
    int kv_dim = dim * config->n_kv_heads / config->n_heads;
    GameBatchState state = {
        .capacity = capacity,
        .horizon = horizon,
        .prompt_count = prompt_count,
        .maximum_context = prompt_count + horizon,
        .kv_dim = kv_dim,
    };
    state.prompt_key = strength_calloc(
        strength_elements(config->n_layers * prompt_count, kv_dim),
        sizeof(*state.prompt_key)
    );
    state.prompt_value = strength_calloc(
        strength_elements(config->n_layers * prompt_count, kv_dim),
        sizeof(*state.prompt_value)
    );
    state.generated_key = strength_calloc(
        strength_elements(config->n_layers * capacity * horizon, kv_dim),
        sizeof(*state.generated_key)
    );
    state.generated_value = strength_calloc(
        strength_elements(config->n_layers * capacity * horizon, kv_dim),
        sizeof(*state.generated_value)
    );
#define GAME_ALLOC(field, rows, width) \
    state.field = strength_calloc(strength_elements((rows), (width)), \
                                  sizeof(*state.field))
    GAME_ALLOC(x, capacity, dim);
    GAME_ALLOC(normalized, capacity, dim);
    GAME_ALLOC(query, capacity, dim);
    GAME_ALLOC(key, capacity, kv_dim);
    GAME_ALLOC(value, capacity, kv_dim);
    GAME_ALLOC(attended, capacity, dim);
    GAME_ALLOC(projected, capacity, dim);
    GAME_ALLOC(gate, capacity, hidden_dim);
    GAME_ALLOC(up, capacity, hidden_dim);
    GAME_ALLOC(down, capacity, dim);
    GAME_ALLOC(final_hidden, capacity, dim);
    GAME_ALLOC(logits, capacity, config->vocab_size);
#undef GAME_ALLOC
    state.attention_scores = strength_calloc(
        (size_t)state.maximum_context,
        sizeof(*state.attention_scores)
    );
    return state;
}

static void game_batch_state_free(GameBatchState *state) {
    free(state->attention_scores);
    free(state->logits);
    free(state->final_hidden);
    free(state->down);
    free(state->up);
    free(state->gate);
    free(state->projected);
    free(state->attended);
    free(state->value);
    free(state->key);
    free(state->query);
    free(state->normalized);
    free(state->x);
    free(state->generated_value);
    free(state->generated_key);
    free(state->prompt_value);
    free(state->prompt_key);
    memset(state, 0, sizeof(*state));
}

static void game_rotate_query_key(
    float *query,
    float *key,
    int position,
    int dim,
    int kv_dim,
    int head_size
) {
    for (int lane = 0; lane < dim; lane += 2) {
        int head_dimension = lane % head_size;
        float frequency = 1.0f /
            powf(10000.0f, head_dimension / (float)head_size);
        float angle = position * frequency;
        float real = cosf(angle);
        float imaginary = sinf(angle);
        float q0 = query[lane];
        float q1 = query[lane + 1];
        query[lane] = q0 * real - q1 * imaginary;
        query[lane + 1] = q0 * imaginary + q1 * real;
        if (lane < kv_dim) {
            float k0 = key[lane];
            float k1 = key[lane + 1];
            key[lane] = k0 * real - k1 * imaginary;
            key[lane + 1] = k0 * imaginary + k1 * real;
        }
    }
}

static void game_prefill(GameProgram *program) {
    Transformer *transformer = program->transformer;
    Config *config = &transformer->config;
    TransformerWeights *weights = &transformer->weights;
    GameBatchState *state = &program->batch;
    int count = program->prompt_count;
    int dim = config->dim;
    int hidden_dim = config->hidden_dim;
    int kv_dim = state->kv_dim;
    int head_size = dim / config->n_heads;
    int kv_mul = config->n_heads / config->n_kv_heads;

#define GAME_PROMPT_ALLOC(name, width) \
    float *name = strength_calloc(strength_elements(count, (width)), \
                                  sizeof(*name))
    GAME_PROMPT_ALLOC(x, dim);
    GAME_PROMPT_ALLOC(normalized, dim);
    GAME_PROMPT_ALLOC(query, dim);
    GAME_PROMPT_ALLOC(key, kv_dim);
    GAME_PROMPT_ALLOC(value, kv_dim);
    GAME_PROMPT_ALLOC(attended, dim);
    GAME_PROMPT_ALLOC(projected, dim);
    GAME_PROMPT_ALLOC(gate, hidden_dim);
    GAME_PROMPT_ALLOC(up, hidden_dim);
    GAME_PROMPT_ALLOC(down, dim);
#undef GAME_PROMPT_ALLOC
    float *scores = strength_calloc((size_t)count, sizeof(*scores));

    embedding_company_apply(
        x,
        program->prompt_tokens,
        count,
        weights->token_embedding_table,
        config->vocab_size,
        dim,
        program->ledger,
        embedding_filler()
    );
    for (int layer = 0; layer < config->n_layers; layer++) {
        rms_company_apply(
            normalized,
            x,
            weights->rms_att_weight + (size_t)layer * dim,
            count,
            dim,
            program->ledger,
            layer_filler(layer, FILLER_ATTN_RMS)
        );
        matmul_company_apply(
            query,
            normalized,
            weights->wq + (size_t)layer * dim * dim,
            count,
            dim,
            dim,
            program->ledger,
            layer_filler(layer, FILLER_QUERY)
        );
        matmul_company_apply(
            key,
            normalized,
            weights->wk + (size_t)layer * dim * kv_dim,
            count,
            dim,
            kv_dim,
            program->ledger,
            layer_filler(layer, FILLER_KEY)
        );
        matmul_company_apply(
            value,
            normalized,
            weights->wv + (size_t)layer * dim * kv_dim,
            count,
            dim,
            kv_dim,
            program->ledger,
            layer_filler(layer, FILLER_VALUE)
        );

        for (int position = 0; position < count; position++) {
            game_rotate_query_key(
                query + (size_t)position * dim,
                key + (size_t)position * kv_dim,
                position,
                dim,
                kv_dim,
                head_size
            );
            memcpy(
                state->prompt_key + game_prompt_cache_index(
                    config,
                    state,
                    layer,
                    position
                ),
                key + (size_t)position * kv_dim,
                (size_t)kv_dim * sizeof(*key)
            );
            memcpy(
                state->prompt_value + game_prompt_cache_index(
                    config,
                    state,
                    layer,
                    position
                ),
                value + (size_t)position * kv_dim,
                (size_t)kv_dim * sizeof(*value)
            );
        }

        for (int position = 0; position < count; position++) {
            float *row_output = attended + (size_t)position * dim;
            const float *row_query = query + (size_t)position * dim;
            for (int head = 0; head < config->n_heads; head++) {
                const float *head_query = row_query + head * head_size;
                for (int context = 0; context <= position; context++) {
                    const float *head_key = key + (size_t)context * kv_dim +
                        (head / kv_mul) * head_size;
                    float score = 0.0f;
                    for (int lane = 0; lane < head_size; lane++) {
                        score += head_query[lane] * head_key[lane];
                    }
                    scores[context] = score / sqrtf(head_size);
                }
                softmax(scores, position + 1);
                float *head_output = row_output + head * head_size;
                memset(
                    head_output,
                    0,
                    (size_t)head_size * sizeof(*head_output)
                );
                for (int context = 0; context <= position; context++) {
                    const float *head_value = value +
                        (size_t)context * kv_dim +
                        (head / kv_mul) * head_size;
                    float mass = scores[context];
                    for (int lane = 0; lane < head_size; lane++) {
                        head_output[lane] += mass * head_value[lane];
                    }
                }
            }
        }

        matmul_company_apply(
            projected,
            attended,
            weights->wo + (size_t)layer * dim * dim,
            count,
            dim,
            dim,
            program->ledger,
            layer_filler(layer, FILLER_ATTN_OUTPUT)
        );
        for (size_t index = 0; index < strength_elements(count, dim); index++) {
            x[index] += projected[index];
        }
        rms_company_apply(
            normalized,
            x,
            weights->rms_ffn_weight + (size_t)layer * dim,
            count,
            dim,
            program->ledger,
            layer_filler(layer, FILLER_FFN_RMS)
        );
        matmul_company_apply(
            gate,
            normalized,
            weights->w1 + (size_t)layer * dim * hidden_dim,
            count,
            dim,
            hidden_dim,
            program->ledger,
            layer_filler(layer, FILLER_FFN_GATE)
        );
        matmul_company_apply(
            up,
            normalized,
            weights->w3 + (size_t)layer * dim * hidden_dim,
            count,
            dim,
            hidden_dim,
            program->ledger,
            layer_filler(layer, FILLER_FFN_UP)
        );
        for (size_t index = 0;
             index < strength_elements(count, hidden_dim);
             index++) {
            float raw = gate[index];
            raw *= 1.0f / (1.0f + expf(-raw));
            gate[index] = raw * up[index];
        }
        matmul_company_apply(
            down,
            gate,
            weights->w2 + (size_t)layer * hidden_dim * dim,
            count,
            hidden_dim,
            dim,
            program->ledger,
            layer_filler(layer, FILLER_FFN_DOWN)
        );
        for (size_t index = 0; index < strength_elements(count, dim); index++) {
            x[index] += down[index];
        }
    }

    float *last = x + (size_t)(count - 1) * dim;
    float *final = strength_calloc((size_t)dim, sizeof(*final));
    rms_company_apply(
        final,
        last,
        weights->rms_final_weight,
        1,
        dim,
        program->ledger,
        final_rms_filler(config)
    );
    program->root_logits = strength_calloc(
        (size_t)config->vocab_size,
        sizeof(*program->root_logits)
    );
    matmul_company_apply(
        program->root_logits,
        final,
        weights->wcls,
        1,
        dim,
        config->vocab_size,
        program->ledger,
        classifier_filler(config)
    );

    free(final);
    free(scores);
    free(down);
    free(up);
    free(gate);
    free(projected);
    free(attended);
    free(value);
    free(key);
    free(query);
    free(normalized);
    free(x);
}

static float *game_batch_step(
    GameProgram *program,
    const int *rollout_ids,
    const int *tokens,
    int active_count,
    int generated_position
) {
    Transformer *transformer = program->transformer;
    Config *config = &transformer->config;
    TransformerWeights *weights = &transformer->weights;
    GameBatchState *state = &program->batch;
    int dim = config->dim;
    int hidden_dim = config->hidden_dim;
    int kv_dim = state->kv_dim;
    int head_size = dim / config->n_heads;
    int kv_mul = config->n_heads / config->n_kv_heads;
    int absolute_position = program->prompt_count + generated_position;
    if (active_count <= 0 || active_count > state->capacity) {
        strength_fail("invalid active rollout company");
    }

    embedding_company_apply(
        state->x,
        tokens,
        active_count,
        weights->token_embedding_table,
        config->vocab_size,
        dim,
        program->ledger,
        embedding_filler()
    );
    for (int layer = 0; layer < config->n_layers; layer++) {
        rms_company_apply(
            state->normalized,
            state->x,
            weights->rms_att_weight + (size_t)layer * dim,
            active_count,
            dim,
            program->ledger,
            layer_filler(layer, FILLER_ATTN_RMS)
        );
        matmul_company_apply(
            state->query,
            state->normalized,
            weights->wq + (size_t)layer * dim * dim,
            active_count,
            dim,
            dim,
            program->ledger,
            layer_filler(layer, FILLER_QUERY)
        );
        matmul_company_apply(
            state->key,
            state->normalized,
            weights->wk + (size_t)layer * dim * kv_dim,
            active_count,
            dim,
            kv_dim,
            program->ledger,
            layer_filler(layer, FILLER_KEY)
        );
        matmul_company_apply(
            state->value,
            state->normalized,
            weights->wv + (size_t)layer * dim * kv_dim,
            active_count,
            dim,
            kv_dim,
            program->ledger,
            layer_filler(layer, FILLER_VALUE)
        );

        for (int row = 0; row < active_count; row++) {
            game_rotate_query_key(
                state->query + (size_t)row * dim,
                state->key + (size_t)row * kv_dim,
                absolute_position,
                dim,
                kv_dim,
                head_size
            );
            int rollout = rollout_ids[row];
            memcpy(
                state->generated_key + game_generated_cache_index(
                    config,
                    state,
                    layer,
                    rollout,
                    generated_position
                ),
                state->key + (size_t)row * kv_dim,
                (size_t)kv_dim * sizeof(*state->key)
            );
            memcpy(
                state->generated_value + game_generated_cache_index(
                    config,
                    state,
                    layer,
                    rollout,
                    generated_position
                ),
                state->value + (size_t)row * kv_dim,
                (size_t)kv_dim * sizeof(*state->value)
            );
        }

        int context_count = absolute_position + 1;
        for (int row = 0; row < active_count; row++) {
            int rollout = rollout_ids[row];
            const float *row_query = state->query + (size_t)row * dim;
            float *row_output = state->attended + (size_t)row * dim;
            for (int head = 0; head < config->n_heads; head++) {
                const float *head_query = row_query + head * head_size;
                for (int context = 0; context < context_count; context++) {
                    const float *context_key;
                    if (context < program->prompt_count) {
                        context_key = state->prompt_key +
                            game_prompt_cache_index(
                                config,
                                state,
                                layer,
                                context
                            );
                    } else {
                        context_key = state->generated_key +
                            game_generated_cache_index(
                                config,
                                state,
                                layer,
                                rollout,
                                context - program->prompt_count
                            );
                    }
                    const float *head_key = context_key +
                        (head / kv_mul) * head_size;
                    float score = 0.0f;
                    for (int lane = 0; lane < head_size; lane++) {
                        score += head_query[lane] * head_key[lane];
                    }
                    state->attention_scores[context] =
                        score / sqrtf(head_size);
                }
                softmax(state->attention_scores, context_count);
                float *head_output = row_output + head * head_size;
                memset(
                    head_output,
                    0,
                    (size_t)head_size * sizeof(*head_output)
                );
                for (int context = 0; context < context_count; context++) {
                    const float *context_value;
                    if (context < program->prompt_count) {
                        context_value = state->prompt_value +
                            game_prompt_cache_index(
                                config,
                                state,
                                layer,
                                context
                            );
                    } else {
                        context_value = state->generated_value +
                            game_generated_cache_index(
                                config,
                                state,
                                layer,
                                rollout,
                                context - program->prompt_count
                            );
                    }
                    const float *head_value = context_value +
                        (head / kv_mul) * head_size;
                    float mass = state->attention_scores[context];
                    for (int lane = 0; lane < head_size; lane++) {
                        head_output[lane] += mass * head_value[lane];
                    }
                }
            }
        }

        matmul_company_apply(
            state->projected,
            state->attended,
            weights->wo + (size_t)layer * dim * dim,
            active_count,
            dim,
            dim,
            program->ledger,
            layer_filler(layer, FILLER_ATTN_OUTPUT)
        );
        for (size_t index = 0;
             index < strength_elements(active_count, dim);
             index++) {
            state->x[index] += state->projected[index];
        }
        rms_company_apply(
            state->normalized,
            state->x,
            weights->rms_ffn_weight + (size_t)layer * dim,
            active_count,
            dim,
            program->ledger,
            layer_filler(layer, FILLER_FFN_RMS)
        );
        matmul_company_apply(
            state->gate,
            state->normalized,
            weights->w1 + (size_t)layer * dim * hidden_dim,
            active_count,
            dim,
            hidden_dim,
            program->ledger,
            layer_filler(layer, FILLER_FFN_GATE)
        );
        matmul_company_apply(
            state->up,
            state->normalized,
            weights->w3 + (size_t)layer * dim * hidden_dim,
            active_count,
            dim,
            hidden_dim,
            program->ledger,
            layer_filler(layer, FILLER_FFN_UP)
        );
        for (size_t index = 0;
             index < strength_elements(active_count, hidden_dim);
             index++) {
            float raw = state->gate[index];
            raw *= 1.0f / (1.0f + expf(-raw));
            state->gate[index] = raw * state->up[index];
        }
        matmul_company_apply(
            state->down,
            state->gate,
            weights->w2 + (size_t)layer * hidden_dim * dim,
            active_count,
            hidden_dim,
            dim,
            program->ledger,
            layer_filler(layer, FILLER_FFN_DOWN)
        );
        for (size_t index = 0;
             index < strength_elements(active_count, dim);
             index++) {
            state->x[index] += state->down[index];
        }
    }

    rms_company_apply(
        state->final_hidden,
        state->x,
        weights->rms_final_weight,
        active_count,
        dim,
        program->ledger,
        final_rms_filler(config)
    );
    matmul_company_apply(
        state->logits,
        state->final_hidden,
        weights->wcls,
        active_count,
        dim,
        config->vocab_size,
        program->ledger,
        classifier_filler(config)
    );
    return state->logits;
}

static bool game_logit_precedes(
    const float *logits,
    int left,
    int right
) {
    if (logits[left] > logits[right]) return true;
    if (logits[left] < logits[right]) return false;
    return left < right;
}

static void game_top_k(
    const float *logits,
    int vocab_size,
    int top_k,
    int *support
) {
    int filled = 0;
    for (int token = 0; token < vocab_size; token++) {
        int insertion = filled;
        while (insertion > 0 &&
               game_logit_precedes(logits, token, support[insertion - 1])) {
            insertion--;
        }
        if (insertion < top_k) {
            int last = filled < top_k ? filled : top_k - 1;
            for (int index = last; index > insertion; index--) {
                support[index] = support[index - 1];
            }
            support[insertion] = token;
            if (filled < top_k) filled++;
        }
    }
    if (filled != top_k) strength_fail("game top-k support incomplete");
}

static double game_log_probability(
    const float *logits,
    int vocab_size,
    int token
) {
    double maximum = -DBL_MAX;
    for (int candidate = 0; candidate < vocab_size; candidate++) {
        if ((double)logits[candidate] > maximum) maximum = logits[candidate];
    }
    double partition = 0.0;
    for (int candidate = 0; candidate < vocab_size; candidate++) {
        partition += exp((double)logits[candidate] - maximum);
    }
    return (double)logits[token] - maximum - log(partition);
}

static int game_sample_token(
    const float *logits,
    int vocab_size,
    int top_k,
    uint64_t *random_state,
    int *support,
    double *log_probability
) {
    game_top_k(logits, vocab_size, top_k, support);
    double maximum = -DBL_MAX;
    for (int rank = 0; rank < top_k; rank++) {
        if ((double)logits[support[rank]] > maximum) {
            maximum = logits[support[rank]];
        }
    }
    double mass = 0.0;
    for (int rank = 0; rank < top_k; rank++) {
        mass += exp((double)logits[support[rank]] - maximum);
    }
    double target = game_random_unit(random_state) * mass;
    int token = support[top_k - 1];
    double cumulative = 0.0;
    for (int rank = 0; rank < top_k; rank++) {
        cumulative += exp((double)logits[support[rank]] - maximum);
        if (target < cumulative) {
            token = support[rank];
            break;
        }
    }
    *log_probability = game_log_probability(logits, vocab_size, token);
    return token;
}

static void game_finish_rollout(
    GameProgram *program,
    GameRollout *rollout
) {
    GameNode *leaf = &program->term.nodes[rollout->node];
    bool new_leaf = !leaf->leaf;
    if (leaf->leaf) {
        if (fabs(leaf->leaf_log_reward - rollout->log_reward) > 1e-10) {
            strength_fail("memoized sampled completion changed reward");
        }
    } else {
        leaf->leaf = true;
        leaf->leaf_log_reward = rollout->log_reward;
        program->unique_completions++;
    }

    leaf->leaf_observations++;
    uint32_t node_id = rollout->node;
    while (node_id != GAME_NO_NODE) {
        GameNode *node = &program->term.nodes[node_id];
        node->backed_log_reward_sum = game_log_add(
            node->backed_log_reward_sum,
            rollout->log_reward,
            node->backed
        );
        node->backed = true;
        node->backed_observation_count++;
        if (new_leaf) node->backed_unique_leaf_count++;

        uint32_t parent_id = node->parent;
        if (parent_id != GAME_NO_NODE) {
            GameNode *parent = &program->term.nodes[parent_id];
            if (parent->selected_child == GAME_NO_NODE) {
                parent->selected_child = node_id;
            } else {
                GameNode *selected =
                    &program->term.nodes[parent->selected_child];
                if (node->backed_log_reward_sum >
                        selected->backed_log_reward_sum ||
                    (node->backed_log_reward_sum ==
                         selected->backed_log_reward_sum &&
                     node->token < selected->token)) {
                    parent->selected_child = node_id;
                }
            }
        }
        node_id = parent_id;
    }
}

static void game_sample_batch(GameProgram *program, int batch_size) {
    Config *config = &program->transformer->config;
    if (batch_size <= 0 || batch_size > program->options.batch_size) {
        strength_fail("invalid sampled rollout batch size");
    }
    int top_k = program->options.top_k;
    int horizon = program->options.horizon;
    GameRollout *rollouts = strength_calloc(
        (size_t)batch_size,
        sizeof(*rollouts)
    );
    int *active_ids = strength_calloc(
        (size_t)batch_size,
        sizeof(*active_ids)
    );
    int *active_tokens = strength_calloc(
        (size_t)batch_size,
        sizeof(*active_tokens)
    );
    int *support = strength_calloc((size_t)top_k, sizeof(*support));
    int *root_support = strength_calloc((size_t)top_k, sizeof(*root_support));
    game_top_k(
        program->root_logits,
        config->vocab_size,
        top_k,
        root_support
    );

    for (int rollout = 0; rollout < batch_size; rollout++) {
        double ignored_log_probability;
        int token = game_sample_token(
            program->root_logits,
            config->vocab_size,
            top_k,
            &program->random_state,
            root_support,
            &ignored_log_probability
        );
        double log_probability = game_log_probability(
            program->root_logits,
            config->vocab_size,
            token
        );
        uint32_t node = game_child(
            &program->term,
            0,
            token,
            log_probability
        );
        rollouts[rollout] = (GameRollout){
            .node = node,
            .current_token = token,
            .length = 1,
            .log_reward = log_probability,
            .active = token != 2 && horizon > 1,
        };
        program->sampled_path_edges++;
    }

    for (int depth = 1; depth < horizon; depth++) {
        int active_count = 0;
        for (int rollout = 0; rollout < batch_size; rollout++) {
            if (!rollouts[rollout].active) continue;
            active_ids[active_count] = rollout;
            active_tokens[active_count] = rollouts[rollout].current_token;
            active_count++;
        }
        if (active_count == 0) break;
        float *logits = game_batch_step(
            program,
            active_ids,
            active_tokens,
            active_count,
            depth - 1
        );
        for (int row = 0; row < active_count; row++) {
            int rollout_index = active_ids[row];
            GameRollout *rollout = &rollouts[rollout_index];
            const float *row_logits = logits +
                (size_t)row * config->vocab_size;
            double log_probability;
            int token = game_sample_token(
                row_logits,
                config->vocab_size,
                top_k,
                &program->random_state,
                support,
                &log_probability
            );
            rollout->node = game_child(
                &program->term,
                rollout->node,
                token,
                log_probability
            );
            rollout->current_token = token;
            rollout->length++;
            rollout->log_reward += log_probability;
            rollout->active = token != 2 && rollout->length < horizon;
            program->sampled_path_edges++;
        }
    }

    for (int rollout = 0; rollout < batch_size; rollout++) {
        game_finish_rollout(program, &rollouts[rollout]);
    }
    program->attempted_rollouts += (uint64_t)batch_size;
    program->rollout_batches++;
    free(root_support);
    free(support);
    free(active_tokens);
    free(active_ids);
    free(rollouts);
}

typedef struct {
    bool backed;
    double log_reward_sum;
    uint64_t observations;
    uint64_t unique_leaves;
} GameAudit;

static GameAudit game_audit_node(const GameTerm *term, uint32_t node_id) {
    const GameNode *node = &term->nodes[node_id];
    GameAudit audit = {0};
    if (node->leaf) {
        if (node->leaf_observations == 0) {
            strength_fail("sampled leaf retained no observations");
        }
        audit.backed = true;
        audit.log_reward_sum = node->leaf_log_reward +
            log((double)node->leaf_observations);
        audit.observations = node->leaf_observations;
        audit.unique_leaves = 1;
    }

    uint32_t expected_selected = GAME_NO_NODE;
    for (uint32_t child = node->first_child;
         child != GAME_NO_NODE;
         child = term->nodes[child].next_sibling) {
        GameAudit child_audit = game_audit_node(term, child);
        const GameNode *child_node = &term->nodes[child];
        if (!child_audit.backed) continue;
        audit.log_reward_sum = game_log_add(
            audit.log_reward_sum,
            child_audit.log_reward_sum,
            audit.backed
        );
        audit.backed = true;
        audit.observations += child_audit.observations;
        audit.unique_leaves += child_audit.unique_leaves;
        if (expected_selected == GAME_NO_NODE) {
            expected_selected = child;
        } else {
            const GameNode *selected = &term->nodes[expected_selected];
            if (child_node->backed_log_reward_sum >
                    selected->backed_log_reward_sum ||
                (child_node->backed_log_reward_sum ==
                     selected->backed_log_reward_sum &&
                 child_node->token < selected->token)) {
                expected_selected = child;
            }
        }
    }

    if (audit.backed != node->backed ||
        audit.observations != node->backed_observation_count ||
        audit.unique_leaves != node->backed_unique_leaf_count ||
        (audit.backed &&
         fabs(audit.log_reward_sum - node->backed_log_reward_sum) > 1e-9)) {
        strength_fail("online sampled reward backup failed its tree audit");
    }
    if (expected_selected != node->selected_child) {
        strength_fail("online sampled selection disagrees with backed rewards");
    }
    return audit;
}

static void game_run(GameProgram *program) {
    long prefill_started = time_in_ms();
    game_prefill(program);
    program->prefill_milliseconds = time_in_ms() - prefill_started;

    uint64_t started = game_monotonic_nanoseconds();
    if (program->options.sample_rollouts > 0) {
        while (program->attempted_rollouts <
               (uint64_t)program->options.sample_rollouts) {
            uint64_t remaining =
                (uint64_t)program->options.sample_rollouts -
                program->attempted_rollouts;
            int batch_size = remaining <
                    (uint64_t)program->options.batch_size ?
                (int)remaining : program->options.batch_size;
            game_sample_batch(program, batch_size);
        }
    } else {
        uint64_t deadline = started +
            (uint64_t)program->options.sample_milliseconds *
                UINT64_C(1000000);
        do {
            game_sample_batch(program, program->options.batch_size);
        } while (game_monotonic_nanoseconds() < deadline);
    }
    program->sampling_milliseconds = (long)(
        (game_monotonic_nanoseconds() - started) / UINT64_C(1000000)
    );
    GameAudit audit = game_audit_node(&program->term, 0);
    if (audit.observations != program->attempted_rollouts ||
        audit.unique_leaves != program->unique_completions) {
        strength_fail("root reachability counts disagree with sampled rollouts");
    }
    if (!program->term.nodes[0].backed ||
        program->term.nodes[0].selected_child == GAME_NO_NODE) {
        strength_fail("sampled game produced no selectable completion");
    }
}

static int game_selected_path(
    const GameTerm *term,
    int *tokens,
    uint32_t *nodes,
    int capacity
) {
    int count = 0;
    uint32_t node = term->nodes[0].selected_child;
    while (node != GAME_NO_NODE) {
        if (count >= capacity) strength_fail("selected game path overflow");
        tokens[count] = term->nodes[node].token;
        nodes[count] = node;
        count++;
        node = term->nodes[node].selected_child;
    }
    if (count == 0 || !term->nodes[nodes[count - 1]].leaf) {
        strength_fail("selected game path did not end at a sampled leaf");
    }
    return count;
}

static int game_best_observed_path(
    const GameTerm *term,
    int *tokens,
    uint32_t *nodes,
    int capacity,
    double *reward
) {
    uint32_t leaf = GAME_NO_NODE;
    double best_reward = -INFINITY;
    for (uint32_t node = 1; node < term->count; node++) {
        if (!term->nodes[node].leaf) continue;
        double candidate = term->nodes[node].leaf_log_reward;
        if (leaf == GAME_NO_NODE || candidate > best_reward ||
            (candidate == best_reward && node < leaf)) {
            leaf = node;
            best_reward = candidate;
        }
    }
    if (leaf == GAME_NO_NODE) {
        strength_fail("sampled game retained no complete path");
    }

    int count = term->nodes[leaf].depth;
    if (count <= 0 || count > capacity) {
        strength_fail("best sampled game path overflow");
    }
    uint32_t node = leaf;
    for (int index = count - 1; index >= 0; index--) {
        tokens[index] = term->nodes[node].token;
        nodes[index] = node;
        node = term->nodes[node].parent;
    }
    if (node != 0) strength_fail("best sampled path lost its root");
    *reward = best_reward;
    return count;
}

static int compare_game_branch(const void *left_raw, const void *right_raw) {
    const GameBranch *left = left_raw;
    const GameBranch *right = right_raw;
    if (left->backed_log_reward_sum > right->backed_log_reward_sum) return -1;
    if (left->backed_log_reward_sum < right->backed_log_reward_sum) return 1;
    if (left->token < right->token) return -1;
    if (left->token > right->token) return 1;
    return 0;
}

static void game_print_completion(
    Tokenizer *tokenizer,
    int previous,
    const int *tokens,
    int count
) {
    for (int index = 0; index < count; index++) {
        print_escaped(decode(tokenizer, previous, tokens[index]));
        previous = tokens[index];
    }
}

static void game_print_decision_trace(
    const GameProgram *program,
    Tokenizer *tokenizer,
    int prompt_last,
    const int *selected_tokens,
    const uint32_t *selected_nodes,
    int selected_count
) {
    puts("backward_reward_trace:");
    uint32_t parent = 0;
    int previous = prompt_last;
    for (int depth = 0; depth < selected_count; depth++) {
        int child_count = 0;
        for (uint32_t child = program->term.nodes[parent].first_child;
             child != GAME_NO_NODE;
             child = program->term.nodes[child].next_sibling) {
            child_count++;
        }
        if (child_count <= 0) {
            strength_fail("selected sampled prefix has no alternatives");
        }
        GameBranch *branches = strength_calloc(
            (size_t)child_count,
            sizeof(*branches)
        );
        int output = 0;
        for (uint32_t child = program->term.nodes[parent].first_child;
             child != GAME_NO_NODE;
             child = program->term.nodes[child].next_sibling) {
            GameNode *node = &program->term.nodes[child];
            branches[output++] = (GameBranch){
                .token = node->token,
                .node = child,
                .backed_log_reward_sum = node->backed_log_reward_sum,
                .observation_count = node->backed_observation_count,
                .unique_leaf_count = node->backed_unique_leaf_count,
            };
        }
        qsort(
            branches,
            (size_t)child_count,
            sizeof(*branches),
            compare_game_branch
        );
        printf(
            "  depth=%d selected_token=%d selected_piece=",
            depth,
            selected_tokens[depth]
        );
        print_piece(tokenizer, previous, selected_tokens[depth]);
        printf(" alternatives=%d\n", child_count);
        int shown = child_count < 5 ? child_count : 5;
        for (int rank = 0; rank < shown; rank++) {
            GameBranch branch = branches[rank];
            printf("    rank=%d token=%d piece=", rank + 1, branch.token);
            print_piece(tokenizer, previous, branch.token);
            printf(
                " backed_log_reward_sum=%.17g "
                "observations=%" PRIu64 " unique_completions=%" PRIu64
                "%s\n",
                branch.backed_log_reward_sum,
                branch.observation_count,
                branch.unique_leaf_count,
                branch.node == selected_nodes[depth] ? " selected=1" : ""
            );
        }
        free(branches);
        parent = selected_nodes[depth];
        previous = selected_tokens[depth];
    }
}

static uint64_t game_ledger_total_crossings(const FillerLedger *ledger) {
    uint64_t total = 0;
    for (int filler = 0; filler < ledger->filler_count; filler++) {
        total += ledger->fillers[filler].crossings;
    }
    return total;
}

static void game_ledger_crossing_range(
    const FillerLedger *ledger,
    uint64_t *minimum,
    uint64_t *maximum
) {
    *minimum = UINT64_MAX;
    *maximum = 0;
    for (int filler = 0; filler < ledger->filler_count; filler++) {
        uint64_t crossings = ledger->fillers[filler].crossings;
        if (crossings < *minimum) *minimum = crossings;
        if (crossings > *maximum) *maximum = crossings;
    }
}

static void game_verify_selected_path(
    GameProgram *program,
    const int *tokens,
    const uint32_t *nodes,
    int count
) {
    Transformer *transformer = program->transformer;
    Config *config = &transformer->config;
    float *logits = NULL;
    for (int position = 0; position < program->prompt_count; position++) {
        logits = forward(
            transformer,
            program->prompt_tokens[position],
            position
        );
    }
    uint64_t failures = 0;
    double maximum_error = 0.0;
    for (int index = 0; index < count; index++) {
        double oracle = game_log_probability(
            logits,
            config->vocab_size,
            tokens[index]
        );
        double actual = program->term.nodes[nodes[index]].edge_log_probability;
        double error = fabs(oracle - actual);
        if (error > maximum_error) maximum_error = error;
        if (error > 1e-10) failures++;
        if (index + 1 < count) {
            logits = forward(
                transformer,
                tokens[index],
                program->prompt_count + index
            );
        }
    }
    printf(
        "llama2_selected_edge_log_probabilities=%d failures=%" PRIu64
        " max_abs_error=%.17g\n",
        count,
        failures,
        maximum_error
    );
    if (failures != 0) strength_fail("batched game disagrees with llama2.c");
}

static int game_parse_integer(const char *text, const char *option) {
    errno = 0;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value < 0 || value > INT32_MAX) {
        fprintf(stderr, "invalid %s value: %s\n", option, text);
        exit(EXIT_FAILURE);
    }
    return (int)value;
}

static void game_usage(const char *program) {
    fprintf(
        stderr,
        "usage: %s CHECKPOINT TOKENIZER PROMPT [-n TOKENS] [-k TOP_K] "
        "[--sample-ms MS | --sample-rollouts COUNT] [--batch SIZE] "
        "[--verify] [--trace-fillers]\n",
        program
    );
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
    if (argc < 4) game_usage(argv[0]);
    GameOptions options = {
        .top_k = 4,
        .horizon = 32,
        .sample_milliseconds = 100,
        .batch_size = 16,
    };
    for (int argument = 4; argument < argc; argument++) {
        const char *flag = argv[argument];
        if (strcmp(flag, "--verify") == 0) {
            options.verify = true;
        } else if (strcmp(flag, "--trace-fillers") == 0) {
            options.trace_fillers = true;
        } else if (strcmp(flag, "-n") == 0 ||
                   strcmp(flag, "-k") == 0 ||
                   strcmp(flag, "--sample-ms") == 0 ||
                   strcmp(flag, "--sample-rollouts") == 0 ||
                   strcmp(flag, "--batch") == 0) {
            if (++argument >= argc) game_usage(argv[0]);
            int value = game_parse_integer(argv[argument], flag);
            if (strcmp(flag, "-n") == 0) options.horizon = value;
            else if (strcmp(flag, "-k") == 0) options.top_k = value;
            else if (strcmp(flag, "--sample-ms") == 0) {
                options.sample_milliseconds = value;
                options.sample_rollouts = 0;
            } else if (strcmp(flag, "--sample-rollouts") == 0) {
                options.sample_rollouts = value;
            } else {
                options.batch_size = value;
            }
        } else {
            game_usage(argv[0]);
        }
    }
    if (options.horizon <= 0 || options.top_k <= 0 ||
        options.sample_milliseconds <= 0 || options.batch_size <= 0 ||
        options.sample_rollouts < 0) {
        strength_fail(
            "horizon, top-k, sample budget, and batch must be positive"
        );
    }

    setvbuf(stdout, NULL, _IOLBF, 0);
    Transformer transformer;
    build_transformer(&transformer, argv[1]);
    if (options.top_k > transformer.config.vocab_size) {
        strength_fail("top-k exceeds model vocabulary");
    }
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, argv[2], transformer.config.vocab_size);
    int *prompt_tokens = strength_calloc(
        strlen(argv[3]) + 3,
        sizeof(*prompt_tokens)
    );
    int prompt_count = 0;
    encode(&tokenizer, argv[3], 1, 0, prompt_tokens, &prompt_count);
    if (prompt_count < 1) strength_fail("prompt encoded to no tokens");
    if (prompt_count + options.horizon > transformer.config.seq_len) {
        strength_fail("prompt plus sampled horizon exceeds sequence length");
    }

    FillerLedger ledger = new_ledger(&transformer.config);
    GameProgram program = {
        .transformer = &transformer,
        .prompt_tokens = prompt_tokens,
        .prompt_count = prompt_count,
        .options = options,
        .ledger = &ledger,
        .random_state = UINT64_C(0x4d595df4d0f33173),
    };
    game_term_init(&program.term);
    program.batch = game_batch_state_new(
        &transformer.config,
        options.batch_size,
        options.horizon,
        prompt_count
    );
    if (game_ledger_total_crossings(&ledger) != 0) {
        strength_fail("a learned filler ran before game_run");
    }
    printf(
        "run_begin horizon=%d top_k=%d sample_ms=%d sample_rollouts=%d "
        "batch_size=%d "
        "learned_filler_crossings_before_run=0\n",
        options.horizon,
        options.top_k,
        options.sample_milliseconds,
        options.sample_rollouts,
        options.batch_size
    );
    long started = time_in_ms();
    game_run(&program);
    long finished = time_in_ms();

    int *selected_tokens = strength_calloc(
        (size_t)options.horizon,
        sizeof(*selected_tokens)
    );
    uint32_t *selected_nodes = strength_calloc(
        (size_t)options.horizon,
        sizeof(*selected_nodes)
    );
    int selected_count = game_selected_path(
        &program.term,
        selected_tokens,
        selected_nodes,
        options.horizon
    );
    GameNode *selected_leaf =
        &program.term.nodes[selected_nodes[selected_count - 1]];
    int *best_tokens = strength_calloc(
        (size_t)options.horizon,
        sizeof(*best_tokens)
    );
    uint32_t *best_nodes = strength_calloc(
        (size_t)options.horizon,
        sizeof(*best_nodes)
    );
    double best_reward;
    int best_count = game_best_observed_path(
        &program.term,
        best_tokens,
        best_nodes,
        options.horizon,
        &best_reward
    );
    printf("prompt: %s\n", argv[3]);
    printf("completion: ");
    game_print_completion(
        &tokenizer,
        prompt_tokens[prompt_count - 1],
        selected_tokens,
        selected_count
    );
    putchar('\n');
    fputs("selected_tokens=[", stdout);
    for (int index = 0; index < selected_count; index++) {
        if (index != 0) putchar(',');
        printf("%d", selected_tokens[index]);
    }
    puts("]");
    printf("best_single_observation_completion: ");
    game_print_completion(
        &tokenizer,
        prompt_tokens[prompt_count - 1],
        best_tokens,
        best_count
    );
    putchar('\n');
    fputs("best_single_observation_tokens=[", stdout);
    for (int index = 0; index < best_count; index++) {
        if (index != 0) putchar(',');
        printf("%d", best_tokens[index]);
    }
    puts("]");
    printf(
        "selected_token_count=%d selected_leaf_log_reward=%.17g "
        "best_single_observation_token_count=%d "
        "best_single_observation_reward=%.17g "
        "root_backed_log_reward_sum=%.17g "
        "root_observations=%" PRIu64 " "
        "root_unique_completions=%" PRIu64 "\n",
        selected_count,
        selected_leaf->leaf_log_reward,
        best_count,
        best_reward,
        program.term.nodes[0].backed_log_reward_sum,
        program.term.nodes[0].backed_observation_count,
        program.term.nodes[0].backed_unique_leaf_count
    );
    puts("selection_term=demanded_sampled_product_of_logit_selections");
    puts("selection_backup=online_logaddexp_of_every_sampled_path_observation");
    puts("selection_witness=argmax_backed_outgoing_reward_at_each_prefix");
    puts("sampling_algorithm=categorical_top_k_path_demand");
    puts("sample_multiplicity=retained");
    puts("posthoc_max_leaf=diagnostic_only");
    puts("sampling_bonus=none");
    if (options.sample_rollouts > 0) {
        printf("sampling_budget=rollouts count=%d\n", options.sample_rollouts);
    } else {
        printf(
            "sampling_budget=milliseconds count=%d\n",
            options.sample_milliseconds
        );
    }
    printf(
        "attempted_rollouts=%" PRIu64 " unique_completions=%" PRIu64
        " prefix_term_nodes=%" PRIu32 " sampled_path_edges=%" PRIu64 "\n",
        program.attempted_rollouts,
        program.unique_completions,
        program.term.count,
        program.sampled_path_edges
    );
    printf(
        "rollout_batches=%" PRIu64 " prefill_milliseconds=%ld "
        "sampling_milliseconds=%ld elapsed_seconds=%.6f\n",
        program.rollout_batches,
        program.prefill_milliseconds,
        program.sampling_milliseconds,
        (finished - started) / 1000.0
    );
    uint64_t minimum_crossings;
    uint64_t maximum_crossings;
    game_ledger_crossing_range(
        &ledger,
        &minimum_crossings,
        &maximum_crossings
    );
    printf(
        "learned_filler_count=%d total_filler_crossings=%" PRIu64
        " crossings_per_filler_min=%" PRIu64
        " crossings_per_filler_max=%" PRIu64
        " coefficient_reads=%" PRIu64 " logical_uses=%" PRIu64 "\n",
        ledger.filler_count,
        game_ledger_total_crossings(&ledger),
        minimum_crossings,
        maximum_crossings,
        ledger_total_reads(&ledger),
        ledger_total_uses(&ledger)
    );
    game_print_decision_trace(
        &program,
        &tokenizer,
        prompt_tokens[prompt_count - 1],
        selected_tokens,
        selected_nodes,
        selected_count
    );
    if (options.trace_fillers) print_ledger(&transformer.config, &ledger);
    if (options.verify) {
        game_verify_selected_path(
            &program,
            selected_tokens,
            selected_nodes,
            selected_count
        );
    }

    free(selected_nodes);
    free(selected_tokens);
    free(best_nodes);
    free(best_tokens);
    free(program.root_logits);
    game_batch_state_free(&program.batch);
    game_term_free(&program.term);
    free(ledger.fillers);
    free(prompt_tokens);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return EXIT_SUCCESS;
}
