/*
 * DO NOT USE AS THE SCALABLE INFERENCE PATH.
 *
 * I should never have treated full vocabulary-prefix materialization as the
 * answer to a task whose central requirement was to sample only demanded
 * continuations.  This bounded program evaluates every branch of a tiny
 * product before selection.  It can remain a numerical oracle, but extending
 * V + V^2 + ... to real horizons simply retries every path and evades the
 * required sampled recursive strength.
 */

#define TESTING
#include "run.c"

#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>

/*
 * The two-token instance of the inside-out evaluator.
 *
 * This is deliberately not a beam search wrapped in a thunk.  Before any
 * learned kernel runs, main constructs a Program.  run_program then forms one
 * prefix-tree company containing
 *
 *   - every position of the forced prompt, and
 *   - one child of the prompt for every possible first completion token.
 *
 * The entire company crosses each learned filler once.  Only after the root
 * and child logits exist do we terminalize the dependent product of token
 * selections.  The product is Escardo's backward-induction equation:
 *
 *   b x = eps_x (\y -> p (x,y))
 *   a   = eps   (\x -> p (x,b x))
 *   eps_product p = (a,b a)
 *
 * In the source calculus both Select_R X = (X -> R) -> X and
 * Cont_R X = (X -> R) -> R are Ran(Const,Const) specializations.  C has no
 * rank-polymorphic type constructors, so the definitions below are their
 * token- and token-pair-specialized representations, not a replacement IR.
 */

typedef double Reward;

typedef struct {
    int first;
    int second;
} TokenPair;

typedef Reward (*TokenObservation)(void *environment, int token);
typedef Reward (*PairObservation)(void *environment, TokenPair pair);

/* Ran(Const Reward, Const Token): the selection monad. */
typedef struct RanConstConstSelectToken RanConstConstSelectToken;
struct RanConstConstSelectToken {
    void *environment;
    int (*apply)(
        void *environment,
        TokenObservation observation,
        void *observation_environment
    );
};

/* Dependent product/strength of two selections. */
typedef struct RanConstConstSelectPair RanConstConstSelectPair;
struct RanConstConstSelectPair {
    void *environment;
    TokenPair (*apply)(
        void *environment,
        PairObservation observation,
        void *observation_environment
    );
};

/* Ran(Const Reward, Const Reward): the continuation monad. */
typedef struct RanConstConstContPair RanConstConstContPair;
struct RanConstConstContPair {
    void *environment;
    Reward (*apply)(
        void *environment,
        PairObservation observation,
        void *observation_environment,
        TokenPair *witness
    );
};

typedef struct {
    uint64_t crossings;
    uint64_t coefficient_reads;
    uint64_t logical_uses;
} FillerCount;

typedef struct {
    int layer_count;
    int filler_count;
    FillerCount *fillers;
} FillerLedger;

enum {
    LAYER_FILLER_COUNT = 9,
    FILLER_ATTN_RMS = 0,
    FILLER_QUERY = 1,
    FILLER_KEY = 2,
    FILLER_VALUE = 3,
    FILLER_ATTN_OUTPUT = 4,
    FILLER_FFN_RMS = 5,
    FILLER_FFN_GATE = 6,
    FILLER_FFN_UP = 7,
    FILLER_FFN_DOWN = 8,
};

typedef struct {
    float *values;
    int row_count;
    int vocab_size;
} LogitCompany;

typedef struct {
    double *first;
    double *second;
    int vocab_size;
} PayoffTable;

typedef struct {
    int token;
    int suffix;
    Reward reward;
} BranchSummary;

typedef struct {
    Transformer *transformer;
    const int *prompt_tokens;
    int prompt_count;
    FillerLedger *ledger;
    bool trace_fillers;
} Program;

typedef struct {
    TokenPair witness;
    Reward reward;
    BranchSummary *branches;
    int branch_count;
    LogitCompany logits;
    PayoffTable payoffs;
} ProgramResult;

typedef struct {
    uint64_t compared;
    uint64_t bit_mismatches;
    uint64_t tolerance_failures;
    float maximum_absolute_error;
    int first_bad_row;
    int first_bad_token;
    float first_bad_company;
    float first_bad_oracle;
} Verification;

static void strength_fail(const char *message) {
    fprintf(stderr, "escardo strength: %s\n", message);
    exit(EXIT_FAILURE);
}

static void *strength_calloc(size_t count, size_t width) {
    if (width != 0 && count > SIZE_MAX / width) {
        strength_fail("allocation size overflow");
    }
    void *result = calloc(count, width);
    if (result == NULL) strength_fail("allocation failed");
    return result;
}

static size_t strength_elements(int count, int width) {
    if (count < 0 || width < 0) strength_fail("negative tensor extent");
    size_t left = (size_t)count;
    size_t right = (size_t)width;
    if (right != 0 && left > SIZE_MAX / right) {
        strength_fail("tensor size overflow");
    }
    return left * right;
}

static uint64_t checked_u64_product(uint64_t left, uint64_t right) {
    if (right != 0 && left > UINT64_MAX / right) {
        strength_fail("counter overflow");
    }
    return left * right;
}

static int embedding_filler(void) {
    return 0;
}

static int layer_filler(int layer, int stage) {
    return 1 + layer * LAYER_FILLER_COUNT + stage;
}

static int final_rms_filler(const Config *config) {
    return 1 + config->n_layers * LAYER_FILLER_COUNT;
}

static int classifier_filler(const Config *config) {
    return final_rms_filler(config) + 1;
}

static FillerLedger new_ledger(const Config *config) {
    FillerLedger result = {
        .layer_count = config->n_layers,
        .filler_count = 3 + config->n_layers * LAYER_FILLER_COUNT,
    };
    result.fillers = strength_calloc(
        (size_t)result.filler_count,
        sizeof(*result.fillers)
    );
    return result;
}

static void note_filler(
    FillerLedger *ledger,
    int filler,
    uint64_t coefficient_reads,
    uint64_t logical_uses
) {
    if (filler < 0 || filler >= ledger->filler_count) {
        strength_fail("invalid learned filler id");
    }
    FillerCount *count = &ledger->fillers[filler];
    count->crossings++;
    count->coefficient_reads += coefficient_reads;
    count->logical_uses += logical_uses;
}

static const char *layer_stage_name(int stage) {
    static const char *names[LAYER_FILLER_COUNT] = {
        "attention_rms",
        "query",
        "key",
        "value",
        "attention_output",
        "ffn_rms",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    };
    if (stage < 0 || stage >= LAYER_FILLER_COUNT) return "invalid";
    return names[stage];
}

static void print_filler_name(const Config *config, int filler) {
    if (filler == embedding_filler()) {
        fputs("embedding", stdout);
        return;
    }
    if (filler == final_rms_filler(config)) {
        fputs("final_rms", stdout);
        return;
    }
    if (filler == classifier_filler(config)) {
        fputs("classifier", stdout);
        return;
    }
    int shifted = filler - 1;
    int layer = shifted / LAYER_FILLER_COUNT;
    int stage = shifted % LAYER_FILLER_COUNT;
    printf("layer_%d.%s", layer, layer_stage_name(stage));
}

/* One embedding filler, with all demanded token occurrences beneath it. */
static void embedding_company_apply(
    float *output,
    const int *tokens,
    int company_count,
    const float *weights,
    int vocab_size,
    int dim,
    FillerLedger *ledger,
    int filler
) {
    int *heads = strength_calloc((size_t)vocab_size, sizeof(*heads));
    int *next = strength_calloc((size_t)company_count, sizeof(*next));
    for (int token = 0; token < vocab_size; token++) heads[token] = -1;
    for (int company = 0; company < company_count; company++) {
        int token = tokens[company];
        if (token < 0 || token >= vocab_size) {
            strength_fail("token outside embedding vocabulary");
        }
        next[company] = heads[token];
        heads[token] = company;
    }

    uint64_t reads = 0;
    for (int token = 0; token < vocab_size; token++) {
        if (heads[token] == -1) continue;
        for (int lane = 0; lane < dim; lane++) {
            float coefficient = weights[(size_t)token * dim + lane];
            reads++;
            for (int company = heads[token]; company != -1;
                 company = next[company]) {
                output[(size_t)company * dim + lane] = coefficient;
            }
        }
    }
    note_filler(
        ledger,
        filler,
        reads,
        checked_u64_product((uint64_t)company_count, (uint64_t)dim)
    );
    free(next);
    free(heads);
}

/* A learned RMS scale is loaded once and broadcast over the company. */
static void rms_company_apply(
    float *output,
    const float *input,
    const float *weights,
    int company_count,
    int width,
    FillerLedger *ledger,
    int filler
) {
    float *resident_weights = strength_calloc(
        (size_t)width,
        sizeof(*resident_weights)
    );
    memcpy(
        resident_weights,
        weights,
        (size_t)width * sizeof(*resident_weights)
    );
    for (int company = 0; company < company_count; company++) {
        const float *row = input + (size_t)company * width;
        float *output_row = output + (size_t)company * width;
        float sum_of_squares = 0.0f;
        for (int lane = 0; lane < width; lane++) {
            sum_of_squares += row[lane] * row[lane];
        }
        sum_of_squares /= width;
        sum_of_squares += 1e-5f;
        float inverse_scale = 1.0f / sqrtf(sum_of_squares);
        for (int lane = 0; lane < width; lane++) {
            output_row[lane] = resident_weights[lane] *
                (inverse_scale * row[lane]);
        }
    }
    note_filler(
        ledger,
        filler,
        (uint64_t)width,
        checked_u64_product((uint64_t)company_count, (uint64_t)width)
    );
    free(resident_weights);
}

/*
 * W : A -> B occurs once.  The company supplies all A-valued positions.
 * Each learned row is streamed into a kernel-resident tile once.  All demanded
 * positions consume that resident row before the next learned row is loaded.
 * The inner accumulation remains j = 0..n-1, exactly as llama2.c's scalar
 * matmul; only accesses to the resident tile repeat, not reads of the learned
 * filler itself.
 */
static void matmul_company_apply(
    float *output,
    const float *input,
    const float *weights,
    int company_count,
    int input_width,
    int output_width,
    FillerLedger *ledger,
    int filler
) {
    float *resident_row = strength_calloc(
        (size_t)input_width,
        sizeof(*resident_row)
    );
    for (int output_lane = 0; output_lane < output_width; output_lane++) {
        memcpy(
            resident_row,
            weights + (size_t)output_lane * input_width,
            (size_t)input_width * sizeof(*resident_row)
        );
        for (int company = 0; company < company_count; company++) {
            const float *input_row = input + (size_t)company * input_width;
            float result = 0.0f;
            for (int input_lane = 0; input_lane < input_width; input_lane++) {
                result += resident_row[input_lane] * input_row[input_lane];
            }
            output[(size_t)company * output_width + output_lane] = result;
        }
    }
    uint64_t coefficients = checked_u64_product(
        (uint64_t)input_width,
        (uint64_t)output_width
    );
    note_filler(
        ledger,
        filler,
        coefficients,
        checked_u64_product(coefficients, (uint64_t)company_count)
    );
    free(resident_row);
}

static int family_position(int company, int prompt_count) {
    return company < prompt_count ? company : prompt_count;
}

static int family_context_count(int company, int prompt_count) {
    return company < prompt_count ? company + 1 : prompt_count + 1;
}

static int family_context_member(
    int company,
    int context_index,
    int prompt_count
) {
    if (company < prompt_count) return context_index;
    return context_index < prompt_count ? context_index : company;
}

static LogitCompany evaluate_prefix_tree_company(Program *program) {
    Transformer *transformer = program->transformer;
    Config *config = &transformer->config;
    TransformerWeights *weights = &transformer->weights;
    int dim = config->dim;
    int hidden_dim = config->hidden_dim;
    int vocab_size = config->vocab_size;
    int prompt_count = program->prompt_count;
    int company_count = prompt_count + vocab_size;
    int output_count = 1 + vocab_size;
    int head_size = dim / config->n_heads;
    int kv_dim = dim * config->n_kv_heads / config->n_heads;
    int kv_mul = config->n_heads / config->n_kv_heads;

    int *company_tokens = strength_calloc(
        (size_t)company_count,
        sizeof(*company_tokens)
    );
    for (int index = 0; index < prompt_count; index++) {
        company_tokens[index] = program->prompt_tokens[index];
    }
    for (int token = 0; token < vocab_size; token++) {
        company_tokens[prompt_count + token] = token;
    }

    float *x = strength_calloc(
        strength_elements(company_count, dim),
        sizeof(*x)
    );
    float *normalized = strength_calloc(
        strength_elements(company_count, dim),
        sizeof(*normalized)
    );
    float *query = strength_calloc(
        strength_elements(company_count, dim),
        sizeof(*query)
    );
    float *key = strength_calloc(
        strength_elements(company_count, kv_dim),
        sizeof(*key)
    );
    float *value = strength_calloc(
        strength_elements(company_count, kv_dim),
        sizeof(*value)
    );
    float *attended = strength_calloc(
        strength_elements(company_count, dim),
        sizeof(*attended)
    );
    float *projected = strength_calloc(
        strength_elements(company_count, dim),
        sizeof(*projected)
    );
    float *gate = strength_calloc(
        strength_elements(company_count, hidden_dim),
        sizeof(*gate)
    );
    float *up = strength_calloc(
        strength_elements(company_count, hidden_dim),
        sizeof(*up)
    );
    float *down = strength_calloc(
        strength_elements(company_count, dim),
        sizeof(*down)
    );
    float *attention_scores = strength_calloc(
        (size_t)(prompt_count + 1),
        sizeof(*attention_scores)
    );

    embedding_company_apply(
        x,
        company_tokens,
        company_count,
        weights->token_embedding_table,
        vocab_size,
        dim,
        program->ledger,
        embedding_filler()
    );

    for (int layer = 0; layer < config->n_layers; layer++) {
        rms_company_apply(
            normalized,
            x,
            weights->rms_att_weight + (size_t)layer * dim,
            company_count,
            dim,
            program->ledger,
            layer_filler(layer, FILLER_ATTN_RMS)
        );
        matmul_company_apply(
            query,
            normalized,
            weights->wq + (size_t)layer * dim * dim,
            company_count,
            dim,
            dim,
            program->ledger,
            layer_filler(layer, FILLER_QUERY)
        );
        matmul_company_apply(
            key,
            normalized,
            weights->wk + (size_t)layer * dim * kv_dim,
            company_count,
            dim,
            kv_dim,
            program->ledger,
            layer_filler(layer, FILLER_KEY)
        );
        matmul_company_apply(
            value,
            normalized,
            weights->wv + (size_t)layer * dim * kv_dim,
            company_count,
            dim,
            kv_dim,
            program->ledger,
            layer_filler(layer, FILLER_VALUE)
        );

        for (int company = 0; company < company_count; company++) {
            int position = family_position(company, prompt_count);
            float *company_query = query + (size_t)company * dim;
            float *company_key = key + (size_t)company * kv_dim;
            for (int lane = 0; lane < dim; lane += 2) {
                int head_dimension = lane % head_size;
                float frequency = 1.0f /
                    powf(10000.0f, head_dimension / (float)head_size);
                float angle = position * frequency;
                float real = cosf(angle);
                float imaginary = sinf(angle);
                float q0 = company_query[lane];
                float q1 = company_query[lane + 1];
                company_query[lane] = q0 * real - q1 * imaginary;
                company_query[lane + 1] = q0 * imaginary + q1 * real;
                if (lane < kv_dim) {
                    float k0 = company_key[lane];
                    float k1 = company_key[lane + 1];
                    company_key[lane] = k0 * real - k1 * imaginary;
                    company_key[lane + 1] = k0 * imaginary + k1 * real;
                }
            }
        }

        for (int company = 0; company < company_count; company++) {
            int context_count = family_context_count(company, prompt_count);
            const float *company_query = query + (size_t)company * dim;
            float *company_output = attended + (size_t)company * dim;
            for (int head = 0; head < config->n_heads; head++) {
                const float *head_query = company_query + head * head_size;
                for (int context_index = 0; context_index < context_count;
                     context_index++) {
                    int member = family_context_member(
                        company,
                        context_index,
                        prompt_count
                    );
                    const float *head_key = key + (size_t)member * kv_dim +
                        (head / kv_mul) * head_size;
                    float score = 0.0f;
                    for (int lane = 0; lane < head_size; lane++) {
                        score += head_query[lane] * head_key[lane];
                    }
                    attention_scores[context_index] =
                        score / sqrtf(head_size);
                }
                softmax(attention_scores, context_count);
                float *head_output = company_output + head * head_size;
                memset(
                    head_output,
                    0,
                    (size_t)head_size * sizeof(*head_output)
                );
                for (int context_index = 0; context_index < context_count;
                     context_index++) {
                    int member = family_context_member(
                        company,
                        context_index,
                        prompt_count
                    );
                    const float *head_value = value +
                        (size_t)member * kv_dim +
                        (head / kv_mul) * head_size;
                    float mass = attention_scores[context_index];
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
            company_count,
            dim,
            dim,
            program->ledger,
            layer_filler(layer, FILLER_ATTN_OUTPUT)
        );
        for (size_t index = 0; index < strength_elements(company_count, dim);
             index++) {
            x[index] += projected[index];
        }

        rms_company_apply(
            normalized,
            x,
            weights->rms_ffn_weight + (size_t)layer * dim,
            company_count,
            dim,
            program->ledger,
            layer_filler(layer, FILLER_FFN_RMS)
        );
        matmul_company_apply(
            gate,
            normalized,
            weights->w1 + (size_t)layer * dim * hidden_dim,
            company_count,
            dim,
            hidden_dim,
            program->ledger,
            layer_filler(layer, FILLER_FFN_GATE)
        );
        matmul_company_apply(
            up,
            normalized,
            weights->w3 + (size_t)layer * dim * hidden_dim,
            company_count,
            dim,
            hidden_dim,
            program->ledger,
            layer_filler(layer, FILLER_FFN_UP)
        );
        for (size_t index = 0;
             index < strength_elements(company_count, hidden_dim);
             index++) {
            float raw_gate = gate[index];
            raw_gate *= 1.0f / (1.0f + expf(-raw_gate));
            gate[index] = raw_gate * up[index];
        }
        matmul_company_apply(
            down,
            gate,
            weights->w2 + (size_t)layer * hidden_dim * dim,
            company_count,
            hidden_dim,
            dim,
            program->ledger,
            layer_filler(layer, FILLER_FFN_DOWN)
        );
        for (size_t index = 0; index < strength_elements(company_count, dim);
             index++) {
            x[index] += down[index];
        }
    }

    float *output_hidden = strength_calloc(
        strength_elements(output_count, dim),
        sizeof(*output_hidden)
    );
    memcpy(
        output_hidden,
        x + (size_t)(prompt_count - 1) * dim,
        (size_t)dim * sizeof(*output_hidden)
    );
    for (int token = 0; token < vocab_size; token++) {
        memcpy(
            output_hidden + (size_t)(token + 1) * dim,
            x + (size_t)(prompt_count + token) * dim,
            (size_t)dim * sizeof(*output_hidden)
        );
    }
    float *final_hidden = strength_calloc(
        strength_elements(output_count, dim),
        sizeof(*final_hidden)
    );
    rms_company_apply(
        final_hidden,
        output_hidden,
        weights->rms_final_weight,
        output_count,
        dim,
        program->ledger,
        final_rms_filler(config)
    );
    float *logits = strength_calloc(
        strength_elements(output_count, vocab_size),
        sizeof(*logits)
    );
    matmul_company_apply(
        logits,
        final_hidden,
        weights->wcls,
        output_count,
        dim,
        vocab_size,
        program->ledger,
        classifier_filler(config)
    );

    free(final_hidden);
    free(output_hidden);
    free(attention_scores);
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
    free(company_tokens);

    return (LogitCompany){
        .values = logits,
        .row_count = output_count,
        .vocab_size = vocab_size,
    };
}

static void logits_to_log_probabilities(
    const float *logits,
    double *log_probabilities,
    int count
) {
    double maximum = -DBL_MAX;
    for (int token = 0; token < count; token++) {
        if ((double)logits[token] > maximum) maximum = logits[token];
    }
    double partition = 0.0;
    for (int token = 0; token < count; token++) {
        partition += exp((double)logits[token] - maximum);
    }
    for (int token = 0; token < count; token++) {
        log_probabilities[token] = (double)logits[token] - maximum -
            log(partition);
    }
}

static PayoffTable make_payoff_table(const LogitCompany *logits) {
    int vocab_size = logits->vocab_size;
    if (logits->row_count != vocab_size + 1) {
        strength_fail("two-token logit company has the wrong shape");
    }
    PayoffTable result = {
        .vocab_size = vocab_size,
        .first = strength_calloc((size_t)vocab_size, sizeof(*result.first)),
        .second = strength_calloc(
            strength_elements(vocab_size, vocab_size),
            sizeof(*result.second)
        ),
    };
    logits_to_log_probabilities(logits->values, result.first, vocab_size);
    for (int first = 0; first < vocab_size; first++) {
        logits_to_log_probabilities(
            logits->values + (size_t)(first + 1) * vocab_size,
            result.second + (size_t)first * vocab_size,
            vocab_size
        );
    }
    return result;
}

typedef struct {
    int count;
} FullTokenDomain;

static int full_domain_select(
    void *raw_environment,
    TokenObservation observation,
    void *observation_environment
) {
    FullTokenDomain *environment = raw_environment;
    bool found = false;
    int best_token = -1;
    Reward best_reward = -INFINITY;
    for (int token = 0; token < environment->count; token++) {
        Reward reward = observation(observation_environment, token);
        if (!found || reward > best_reward) {
            found = true;
            best_token = token;
            best_reward = reward;
        }
    }
    if (!found) strength_fail("selection over an empty token domain");
    return best_token;
}

typedef struct {
    RanConstConstSelectToken head;
    RanConstConstSelectToken (*suffix)(void *environment, int first);
    void *suffix_environment;
    int domain_size;
    int *remembered_suffixes;
    Reward *remembered_rewards;
    unsigned char *remembered;
} Product2;

typedef struct {
    Product2 *product;
    PairObservation observation;
    void *observation_environment;
} ProductFrame;

typedef struct {
    ProductFrame *frame;
    int first;
} SuffixObservationEnvironment;

static Reward observe_suffix_token(void *raw_environment, int second) {
    SuffixObservationEnvironment *environment = raw_environment;
    return environment->frame->observation(
        environment->frame->observation_environment,
        (TokenPair){.first = environment->first, .second = second}
    );
}

static Reward observe_head_token(void *raw_environment, int first) {
    ProductFrame *frame = raw_environment;
    Product2 *product = frame->product;
    if (!product->remembered[first]) {
        RanConstConstSelectToken suffix = product->suffix(
            product->suffix_environment,
            first
        );
        SuffixObservationEnvironment suffix_environment = {
            .frame = frame,
            .first = first,
        };
        int second = suffix.apply(
            suffix.environment,
            observe_suffix_token,
            &suffix_environment
        );
        product->remembered_suffixes[first] = second;
        product->remembered_rewards[first] = frame->observation(
            frame->observation_environment,
            (TokenPair){.first = first, .second = second}
        );
        product->remembered[first] = 1;
    }
    return product->remembered_rewards[first];
}

static TokenPair product2_apply(
    void *raw_environment,
    PairObservation observation,
    void *observation_environment
) {
    Product2 *product = raw_environment;
    ProductFrame frame = {
        .product = product,
        .observation = observation,
        .observation_environment = observation_environment,
    };
    int first = product->head.apply(
        product->head.environment,
        observe_head_token,
        &frame
    );
    if (!product->remembered[first]) {
        strength_fail("backward induction forgot its selected suffix");
    }
    return (TokenPair){
        .first = first,
        .second = product->remembered_suffixes[first],
    };
}

static RanConstConstSelectToken constant_suffix_selection(
    void *environment,
    int first
) {
    (void)first;
    return (RanConstConstSelectToken){
        .environment = environment,
        .apply = full_domain_select,
    };
}

static RanConstConstSelectPair dependent_product_strength(
    RanConstConstSelectToken head,
    RanConstConstSelectToken (*suffix)(void *environment, int first),
    void *suffix_environment,
    int domain_size,
    Product2 *storage
) {
    *storage = (Product2){
        .head = head,
        .suffix = suffix,
        .suffix_environment = suffix_environment,
        .domain_size = domain_size,
        .remembered_suffixes = strength_calloc(
            (size_t)domain_size,
            sizeof(*storage->remembered_suffixes)
        ),
        .remembered_rewards = strength_calloc(
            (size_t)domain_size,
            sizeof(*storage->remembered_rewards)
        ),
        .remembered = strength_calloc(
            (size_t)domain_size,
            sizeof(*storage->remembered)
        ),
    };
    return (RanConstConstSelectPair){
        .environment = storage,
        .apply = product2_apply,
    };
}

typedef struct {
    RanConstConstSelectPair selection;
} TauEnvironment;

static Reward tau_apply(
    void *raw_environment,
    PairObservation observation,
    void *observation_environment,
    TokenPair *witness
) {
    TauEnvironment *environment = raw_environment;
    *witness = environment->selection.apply(
        environment->selection.environment,
        observation,
        observation_environment
    );
    return observation(observation_environment, *witness);
}

/* tau eps p = p (eps p): terminalize Select into Cont exactly once. */
static RanConstConstContPair selection_to_continuation(
    RanConstConstSelectPair selection,
    TauEnvironment *storage
) {
    *storage = (TauEnvironment){.selection = selection};
    return (RanConstConstContPair){
        .environment = storage,
        .apply = tau_apply,
    };
}

static Reward observe_pair_payoff(void *raw_environment, TokenPair pair) {
    PayoffTable *table = raw_environment;
    if (pair.first < 0 || pair.first >= table->vocab_size ||
        pair.second < 0 || pair.second >= table->vocab_size) {
        strength_fail("pair observer received a token outside its domain");
    }
    return table->first[pair.first] +
        table->second[(size_t)pair.first * table->vocab_size + pair.second];
}

static int compare_branch_summary(const void *left_raw, const void *right_raw) {
    const BranchSummary *left = left_raw;
    const BranchSummary *right = right_raw;
    if (left->reward > right->reward) return -1;
    if (left->reward < right->reward) return 1;
    if (left->token < right->token) return -1;
    if (left->token > right->token) return 1;
    return 0;
}

static ProgramResult run_program(Program *program) {
    LogitCompany logits = evaluate_prefix_tree_company(program);
    PayoffTable payoffs = make_payoff_table(&logits);
    int vocab_size = payoffs.vocab_size;
    FullTokenDomain domain = {.count = vocab_size};
    RanConstConstSelectToken token_selection = {
        .environment = &domain,
        .apply = full_domain_select,
    };
    Product2 product_storage;
    RanConstConstSelectPair product = dependent_product_strength(
        token_selection,
        constant_suffix_selection,
        &domain,
        vocab_size,
        &product_storage
    );
    TauEnvironment tau_storage;
    RanConstConstContPair continuation = selection_to_continuation(
        product,
        &tau_storage
    );

    TokenPair witness;
    Reward reward = continuation.apply(
        continuation.environment,
        observe_pair_payoff,
        &payoffs,
        &witness
    );

    BranchSummary *branches = strength_calloc(
        (size_t)vocab_size,
        sizeof(*branches)
    );
    for (int first = 0; first < vocab_size; first++) {
        if (!product_storage.remembered[first]) {
            strength_fail("strength did not retain every reached suffix");
        }
        branches[first] = (BranchSummary){
            .token = first,
            .suffix = product_storage.remembered_suffixes[first],
            .reward = product_storage.remembered_rewards[first],
        };
    }
    qsort(
        branches,
        (size_t)vocab_size,
        sizeof(*branches),
        compare_branch_summary
    );

    free(product_storage.remembered);
    free(product_storage.remembered_rewards);
    free(product_storage.remembered_suffixes);
    return (ProgramResult){
        .witness = witness,
        .reward = reward,
        .branches = branches,
        .branch_count = vocab_size,
        .logits = logits,
        .payoffs = payoffs,
    };
}

static void compare_logit_row(
    Verification *verification,
    int row,
    const float *company,
    const float *oracle,
    int vocab_size
) {
    for (int token = 0; token < vocab_size; token++) {
        float absolute_error = fabsf(company[token] - oracle[token]);
        verification->compared++;
        if (memcmp(&company[token], &oracle[token], sizeof(float)) != 0) {
            verification->bit_mismatches++;
        }
        if (absolute_error > verification->maximum_absolute_error) {
            verification->maximum_absolute_error = absolute_error;
        }
        if (absolute_error > 1e-5f) {
            if (verification->tolerance_failures == 0) {
                verification->first_bad_row = row;
                verification->first_bad_token = token;
                verification->first_bad_company = company[token];
                verification->first_bad_oracle = oracle[token];
            }
            verification->tolerance_failures++;
        }
    }
}

/* A separate, explicitly eager diagnostic oracle; never part of Program.run. */
static Verification verify_against_llama2_forward(
    Transformer *transformer,
    const int *prompt_tokens,
    int prompt_count,
    const LogitCompany *company
) {
    Verification result = {
        .first_bad_row = -1,
        .first_bad_token = -1,
    };
    int vocab_size = transformer->config.vocab_size;
    for (int position = 0; position < prompt_count; position++) {
        float *oracle = forward(transformer, prompt_tokens[position], position);
        if (position == prompt_count - 1) {
            compare_logit_row(
                &result,
                0,
                company->values,
                oracle,
                vocab_size
            );
        }
    }
    for (int first = 0; first < vocab_size; first++) {
        float *oracle = forward(transformer, first, prompt_count);
        compare_logit_row(
            &result,
            first + 1,
            company->values + (size_t)(first + 1) * vocab_size,
            oracle,
            vocab_size
        );
    }
    return result;
}

static void print_escaped(const char *text) {
    for (const unsigned char *cursor = (const unsigned char *)text;
         *cursor != '\0'; cursor++) {
        unsigned char byte = *cursor;
        if (byte == '\\' || byte == '"') {
            putchar('\\');
            putchar(byte);
        } else if (byte == '\n') {
            fputs("\\n", stdout);
        } else if (byte == '\r') {
            fputs("\\r", stdout);
        } else if (byte == '\t') {
            fputs("\\t", stdout);
        } else if (byte >= 0x20 && byte != 0x7f) {
            putchar(byte);
        } else {
            printf("\\x%02x", byte);
        }
    }
}

static void print_piece(Tokenizer *tokenizer, int previous, int token) {
    putchar('"');
    print_escaped(decode(tokenizer, previous, token));
    putchar('"');
}

static void print_completion(
    Tokenizer *tokenizer,
    int prompt_last_token,
    TokenPair pair
) {
    print_escaped(decode(tokenizer, prompt_last_token, pair.first));
    print_escaped(decode(tokenizer, pair.first, pair.second));
}

static uint64_t ledger_total_reads(const FillerLedger *ledger) {
    uint64_t total = 0;
    for (int filler = 0; filler < ledger->filler_count; filler++) {
        total += ledger->fillers[filler].coefficient_reads;
    }
    return total;
}

static uint64_t ledger_total_uses(const FillerLedger *ledger) {
    uint64_t total = 0;
    for (int filler = 0; filler < ledger->filler_count; filler++) {
        total += ledger->fillers[filler].logical_uses;
    }
    return total;
}

static void verify_one_shot_ledger(const FillerLedger *ledger) {
    for (int filler = 0; filler < ledger->filler_count; filler++) {
        if (ledger->fillers[filler].crossings != 1) {
            strength_fail("a learned filler did not cross its company once");
        }
    }
}

static void print_ledger(const Config *config, const FillerLedger *ledger) {
    puts("filler_ledger:");
    for (int filler = 0; filler < ledger->filler_count; filler++) {
        fputs("  filler=", stdout);
        print_filler_name(config, filler);
        printf(
            " crossings=%" PRIu64 " coefficient_reads=%" PRIu64
            " logical_uses=%" PRIu64 "\n",
            ledger->fillers[filler].crossings,
            ledger->fillers[filler].coefficient_reads,
            ledger->fillers[filler].logical_uses
        );
    }
}

static void strength_usage(const char *program) {
    fprintf(
        stderr,
        "usage: %s CHECKPOINT TOKENIZER PROMPT [--verify] [--trace-fillers]\n",
        program
    );
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 6) strength_usage(argv[0]);
    bool verify = false;
    bool trace_fillers = false;
    for (int argument = 4; argument < argc; argument++) {
        if (strcmp(argv[argument], "--verify") == 0) {
            verify = true;
        } else if (strcmp(argv[argument], "--trace-fillers") == 0) {
            trace_fillers = true;
        } else {
            strength_usage(argv[0]);
        }
    }

    Transformer transformer;
    build_transformer(&transformer, argv[1]);
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, argv[2], transformer.config.vocab_size);

    int *prompt_tokens = strength_calloc(
        strlen(argv[3]) + 3,
        sizeof(*prompt_tokens)
    );
    int prompt_count = 0;
    encode(&tokenizer, argv[3], 1, 0, prompt_tokens, &prompt_count);
    if (prompt_count < 1) strength_fail("prompt encoded to no tokens");
    if (prompt_count + 1 >= transformer.config.seq_len) {
        strength_fail("prompt is too long for a two-token completion");
    }

    FillerLedger ledger = new_ledger(&transformer.config);
    Program program = {
        .transformer = &transformer,
        .prompt_tokens = prompt_tokens,
        .prompt_count = prompt_count,
        .ledger = &ledger,
        .trace_fillers = trace_fillers,
    };

    if (ledger_total_reads(&ledger) != 0) {
        strength_fail("a learned filler ran before run_program");
    }
    long start = time_in_ms();
    ProgramResult result = run_program(&program);
    long end = time_in_ms();
    verify_one_shot_ledger(&ledger);

    int prompt_last = prompt_tokens[prompt_count - 1];
    printf("prompt: %s\n", argv[3]);
    printf("prompt_token_count=%d\n", prompt_count);
    printf("completion: ");
    print_completion(&tokenizer, prompt_last, result.witness);
    putchar('\n');
    printf(
        "selected_tokens=[%d,%d]\nselected_reward=%.17g\n",
        result.witness.first,
        result.witness.second,
        result.reward
    );
    puts("selection_term=Ran(Const_Reward,Const_TokenPair)");
    puts("observation_term=Ran(Const_Reward,Const_Reward)");
    puts("terminalization=tau_once");
    printf(
        "demanded_prefix_tree_nodes=%d output_contexts=%d reached_pairs=%zu\n",
        prompt_count + transformer.config.vocab_size,
        transformer.config.vocab_size + 1,
        (size_t)transformer.config.vocab_size *
            transformer.config.vocab_size
    );
    printf("elapsed_seconds=%.6f\n", (end - start) / 1000.0);
    puts("backward_induction_top5:");
    int shown = result.branch_count < 5 ? result.branch_count : 5;
    for (int rank = 0; rank < shown; rank++) {
        BranchSummary branch = result.branches[rank];
        printf("  rank=%d first=%d first_piece=", rank + 1, branch.token);
        print_piece(&tokenizer, prompt_last, branch.token);
        printf(" suffix=%d suffix_piece=", branch.suffix);
        print_piece(&tokenizer, branch.token, branch.suffix);
        printf(" backed_up_reward=%.17g\n", branch.reward);
    }

    uint64_t reads = ledger_total_reads(&ledger);
    uint64_t uses = ledger_total_uses(&ledger);
    printf(
        "learned_filler_count=%d one_shot_crossings=%d "
        "coefficient_reads=%" PRIu64 " logical_uses=%" PRIu64
        " reuse_ratio=%.6f\n",
        ledger.filler_count,
        ledger.filler_count,
        reads,
        uses,
        reads == 0 ? 0.0 : (double)uses / (double)reads
    );
    if (trace_fillers) print_ledger(&transformer.config, &ledger);

    if (verify) {
        Verification verification = verify_against_llama2_forward(
            &transformer,
            prompt_tokens,
            prompt_count,
            &result.logits
        );
        printf(
            "llama2_oracle_logits=%" PRIu64 " bit_mismatches=%" PRIu64
            " tolerance_failures=%" PRIu64 " max_abs_error=%.9g\n",
            verification.compared,
            verification.bit_mismatches,
            verification.tolerance_failures,
            verification.maximum_absolute_error
        );
        if (verification.tolerance_failures != 0) {
            fprintf(
                stderr,
                "first oracle mismatch row=%d token=%d company=%.9g "
                "oracle=%.9g\n",
                verification.first_bad_row,
                verification.first_bad_token,
                verification.first_bad_company,
                verification.first_bad_oracle
            );
            return EXIT_FAILURE;
        }
    }

    free(result.payoffs.second);
    free(result.payoffs.first);
    free(result.logits.values);
    free(result.branches);
    free(ledger.fillers);
    free(prompt_tokens);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return EXIT_SUCCESS;
}
