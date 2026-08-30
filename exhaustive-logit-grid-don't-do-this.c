/*
 * DO NOT USE AS THE SCALABLE INFERENCE PATH.
 *
 * I should never have expanded the complete depth-three vocabulary grid and
 * presented that exhaustive materialization as progress toward sampled
 * Escardo inference.  It is a finite correctness oracle only; its V^2 state
 * is precisely the all-path enumeration the requested algorithm avoids.
 * It is kept only as a reminder of the rejected approach.
 */

#define main escardo_strength_depth2_main
#include "exhaustive-prefix-company-don't-do-this.c"
#undef main

/*
 * A complete, three-logit instance of Escardo's dependent product.
 *
 * The older two-token control program is included above so this experiment
 * reuses exactly its weight-stationary numerical leaves.  The carrier of the
 * selection below is not a bare token.  It is the model-produced logit value,
 * with the token retained as the logit's finite-domain coordinate:
 *
 *     Select_R Logit = (Logit -> R) -> Logit
 *
 * Before the selection is terminalized, the complete depth-three prefix
 * company is passed through every learned filler.  For vocabulary V it holds
 * the prompt, V possible first tokens, and V^2 possible token pairs.  This is
 * intentionally expensive and bounded: it is a correctness experiment, not
 * a long-horizon implementation.
 */

typedef struct {
    int node_count;
    int output_count;
    int maximum_context;
    int *tokens;
    int *positions;
    int *parents;
    int *output_nodes;
} Depth3Shape;

typedef struct {
    int count;
    int row_count;
    int width;
    float **hidden;
} Depth3Scales;

typedef struct {
    LogitCompany logits;
    Depth3Scales scales;
} Depth3Evaluation;

static int checked_int_sum(int left, int right) {
    if (left < 0 || right < 0 || left > INT32_MAX - right) {
        strength_fail("integer extent overflow");
    }
    return left + right;
}

static int checked_int_product(int left, int right) {
    if (left < 0 || right < 0 ||
        (right != 0 && left > INT32_MAX / right)) {
        strength_fail("integer extent overflow");
    }
    return left * right;
}

static Depth3Shape make_depth3_shape(
    const int *prompt_tokens,
    int prompt_count,
    int vocab_size
) {
    int pair_count = checked_int_product(vocab_size, vocab_size);
    int first_offset = prompt_count;
    int pair_offset = checked_int_sum(first_offset, vocab_size);
    int node_count = checked_int_sum(pair_offset, pair_count);
    int output_count = checked_int_sum(checked_int_sum(1, vocab_size), pair_count);
    Depth3Shape shape = {
        .node_count = node_count,
        .output_count = output_count,
        .maximum_context = checked_int_sum(prompt_count, 2),
        .tokens = strength_calloc((size_t)node_count, sizeof(*shape.tokens)),
        .positions = strength_calloc((size_t)node_count, sizeof(*shape.positions)),
        .parents = strength_calloc((size_t)node_count, sizeof(*shape.parents)),
        .output_nodes = strength_calloc(
            (size_t)output_count,
            sizeof(*shape.output_nodes)
        ),
    };

    for (int position = 0; position < prompt_count; position++) {
        shape.tokens[position] = prompt_tokens[position];
        shape.positions[position] = position;
        shape.parents[position] = position == 0 ? -1 : position - 1;
    }
    shape.output_nodes[0] = prompt_count - 1;

    for (int first = 0; first < vocab_size; first++) {
        int node = first_offset + first;
        shape.tokens[node] = first;
        shape.positions[node] = prompt_count;
        shape.parents[node] = prompt_count - 1;
        shape.output_nodes[1 + first] = node;
    }

    for (int first = 0; first < vocab_size; first++) {
        for (int second = 0; second < vocab_size; second++) {
            int pair = first * vocab_size + second;
            int node = pair_offset + pair;
            shape.tokens[node] = second;
            shape.positions[node] = prompt_count + 1;
            shape.parents[node] = first_offset + first;
            shape.output_nodes[1 + vocab_size + pair] = node;
        }
    }
    return shape;
}

static void free_depth3_shape(Depth3Shape *shape) {
    free(shape->output_nodes);
    free(shape->parents);
    free(shape->positions);
    free(shape->tokens);
    memset(shape, 0, sizeof(*shape));
}

static int depth3_context_member(
    const Depth3Shape *shape,
    int node,
    int context_index
) {
    int position = shape->positions[node];
    if (context_index < 0 || context_index > position) {
        strength_fail("attention context index outside prefix");
    }
    int steps = position - context_index;
    int member = node;
    while (steps-- > 0) {
        member = shape->parents[member];
        if (member < 0) strength_fail("broken prefix-company parent chain");
    }
    if (shape->positions[member] != context_index) {
        strength_fail("prefix-company position mismatch");
    }
    return member;
}

static Depth3Scales new_depth3_scales(
    int count,
    int row_count,
    int width
) {
    Depth3Scales scales = {
        .count = count,
        .row_count = row_count,
        .width = width,
        .hidden = strength_calloc((size_t)count, sizeof(*scales.hidden)),
    };
    for (int scale = 0; scale < count; scale++) {
        scales.hidden[scale] = strength_calloc(
            strength_elements(row_count, width),
            sizeof(**scales.hidden)
        );
    }
    return scales;
}

static void capture_depth3_scale(
    Depth3Scales *scales,
    int scale,
    const float *hidden
) {
    if (scale < 0 || scale >= scales->count) {
        strength_fail("invalid retained layer scale");
    }
    memcpy(
        scales->hidden[scale],
        hidden,
        strength_elements(scales->row_count, scales->width) * sizeof(float)
    );
}

static void free_depth3_scales(Depth3Scales *scales) {
    for (int scale = 0; scale < scales->count; scale++) {
        free(scales->hidden[scale]);
    }
    free(scales->hidden);
    memset(scales, 0, sizeof(*scales));
}

static Depth3Evaluation evaluate_depth3_company(
    Program *program,
    const Depth3Shape *shape
) {
    Transformer *transformer = program->transformer;
    Config *config = &transformer->config;
    TransformerWeights *weights = &transformer->weights;
    int dim = config->dim;
    int hidden_dim = config->hidden_dim;
    int vocab_size = config->vocab_size;
    int company_count = shape->node_count;
    int output_count = shape->output_count;
    int head_size = dim / config->n_heads;
    int kv_dim = dim * config->n_kv_heads / config->n_heads;
    int kv_mul = config->n_heads / config->n_kv_heads;

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
        (size_t)shape->maximum_context,
        sizeof(*attention_scores)
    );
    Depth3Scales scales = new_depth3_scales(
        config->n_layers + 1,
        company_count,
        dim
    );

    embedding_company_apply(
        x,
        shape->tokens,
        company_count,
        weights->token_embedding_table,
        vocab_size,
        dim,
        program->ledger,
        embedding_filler()
    );
    capture_depth3_scale(&scales, 0, x);

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
            int position = shape->positions[company];
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
            int context_count = shape->positions[company] + 1;
            const float *company_query = query + (size_t)company * dim;
            float *company_output = attended + (size_t)company * dim;
            for (int head = 0; head < config->n_heads; head++) {
                const float *head_query = company_query + head * head_size;
                for (int context_index = 0; context_index < context_count;
                     context_index++) {
                    int member = depth3_context_member(
                        shape,
                        company,
                        context_index
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
                    int member = depth3_context_member(
                        shape,
                        company,
                        context_index
                    );
                    const float *head_value = value + (size_t)member * kv_dim +
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
        capture_depth3_scale(&scales, layer + 1, x);
    }

    float *output_hidden = strength_calloc(
        strength_elements(output_count, dim),
        sizeof(*output_hidden)
    );
    for (int output = 0; output < output_count; output++) {
        memcpy(
            output_hidden + (size_t)output * dim,
            x + (size_t)shape->output_nodes[output] * dim,
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

    return (Depth3Evaluation){
        .logits = {
            .values = logits,
            .row_count = output_count,
            .vocab_size = vocab_size,
        },
        .scales = scales,
    };
}

/*
 * A logit is a value in a particular model-produced vocabulary row.  The
 * token is not a replacement carrier: it is the coordinate needed to name
 * the logit and hence to obtain the dependent row at the next position.
 */
typedef struct {
    int row;
    int token;
    float value;
    double log_probability;
} ModelLogit;

typedef struct {
    ModelLogit first;
    ModelLogit second;
} ModelLogitPair;

typedef struct {
    ModelLogit first;
    ModelLogit second;
    ModelLogit third;
} ModelLogitTriple;

typedef Reward (*LogitObservation)(void *environment, ModelLogit logit);
typedef Reward (*LogitPairObservation)(
    void *environment,
    ModelLogitPair pair
);
typedef Reward (*LogitTripleObservation)(
    void *environment,
    ModelLogitTriple triple
);

/* Select_R Logit = (Logit -> R) -> Logit. */
typedef struct RanConstConstSelectLogit RanConstConstSelectLogit;
struct RanConstConstSelectLogit {
    void *environment;
    ModelLogit (*apply)(
        void *environment,
        LogitObservation observation,
        void *observation_environment
    );
};

/* The two dependent-product codomains used by this finite instance. */
typedef struct RanConstConstSelectLogitPair RanConstConstSelectLogitPair;
struct RanConstConstSelectLogitPair {
    void *environment;
    ModelLogitPair (*apply)(
        void *environment,
        LogitPairObservation observation,
        void *observation_environment
    );
};

typedef struct RanConstConstSelectLogitTriple RanConstConstSelectLogitTriple;
struct RanConstConstSelectLogitTriple {
    void *environment;
    ModelLogitTriple (*apply)(
        void *environment,
        LogitTripleObservation observation,
        void *observation_environment
    );
};

/* Cont_R Triple = (Triple -> R) -> R. */
typedef struct RanConstConstContLogitTriple RanConstConstContLogitTriple;
struct RanConstConstContLogitTriple {
    void *environment;
    Reward (*apply)(
        void *environment,
        LogitTripleObservation observation,
        void *observation_environment,
        ModelLogitTriple *witness
    );
};

typedef struct {
    const LogitCompany *company;
    double *log_partitions;
} LogitTable;

typedef struct {
    uint64_t row_selection_applications;
    uint64_t candidate_observations;
    uint64_t terminal_observations;
} LogitSelectionStats;

typedef struct {
    const LogitTable *table;
    int support_count;
    int *tokens;
    unsigned char *ready;
} LogitSupportCache;

static LogitTable make_logit_table(const LogitCompany *company) {
    LogitTable result = {
        .company = company,
        .log_partitions = strength_calloc(
            (size_t)company->row_count,
            sizeof(*result.log_partitions)
        ),
    };
    for (int row = 0; row < company->row_count; row++) {
        const float *values = company->values +
            (size_t)row * company->vocab_size;
        double maximum = -DBL_MAX;
        for (int token = 0; token < company->vocab_size; token++) {
            if ((double)values[token] > maximum) maximum = values[token];
        }
        double partition = 0.0;
        for (int token = 0; token < company->vocab_size; token++) {
            partition += exp((double)values[token] - maximum);
        }
        result.log_partitions[row] = maximum + log(partition);
    }
    return result;
}

static void free_logit_table(LogitTable *table) {
    free(table->log_partitions);
    memset(table, 0, sizeof(*table));
}

static ModelLogit table_logit(
    const LogitTable *table,
    int row,
    int token
) {
    if (row < 0 || row >= table->company->row_count ||
        token < 0 || token >= table->company->vocab_size) {
        strength_fail("logit coordinate outside the complete company");
    }
    float value = table->company->values[
        (size_t)row * table->company->vocab_size + token
    ];
    return (ModelLogit){
        .row = row,
        .token = token,
        .value = value,
        .log_probability = (double)value - table->log_partitions[row],
    };
}

static LogitSupportCache make_logit_support_cache(
    const LogitTable *table,
    int support_count
) {
    int vocab_size = table->company->vocab_size;
    if (support_count <= 0 || support_count > vocab_size) {
        strength_fail("top-k support outside the logit vocabulary");
    }
    LogitSupportCache cache = {
        .table = table,
        .support_count = support_count,
    };
    if (support_count < vocab_size) {
        cache.tokens = strength_calloc(
            strength_elements(table->company->row_count, support_count),
            sizeof(*cache.tokens)
        );
        cache.ready = strength_calloc(
            (size_t)table->company->row_count,
            sizeof(*cache.ready)
        );
    }
    return cache;
}

static void free_logit_support_cache(LogitSupportCache *cache) {
    free(cache->ready);
    free(cache->tokens);
    memset(cache, 0, sizeof(*cache));
}

static bool logit_precedes(
    const float *row,
    int left_token,
    int right_token
) {
    if (row[left_token] > row[right_token]) return true;
    if (row[left_token] < row[right_token]) return false;
    return left_token < right_token;
}

static void prepare_logit_support(LogitSupportCache *cache, int row_index) {
    if (row_index < 0 || row_index >= cache->table->company->row_count) {
        strength_fail("top-k row outside the logit company");
    }
    if (cache->tokens == NULL || cache->ready[row_index]) return;
    int count = cache->support_count;
    int vocab_size = cache->table->company->vocab_size;
    int *support = cache->tokens + (size_t)row_index * count;
    const float *row = cache->table->company->values +
        (size_t)row_index * vocab_size;
    int filled = 0;
    for (int token = 0; token < vocab_size; token++) {
        int insertion = filled;
        while (insertion > 0 &&
               logit_precedes(row, token, support[insertion - 1])) {
            insertion--;
        }
        if (insertion < count) {
            int last = filled < count ? filled : count - 1;
            for (int index = last; index > insertion; index--) {
                support[index] = support[index - 1];
            }
            support[insertion] = token;
            if (filled < count) filled++;
        }
    }
    if (filled != count) strength_fail("top-k logit support incomplete");
    cache->ready[row_index] = 1;
}

static int logit_support_token(
    LogitSupportCache *cache,
    int row,
    int support_index
) {
    if (support_index < 0 || support_index >= cache->support_count) {
        strength_fail("top-k support index outside its domain");
    }
    if (cache->tokens == NULL) return support_index;
    prepare_logit_support(cache, row);
    return cache->tokens[(size_t)row * cache->support_count + support_index];
}

typedef struct {
    const LogitTable *table;
    LogitSupportCache *support;
    LogitSelectionStats *stats;
    int row;
} LogitRowSelection;

static ModelLogit select_logit_row(
    void *raw_environment,
    LogitObservation observation,
    void *observation_environment
) {
    LogitRowSelection *environment = raw_environment;
    environment->stats->row_selection_applications++;
    bool found = false;
    ModelLogit best = {0};
    Reward best_reward = -INFINITY;
    for (int index = 0; index < environment->support->support_count; index++) {
        int token = logit_support_token(
            environment->support,
            environment->row,
            index
        );
        ModelLogit candidate = table_logit(
            environment->table,
            environment->row,
            token
        );
        Reward reward = observation(observation_environment, candidate);
        environment->stats->candidate_observations++;
        if (isnan(reward)) strength_fail("observer returned NaN");
        if (!found || reward > best_reward) {
            found = true;
            best = candidate;
            best_reward = reward;
        }
    }
    if (!found) strength_fail("selection over an empty logit row");
    return best;
}

static RanConstConstSelectLogit logit_row_selection(
    LogitRowSelection *environment
) {
    return (RanConstConstSelectLogit){
        .environment = environment,
        .apply = select_logit_row,
    };
}

typedef RanConstConstSelectLogit (*DependentLogitSelection)(
    void *environment,
    ModelLogit head
);

typedef struct {
    RanConstConstSelectLogit head;
    DependentLogitSelection suffix;
    void *suffix_environment;
    int domain_size;
    ModelLogit *remembered_suffixes;
    Reward *remembered_rewards;
    unsigned char *remembered;
} LogitPairProduct;

typedef struct {
    LogitPairProduct *product;
    LogitPairObservation observation;
    void *observation_environment;
} LogitPairFrame;

typedef struct {
    LogitPairFrame *frame;
    ModelLogit head;
} LogitPairSuffixFrame;

static Reward observe_logit_pair_suffix(
    void *raw_environment,
    ModelLogit suffix
) {
    LogitPairSuffixFrame *environment = raw_environment;
    return environment->frame->observation(
        environment->frame->observation_environment,
        (ModelLogitPair){
            .first = environment->head,
            .second = suffix,
        }
    );
}

static Reward observe_logit_pair_head(
    void *raw_environment,
    ModelLogit head
) {
    LogitPairFrame *frame = raw_environment;
    LogitPairProduct *product = frame->product;
    int coordinate = head.token;
    if (coordinate < 0 || coordinate >= product->domain_size) {
        strength_fail("pair-product head coordinate outside its domain");
    }
    if (!product->remembered[coordinate]) {
        RanConstConstSelectLogit suffix = product->suffix(
            product->suffix_environment,
            head
        );
        LogitPairSuffixFrame suffix_frame = {
            .frame = frame,
            .head = head,
        };
        ModelLogit selected_suffix = suffix.apply(
            suffix.environment,
            observe_logit_pair_suffix,
            &suffix_frame
        );
        product->remembered_suffixes[coordinate] = selected_suffix;
        product->remembered_rewards[coordinate] = frame->observation(
            frame->observation_environment,
            (ModelLogitPair){
                .first = head,
                .second = selected_suffix,
            }
        );
        product->remembered[coordinate] = 1;
    }
    return product->remembered_rewards[coordinate];
}

static ModelLogitPair apply_logit_pair_product(
    void *raw_environment,
    LogitPairObservation observation,
    void *observation_environment
) {
    LogitPairProduct *product = raw_environment;
    LogitPairFrame frame = {
        .product = product,
        .observation = observation,
        .observation_environment = observation_environment,
    };
    ModelLogit head = product->head.apply(
        product->head.environment,
        observe_logit_pair_head,
        &frame
    );
    if (!product->remembered[head.token]) {
        strength_fail("pair strength forgot its selected suffix");
    }
    return (ModelLogitPair){
        .first = head,
        .second = product->remembered_suffixes[head.token],
    };
}

static RanConstConstSelectLogitPair dependent_logit_pair_strength(
    RanConstConstSelectLogit head,
    DependentLogitSelection suffix,
    void *suffix_environment,
    int domain_size,
    LogitPairProduct *storage
) {
    *storage = (LogitPairProduct){
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
    return (RanConstConstSelectLogitPair){
        .environment = storage,
        .apply = apply_logit_pair_product,
    };
}

static void free_logit_pair_product(LogitPairProduct *product) {
    free(product->remembered);
    free(product->remembered_rewards);
    free(product->remembered_suffixes);
    memset(product, 0, sizeof(*product));
}

typedef struct LogitTailFamily LogitTailFamily;

typedef struct {
    LogitTailFamily *family;
    int first_token;
} ThirdLogitFamily;

struct LogitTailFamily {
    const LogitTable *table;
    LogitSupportCache *support;
    LogitSelectionStats *stats;
    int vocab_size;
    LogitRowSelection *second_rows;
    LogitRowSelection *third_rows;
    ThirdLogitFamily *third_families;
    LogitPairProduct *products;
    RanConstConstSelectLogitPair *selections;
};

static RanConstConstSelectLogit third_logit_selection(
    void *raw_environment,
    ModelLogit second
) {
    ThirdLogitFamily *environment = raw_environment;
    LogitTailFamily *family = environment->family;
    if (second.token < 0 || second.token >= family->vocab_size) {
        strength_fail("second-logit coordinate outside its domain");
    }
    int pair = environment->first_token * family->vocab_size + second.token;
    return logit_row_selection(&family->third_rows[pair]);
}

static LogitTailFamily make_logit_tail_family(
    const LogitTable *table,
    LogitSupportCache *support,
    LogitSelectionStats *stats
) {
    int vocab_size = table->company->vocab_size;
    int pair_count = checked_int_product(vocab_size, vocab_size);
    LogitTailFamily family = {
        .table = table,
        .support = support,
        .stats = stats,
        .vocab_size = vocab_size,
        .second_rows = strength_calloc(
            (size_t)vocab_size,
            sizeof(*family.second_rows)
        ),
        .third_rows = strength_calloc(
            (size_t)pair_count,
            sizeof(*family.third_rows)
        ),
        .third_families = strength_calloc(
            (size_t)vocab_size,
            sizeof(*family.third_families)
        ),
        .products = strength_calloc(
            (size_t)vocab_size,
            sizeof(*family.products)
        ),
        .selections = strength_calloc(
            (size_t)vocab_size,
            sizeof(*family.selections)
        ),
    };

    for (int first = 0; first < vocab_size; first++) {
        family.second_rows[first] = (LogitRowSelection){
            .table = table,
            .support = support,
            .stats = stats,
            .row = 1 + first,
        };
        family.third_families[first] = (ThirdLogitFamily){
            .family = NULL,
            .first_token = first,
        };
    }
    for (int pair = 0; pair < pair_count; pair++) {
        family.third_rows[pair] = (LogitRowSelection){
            .table = table,
            .support = support,
            .stats = stats,
            .row = 1 + vocab_size + pair,
        };
    }

    /*
     * family is returned by value, so fix the self pointers after its final
     * address is known in finish_logit_tail_family.
     */
    return family;
}

static void finish_logit_tail_family(LogitTailFamily *family) {
    for (int first = 0; first < family->vocab_size; first++) {
        family->third_families[first].family = family;
        family->selections[first] = dependent_logit_pair_strength(
            logit_row_selection(&family->second_rows[first]),
            third_logit_selection,
            &family->third_families[first],
            family->vocab_size,
            &family->products[first]
        );
    }
}

static void free_logit_tail_family(LogitTailFamily *family) {
    for (int first = 0; first < family->vocab_size; first++) {
        free_logit_pair_product(&family->products[first]);
    }
    free(family->selections);
    free(family->products);
    free(family->third_families);
    free(family->third_rows);
    free(family->second_rows);
    memset(family, 0, sizeof(*family));
}

static RanConstConstSelectLogitPair tail_pair_selection(
    void *raw_environment,
    ModelLogit first
) {
    LogitTailFamily *family = raw_environment;
    if (first.token < 0 || first.token >= family->vocab_size) {
        strength_fail("first-logit coordinate outside its domain");
    }
    return family->selections[first.token];
}

typedef RanConstConstSelectLogitPair (*DependentLogitPairSelection)(
    void *environment,
    ModelLogit head
);

typedef struct {
    RanConstConstSelectLogit head;
    DependentLogitPairSelection suffix;
    void *suffix_environment;
    int domain_size;
    ModelLogitPair *remembered_suffixes;
    Reward *remembered_rewards;
    unsigned char *remembered;
} LogitTripleProduct;

typedef struct {
    LogitTripleProduct *product;
    LogitTripleObservation observation;
    void *observation_environment;
} LogitTripleFrame;

typedef struct {
    LogitTripleFrame *frame;
    ModelLogit head;
} LogitTripleSuffixFrame;

static Reward observe_logit_triple_suffix(
    void *raw_environment,
    ModelLogitPair suffix
) {
    LogitTripleSuffixFrame *environment = raw_environment;
    return environment->frame->observation(
        environment->frame->observation_environment,
        (ModelLogitTriple){
            .first = environment->head,
            .second = suffix.first,
            .third = suffix.second,
        }
    );
}

static Reward observe_logit_triple_head(
    void *raw_environment,
    ModelLogit head
) {
    LogitTripleFrame *frame = raw_environment;
    LogitTripleProduct *product = frame->product;
    int coordinate = head.token;
    if (coordinate < 0 || coordinate >= product->domain_size) {
        strength_fail("triple-product head coordinate outside its domain");
    }
    if (!product->remembered[coordinate]) {
        RanConstConstSelectLogitPair suffix = product->suffix(
            product->suffix_environment,
            head
        );
        LogitTripleSuffixFrame suffix_frame = {
            .frame = frame,
            .head = head,
        };
        ModelLogitPair selected_suffix = suffix.apply(
            suffix.environment,
            observe_logit_triple_suffix,
            &suffix_frame
        );
        product->remembered_suffixes[coordinate] = selected_suffix;
        product->remembered_rewards[coordinate] = frame->observation(
            frame->observation_environment,
            (ModelLogitTriple){
                .first = head,
                .second = selected_suffix.first,
                .third = selected_suffix.second,
            }
        );
        product->remembered[coordinate] = 1;
    }
    return product->remembered_rewards[coordinate];
}

static ModelLogitTriple apply_logit_triple_product(
    void *raw_environment,
    LogitTripleObservation observation,
    void *observation_environment
) {
    LogitTripleProduct *product = raw_environment;
    LogitTripleFrame frame = {
        .product = product,
        .observation = observation,
        .observation_environment = observation_environment,
    };
    ModelLogit head = product->head.apply(
        product->head.environment,
        observe_logit_triple_head,
        &frame
    );
    if (!product->remembered[head.token]) {
        strength_fail("triple strength forgot its selected suffix");
    }
    ModelLogitPair suffix = product->remembered_suffixes[head.token];
    return (ModelLogitTriple){
        .first = head,
        .second = suffix.first,
        .third = suffix.second,
    };
}

static RanConstConstSelectLogitTriple dependent_logit_triple_strength(
    RanConstConstSelectLogit head,
    DependentLogitPairSelection suffix,
    void *suffix_environment,
    int domain_size,
    LogitTripleProduct *storage
) {
    *storage = (LogitTripleProduct){
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
    return (RanConstConstSelectLogitTriple){
        .environment = storage,
        .apply = apply_logit_triple_product,
    };
}

static void free_logit_triple_product(LogitTripleProduct *product) {
    free(product->remembered);
    free(product->remembered_rewards);
    free(product->remembered_suffixes);
    memset(product, 0, sizeof(*product));
}

typedef struct {
    RanConstConstSelectLogitTriple selection;
} LogitTauEnvironment;

static Reward apply_logit_tau(
    void *raw_environment,
    LogitTripleObservation observation,
    void *observation_environment,
    ModelLogitTriple *witness
) {
    LogitTauEnvironment *environment = raw_environment;
    *witness = environment->selection.apply(
        environment->selection.environment,
        observation,
        observation_environment
    );
    return observation(observation_environment, *witness);
}

/* tau eps p = p (eps p), and this is the only terminalization. */
static RanConstConstContLogitTriple logit_selection_to_continuation(
    RanConstConstSelectLogitTriple selection,
    LogitTauEnvironment *storage
) {
    *storage = (LogitTauEnvironment){.selection = selection};
    return (RanConstConstContLogitTriple){
        .environment = storage,
        .apply = apply_logit_tau,
    };
}

typedef struct {
    int vocab_size;
    LogitSelectionStats *stats;
} LogProbabilityObserver;

static Reward observe_log_probability_triple(
    void *raw_environment,
    ModelLogitTriple triple
) {
    LogProbabilityObserver *environment = raw_environment;
    int vocab_size = environment->vocab_size;
    int expected_second_row = 1 + triple.first.token;
    int expected_third_row = 1 + vocab_size +
        triple.first.token * vocab_size + triple.second.token;
    if (triple.first.row != 0 || triple.second.row != expected_second_row ||
        triple.third.row != expected_third_row) {
        strength_fail("observer received a non-causal logit triple");
    }
    environment->stats->terminal_observations++;
    return triple.first.log_probability + triple.second.log_probability +
        triple.third.log_probability;
}

typedef struct {
    ModelLogitTriple triple;
    Reward reward;
    Reward covered_log_probability_mass;
    uint64_t path_count;
} LogitTripleBranch;

typedef struct {
    ModelLogitPair pair;
    Reward reward;
    Reward covered_log_probability_mass;
    uint64_t path_count;
} LogitPairBranch;

static int compare_logit_triple_branch(
    const void *left_raw,
    const void *right_raw
) {
    const LogitTripleBranch *left = left_raw;
    const LogitTripleBranch *right = right_raw;
    if (left->reward > right->reward) return -1;
    if (left->reward < right->reward) return 1;
    if (left->triple.first.token < right->triple.first.token) return -1;
    if (left->triple.first.token > right->triple.first.token) return 1;
    return 0;
}

static int compare_logit_pair_branch(
    const void *left_raw,
    const void *right_raw
) {
    const LogitPairBranch *left = left_raw;
    const LogitPairBranch *right = right_raw;
    if (left->reward > right->reward) return -1;
    if (left->reward < right->reward) return 1;
    if (left->pair.first.token < right->pair.first.token) return -1;
    if (left->pair.first.token > right->pair.first.token) return 1;
    return 0;
}

typedef struct {
    Program *program;
    const Depth3Shape *shape;
    int top_k;
    int sample_milliseconds;
} Depth3LogitTerm;

typedef struct {
    Depth3Evaluation evaluation;
    ModelLogitTriple witness;
    Reward reward;
    Reward backed_reward;
    Reward covered_log_probability_mass;
    Reward selected_pair_backed_reward;
    Reward selected_pair_covered_log_probability_mass;
    uint64_t selected_pair_path_count;
    uint64_t selected_root_path_count;
    LogitTripleBranch *root_branches;
    LogitPairBranch *selected_tail_branches;
    int branch_count;
    int tail_branch_count;
    LogitSelectionStats stats;
    bool sampled;
    bool sampled_support_exhausted;
    uint64_t sampled_attempts;
    uint64_t sampled_unique_roots;
    uint64_t sampled_unique_pairs;
    uint64_t sampled_unique_triples;
    uint64_t sampled_logit_draws;
    long company_milliseconds;
    long selection_milliseconds;
} Depth3LogitResult;

typedef struct {
    bool reached;
    uint64_t path_count;
    Reward best_reward;
    Reward log_reward_sum;
} SampledRootBackup;

typedef struct {
    bool reached;
    uint64_t path_count;
    Reward best_reward;
    Reward log_reward_sum;
} SampledPairBackup;

static uint64_t monotonic_nanoseconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        strength_fail("could not read the sampling clock");
    }
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) +
        (uint64_t)now.tv_nsec;
}

static uint64_t next_sample_random(uint64_t *state) {
    uint64_t value = *state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    return value * UINT64_C(2685821657736338717);
}

static int sample_support_index(uint64_t *state, int count) {
    if (count <= 0) strength_fail("sampled an empty support");
    uint64_t bound = (uint64_t)count;
    uint64_t threshold = (uint64_t)(-bound) % bound;
    uint64_t value;
    do {
        value = next_sample_random(state);
    } while (value < threshold);
    return (int)(value % bound);
}

static Reward log_add_reward(Reward total, Reward reward, bool has_total) {
    if (!has_total) return reward;
    Reward maximum = total > reward ? total : reward;
    Reward minimum = total > reward ? reward : total;
    return maximum + log1p(exp(minimum - maximum));
}

static int best_sampled_second_rank(
    const SampledPairBackup *pairs,
    int top_k,
    int first_rank
) {
    int best = -1;
    Reward best_reward = -INFINITY;
    for (int second_rank = 0; second_rank < top_k; second_rank++) {
        const SampledPairBackup *pair =
            &pairs[first_rank * top_k + second_rank];
        if (!pair->reached) continue;
        if (best < 0 || pair->best_reward > best_reward ||
            (pair->best_reward == best_reward && second_rank < best)) {
            best = second_rank;
            best_reward = pair->best_reward;
        }
    }
    return best;
}

static int best_sampled_third_rank(
    const unsigned char *reached,
    const Reward *leaf_rewards,
    int top_k,
    int first_rank,
    int second_rank
) {
    int best = -1;
    Reward best_reward = -INFINITY;
    for (int third_rank = 0; third_rank < top_k; third_rank++) {
        uint64_t path =
            ((uint64_t)first_rank * (uint64_t)top_k +
             (uint64_t)second_rank) * (uint64_t)top_k +
            (uint64_t)third_rank;
        unsigned char bit = (unsigned char)(1u << (path & 7));
        if ((reached[path >> 3] & bit) == 0) continue;
        Reward reward = leaf_rewards[path];
        if (best < 0 || reward > best_reward ||
            (reward == best_reward && third_rank < best)) {
            best = third_rank;
            best_reward = reward;
        }
    }
    return best;
}

/*
 * Sample a finite reachability subterm, then retain the exact max-reward
 * continuation at every reached pair and root.  Randomness chooses coverage;
 * it never changes, augments, or normalizes the observer's reward.
 */
static void run_sampled_logit_strength(
    Depth3LogitResult *result,
    const LogitTable *table,
    LogitSupportCache *support,
    int sample_milliseconds
) {
    int vocab_size = table->company->vocab_size;
    int top_k = support->support_count;
    uint64_t support_paths = checked_u64_product(
        checked_u64_product((uint64_t)top_k, (uint64_t)top_k),
        (uint64_t)top_k
    );
    size_t reached_bytes = (size_t)((support_paths + 7) / 8);
    unsigned char *reached_triples = strength_calloc(
        reached_bytes,
        sizeof(*reached_triples)
    );
    Reward *leaf_rewards = strength_calloc(
        (size_t)support_paths,
        sizeof(*leaf_rewards)
    );
    SampledRootBackup *roots = strength_calloc(
        (size_t)top_k,
        sizeof(*roots)
    );
    SampledPairBackup *pairs = strength_calloc(
        strength_elements(top_k, top_k),
        sizeof(*pairs)
    );
    LogProbabilityObserver observer = {
        .vocab_size = vocab_size,
        .stats = &result->stats,
    };
    uint64_t random_state = UINT64_C(0x4d595df4d0f33173);
    uint64_t started = monotonic_nanoseconds();
    uint64_t deadline = started +
        (uint64_t)sample_milliseconds * UINT64_C(1000000);

    do {
        int first_rank = sample_support_index(&random_state, top_k);
        int first_token = logit_support_token(support, 0, first_rank);
        int second_row = 1 + first_token;
        int second_rank = sample_support_index(&random_state, top_k);
        int second_token = logit_support_token(
            support,
            second_row,
            second_rank
        );
        int third_row = 1 + vocab_size +
            first_token * vocab_size + second_token;
        int third_rank = sample_support_index(&random_state, top_k);
        int third_token = logit_support_token(
            support,
            third_row,
            third_rank
        );
        result->sampled_attempts++;
        result->sampled_logit_draws += 3;

        uint64_t support_path =
            ((uint64_t)first_rank * (uint64_t)top_k +
             (uint64_t)second_rank) * (uint64_t)top_k +
            (uint64_t)third_rank;
        size_t byte = (size_t)(support_path >> 3);
        unsigned char bit = (unsigned char)(1u << (support_path & 7));
        if ((reached_triples[byte] & bit) == 0) {
            reached_triples[byte] |= bit;
            result->sampled_unique_triples++;
            ModelLogitTriple triple = {
                .first = table_logit(table, 0, first_token),
                .second = table_logit(table, second_row, second_token),
                .third = table_logit(table, third_row, third_token),
            };
            Reward reward = observe_log_probability_triple(
                &observer,
                triple
            );

            leaf_rewards[support_path] = reward;
            int pair_index = first_rank * top_k + second_rank;
            SampledPairBackup *pair = &pairs[pair_index];
            if (!pair->reached) {
                result->sampled_unique_pairs++;
                pair->best_reward = reward;
            } else if (reward > pair->best_reward) {
                pair->best_reward = reward;
            }
            pair->log_reward_sum = log_add_reward(
                pair->log_reward_sum,
                reward,
                pair->reached
            );
            pair->reached = true;
            pair->path_count++;

            SampledRootBackup *root = &roots[first_rank];
            if (!root->reached) {
                result->sampled_unique_roots++;
                root->best_reward = reward;
            } else if (reward > root->best_reward) {
                root->best_reward = reward;
            }
            root->log_reward_sum = log_add_reward(
                root->log_reward_sum,
                reward,
                root->reached
            );
            root->reached = true;
            root->path_count++;
        }
        if (result->sampled_unique_triples == support_paths) {
            result->sampled_support_exhausted = true;
            break;
        }
    } while (monotonic_nanoseconds() < deadline);

    bool found = false;
    Reward best_backed_reward = -INFINITY;
    int best_first_rank = -1;
    for (int first_rank = 0; first_rank < top_k; first_rank++) {
        if (!roots[first_rank].reached) continue;
        if (!found ||
            roots[first_rank].best_reward > best_backed_reward ||
            (roots[first_rank].best_reward == best_backed_reward &&
             first_rank < best_first_rank)) {
            found = true;
            best_backed_reward = roots[first_rank].best_reward;
            best_first_rank = first_rank;
        }
    }
    if (!found) strength_fail("sampling reached no complete logit path");

    int best_second_rank = -1;
    Reward best_pair_reward = -INFINITY;
    for (int second_rank = 0; second_rank < top_k; second_rank++) {
        SampledPairBackup *pair =
            &pairs[best_first_rank * top_k + second_rank];
        if (!pair->reached) continue;
        if (best_second_rank < 0 ||
            pair->best_reward > best_pair_reward ||
            (pair->best_reward == best_pair_reward &&
             second_rank < best_second_rank)) {
            best_second_rank = second_rank;
            best_pair_reward = pair->best_reward;
        }
    }
    if (best_second_rank < 0) {
        strength_fail("selected sampled root has no reached pair");
    }

    int best_third_rank = -1;
    Reward best_leaf_reward = -INFINITY;
    for (int third_rank = 0; third_rank < top_k; third_rank++) {
        uint64_t path =
            ((uint64_t)best_first_rank * (uint64_t)top_k +
             (uint64_t)best_second_rank) * (uint64_t)top_k +
            (uint64_t)third_rank;
        unsigned char path_bit = (unsigned char)(1u << (path & 7));
        if ((reached_triples[path >> 3] & path_bit) == 0) continue;
        Reward leaf_reward = leaf_rewards[path];
        if (best_third_rank < 0 || leaf_reward > best_leaf_reward ||
            (leaf_reward == best_leaf_reward &&
             third_rank < best_third_rank)) {
            best_third_rank = third_rank;
            best_leaf_reward = leaf_reward;
        }
    }
    if (best_third_rank < 0) {
        strength_fail("selected sampled pair has no reached leaf");
    }

    int best_first = logit_support_token(support, 0, best_first_rank);
    int best_second_row = 1 + best_first;
    int best_second = logit_support_token(
        support,
        best_second_row,
        best_second_rank
    );
    int best_third_row = 1 + vocab_size +
        best_first * vocab_size + best_second;
    int best_third = logit_support_token(
        support,
        best_third_row,
        best_third_rank
    );
    result->witness = (ModelLogitTriple){
        .first = table_logit(table, 0, best_first),
        .second = table_logit(table, best_second_row, best_second),
        .third = table_logit(table, best_third_row, best_third),
    };
    result->backed_reward = best_backed_reward;
    result->covered_log_probability_mass =
        roots[best_first_rank].log_reward_sum;
    result->selected_pair_backed_reward = best_pair_reward;
    result->selected_pair_covered_log_probability_mass =
        pairs[best_first_rank * top_k + best_second_rank].log_reward_sum;
    result->selected_pair_path_count =
        pairs[best_first_rank * top_k + best_second_rank].path_count;
    result->selected_root_path_count = roots[best_first_rank].path_count;

    /* Observe the recursively selected path once after the backward pass. */
    result->reward = observe_log_probability_triple(
        &observer,
        result->witness
    );

    result->branch_count = (int)result->sampled_unique_roots;
    result->root_branches = strength_calloc(
        (size_t)result->branch_count,
        sizeof(*result->root_branches)
    );
    int root_output = 0;
    for (int first_rank = 0; first_rank < top_k; first_rank++) {
        if (!roots[first_rank].reached) continue;
        int second_rank = best_sampled_second_rank(
            pairs,
            top_k,
            first_rank
        );
        if (second_rank < 0) {
            strength_fail("reached sampled root lost its pair");
        }
        int third_rank = best_sampled_third_rank(
            reached_triples,
            leaf_rewards,
            top_k,
            first_rank,
            second_rank
        );
        if (third_rank < 0) {
            strength_fail("reached sampled root lost its continuation");
        }
        int first = logit_support_token(support, 0, first_rank);
        int second_row = 1 + first;
        int second = logit_support_token(
            support,
            second_row,
            second_rank
        );
        int third_row = 1 + vocab_size + first * vocab_size + second;
        int third = logit_support_token(
            support,
            third_row,
            third_rank
        );
        result->root_branches[root_output++] = (LogitTripleBranch){
            .triple = {
                .first = table_logit(table, 0, first),
                .second = table_logit(table, second_row, second),
                .third = table_logit(table, third_row, third),
            },
            .reward = roots[first_rank].best_reward,
            .covered_log_probability_mass =
                roots[first_rank].log_reward_sum,
            .path_count = roots[first_rank].path_count,
        };
    }
    if (root_output != result->branch_count) {
        strength_fail("sampled root reachability count changed");
    }
    qsort(
        result->root_branches,
        (size_t)result->branch_count,
        sizeof(*result->root_branches),
        compare_logit_triple_branch
    );

    result->tail_branch_count = 0;
    int selected_second_row = 1 + best_first;
    for (int second_rank = 0; second_rank < top_k; second_rank++) {
        if (pairs[best_first_rank * top_k + second_rank].reached) {
            result->tail_branch_count++;
        }
    }
    result->selected_tail_branches = strength_calloc(
        (size_t)result->tail_branch_count,
        sizeof(*result->selected_tail_branches)
    );
    int tail_output = 0;
    for (int second_rank = 0; second_rank < top_k; second_rank++) {
        int second = logit_support_token(
            support,
            selected_second_row,
            second_rank
        );
        SampledPairBackup *pair =
            &pairs[best_first_rank * top_k + second_rank];
        if (!pair->reached) continue;
        int third_rank = best_sampled_third_rank(
            reached_triples,
            leaf_rewards,
            top_k,
            best_first_rank,
            second_rank
        );
        if (third_rank < 0) {
            strength_fail("reached sampled pair lost its leaf");
        }
        int third_row = 1 + vocab_size + best_first * vocab_size + second;
        int third = logit_support_token(
            support,
            third_row,
            third_rank
        );
        result->selected_tail_branches[tail_output++] = (LogitPairBranch){
            .pair = {
                .first = table_logit(table, selected_second_row, second),
                .second = table_logit(table, third_row, third),
            },
            .reward = pair->best_reward,
            .covered_log_probability_mass = pair->log_reward_sum,
            .path_count = pair->path_count,
        };
    }
    if (tail_output != result->tail_branch_count) {
        strength_fail("sampled pair reachability count changed");
    }
    qsort(
        result->selected_tail_branches,
        (size_t)result->tail_branch_count,
        sizeof(*result->selected_tail_branches),
        compare_logit_pair_branch
    );

    free(pairs);
    free(roots);
    free(leaf_rewards);
    free(reached_triples);
}

/*
 * This is the sole evaluator for the composed finite term.  It first lowers
 * the already-formed causal company through each learned filler once, then
 * applies the two dependent strengths, and finally applies tau once.
 */
static Depth3LogitResult run_depth3_logit_term(Depth3LogitTerm *term) {
    Depth3LogitResult result = {0};
    long company_started = time_in_ms();
    result.evaluation = evaluate_depth3_company(term->program, term->shape);
    result.company_milliseconds = time_in_ms() - company_started;
    long selection_started = time_in_ms();
    LogitTable table = make_logit_table(&result.evaluation.logits);
    int vocab_size = table.company->vocab_size;
    LogitSupportCache support = make_logit_support_cache(&table, term->top_k);

    if (term->sample_milliseconds > 0) {
        result.sampled = true;
        run_sampled_logit_strength(
            &result,
            &table,
            &support,
            term->sample_milliseconds
        );
        result.selection_milliseconds = time_in_ms() - selection_started;
        free_logit_support_cache(&support);
        free_logit_table(&table);
        return result;
    }

    LogitRowSelection root_row = {
        .table = &table,
        .support = &support,
        .stats = &result.stats,
        .row = 0,
    };
    LogitTailFamily tails = make_logit_tail_family(
        &table,
        &support,
        &result.stats
    );
    finish_logit_tail_family(&tails);

    LogitTripleProduct triple_storage;
    RanConstConstSelectLogitTriple selection =
        dependent_logit_triple_strength(
            logit_row_selection(&root_row),
            tail_pair_selection,
            &tails,
            vocab_size,
            &triple_storage
        );
    LogitTauEnvironment tau_storage;
    RanConstConstContLogitTriple continuation =
        logit_selection_to_continuation(selection, &tau_storage);
    LogProbabilityObserver observer = {
        .vocab_size = vocab_size,
        .stats = &result.stats,
    };
    result.reward = continuation.apply(
        continuation.environment,
        observe_log_probability_triple,
        &observer,
        &result.witness
    );

    result.branch_count = term->top_k;
    result.tail_branch_count = term->top_k;
    result.root_branches = strength_calloc(
        (size_t)result.branch_count,
        sizeof(*result.root_branches)
    );
    for (int index = 0; index < result.branch_count; index++) {
        int first = logit_support_token(&support, 0, index);
        if (!triple_storage.remembered[first]) {
            strength_fail("root strength did not visit its top-k support");
        }
        ModelLogitPair suffix = triple_storage.remembered_suffixes[first];
        result.root_branches[index] = (LogitTripleBranch){
            .triple = {
                .first = table_logit(&table, 0, first),
                .second = suffix.first,
                .third = suffix.second,
            },
            .reward = triple_storage.remembered_rewards[first],
        };
    }
    qsort(
        result.root_branches,
        (size_t)result.branch_count,
        sizeof(*result.root_branches),
        compare_logit_triple_branch
    );

    result.selected_tail_branches = strength_calloc(
        (size_t)result.branch_count,
        sizeof(*result.selected_tail_branches)
    );
    LogitPairProduct *selected_tail =
        &tails.products[result.witness.first.token];
    int selected_second_row = 1 + result.witness.first.token;
    for (int index = 0; index < result.branch_count; index++) {
        int second = logit_support_token(
            &support,
            selected_second_row,
            index
        );
        if (!selected_tail->remembered[second]) {
            strength_fail("tail strength did not visit its top-k support");
        }
        result.selected_tail_branches[index] = (LogitPairBranch){
            .pair = {
                .first = table_logit(
                    &table,
                    1 + result.witness.first.token,
                    second
                ),
                .second = selected_tail->remembered_suffixes[second],
            },
            .reward = selected_tail->remembered_rewards[second],
        };
    }
    qsort(
        result.selected_tail_branches,
        (size_t)result.branch_count,
        sizeof(*result.selected_tail_branches),
        compare_logit_pair_branch
    );

    free_logit_triple_product(&triple_storage);
    free_logit_tail_family(&tails);
    free_logit_support_cache(&support);
    free_logit_table(&table);
    result.selection_milliseconds = time_in_ms() - selection_started;
    return result;
}

static void free_depth3_logit_result(Depth3LogitResult *result) {
    free(result->selected_tail_branches);
    free(result->root_branches);
    free(result->evaluation.logits.values);
    free_depth3_scales(&result->evaluation.scales);
    memset(result, 0, sizeof(*result));
}

/* An eager stock forward pass used only after run, as a numerical oracle. */
static Verification verify_depth3_against_llama2_forward(
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
    int expected_rows = 1 + vocab_size + vocab_size * vocab_size;
    if (company->row_count != expected_rows) {
        strength_fail("depth-three logit company has the wrong row count");
    }
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
        int second_row = 1 + first;
        compare_logit_row(
            &result,
            second_row,
            company->values + (size_t)second_row * vocab_size,
            oracle,
            vocab_size
        );
        for (int second = 0; second < vocab_size; second++) {
            oracle = forward(transformer, second, prompt_count + 1);
            int third_row = 1 + vocab_size + first * vocab_size + second;
            compare_logit_row(
                &result,
                third_row,
                company->values + (size_t)third_row * vocab_size,
                oracle,
                vocab_size
            );
        }
    }
    return result;
}

static int logit_local_rank(
    const LogitCompany *company,
    ModelLogit selected
) {
    const float *row = company->values +
        (size_t)selected.row * company->vocab_size;
    int rank = 1;
    for (int token = 0; token < company->vocab_size; token++) {
        if (row[token] > selected.value ||
            (row[token] == selected.value && token < selected.token)) {
            rank++;
        }
    }
    return rank;
}

static void print_logit_completion(
    Tokenizer *tokenizer,
    int prompt_last_token,
    ModelLogitTriple triple
) {
    print_escaped(decode(tokenizer, prompt_last_token, triple.first.token));
    print_escaped(decode(tokenizer, triple.first.token, triple.second.token));
    print_escaped(decode(tokenizer, triple.second.token, triple.third.token));
}

static void print_logit_triple_text(
    Tokenizer *tokenizer,
    int prompt_last_token,
    ModelLogitTriple triple
) {
    putchar('"');
    print_logit_completion(tokenizer, prompt_last_token, triple);
    putchar('"');
}

static void print_selected_logit(
    const char *position,
    Tokenizer *tokenizer,
    int previous_token,
    const LogitCompany *company,
    ModelLogit logit
) {
    printf(
        "  position=%s row=%d token=%d piece=",
        position,
        logit.row,
        logit.token
    );
    print_piece(tokenizer, previous_token, logit.token);
    printf(
        " raw_logit=%.9g log_probability=%.17g local_rank=%d\n",
        logit.value,
        logit.log_probability,
        logit_local_rank(company, logit)
    );
}

static void logit_strength_usage(const char *program) {
    fprintf(
        stderr,
        "usage: %s CHECKPOINT TOKENIZER PROMPT [-k TOP_K] "
        "[--sample-ms MILLISECONDS] [--verify] [--trace-fillers]\n",
        program
    );
    exit(EXIT_FAILURE);
}

static int parse_logit_strength_integer(
    const char *text,
    const char *option
) {
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

int main(int argc, char **argv) {
    if (argc < 4) logit_strength_usage(argv[0]);
    bool verify = false;
    bool trace_fillers = false;
    int top_k = -1;
    int sample_milliseconds = 0;
    for (int argument = 4; argument < argc; argument++) {
        if (strcmp(argv[argument], "--verify") == 0) {
            verify = true;
        } else if (strcmp(argv[argument], "--trace-fillers") == 0) {
            trace_fillers = true;
        } else if (strcmp(argv[argument], "-k") == 0 ||
                   strcmp(argv[argument], "--sample-ms") == 0) {
            const char *option = argv[argument];
            if (++argument >= argc) logit_strength_usage(argv[0]);
            int value = parse_logit_strength_integer(
                argv[argument],
                option
            );
            if (strcmp(option, "-k") == 0) {
                top_k = value;
            } else {
                sample_milliseconds = value;
            }
        } else {
            logit_strength_usage(argv[0]);
        }
    }

    setvbuf(stdout, NULL, _IOLBF, 0);
    Transformer transformer;
    build_transformer(&transformer, argv[1]);
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, argv[2], transformer.config.vocab_size);
    if (transformer.config.vocab_size > 512) {
        strength_fail(
            "complete depth-three experiment is bounded to vocab <= 512"
        );
    }
    if (top_k < 0) top_k = transformer.config.vocab_size;
    if (top_k == 0) strength_fail("top-k must be positive");
    if (top_k > transformer.config.vocab_size) {
        strength_fail("top-k exceeds the model vocabulary");
    }

    int *prompt_tokens = strength_calloc(
        strlen(argv[3]) + 3,
        sizeof(*prompt_tokens)
    );
    int prompt_count = 0;
    encode(&tokenizer, argv[3], 1, 0, prompt_tokens, &prompt_count);
    if (prompt_count < 1) strength_fail("prompt encoded to no tokens");
    if (prompt_count + 1 >= transformer.config.seq_len) {
        strength_fail("prompt is too long for a three-token completion");
    }

    Depth3Shape shape = make_depth3_shape(
        prompt_tokens,
        prompt_count,
        transformer.config.vocab_size
    );
    FillerLedger ledger = new_ledger(&transformer.config);
    Program program = {
        .transformer = &transformer,
        .prompt_tokens = prompt_tokens,
        .prompt_count = prompt_count,
        .ledger = &ledger,
        .trace_fillers = trace_fillers,
    };
    Depth3LogitTerm term = {
        .program = &program,
        .shape = &shape,
        .top_k = top_k,
        .sample_milliseconds = sample_milliseconds,
    };

    if (ledger_total_reads(&ledger) != 0) {
        strength_fail("a learned filler ran before the sole run");
    }
    uint64_t reached_paths = checked_u64_product(
        checked_u64_product(
            (uint64_t)transformer.config.vocab_size,
            (uint64_t)transformer.config.vocab_size
        ),
        (uint64_t)transformer.config.vocab_size
    );
    uint64_t support_paths = checked_u64_product(
        checked_u64_product((uint64_t)top_k, (uint64_t)top_k),
        (uint64_t)top_k
    );
    printf(
        "run_begin company_nodes=%d output_rows=%d reached_paths=%" PRIu64
        " top_k=%d support_paths=%" PRIu64
        " sample_ms=%d learned_reads_before_run=0\n",
        shape.node_count,
        shape.output_count,
        reached_paths,
        top_k,
        support_paths,
        sample_milliseconds
    );
    long start = time_in_ms();
    Depth3LogitResult result = run_depth3_logit_term(&term);
    long end = time_in_ms();
    verify_one_shot_ledger(&ledger);

    int prompt_last = prompt_tokens[prompt_count - 1];
    printf("prompt: %s\n", argv[3]);
    printf("prompt_token_count=%d\n", prompt_count);
    printf("completion: ");
    print_logit_completion(&tokenizer, prompt_last, result.witness);
    putchar('\n');
    printf(
        "selected_tokens=[%d,%d,%d]\nselected_reward=%.17g\n",
        result.witness.first.token,
        result.witness.second.token,
        result.witness.third.token,
        result.reward
    );
    puts("selection_carrier=ModelLogit(row,token,value,log_probability)");
    if (result.sampled) {
        puts("selection_mode=sampled_reachability_backward_induction");
        puts("selection_term=sampled_dependent_product_3(Select_Reward_Logit)");
        puts("selection_backup=max_complete_observer_reward");
        puts("coverage_diagnostic=logsumexp_of_distinct_reached_path_probabilities");
    } else {
        puts("selection_mode=exact_top_k_backward_induction");
        puts("selection_term=dependent_product_3(Select_Reward_Logit)");
    }
    puts("observation_term=Cont_Reward_LogitTriple");
    puts("terminal_observer=sum_log_probabilities_diagnostic");
    puts(result.sampled ?
        "terminalization=tau_on_finite_reached_support" :
        "terminalization=tau_once");
    printf(
        "company_nodes=%d output_logit_rows=%d retained_layer_scales=%d "
        "retained_rows_per_scale=%d\n",
        shape.node_count,
        shape.output_count,
        result.evaluation.scales.count,
        result.evaluation.scales.row_count
    );
    printf(
        "row_selection_applications=%" PRIu64
        " candidate_observations=%" PRIu64
        " terminal_observations=%" PRIu64 "\n",
        result.stats.row_selection_applications,
        result.stats.candidate_observations,
        result.stats.terminal_observations
    );
    printf(
        "top_k=%d support_paths=%" PRIu64
        " company_milliseconds=%ld selection_milliseconds=%ld\n",
        top_k,
        support_paths,
        result.company_milliseconds,
        result.selection_milliseconds
    );
    if (result.sampled) {
        printf(
            "sampling_budget_ms=%d sampling_seed=%" PRIu64
            " sampling_stop_reason=%s\n",
            sample_milliseconds,
            UINT64_C(0x4d595df4d0f33173),
            result.sampled_support_exhausted ?
                "support_exhausted" : "deadline"
        );
        printf(
            "sampled_attempts=%" PRIu64
            " sampled_logit_draws=%" PRIu64
            " sampled_unique_roots=%" PRIu64
            " sampled_unique_pairs=%" PRIu64
            " sampled_unique_triples=%" PRIu64 "\n",
            result.sampled_attempts,
            result.sampled_logit_draws,
            result.sampled_unique_roots,
            result.sampled_unique_pairs,
            result.sampled_unique_triples
        );
        printf(
            "selected_backup leaf_log_reward=%.17g "
            "pair_best_reward=%.17g "
            "pair_covered_log_probability_mass=%.17g "
            "pair_paths=%" PRIu64 " root_best_reward=%.17g "
            "root_covered_log_probability_mass=%.17g "
            "root_paths=%" PRIu64 "\n",
            result.reward,
            result.selected_pair_backed_reward,
            result.selected_pair_covered_log_probability_mass,
            result.selected_pair_path_count,
            result.backed_reward,
            result.covered_log_probability_mass,
            result.selected_root_path_count
        );
    }
    printf("elapsed_seconds=%.6f\n", (end - start) / 1000.0);

    puts("selected_logits:");
    print_selected_logit(
        "first",
        &tokenizer,
        prompt_last,
        &result.evaluation.logits,
        result.witness.first
    );
    print_selected_logit(
        "second",
        &tokenizer,
        result.witness.first.token,
        &result.evaluation.logits,
        result.witness.second
    );
    print_selected_logit(
        "third",
        &tokenizer,
        result.witness.second.token,
        &result.evaluation.logits,
        result.witness.third
    );

    puts("backward_induction_root_top5:");
    int shown = result.branch_count < 5 ? result.branch_count : 5;
    for (int rank = 0; rank < shown; rank++) {
        LogitTripleBranch branch = result.root_branches[rank];
        printf(
            "  rank=%d tokens=[%d,%d,%d] text=",
            rank + 1,
            branch.triple.first.token,
            branch.triple.second.token,
            branch.triple.third.token
        );
        print_logit_triple_text(&tokenizer, prompt_last, branch.triple);
        if (result.sampled) {
            printf(
                " backed_best_reward=%.17g "
                "covered_log_probability_mass=%.17g "
                "reached_paths=%" PRIu64 "\n",
                branch.reward,
                branch.covered_log_probability_mass,
                branch.path_count
            );
        } else {
            printf(" backed_up_reward=%.17g\n", branch.reward);
        }
    }

    puts("selected_first_tail_top5:");
    int tail_shown = result.tail_branch_count < 5 ?
        result.tail_branch_count : 5;
    for (int rank = 0; rank < tail_shown; rank++) {
        LogitPairBranch branch = result.selected_tail_branches[rank];
        ModelLogitTriple triple = {
            .first = result.witness.first,
            .second = branch.pair.first,
            .third = branch.pair.second,
        };
        printf(
            "  rank=%d tokens=[%d,%d,%d] text=",
            rank + 1,
            triple.first.token,
            triple.second.token,
            triple.third.token
        );
        print_logit_triple_text(&tokenizer, prompt_last, triple);
        if (result.sampled) {
            printf(
                " backed_best_reward=%.17g "
                "covered_log_probability_mass=%.17g "
                "reached_paths=%" PRIu64 "\n",
                branch.reward,
                branch.covered_log_probability_mass,
                branch.path_count
            );
        } else {
            printf(" backed_up_reward=%.17g\n", branch.reward);
        }
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

    int exit_status = EXIT_SUCCESS;
    if (verify) {
        puts("llama2_oracle_verification=begin");
        Verification verification = verify_depth3_against_llama2_forward(
            &transformer,
            prompt_tokens,
            prompt_count,
            &result.evaluation.logits
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
            exit_status = EXIT_FAILURE;
        }
    }

    free_depth3_logit_result(&result);
    free(ledger.fillers);
    free_depth3_shape(&shape);
    free(prompt_tokens);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return exit_status;
}
