#define CANDIDATE_PROBE_LIBRARY
#include "candidate_probe.c"

#include <stdbool.h>
#include <limits.h>

typedef struct {
    int positions;
    int heads;
    double attention_entropy;
    double attention_max_mass;
    double attention_same_token_mass;
    double attention_prompt_mass;
    double attention_distance;
    double attention_update_ratio;
    double attention_update_alignment;
    double ffn_update_ratio;
    double ffn_update_alignment;
    double residual_inertia;
    double prior_state_similarity;
    double same_token_state_similarity;
    int same_token_states;
} LayerCompany;

typedef struct {
    int prompt_tokens;
    const int *tokens;
    float *hidden;
    LayerCompany *layers;
    LayerCompany *position_layers;
} CompanyTrace;

typedef struct {
    const char *prompt;
    const char *completion;
    char *owned_prompt;
    char *owned_completion;
    const char *trace_path;
    int checkpoint_every;
    int count_only;
} CompanyOptions;

typedef struct {
    int count;
    int dim;
    double *mean;
    double *scatter;
    double *first;
    double *previous;
    double path_length;
} AffineAccumulator;

static double vector_norm(const float *values, int width) {
    double square_sum = 0.0;
    for (int index = 0; index < width; index++) {
        square_sum += (double)values[index] * values[index];
    }
    return sqrt(square_sum);
}

static double vector_cosine(
    const float *left,
    const float *right,
    int width
) {
    double dot = 0.0;
    double left_square = 0.0;
    double right_square = 0.0;
    for (int index = 0; index < width; index++) {
        dot += (double)left[index] * right[index];
        left_square += (double)left[index] * left[index];
        right_square += (double)right[index] * right[index];
    }
    if (left_square == 0.0 || right_square == 0.0) return 0.0;
    return dot / sqrt(left_square * right_square);
}

static void accumulate_update(
    const float *boundary,
    const float *update,
    int width,
    double *ratio,
    double *alignment
) {
    double boundary_norm = vector_norm(boundary, width);
    double update_norm = vector_norm(update, width);
    *ratio += boundary_norm == 0.0 ? 0.0 : update_norm / boundary_norm;
    *alignment += vector_cosine(boundary, update, width);
}

/* This is llama2.c's numerical path with observations retained at the actual
 * affine residual and attention boundaries. It does not alter a kernel or
 * insert a linguistic span boundary. */
static float *company_forward(
    Transformer *transformer,
    int token,
    int position,
    CompanyTrace *trace
) {
    Config *config = &transformer->config;
    TransformerWeights *weights = &transformer->weights;
    RunState *state = &transformer->state;
    int dim = config->dim;
    int kv_dim = dim * config->n_kv_heads / config->n_heads;
    int kv_mul = config->n_heads / config->n_kv_heads;
    int head_size = dim / config->n_heads;
    bool retained = position >= trace->prompt_tokens;

    memcpy(
        state->x,
        weights->token_embedding_table + (size_t)token * dim,
        (size_t)dim * sizeof(*state->x)
    );

    for (int layer = 0; layer < config->n_layers; layer++) {
        LayerCompany *company = &trace->layers[layer];
        LayerCompany *position_company = retained ?
            trace->position_layers +
                (size_t)layer * config->seq_len + position :
            NULL;
        float layer_input[dim];
        memcpy(layer_input, state->x, sizeof(layer_input));

        rmsnorm(
            state->xb,
            state->x,
            weights->rms_att_weight + (size_t)layer * dim,
            dim
        );
        size_t layer_offset =
            (size_t)layer * config->seq_len * kv_dim;
        state->k = state->key_cache + layer_offset +
            (size_t)position * kv_dim;
        state->v = state->value_cache + layer_offset +
            (size_t)position * kv_dim;
        matmul(
            state->q,
            state->xb,
            weights->wq + (size_t)layer * dim * dim,
            dim,
            dim
        );
        matmul(
            state->k,
            state->xb,
            weights->wk + (size_t)layer * dim * kv_dim,
            dim,
            kv_dim
        );
        matmul(
            state->v,
            state->xb,
            weights->wv + (size_t)layer * dim * kv_dim,
            dim,
            kv_dim
        );

        for (int index = 0; index < dim; index += 2) {
            int head_dimension = index % head_size;
            float frequency = 1.0f /
                powf(10000.0f, head_dimension / (float)head_size);
            float angle = position * frequency;
            float real = cosf(angle);
            float imaginary = sinf(angle);
            int rotations = index < kv_dim ? 2 : 1;
            for (int vector = 0; vector < rotations; vector++) {
                float *values = vector == 0 ? state->q : state->k;
                float first = values[index];
                float second = values[index + 1];
                values[index] = first * real - second * imaginary;
                values[index + 1] = first * imaginary + second * real;
            }
        }

        for (int head = 0; head < config->n_heads; head++) {
            float *query = state->q + head * head_size;
            float *attention = state->att +
                (size_t)head * config->seq_len;
            for (int timestep = 0; timestep <= position; timestep++) {
                float *key = state->key_cache + layer_offset +
                    (size_t)timestep * kv_dim +
                    (head / kv_mul) * head_size;
                float score = 0.0f;
                for (int lane = 0; lane < head_size; lane++) {
                    score += query[lane] * key[lane];
                }
                attention[timestep] = score / sqrtf(head_size);
            }
            softmax(attention, position + 1);

            if (retained) {
                double entropy = 0.0;
                double maximum = 0.0;
                double same_token_mass = 0.0;
                double prompt_mass = 0.0;
                double distance = 0.0;
                for (int timestep = 0; timestep <= position; timestep++) {
                    double mass = attention[timestep];
                    if (mass > 0.0) entropy -= mass * log(mass);
                    if (mass > maximum) maximum = mass;
                    if (trace->tokens[timestep] == token) {
                        same_token_mass += mass;
                    }
                    if (timestep < trace->prompt_tokens) prompt_mass += mass;
                    distance += mass * (position - timestep);
                }
                company->attention_entropy += entropy;
                company->attention_max_mass += maximum;
                company->attention_same_token_mass += same_token_mass;
                company->attention_prompt_mass += prompt_mass;
                company->attention_distance += distance;
                company->heads++;
                position_company->attention_entropy += entropy;
                position_company->attention_max_mass += maximum;
                position_company->attention_same_token_mass +=
                    same_token_mass;
                position_company->attention_prompt_mass += prompt_mass;
                position_company->attention_distance += distance;
                position_company->heads++;
            }

            float *head_output = state->xb + head * head_size;
            memset(head_output, 0, (size_t)head_size * sizeof(*head_output));
            for (int timestep = 0; timestep <= position; timestep++) {
                float *value = state->value_cache + layer_offset +
                    (size_t)timestep * kv_dim +
                    (head / kv_mul) * head_size;
                float mass = attention[timestep];
                for (int lane = 0; lane < head_size; lane++) {
                    head_output[lane] += mass * value[lane];
                }
            }
        }

        matmul(
            state->xb2,
            state->xb,
            weights->wo + (size_t)layer * dim * dim,
            dim,
            dim
        );
        if (retained) {
            accumulate_update(
                layer_input,
                state->xb2,
                dim,
                &company->attention_update_ratio,
                &company->attention_update_alignment
            );
            accumulate_update(
                layer_input,
                state->xb2,
                dim,
                &position_company->attention_update_ratio,
                &position_company->attention_update_alignment
            );
        }
        for (int lane = 0; lane < dim; lane++) {
            state->x[lane] += state->xb2[lane];
        }

        float ffn_boundary[dim];
        memcpy(ffn_boundary, state->x, sizeof(ffn_boundary));
        rmsnorm(
            state->xb,
            state->x,
            weights->rms_ffn_weight + (size_t)layer * dim,
            dim
        );
        matmul(
            state->hb,
            state->xb,
            weights->w1 + (size_t)layer * dim * config->hidden_dim,
            dim,
            config->hidden_dim
        );
        matmul(
            state->hb2,
            state->xb,
            weights->w3 + (size_t)layer * dim * config->hidden_dim,
            dim,
            config->hidden_dim
        );
        for (int lane = 0; lane < config->hidden_dim; lane++) {
            float value = state->hb[lane];
            value *= 1.0f / (1.0f + expf(-value));
            state->hb[lane] = value * state->hb2[lane];
        }
        matmul(
            state->xb,
            state->hb,
            weights->w2 + (size_t)layer * dim * config->hidden_dim,
            config->hidden_dim,
            dim
        );
        if (retained) {
            accumulate_update(
                ffn_boundary,
                state->xb,
                dim,
                &company->ffn_update_ratio,
                &company->ffn_update_alignment
            );
            accumulate_update(
                ffn_boundary,
                state->xb,
                dim,
                &position_company->ffn_update_ratio,
                &position_company->ffn_update_alignment
            );
        }
        for (int lane = 0; lane < dim; lane++) {
            state->x[lane] += state->xb[lane];
        }

        float *stored = trace->hidden +
            ((size_t)layer * config->seq_len + position) * dim;
        memcpy(stored, state->x, (size_t)dim * sizeof(*stored));
        if (retained) {
            company->positions++;
            position_company->positions++;
            company->residual_inertia += vector_cosine(
                layer_input,
                state->x,
                dim
            );
            position_company->residual_inertia += vector_cosine(
                layer_input,
                state->x,
                dim
            );
            double most_similar = -1.0;
            double most_similar_same_token = -1.0;
            bool found_same_token = false;
            for (int timestep = 0; timestep < position; timestep++) {
                float *prior = trace->hidden +
                    ((size_t)layer * config->seq_len + timestep) * dim;
                double similarity = vector_cosine(state->x, prior, dim);
                if (similarity > most_similar) most_similar = similarity;
                if (trace->tokens[timestep] == token &&
                    similarity > most_similar_same_token) {
                    most_similar_same_token = similarity;
                    found_same_token = true;
                }
            }
            if (most_similar > -1.0) {
                company->prior_state_similarity += most_similar;
                position_company->prior_state_similarity += most_similar;
            }
            if (found_same_token) {
                company->same_token_state_similarity +=
                    most_similar_same_token;
                company->same_token_states++;
                position_company->same_token_state_similarity +=
                    most_similar_same_token;
                position_company->same_token_states++;
            }
        }
    }

    rmsnorm(
        state->x,
        state->x,
        weights->rms_final_weight,
        config->dim
    );
    matmul(
        state->logits,
        state->x,
        weights->wcls,
        config->dim,
        config->vocab_size
    );
    return state->logits;
}

static void company_usage(const char *program) {
    fprintf(
        stderr,
        "usage:\n"
        "  %s CHECKPOINT TOKENIZER PROMPT COMPLETION [OPTIONS]\n"
        "  %s CHECKPOINT TOKENIZER --files PROMPT_FILE COMPLETION_FILE "
        "[OPTIONS]\n"
        "options:\n"
        "  --trace PATH              flushed JSONL token/layer trace\n"
        "  --checkpoint-every N      affine prefix interval (default 32)\n"
        "  --count-only              print token counts without inference\n",
        program,
        program
    );
    exit(EXIT_FAILURE);
}

static char *read_text_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "could not open text file %s\n", path);
        exit(EXIT_FAILURE);
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fprintf(stderr, "could not seek text file %s\n", path);
        exit(EXIT_FAILURE);
    }
    long length = ftell(file);
    if (length < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "could not measure text file %s\n", path);
        exit(EXIT_FAILURE);
    }
    char *text = malloc((size_t)length + 1);
    if (text == NULL) {
        fprintf(stderr, "could not allocate text file %s\n", path);
        exit(EXIT_FAILURE);
    }
    size_t read = fread(text, 1, (size_t)length, file);
    if (read != (size_t)length || fclose(file) != 0) {
        fprintf(stderr, "could not read text file %s\n", path);
        exit(EXIT_FAILURE);
    }
    text[length] = '\0';
    return text;
}

static CompanyOptions parse_company_options(int argc, char **argv) {
    if (argc < 5) company_usage(argv[0]);
    CompanyOptions options = {.checkpoint_every = 32};
    int index = 3;
    if (strcmp(argv[index], "--files") == 0) {
        if (argc < 6) company_usage(argv[0]);
        options.owned_prompt = read_text_file(argv[index + 1]);
        options.owned_completion = read_text_file(argv[index + 2]);
        options.prompt = options.owned_prompt;
        options.completion = options.owned_completion;
        index += 3;
    } else {
        if (argc < 5) company_usage(argv[0]);
        options.prompt = argv[index];
        options.completion = argv[index + 1];
        index += 2;
    }
    while (index < argc) {
        if (strcmp(argv[index], "--trace") == 0 && index + 1 < argc) {
            options.trace_path = argv[index + 1];
            index += 2;
        } else if (strcmp(argv[index], "--checkpoint-every") == 0 &&
                   index + 1 < argc) {
            long interval = parse_long(argv[index + 1], "checkpoint interval");
            if (interval <= 0 || interval > INT_MAX) {
                fprintf(stderr, "checkpoint interval must be positive\n");
                exit(EXIT_FAILURE);
            }
            options.checkpoint_every = (int)interval;
            index += 2;
        } else if (strcmp(argv[index], "--count-only") == 0) {
            options.count_only = 1;
            index++;
        } else {
            company_usage(argv[0]);
        }
    }
    return options;
}

static void fprint_json_string(FILE *file, const char *text) {
    fputc('"', file);
    for (const unsigned char *cursor = (const unsigned char *)text;
         *cursor != '\0'; cursor++) {
        unsigned char byte = *cursor;
        if (byte == '"' || byte == '\\') {
            fputc('\\', file);
            fputc(byte, file);
        } else if (byte == '\n') {
            fputs("\\n", file);
        } else if (byte == '\r') {
            fputs("\\r", file);
        } else if (byte == '\t') {
            fputs("\\t", file);
        } else if (byte >= 0x20) {
            fputc(byte, file);
        } else {
            fprintf(file, "\\u%04x", byte);
        }
    }
    fputc('"', file);
}

static double distribution_entropy(const float *logits, int count) {
    double maximum = -DBL_MAX;
    for (int index = 0; index < count; index++) {
        if ((double)logits[index] > maximum) maximum = logits[index];
    }
    double normalizer = 0.0;
    for (int index = 0; index < count; index++) {
        normalizer += exp((double)logits[index] - maximum);
    }
    double entropy = 0.0;
    for (int index = 0; index < count; index++) {
        double probability = exp((double)logits[index] - maximum) /
            normalizer;
        if (probability > 0.0) entropy -= probability * log(probability);
    }
    return entropy;
}

static int maximum_logit_token(const float *logits, int count) {
    int best = 0;
    for (int token = 1; token < count; token++) {
        if (logits[token] > logits[best]) best = token;
    }
    return best;
}

static void write_token_trace(
    FILE *file,
    Tokenizer *tokenizer,
    const float *logits,
    int vocab_size,
    int previous_token,
    int token,
    int completion_index,
    int context_position,
    double cumulative_log_probability
) {
    double log_probability = token_log_probability(logits, vocab_size, token);
    double entropy = distribution_entropy(logits, vocab_size);
    int top = maximum_logit_token(logits, vocab_size);
    fprintf(
        file,
        "{\"kind\":\"token\",\"completion_index\":%d,"
        "\"context_position\":%d,\"token_position\":%d,"
        "\"token_id\":%d,\"piece\":",
        completion_index,
        context_position,
        context_position + 1,
        token
    );
    fprint_json_string(file, decode(tokenizer, previous_token, token));
    fprintf(
        file,
        ",\"log_probability\":%.17g,\"local_rank\":%d,"
        "\"entropy\":%.17g,\"surprisal_minus_entropy\":%.17g,"
        "\"cumulative_log_probability\":%.17g,"
        "\"mean_log_probability\":%.17g,\"top_token_id\":%d,"
        "\"top_piece\":",
        log_probability,
        local_rank(logits, vocab_size, token),
        entropy,
        -log_probability - entropy,
        cumulative_log_probability,
        cumulative_log_probability / (completion_index + 1),
        top
    );
    fprint_json_string(file, decode(tokenizer, previous_token, top));
    fprintf(
        file,
        ",\"top_log_probability\":%.17g}\n",
        token_log_probability(logits, vocab_size, top)
    );
    fflush(file);
}

static void write_layer_trace(
    FILE *file,
    const LayerCompany *company,
    int layer,
    int completion_index,
    int position
) {
    double heads = company->heads;
    double positions = company->positions;
    double entropy = company->attention_entropy / heads;
    double entropy_limit = position > 0 ? log((double)position + 1.0) : 0.0;
    fprintf(
        file,
        "{\"kind\":\"layer\",\"completion_index\":%d,"
        "\"token_position\":%d,\"layer\":%d,"
        "\"attention_entropy\":%.17g,"
        "\"attention_entropy_fraction\":%.17g,"
        "\"attention_max_mass\":%.17g,"
        "\"attention_same_token_mass\":%.17g,"
        "\"attention_prompt_mass\":%.17g,"
        "\"attention_distance\":%.17g,"
        "\"attention_update_ratio\":%.17g,"
        "\"attention_update_alignment\":%.17g,"
        "\"ffn_update_ratio\":%.17g,"
        "\"ffn_update_alignment\":%.17g,"
        "\"residual_inertia\":%.17g,"
        "\"prior_state_similarity\":%.17g,"
        "\"same_token_state_similarity\":",
        completion_index,
        position,
        layer,
        entropy,
        entropy_limit == 0.0 ? 0.0 : entropy / entropy_limit,
        company->attention_max_mass / heads,
        company->attention_same_token_mass / heads,
        company->attention_prompt_mass / heads,
        company->attention_distance / heads,
        company->attention_update_ratio / positions,
        company->attention_update_alignment / positions,
        company->ffn_update_ratio / positions,
        company->ffn_update_alignment / positions,
        company->residual_inertia / positions,
        company->prior_state_similarity / positions
    );
    if (company->same_token_states == 0) {
        fputs("null", file);
    } else {
        fprintf(
            file,
            "%.17g",
            company->same_token_state_similarity /
                company->same_token_states
        );
    }
    fputs("}\n", file);
    fflush(file);
}

static void affine_accumulator_init(AffineAccumulator *accumulator, int dim) {
    accumulator->dim = dim;
    accumulator->mean = calloc((size_t)dim, sizeof(*accumulator->mean));
    accumulator->scatter = calloc(
        (size_t)dim * dim,
        sizeof(*accumulator->scatter)
    );
    accumulator->first = calloc((size_t)dim, sizeof(*accumulator->first));
    accumulator->previous = calloc(
        (size_t)dim,
        sizeof(*accumulator->previous)
    );
    if (accumulator->mean == NULL || accumulator->scatter == NULL ||
        accumulator->first == NULL || accumulator->previous == NULL) {
        fprintf(stderr, "could not allocate affine accumulator\n");
        exit(EXIT_FAILURE);
    }
}

static void affine_accumulator_free(AffineAccumulator *accumulator) {
    free(accumulator->previous);
    free(accumulator->first);
    free(accumulator->scatter);
    free(accumulator->mean);
}

static void affine_accumulator_add(
    AffineAccumulator *accumulator,
    const float *state
) {
    int dim = accumulator->dim;
    if (accumulator->count == 0) {
        for (int lane = 0; lane < dim; lane++) {
            accumulator->first[lane] = state[lane];
            accumulator->previous[lane] = state[lane];
        }
    } else {
        double step_square = 0.0;
        for (int lane = 0; lane < dim; lane++) {
            double difference = state[lane] - accumulator->previous[lane];
            step_square += difference * difference;
            accumulator->previous[lane] = state[lane];
        }
        accumulator->path_length += sqrt(step_square);
    }

    double delta[dim];
    int new_count = accumulator->count + 1;
    for (int lane = 0; lane < dim; lane++) {
        delta[lane] = state[lane] - accumulator->mean[lane];
        accumulator->mean[lane] += delta[lane] / new_count;
    }
    for (int row = 0; row < dim; row++) {
        for (int column = 0; column < dim; column++) {
            double delta_after = state[column] - accumulator->mean[column];
            accumulator->scatter[(size_t)row * dim + column] +=
                delta[row] * delta_after;
        }
    }
    accumulator->count = new_count;
}

static void affine_accumulator_values(
    const AffineAccumulator *accumulator,
    double *effective_dimension,
    double *variance,
    double *path_efficiency
) {
    int dim = accumulator->dim;
    double covariance_trace = 0.0;
    double covariance_square_trace = 0.0;
    for (int row = 0; row < dim; row++) {
        covariance_trace +=
            accumulator->scatter[(size_t)row * dim + row];
        for (int column = 0; column < dim; column++) {
            double entry = accumulator->scatter[
                (size_t)row * dim + column
            ] / accumulator->count;
            double transposed = accumulator->scatter[
                (size_t)column * dim + row
            ] / accumulator->count;
            covariance_square_trace += entry * transposed;
        }
    }
    covariance_trace /= accumulator->count;
    *variance = covariance_trace;
    *effective_dimension = covariance_square_trace == 0.0 ? 0.0 :
        covariance_trace * covariance_trace / covariance_square_trace;
    double displacement_square = 0.0;
    for (int lane = 0; lane < dim; lane++) {
        double difference = accumulator->previous[lane] -
            accumulator->first[lane];
        displacement_square += difference * difference;
    }
    *path_efficiency = accumulator->path_length == 0.0 ? 0.0 :
        sqrt(displacement_square) / accumulator->path_length;
}

static void print_affine_company(
    const CompanyTrace *trace,
    const Config *config,
    int total_tokens
) {
    int first = trace->prompt_tokens;
    int dim = config->dim;
    for (int layer = 0; layer < config->n_layers; layer++) {
        const float *states = trace->hidden +
            (size_t)layer * config->seq_len * dim;
        AffineAccumulator accumulator = {0};
        affine_accumulator_init(&accumulator, dim);
        for (int position = first; position < total_tokens; position++) {
            const float *state = states + (size_t)position * dim;
            affine_accumulator_add(&accumulator, state);
        }
        double effective_dimension;
        double covariance_trace;
        double path_efficiency;
        affine_accumulator_values(
            &accumulator,
            &effective_dimension,
            &covariance_trace,
            &path_efficiency
        );
        printf(
            "  layer=%d affine_effective_dimension=%.9g "
            "affine_variance=%.9g path_efficiency=%.9g\n",
            layer,
            effective_dimension,
            covariance_trace,
            path_efficiency
        );
        affine_accumulator_free(&accumulator);
    }
}

static void write_affine_checkpoints(
    FILE *file,
    const CompanyTrace *trace,
    const Config *config,
    int total_tokens,
    int checkpoint_every
) {
    int first = trace->prompt_tokens;
    int dim = config->dim;
    AffineAccumulator *accumulators = calloc(
        (size_t)config->n_layers,
        sizeof(*accumulators)
    );
    if (accumulators == NULL) {
        fprintf(stderr, "could not allocate affine checkpoints\n");
        exit(EXIT_FAILURE);
    }
    for (int layer = 0; layer < config->n_layers; layer++) {
        affine_accumulator_init(&accumulators[layer], dim);
    }
    for (int position = first; position < total_tokens; position++) {
        int prefix_tokens = position - first + 1;
        for (int layer = 0; layer < config->n_layers; layer++) {
            const float *state = trace->hidden +
                ((size_t)layer * config->seq_len + position) * dim;
            affine_accumulator_add(&accumulators[layer], state);
        }
        if (prefix_tokens % checkpoint_every != 0 &&
            position + 1 != total_tokens) {
            continue;
        }
        for (int layer = 0; layer < config->n_layers; layer++) {
            double effective_dimension;
            double variance;
            double path_efficiency;
            affine_accumulator_values(
                &accumulators[layer],
                &effective_dimension,
                &variance,
                &path_efficiency
            );
            fprintf(
                file,
                "{\"kind\":\"affine_prefix\","
                "\"completion_tokens\":%d,\"through_position\":%d,"
                "\"layer\":%d,\"effective_dimension\":%.17g,"
                "\"variance\":%.17g,\"path_efficiency\":%.17g}\n",
                prefix_tokens,
                position,
                layer,
                effective_dimension,
                variance,
                path_efficiency
            );
            fflush(file);
        }
    }
    for (int layer = 0; layer < config->n_layers; layer++) {
        affine_accumulator_free(&accumulators[layer]);
    }
    free(accumulators);
}

int main(int argc, char **argv) {
    CompanyOptions options = parse_company_options(argc, argv);

    Transformer transformer;
    build_transformer(&transformer, argv[1]);
    Tokenizer tokenizer;
    build_tokenizer(
        &tokenizer,
        argv[2],
        transformer.config.vocab_size
    );

    const char *prompt = options.prompt;
    const char *completion = options.completion;
    size_t prompt_bytes = strlen(prompt);
    size_t completion_bytes = strlen(completion);
    char *text = malloc(prompt_bytes + completion_bytes + 1);
    int *prompt_tokens = malloc((prompt_bytes + 3) * sizeof(*prompt_tokens));
    int *tokens = malloc(
        (prompt_bytes + completion_bytes + 3) * sizeof(*tokens)
    );
    if (text == NULL || prompt_tokens == NULL || tokens == NULL) {
        fprintf(stderr, "could not allocate encoded company\n");
        exit(EXIT_FAILURE);
    }
    memcpy(text, prompt, prompt_bytes);
    memcpy(text + prompt_bytes, completion, completion_bytes + 1);
    int prompt_count = 0;
    int count = 0;
    encode(&tokenizer, (char *)prompt, 1, 0, prompt_tokens, &prompt_count);
    encode(&tokenizer, text, 1, 0, tokens, &count);
    if (prompt_count <= 0 || count <= prompt_count) {
        fprintf(stderr, "company needs a nonempty prompt and completion\n");
        exit(EXIT_FAILURE);
    }
    for (int index = 0; index < prompt_count; index++) {
        if (prompt_tokens[index] != tokens[index]) {
            fprintf(stderr, "prompt is not a token prefix of company text\n");
            exit(EXIT_FAILURE);
        }
    }
    int completion_count = count - prompt_count;
    if (options.count_only) {
        printf(
            "prompt_tokens=%d completion_tokens=%d total_tokens=%d "
            "max_context_tokens=%d fits=%d\n",
            prompt_count,
            completion_count,
            count,
            transformer.config.seq_len,
            count <= transformer.config.seq_len
        );
        free(tokens);
        free(prompt_tokens);
        free(text);
        free(options.owned_completion);
        free(options.owned_prompt);
        free_tokenizer(&tokenizer);
        free_transformer(&transformer);
        return 0;
    }
    if (count > transformer.config.seq_len) {
        fprintf(
            stderr,
            "company has %d tokens but model context holds %d\n",
            count,
            transformer.config.seq_len
        );
        exit(EXIT_FAILURE);
    }

    FILE *trace_file = NULL;
    if (options.trace_path != NULL) {
        trace_file = fopen(options.trace_path, "w");
        if (trace_file == NULL) {
            fprintf(stderr, "could not open trace %s\n", options.trace_path);
            exit(EXIT_FAILURE);
        }
        setvbuf(trace_file, NULL, _IOLBF, 0);
        fprintf(
            trace_file,
            "{\"kind\":\"meta\",\"prompt_tokens\":%d,"
            "\"completion_tokens\":%d,\"total_tokens\":%d,"
            "\"max_context_tokens\":%d,\"layers\":%d,\"dim\":%d,"
            "\"checkpoint_every\":%d,"
            "\"sequence_delimiter_token_id\":1}\n",
            prompt_count,
            completion_count,
            count,
            transformer.config.seq_len,
            transformer.config.n_layers,
            transformer.config.dim,
            options.checkpoint_every
        );
        fflush(trace_file);
    }

    CompanyTrace trace = {
        .prompt_tokens = prompt_count,
        .tokens = tokens,
        .hidden = calloc(
            (size_t)transformer.config.n_layers *
                transformer.config.seq_len * transformer.config.dim,
            sizeof(*trace.hidden)
        ),
        .layers = calloc(
            (size_t)transformer.config.n_layers,
            sizeof(*trace.layers)
        ),
        .position_layers = calloc(
            (size_t)transformer.config.n_layers * transformer.config.seq_len,
            sizeof(*trace.position_layers)
        ),
    };
    if (trace.hidden == NULL || trace.layers == NULL ||
        trace.position_layers == NULL) {
        fprintf(stderr, "could not allocate company trace\n");
        exit(EXIT_FAILURE);
    }

    double completion_score = 0.0;
    int token = tokens[0];
    float *logits = NULL;
    for (int position = 0; position < count; position++) {
        logits = company_forward(&transformer, token, position, &trace);
        if (trace_file != NULL && position >= prompt_count) {
            int completion_index = position - prompt_count;
            for (int layer = 0; layer < transformer.config.n_layers; layer++) {
                const LayerCompany *company = trace.position_layers +
                    (size_t)layer * transformer.config.seq_len + position;
                write_layer_trace(
                    trace_file,
                    company,
                    layer,
                    completion_index,
                    position
                );
            }
        }
        if (position + 1 < count) {
            int next = tokens[position + 1];
            if (position >= prompt_count - 1) {
                double log_probability = token_log_probability(
                    logits,
                    transformer.config.vocab_size,
                    next
                );
                completion_score += log_probability;
                if (trace_file != NULL) {
                    int completion_index = position - (prompt_count - 1);
                    write_token_trace(
                        trace_file,
                        &tokenizer,
                        logits,
                        transformer.config.vocab_size,
                        token,
                        next,
                        completion_index,
                        position,
                        completion_score
                    );
                }
            }
            token = next;
        }
    }

    double delimiter_log_probability = token_log_probability(
        logits,
        transformer.config.vocab_size,
        1
    );
    int delimiter_rank = local_rank(
        logits,
        transformer.config.vocab_size,
        1
    );
    if (trace_file != NULL) {
        fprintf(
            trace_file,
            "{\"kind\":\"terminal\",\"context_position\":%d,"
            "\"sequence_delimiter_token_id\":1,"
            "\"delimiter_log_probability\":%.17g,"
            "\"delimiter_rank\":%d}\n",
            count - 1,
            delimiter_log_probability,
            delimiter_rank
        );
        fflush(trace_file);
        write_affine_checkpoints(
            trace_file,
            &trace,
            &transformer.config,
            count,
            options.checkpoint_every
        );
    }

    printf(
        "prompt_tokens=%d completion_tokens=%d "
        "completion_log_probability=%.17g "
        "delimiter_log_probability=%.17g delimiter_rank=%d\n",
        prompt_count,
        completion_count,
        completion_score,
        delimiter_log_probability,
        delimiter_rank
    );
    puts("layer_company_profiles:");
    for (int layer = 0; layer < transformer.config.n_layers; layer++) {
        LayerCompany *company = &trace.layers[layer];
        double positions = company->positions;
        double heads = company->heads;
        printf(
            "  layer=%d positions=%d "
            "attention_entropy=%.9g attention_max_mass=%.9g "
            "attention_same_token_mass=%.9g attention_prompt_mass=%.9g "
            "attention_distance=%.9g "
            "attention_update_ratio=%.9g "
            "attention_update_alignment=%.9g "
            "ffn_update_ratio=%.9g ffn_update_alignment=%.9g "
            "residual_inertia=%.9g prior_state_similarity=%.9g "
            "same_token_state_similarity=%.9g same_token_states=%d\n",
            layer,
            company->positions,
            company->attention_entropy / heads,
            company->attention_max_mass / heads,
            company->attention_same_token_mass / heads,
            company->attention_prompt_mass / heads,
            company->attention_distance / heads,
            company->attention_update_ratio / positions,
            company->attention_update_alignment / positions,
            company->ffn_update_ratio / positions,
            company->ffn_update_alignment / positions,
            company->residual_inertia / positions,
            company->prior_state_similarity / positions,
            company->same_token_states == 0 ? 0.0 :
                company->same_token_state_similarity /
                    company->same_token_states,
            company->same_token_states
        );
    }
    puts("affine_company_profiles:");
    print_affine_company(&trace, &transformer.config, count);

    if (trace_file != NULL && fclose(trace_file) != 0) {
        fprintf(stderr, "could not close trace %s\n", options.trace_path);
        exit(EXIT_FAILURE);
    }
    free(trace.position_layers);
    free(trace.layers);
    free(trace.hidden);
    free(tokens);
    free(prompt_tokens);
    free(text);
    free(options.owned_completion);
    free(options.owned_prompt);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
