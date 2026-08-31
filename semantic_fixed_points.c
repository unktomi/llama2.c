/*
 * Contextual fixed-mode diagnostic for the frozen llama2.c transformer.
 *
 * Four equal-tokenized texts form a semantic square:
 *
 *     text00  -- edit A -->  text10
 *       |                       |
 *     edit B                  edit B
 *       |                       |
 *       v                       v
 *     text01  -- edit A -->  text11
 *
 * The program retains the two torsor arrows and their mixed four-corner
 * difference at every numerical boundary of the later edited token.  At
 * residual-stream endomorphisms it also restricts the operation to the two
 * actual edit directions and solves the resulting two-dimensional fixed-mode
 * problem.  These measurements are not a completion score and are never fed
 * back into inference.
 */

#define TESTING
#include "run.c"

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <stdbool.h>

typedef enum {
    STAGE_LAYER_INPUT,
    STAGE_ATTENTION_RMS,
    STAGE_QUERY,
    STAGE_ATTENTION_SCORES,
    STAGE_ATTENTION_PROBABILITIES,
    STAGE_ATTENTION_VALUE_SUM,
    STAGE_ATTENTION_UPDATE,
    STAGE_POST_ATTENTION,
    STAGE_FFN_RMS,
    STAGE_W1,
    STAGE_W3,
    STAGE_SILU,
    STAGE_SWIGLU_PRODUCT,
    STAGE_FFN_UPDATE,
    STAGE_LAYER_OUTPUT,
    STAGE_COUNT
} Stage;

static const char *const stage_names[STAGE_COUNT] = {
    "layer_input",
    "attention_rms",
    "query",
    "attention_scores",
    "attention_probabilities",
    "attention_value_sum",
    "attention_update",
    "post_attention",
    "ffn_rms",
    "w1",
    "w3",
    "silu",
    "swiglu_product",
    "ffn_update",
    "layer_output"
};

typedef struct {
    int width;
    float *values;
} StageBuffer;

typedef struct {
    StageBuffer stages[STAGE_COUNT];
} LayerCapture;

typedef struct {
    LayerCapture *layers;
} ContextCapture;

typedef struct {
    const char *text;
    int *tokens;
    int count;
} EncodedContext;

typedef struct {
    const char *trace_path;
    int observe_position;
} Options;

typedef struct {
    double edit_a_norm;
    double edit_b_norm;
    double edit_cosine;
    double interaction_norm;
    double interaction_fraction;
} CornerStats;

typedef struct {
    double edit_a_norm;
    double edit_b_norm;
    double edit_cosine;
} DirectionStats;

typedef struct {
    int input_rank;
    int output_rank;
    bool edit_a_defined;
    bool edit_b_defined;
    bool modes_defined;
    double edit_a_gain;
    double edit_b_gain;
    double edit_a_alignment;
    double edit_b_alignment;
    double edit_a_fixed_defect;
    double edit_b_fixed_defect;
    double closest_fixed_defect;
    double closest_fixed_a;
    double closest_fixed_b;
    double closest_discarded_gain;
    double closest_discarded_a;
    double closest_discarded_b;
    double transfer[4];
    double transfer_fixed_defect;
    double subspace_escape;
} TransitionStats;

typedef struct {
    double value;
    double coefficient_a;
    double coefficient_b;
} GeneralizedMode;

typedef struct {
    int rank;
    double eigenvalues[2];
    double principal[2];
    double pseudoinverse[4];
} SymmetricBasis;

static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

static void *checked_calloc(size_t count, size_t width) {
    void *memory = calloc(count, width);
    if (memory == NULL) fail("allocation failed");
    return memory;
}

static long parse_long_value(const char *text, const char *name) {
    errno = 0;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        fprintf(stderr, "%s must be an integer\n", name);
        exit(EXIT_FAILURE);
    }
    return value;
}

static void usage(const char *program) {
    fprintf(
        stderr,
        "usage: %s CHECKPOINT TOKENIZER TEXT00 TEXT10 TEXT01 TEXT11 "
        "[--observe-position N] [--trace PATH]\n"
        "\n"
        "TEXT10 applies edit A to TEXT00, TEXT01 applies edit B, and "
        "TEXT11 applies both. The four token sequences must have equal "
        "length and exactly one independently changed token per edit.\n",
        program
    );
    exit(EXIT_FAILURE);
}

static Options parse_options(int argc, char **argv) {
    if (argc < 7) usage(argv[0]);
    Options options = {.observe_position = -1};
    int index = 7;
    while (index < argc) {
        if (strcmp(argv[index], "--trace") == 0 && index + 1 < argc) {
            options.trace_path = argv[index + 1];
            index += 2;
        } else if (strcmp(argv[index], "--observe-position") == 0 &&
                   index + 1 < argc) {
            long parsed = parse_long_value(argv[index + 1], "observe position");
            if (parsed < 0 || parsed > INT_MAX) usage(argv[0]);
            options.observe_position = (int)parsed;
            index += 2;
        } else {
            usage(argv[0]);
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

static EncodedContext encode_context(
    Tokenizer *tokenizer,
    const char *text
) {
    EncodedContext context = {.text = text};
    size_t capacity = strlen(text) + 3U;
    context.tokens = checked_calloc(capacity, sizeof(*context.tokens));
    encode(
        tokenizer,
        (char *)text,
        1,
        0,
        context.tokens,
        &context.count
    );
    if (context.count <= 0) fail("tokenizer produced an empty context");
    return context;
}

static void free_context(EncodedContext *context) {
    free(context->tokens);
    context->tokens = NULL;
}

static int stage_width(
    Stage stage,
    const Config *config,
    int observe_position
) {
    switch (stage) {
        case STAGE_ATTENTION_SCORES:
        case STAGE_ATTENTION_PROBABILITIES:
            return config->n_heads * (observe_position + 1);
        case STAGE_W1:
        case STAGE_W3:
        case STAGE_SILU:
        case STAGE_SWIGLU_PRODUCT:
            return config->hidden_dim;
        default:
            return config->dim;
    }
}

static ContextCapture allocate_capture(
    const Config *config,
    int observe_position
) {
    ContextCapture capture;
    capture.layers = checked_calloc(
        (size_t)config->n_layers,
        sizeof(*capture.layers)
    );
    for (int layer = 0; layer < config->n_layers; layer++) {
        for (int stage = 0; stage < STAGE_COUNT; stage++) {
            StageBuffer *buffer = &capture.layers[layer].stages[stage];
            buffer->width = stage_width((Stage)stage, config, observe_position);
            buffer->values = checked_calloc(
                (size_t)buffer->width,
                sizeof(*buffer->values)
            );
        }
    }
    return capture;
}

static void free_capture(ContextCapture *capture, int layers) {
    if (capture->layers == NULL) return;
    for (int layer = 0; layer < layers; layer++) {
        for (int stage = 0; stage < STAGE_COUNT; stage++) {
            free(capture->layers[layer].stages[stage].values);
        }
    }
    free(capture->layers);
    capture->layers = NULL;
}

static void retain_stage(
    LayerCapture *capture,
    Stage stage,
    const float *values,
    int width
) {
    StageBuffer *buffer = &capture->stages[stage];
    if (buffer->width != width) fail("capture stage width changed");
    memcpy(buffer->values, values, (size_t)width * sizeof(*values));
}

static void rmsnorm_jvp(
    float *output_tangent,
    const float *input,
    const float *input_tangent,
    const float *weight,
    int size
) {
    float mean_square = 0.0f;
    float mean_product = 0.0f;
    for (int lane = 0; lane < size; lane++) {
        mean_square += input[lane] * input[lane];
        mean_product += input[lane] * input_tangent[lane];
    }
    mean_square = mean_square / size + 1e-5f;
    mean_product /= size;
    float inverse_rms = 1.0f / sqrtf(mean_square);
    float radial_derivative =
        inverse_rms * inverse_rms * inverse_rms * mean_product;
    for (int lane = 0; lane < size; lane++) {
        output_tangent[lane] = weight[lane] *
            (inverse_rms * input_tangent[lane] -
             radial_derivative * input[lane]);
    }
}

static void softmax_jvp(
    float *score_tangent,
    const float *probability,
    int size
) {
    float expected_tangent = 0.0f;
    for (int index = 0; index < size; index++) {
        expected_tangent += probability[index] * score_tangent[index];
    }
    for (int index = 0; index < size; index++) {
        score_tangent[index] = probability[index] *
            (score_tangent[index] - expected_tangent);
    }
}

/*
 * This is run.c's layer kernel with observational copies at its existing
 * boundaries. It deliberately omits the final token classifier: the
 * diagnostic concerns the hidden-state layer endomorphisms, not an eager
 * token score.
 */
static void forward_capture_boundaries(
    Transformer *transformer,
    int token,
    int position,
    LayerCapture *captures
) {
    Config *config = &transformer->config;
    TransformerWeights *weights = &transformer->weights;
    RunState *state = &transformer->state;
    int dim = config->dim;
    int hidden_dim = config->hidden_dim;
    int kv_dim = dim * config->n_kv_heads / config->n_heads;
    int kv_mul = config->n_heads / config->n_kv_heads;
    int head_size = dim / config->n_heads;
    int score_width = config->n_heads * (position + 1);

    memcpy(
        state->x,
        weights->token_embedding_table + (size_t)token * dim,
        (size_t)dim * sizeof(*state->x)
    );

    for (int layer = 0; layer < config->n_layers; layer++) {
        LayerCapture *capture = captures == NULL ? NULL : &captures[layer];
        if (capture != NULL) {
            retain_stage(capture, STAGE_LAYER_INPUT, state->x, dim);
        }

        rmsnorm(
            state->xb,
            state->x,
            weights->rms_att_weight + (size_t)layer * dim,
            dim
        );
        if (capture != NULL) {
            retain_stage(capture, STAGE_ATTENTION_RMS, state->xb, dim);
        }

        size_t layer_offset = (size_t)layer * config->seq_len * kv_dim;
        state->k = state->key_cache + layer_offset + (size_t)position * kv_dim;
        state->v = state->value_cache + layer_offset + (size_t)position * kv_dim;
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
        if (capture != NULL) {
            retain_stage(capture, STAGE_QUERY, state->q, dim);
        }

        for (int head = 0; head < config->n_heads; head++) {
            float *query = state->q + head * head_size;
            float *attention = state->att + (size_t)head * config->seq_len;
            for (int timestep = 0; timestep <= position; timestep++) {
                float *key = state->key_cache + layer_offset +
                    (size_t)timestep * kv_dim +
                    (head / kv_mul) * head_size;
                float score = 0.0f;
                for (int lane = 0; lane < head_size; lane++) {
                    score += query[lane] * key[lane];
                }
                attention[timestep] = score / sqrtf((float)head_size);
            }
            if (capture != NULL) {
                memcpy(
                    capture->stages[STAGE_ATTENTION_SCORES].values +
                        (size_t)head * (position + 1),
                    attention,
                    (size_t)(position + 1) * sizeof(*attention)
                );
            }
            softmax(attention, position + 1);
            if (capture != NULL) {
                memcpy(
                    capture->stages[STAGE_ATTENTION_PROBABILITIES].values +
                        (size_t)head * (position + 1),
                    attention,
                    (size_t)(position + 1) * sizeof(*attention)
                );
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
        if (capture != NULL) {
            if (capture->stages[STAGE_ATTENTION_SCORES].width != score_width) {
                fail("attention score width changed");
            }
            retain_stage(capture, STAGE_ATTENTION_VALUE_SUM, state->xb, dim);
        }

        matmul(
            state->xb2,
            state->xb,
            weights->wo + (size_t)layer * dim * dim,
            dim,
            dim
        );
        if (capture != NULL) {
            retain_stage(capture, STAGE_ATTENTION_UPDATE, state->xb2, dim);
        }
        for (int lane = 0; lane < dim; lane++) state->x[lane] += state->xb2[lane];
        if (capture != NULL) {
            retain_stage(capture, STAGE_POST_ATTENTION, state->x, dim);
        }

        rmsnorm(
            state->xb,
            state->x,
            weights->rms_ffn_weight + (size_t)layer * dim,
            dim
        );
        if (capture != NULL) {
            retain_stage(capture, STAGE_FFN_RMS, state->xb, dim);
        }
        matmul(
            state->hb,
            state->xb,
            weights->w1 + (size_t)layer * dim * hidden_dim,
            dim,
            hidden_dim
        );
        matmul(
            state->hb2,
            state->xb,
            weights->w3 + (size_t)layer * dim * hidden_dim,
            dim,
            hidden_dim
        );
        if (capture != NULL) {
            retain_stage(capture, STAGE_W1, state->hb, hidden_dim);
            retain_stage(capture, STAGE_W3, state->hb2, hidden_dim);
        }
        for (int lane = 0; lane < hidden_dim; lane++) {
            float value = state->hb[lane];
            value *= 1.0f / (1.0f + expf(-value));
            if (capture != NULL) {
                capture->stages[STAGE_SILU].values[lane] = value;
            }
            value *= state->hb2[lane];
            state->hb[lane] = value;
        }
        if (capture != NULL) {
            retain_stage(
                capture,
                STAGE_SWIGLU_PRODUCT,
                state->hb,
                hidden_dim
            );
        }
        matmul(
            state->xb,
            state->hb,
            weights->w2 + (size_t)layer * dim * hidden_dim,
            hidden_dim,
            dim
        );
        if (capture != NULL) {
            retain_stage(capture, STAGE_FFN_UPDATE, state->xb, dim);
        }
        for (int lane = 0; lane < dim; lane++) state->x[lane] += state->xb[lane];
        if (capture != NULL) {
            retain_stage(capture, STAGE_LAYER_OUTPUT, state->x, dim);
        }
    }
}

/*
 * Evaluate the unedited context and carry two forward tangents through the
 * same numerical operations. The tangent seeds are finite embedding arrows,
 * but every value retained below is the analytic directional derivative at
 * the unedited context. Tangent KV caches make earlier-token edits visible to
 * later-token attention without replacing them by a current-token surrogate.
 */
static void forward_capture_with_tangents(
    Transformer *transformer,
    RunState tangent_states[2],
    int token,
    const int edit_tokens[2],
    int position,
    LayerCapture *captures,
    LayerCapture *tangent_captures[2]
) {
    Config *config = &transformer->config;
    TransformerWeights *weights = &transformer->weights;
    RunState *state = &transformer->state;
    int dim = config->dim;
    int hidden_dim = config->hidden_dim;
    int kv_dim = dim * config->n_kv_heads / config->n_heads;
    int kv_mul = config->n_heads / config->n_kv_heads;
    int head_size = dim / config->n_heads;
    int score_width = config->n_heads * (position + 1);
    const float *embedding =
        weights->token_embedding_table + (size_t)token * dim;

    memcpy(state->x, embedding, (size_t)dim * sizeof(*state->x));
    for (int direction = 0; direction < 2; direction++) {
        const float *edited_embedding = weights->token_embedding_table +
            (size_t)edit_tokens[direction] * dim;
        for (int lane = 0; lane < dim; lane++) {
            tangent_states[direction].x[lane] =
                edited_embedding[lane] - embedding[lane];
        }
    }

    for (int layer = 0; layer < config->n_layers; layer++) {
        LayerCapture *capture = captures == NULL ? NULL : &captures[layer];
        LayerCapture *tangent_layer_captures[2];
        for (int direction = 0; direction < 2; direction++) {
            tangent_layer_captures[direction] =
                tangent_captures[direction] == NULL ? NULL :
                &tangent_captures[direction][layer];
        }
        if (capture != NULL) {
            retain_stage(capture, STAGE_LAYER_INPUT, state->x, dim);
        }
        for (int direction = 0; direction < 2; direction++) {
            if (tangent_layer_captures[direction] != NULL) {
                retain_stage(
                    tangent_layer_captures[direction],
                    STAGE_LAYER_INPUT,
                    tangent_states[direction].x,
                    dim
                );
            }
        }

        const float *attention_weight =
            weights->rms_att_weight + (size_t)layer * dim;
        rmsnorm(state->xb, state->x, (float *)attention_weight, dim);
        for (int direction = 0; direction < 2; direction++) {
            rmsnorm_jvp(
                tangent_states[direction].xb,
                state->x,
                tangent_states[direction].x,
                attention_weight,
                dim
            );
        }
        if (capture != NULL) {
            retain_stage(capture, STAGE_ATTENTION_RMS, state->xb, dim);
        }
        for (int direction = 0; direction < 2; direction++) {
            if (tangent_layer_captures[direction] != NULL) {
                retain_stage(
                    tangent_layer_captures[direction],
                    STAGE_ATTENTION_RMS,
                    tangent_states[direction].xb,
                    dim
                );
            }
        }

        size_t layer_offset = (size_t)layer * config->seq_len * kv_dim;
        state->k = state->key_cache + layer_offset + (size_t)position * kv_dim;
        state->v = state->value_cache + layer_offset + (size_t)position * kv_dim;
        for (int direction = 0; direction < 2; direction++) {
            tangent_states[direction].k =
                tangent_states[direction].key_cache + layer_offset +
                (size_t)position * kv_dim;
            tangent_states[direction].v =
                tangent_states[direction].value_cache + layer_offset +
                (size_t)position * kv_dim;
        }
        float *wq = weights->wq + (size_t)layer * dim * dim;
        float *wk = weights->wk + (size_t)layer * dim * kv_dim;
        float *wv = weights->wv + (size_t)layer * dim * kv_dim;
        matmul(state->q, state->xb, wq, dim, dim);
        matmul(state->k, state->xb, wk, dim, kv_dim);
        matmul(state->v, state->xb, wv, dim, kv_dim);
        for (int direction = 0; direction < 2; direction++) {
            matmul(
                tangent_states[direction].q,
                tangent_states[direction].xb,
                wq,
                dim,
                dim
            );
            matmul(
                tangent_states[direction].k,
                tangent_states[direction].xb,
                wk,
                dim,
                kv_dim
            );
            matmul(
                tangent_states[direction].v,
                tangent_states[direction].xb,
                wv,
                dim,
                kv_dim
            );
        }

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
                for (int direction = 0; direction < 2; direction++) {
                    float *tangent_values = vector == 0 ?
                        tangent_states[direction].q :
                        tangent_states[direction].k;
                    float tangent_first = tangent_values[index];
                    float tangent_second = tangent_values[index + 1];
                    tangent_values[index] =
                        tangent_first * real - tangent_second * imaginary;
                    tangent_values[index + 1] =
                        tangent_first * imaginary + tangent_second * real;
                }
            }
        }
        if (capture != NULL) {
            retain_stage(capture, STAGE_QUERY, state->q, dim);
        }
        for (int direction = 0; direction < 2; direction++) {
            if (tangent_layer_captures[direction] != NULL) {
                retain_stage(
                    tangent_layer_captures[direction],
                    STAGE_QUERY,
                    tangent_states[direction].q,
                    dim
                );
            }
        }

        float score_scale = 1.0f / sqrtf((float)head_size);
        for (int head = 0; head < config->n_heads; head++) {
            float *query = state->q + head * head_size;
            float *attention = state->att + (size_t)head * config->seq_len;
            float *tangent_attention[2] = {
                tangent_states[0].att + (size_t)head * config->seq_len,
                tangent_states[1].att + (size_t)head * config->seq_len
            };
            for (int timestep = 0; timestep <= position; timestep++) {
                float *key = state->key_cache + layer_offset +
                    (size_t)timestep * kv_dim +
                    (head / kv_mul) * head_size;
                float score = 0.0f;
                float score_tangent[2] = {0.0f, 0.0f};
                for (int lane = 0; lane < head_size; lane++) {
                    score += query[lane] * key[lane];
                    for (int direction = 0; direction < 2; direction++) {
                        float *tangent_query =
                            tangent_states[direction].q + head * head_size;
                        float *tangent_key =
                            tangent_states[direction].key_cache + layer_offset +
                            (size_t)timestep * kv_dim +
                            (head / kv_mul) * head_size;
                        score_tangent[direction] +=
                            tangent_query[lane] * key[lane] +
                            query[lane] * tangent_key[lane];
                    }
                }
                attention[timestep] = score * score_scale;
                for (int direction = 0; direction < 2; direction++) {
                    tangent_attention[direction][timestep] =
                        score_tangent[direction] * score_scale;
                }
            }
            if (capture != NULL) {
                memcpy(
                    capture->stages[STAGE_ATTENTION_SCORES].values +
                        (size_t)head * (position + 1),
                    attention,
                    (size_t)(position + 1) * sizeof(*attention)
                );
            }
            for (int direction = 0; direction < 2; direction++) {
                if (tangent_layer_captures[direction] != NULL) {
                    memcpy(
                        tangent_layer_captures[direction]
                            ->stages[STAGE_ATTENTION_SCORES].values +
                            (size_t)head * (position + 1),
                        tangent_attention[direction],
                        (size_t)(position + 1) * sizeof(float)
                    );
                }
            }

            softmax(attention, position + 1);
            for (int direction = 0; direction < 2; direction++) {
                softmax_jvp(
                    tangent_attention[direction],
                    attention,
                    position + 1
                );
            }
            if (capture != NULL) {
                memcpy(
                    capture->stages[STAGE_ATTENTION_PROBABILITIES].values +
                        (size_t)head * (position + 1),
                    attention,
                    (size_t)(position + 1) * sizeof(*attention)
                );
            }
            for (int direction = 0; direction < 2; direction++) {
                if (tangent_layer_captures[direction] != NULL) {
                    memcpy(
                        tangent_layer_captures[direction]
                            ->stages[STAGE_ATTENTION_PROBABILITIES].values +
                            (size_t)head * (position + 1),
                        tangent_attention[direction],
                        (size_t)(position + 1) * sizeof(float)
                    );
                }
            }

            float *head_output = state->xb + head * head_size;
            float *tangent_head_output[2] = {
                tangent_states[0].xb + head * head_size,
                tangent_states[1].xb + head * head_size
            };
            memset(head_output, 0, (size_t)head_size * sizeof(*head_output));
            for (int direction = 0; direction < 2; direction++) {
                memset(
                    tangent_head_output[direction],
                    0,
                    (size_t)head_size * sizeof(float)
                );
            }
            for (int timestep = 0; timestep <= position; timestep++) {
                float *value = state->value_cache + layer_offset +
                    (size_t)timestep * kv_dim +
                    (head / kv_mul) * head_size;
                float mass = attention[timestep];
                for (int lane = 0; lane < head_size; lane++) {
                    head_output[lane] += mass * value[lane];
                    for (int direction = 0; direction < 2; direction++) {
                        float *tangent_value =
                            tangent_states[direction].value_cache +
                            layer_offset + (size_t)timestep * kv_dim +
                            (head / kv_mul) * head_size;
                        tangent_head_output[direction][lane] +=
                            tangent_attention[direction][timestep] *
                                value[lane] +
                            mass * tangent_value[lane];
                    }
                }
            }
        }
        if (capture != NULL) {
            if (capture->stages[STAGE_ATTENTION_SCORES].width != score_width) {
                fail("attention score width changed");
            }
            retain_stage(capture, STAGE_ATTENTION_VALUE_SUM, state->xb, dim);
        }
        for (int direction = 0; direction < 2; direction++) {
            if (tangent_layer_captures[direction] != NULL) {
                retain_stage(
                    tangent_layer_captures[direction],
                    STAGE_ATTENTION_VALUE_SUM,
                    tangent_states[direction].xb,
                    dim
                );
            }
        }

        float *wo = weights->wo + (size_t)layer * dim * dim;
        matmul(state->xb2, state->xb, wo, dim, dim);
        for (int direction = 0; direction < 2; direction++) {
            matmul(
                tangent_states[direction].xb2,
                tangent_states[direction].xb,
                wo,
                dim,
                dim
            );
        }
        if (capture != NULL) {
            retain_stage(capture, STAGE_ATTENTION_UPDATE, state->xb2, dim);
        }
        for (int direction = 0; direction < 2; direction++) {
            if (tangent_layer_captures[direction] != NULL) {
                retain_stage(
                    tangent_layer_captures[direction],
                    STAGE_ATTENTION_UPDATE,
                    tangent_states[direction].xb2,
                    dim
                );
            }
        }
        for (int lane = 0; lane < dim; lane++) {
            state->x[lane] += state->xb2[lane];
            for (int direction = 0; direction < 2; direction++) {
                tangent_states[direction].x[lane] +=
                    tangent_states[direction].xb2[lane];
            }
        }
        if (capture != NULL) {
            retain_stage(capture, STAGE_POST_ATTENTION, state->x, dim);
        }
        for (int direction = 0; direction < 2; direction++) {
            if (tangent_layer_captures[direction] != NULL) {
                retain_stage(
                    tangent_layer_captures[direction],
                    STAGE_POST_ATTENTION,
                    tangent_states[direction].x,
                    dim
                );
            }
        }

        const float *ffn_weight =
            weights->rms_ffn_weight + (size_t)layer * dim;
        rmsnorm(state->xb, state->x, (float *)ffn_weight, dim);
        for (int direction = 0; direction < 2; direction++) {
            rmsnorm_jvp(
                tangent_states[direction].xb,
                state->x,
                tangent_states[direction].x,
                ffn_weight,
                dim
            );
        }
        if (capture != NULL) {
            retain_stage(capture, STAGE_FFN_RMS, state->xb, dim);
        }
        for (int direction = 0; direction < 2; direction++) {
            if (tangent_layer_captures[direction] != NULL) {
                retain_stage(
                    tangent_layer_captures[direction],
                    STAGE_FFN_RMS,
                    tangent_states[direction].xb,
                    dim
                );
            }
        }

        float *w1 = weights->w1 + (size_t)layer * dim * hidden_dim;
        float *w3 = weights->w3 + (size_t)layer * dim * hidden_dim;
        matmul(state->hb, state->xb, w1, dim, hidden_dim);
        matmul(state->hb2, state->xb, w3, dim, hidden_dim);
        for (int direction = 0; direction < 2; direction++) {
            matmul(
                tangent_states[direction].hb,
                tangent_states[direction].xb,
                w1,
                dim,
                hidden_dim
            );
            matmul(
                tangent_states[direction].hb2,
                tangent_states[direction].xb,
                w3,
                dim,
                hidden_dim
            );
        }
        if (capture != NULL) {
            retain_stage(capture, STAGE_W1, state->hb, hidden_dim);
            retain_stage(capture, STAGE_W3, state->hb2, hidden_dim);
        }
        for (int direction = 0; direction < 2; direction++) {
            if (tangent_layer_captures[direction] != NULL) {
                retain_stage(
                    tangent_layer_captures[direction],
                    STAGE_W1,
                    tangent_states[direction].hb,
                    hidden_dim
                );
                retain_stage(
                    tangent_layer_captures[direction],
                    STAGE_W3,
                    tangent_states[direction].hb2,
                    hidden_dim
                );
            }
        }

        for (int lane = 0; lane < hidden_dim; lane++) {
            float first = state->hb[lane];
            float second = state->hb2[lane];
            float sigmoid = 1.0f / (1.0f + expf(-first));
            float silu = first * sigmoid;
            float silu_slope = sigmoid +
                first * sigmoid * (1.0f - sigmoid);
            if (capture != NULL) {
                capture->stages[STAGE_SILU].values[lane] = silu;
            }
            for (int direction = 0; direction < 2; direction++) {
                float silu_tangent = silu_slope *
                    tangent_states[direction].hb[lane];
                if (tangent_layer_captures[direction] != NULL) {
                    tangent_layer_captures[direction]
                        ->stages[STAGE_SILU].values[lane] = silu_tangent;
                }
                tangent_states[direction].hb[lane] =
                    silu_tangent * second +
                    silu * tangent_states[direction].hb2[lane];
            }
            state->hb[lane] = silu * second;
        }
        if (capture != NULL) {
            retain_stage(
                capture,
                STAGE_SWIGLU_PRODUCT,
                state->hb,
                hidden_dim
            );
        }
        for (int direction = 0; direction < 2; direction++) {
            if (tangent_layer_captures[direction] != NULL) {
                retain_stage(
                    tangent_layer_captures[direction],
                    STAGE_SWIGLU_PRODUCT,
                    tangent_states[direction].hb,
                    hidden_dim
                );
            }
        }

        float *w2 = weights->w2 + (size_t)layer * dim * hidden_dim;
        matmul(state->xb, state->hb, w2, hidden_dim, dim);
        for (int direction = 0; direction < 2; direction++) {
            matmul(
                tangent_states[direction].xb,
                tangent_states[direction].hb,
                w2,
                hidden_dim,
                dim
            );
        }
        if (capture != NULL) {
            retain_stage(capture, STAGE_FFN_UPDATE, state->xb, dim);
        }
        for (int direction = 0; direction < 2; direction++) {
            if (tangent_layer_captures[direction] != NULL) {
                retain_stage(
                    tangent_layer_captures[direction],
                    STAGE_FFN_UPDATE,
                    tangent_states[direction].xb,
                    dim
                );
            }
        }
        for (int lane = 0; lane < dim; lane++) {
            state->x[lane] += state->xb[lane];
            for (int direction = 0; direction < 2; direction++) {
                tangent_states[direction].x[lane] +=
                    tangent_states[direction].xb[lane];
            }
        }
        if (capture != NULL) {
            retain_stage(capture, STAGE_LAYER_OUTPUT, state->x, dim);
        }
        for (int direction = 0; direction < 2; direction++) {
            if (tangent_layer_captures[direction] != NULL) {
                retain_stage(
                    tangent_layer_captures[direction],
                    STAGE_LAYER_OUTPUT,
                    tangent_states[direction].x,
                    dim
                );
            }
        }
    }
}

static void capture_context_with_tangents(
    Transformer *transformer,
    RunState tangent_states[2],
    const EncodedContext *base_context,
    const EncodedContext *edit_a_context,
    const EncodedContext *edit_b_context,
    int observe_position,
    ContextCapture *base_capture,
    ContextCapture tangent_capture[2]
) {
    Config *config = &transformer->config;
    int kv_dim = config->dim * config->n_kv_heads / config->n_heads;
    size_t cache_count =
        (size_t)config->n_layers * config->seq_len * kv_dim;
    memset(transformer->state.key_cache, 0, cache_count * sizeof(float));
    memset(transformer->state.value_cache, 0, cache_count * sizeof(float));
    for (int direction = 0; direction < 2; direction++) {
        memset(
            tangent_states[direction].key_cache,
            0,
            cache_count * sizeof(float)
        );
        memset(
            tangent_states[direction].value_cache,
            0,
            cache_count * sizeof(float)
        );
    }
    for (int position = 0; position <= observe_position; position++) {
        int edit_tokens[2] = {
            edit_a_context->tokens[position],
            edit_b_context->tokens[position]
        };
        LayerCapture *tangent_layers[2] = {NULL, NULL};
        LayerCapture *base_layers = NULL;
        if (position == observe_position) {
            base_layers = base_capture->layers;
            tangent_layers[0] = tangent_capture[0].layers;
            tangent_layers[1] = tangent_capture[1].layers;
        }
        forward_capture_with_tangents(
            transformer,
            tangent_states,
            base_context->tokens[position],
            edit_tokens,
            position,
            base_layers,
            tangent_layers
        );
    }
}

static void capture_context(
    Transformer *transformer,
    const EncodedContext *context,
    int observe_position,
    ContextCapture *capture
) {
    Config *config = &transformer->config;
    int kv_dim = config->dim * config->n_kv_heads / config->n_heads;
    size_t cache_count =
        (size_t)config->n_layers * config->seq_len * kv_dim;
    memset(transformer->state.key_cache, 0, cache_count * sizeof(float));
    memset(transformer->state.value_cache, 0, cache_count * sizeof(float));
    for (int position = 0; position <= observe_position; position++) {
        forward_capture_boundaries(
            transformer,
            context->tokens[position],
            position,
            position == observe_position ? capture->layers : NULL
        );
    }
}

static CornerStats measure_corners(
    const float *const values[4],
    int width
) {
    double aa = 0.0;
    double bb = 0.0;
    double ab = 0.0;
    double interaction_square = 0.0;
    for (int lane = 0; lane < width; lane++) {
        double edit_a = (double)values[1][lane] - values[0][lane];
        double edit_b = (double)values[2][lane] - values[0][lane];
        double interaction = (double)values[3][lane] - values[1][lane] -
            values[2][lane] + values[0][lane];
        aa += edit_a * edit_a;
        bb += edit_b * edit_b;
        ab += edit_a * edit_b;
        interaction_square += interaction * interaction;
    }
    CornerStats stats;
    stats.edit_a_norm = sqrt(aa);
    stats.edit_b_norm = sqrt(bb);
    stats.edit_cosine = aa == 0.0 || bb == 0.0 ? 0.0 :
        ab / sqrt(aa * bb);
    stats.interaction_norm = sqrt(interaction_square);
    double scale = stats.edit_a_norm + stats.edit_b_norm;
    stats.interaction_fraction = scale == 0.0 ? 0.0 :
        stats.interaction_norm / scale;
    return stats;
}

static DirectionStats measure_directions(
    const float *edit_a,
    const float *edit_b,
    int width
) {
    double aa = 0.0;
    double bb = 0.0;
    double ab = 0.0;
    for (int lane = 0; lane < width; lane++) {
        double a = edit_a[lane];
        double b = edit_b[lane];
        aa += a * a;
        bb += b * b;
        ab += a * b;
    }
    return (DirectionStats){
        .edit_a_norm = sqrt(aa),
        .edit_b_norm = sqrt(bb),
        .edit_cosine = aa == 0.0 || bb == 0.0 ? 0.0 :
            ab / sqrt(aa * bb)
    };
}

static GeneralizedMode generalized_smallest(
    double a00,
    double a01,
    double a11,
    double b00,
    double b01,
    double b11
) {
    double original_trace = b00 + b11;
    double regularizer = DBL_EPSILON * 1024.0 *
        (original_trace > 1.0 ? original_trace : 1.0);
    b00 += regularizer;
    b11 += regularizer;
    double l00 = sqrt(b00);
    double l10 = b01 / l00;
    double remainder = b11 - l10 * l10;
    if (remainder < regularizer) remainder = regularizer;
    double l11 = sqrt(remainder);
    double inv00 = 1.0 / l00;
    double inv10 = -l10 / (l00 * l11);
    double inv11 = 1.0 / l11;

    double c00 = inv00 * inv00 * a00;
    double c01 = inv00 * (inv10 * a00 + inv11 * a01);
    double c11 = inv10 * inv10 * a00 +
        2.0 * inv10 * inv11 * a01 + inv11 * inv11 * a11;
    double radius = hypot(c00 - c11, 2.0 * c01);
    double eigenvalue = 0.5 * (c00 + c11 - radius);
    if (eigenvalue < 0.0 && eigenvalue > -1e-12) eigenvalue = 0.0;

    double u0;
    double u1;
    if (fabs(c01) > 1e-30) {
        u0 = c01;
        u1 = eigenvalue - c00;
    } else if (c00 <= c11) {
        u0 = 1.0;
        u1 = 0.0;
    } else {
        u0 = 0.0;
        u1 = 1.0;
    }
    double u_norm = hypot(u0, u1);
    if (u_norm == 0.0) {
        u0 = 1.0;
        u1 = 0.0;
    } else {
        u0 /= u_norm;
        u1 /= u_norm;
    }
    double coefficient_a = inv00 * u0 + inv10 * u1;
    double coefficient_b = inv11 * u1;
    double coefficient_scale = fmax(fabs(coefficient_a), fabs(coefficient_b));
    if (coefficient_scale > 0.0) {
        coefficient_a /= coefficient_scale;
        coefficient_b /= coefficient_scale;
    }
    return (GeneralizedMode){
        .value = eigenvalue < 0.0 ? 0.0 : eigenvalue,
        .coefficient_a = coefficient_a,
        .coefficient_b = coefficient_b
    };
}

static SymmetricBasis symmetric_basis(
    double a00,
    double a01,
    double a11
) {
    SymmetricBasis basis = {0};
    double radius = hypot(a00 - a11, 2.0 * a01);
    basis.eigenvalues[0] = 0.5 * (a00 + a11 + radius);
    basis.eigenvalues[1] = 0.5 * (a00 + a11 - radius);
    if (basis.eigenvalues[1] < 0.0 && basis.eigenvalues[1] > -1e-12) {
        basis.eigenvalues[1] = 0.0;
    }
    if (fabs(a01) > 1e-30) {
        basis.principal[0] = a01;
        basis.principal[1] = basis.eigenvalues[0] - a00;
    } else if (a00 >= a11) {
        basis.principal[0] = 1.0;
        basis.principal[1] = 0.0;
    } else {
        basis.principal[0] = 0.0;
        basis.principal[1] = 1.0;
    }
    double norm = hypot(basis.principal[0], basis.principal[1]);
    if (norm == 0.0) {
        basis.principal[0] = 1.0;
        basis.principal[1] = 0.0;
    } else {
        basis.principal[0] /= norm;
        basis.principal[1] /= norm;
    }
    double threshold = fmax(1e-24, basis.eigenvalues[0] * 1e-10);
    double vectors[2][2] = {
        {basis.principal[0], basis.principal[1]},
        {-basis.principal[1], basis.principal[0]}
    };
    for (int mode = 0; mode < 2; mode++) {
        double eigenvalue = basis.eigenvalues[mode];
        if (eigenvalue <= threshold) continue;
        basis.rank++;
        double inverse = 1.0 / eigenvalue;
        basis.pseudoinverse[0] +=
            inverse * vectors[mode][0] * vectors[mode][0];
        basis.pseudoinverse[1] +=
            inverse * vectors[mode][0] * vectors[mode][1];
        basis.pseudoinverse[2] +=
            inverse * vectors[mode][1] * vectors[mode][0];
        basis.pseudoinverse[3] +=
            inverse * vectors[mode][1] * vectors[mode][1];
    }
    return basis;
}

static GeneralizedMode rank_one_mode(
    double a00,
    double a01,
    double a11,
    double b00,
    double b01,
    double b11,
    const SymmetricBasis *input_basis
) {
    double c0 = input_basis->principal[0];
    double c1 = input_basis->principal[1];
    double numerator = c0 * c0 * a00 + 2.0 * c0 * c1 * a01 +
        c1 * c1 * a11;
    double denominator = c0 * c0 * b00 + 2.0 * c0 * c1 * b01 +
        c1 * c1 * b11;
    double scale = fmax(fabs(c0), fabs(c1));
    if (scale > 0.0) {
        c0 /= scale;
        c1 /= scale;
    }
    return (GeneralizedMode){
        .value = denominator <= 0.0 ? 0.0 : fmax(0.0, numerator / denominator),
        .coefficient_a = c0,
        .coefficient_b = c1
    };
}

static TransitionStats measure_direction_map(
    const float *input_origin,
    const float *input_a_endpoint,
    const float *input_b_endpoint,
    const float *output_origin,
    const float *output_a_endpoint,
    const float *output_b_endpoint,
    int width
) {
    double b00 = 0.0;
    double b01 = 0.0;
    double b11 = 0.0;
    double error00 = 0.0;
    double error01 = 0.0;
    double error11 = 0.0;
    double output00 = 0.0;
    double output01 = 0.0;
    double output11 = 0.0;
    double cross00 = 0.0;
    double cross01 = 0.0;
    double cross10 = 0.0;
    double cross11 = 0.0;
    double in_a_out_a = 0.0;
    double in_b_out_b = 0.0;

    for (int lane = 0; lane < width; lane++) {
        double in_origin = input_origin == NULL ? 0.0 : input_origin[lane];
        double out_origin = output_origin == NULL ? 0.0 : output_origin[lane];
        double in_a = (double)input_a_endpoint[lane] - in_origin;
        double in_b = (double)input_b_endpoint[lane] - in_origin;
        double out_a = (double)output_a_endpoint[lane] - out_origin;
        double out_b = (double)output_b_endpoint[lane] - out_origin;
        double error_a = out_a - in_a;
        double error_b = out_b - in_b;
        b00 += in_a * in_a;
        b01 += in_a * in_b;
        b11 += in_b * in_b;
        error00 += error_a * error_a;
        error01 += error_a * error_b;
        error11 += error_b * error_b;
        output00 += out_a * out_a;
        output01 += out_a * out_b;
        output11 += out_b * out_b;
        cross00 += in_a * out_a;
        cross01 += in_a * out_b;
        cross10 += in_b * out_a;
        cross11 += in_b * out_b;
        in_a_out_a += in_a * out_a;
        in_b_out_b += in_b * out_b;
    }

    SymmetricBasis input_basis = symmetric_basis(b00, b01, b11);
    SymmetricBasis output_basis = symmetric_basis(
        output00, output01, output11
    );
    GeneralizedMode fixed = {0};
    GeneralizedMode discarded = {0};
    if (input_basis.rank == 1) {
        fixed = rank_one_mode(
            error00, error01, error11, b00, b01, b11, &input_basis
        );
        discarded = rank_one_mode(
            output00, output01, output11, b00, b01, b11, &input_basis
        );
    } else if (input_basis.rank == 2) {
        fixed = generalized_smallest(
            error00, error01, error11, b00, b01, b11
        );
        discarded = generalized_smallest(
            output00, output01, output11, b00, b01, b11
        );
    }

    TransitionStats stats = {0};
    stats.input_rank = input_basis.rank;
    stats.output_rank = output_basis.rank;
    stats.edit_a_defined = b00 > 0.0;
    stats.edit_b_defined = b11 > 0.0;
    stats.modes_defined = input_basis.rank > 0;
    stats.transfer[0] = input_basis.pseudoinverse[0] * cross00 +
        input_basis.pseudoinverse[1] * cross10;
    stats.transfer[1] = input_basis.pseudoinverse[0] * cross01 +
        input_basis.pseudoinverse[1] * cross11;
    stats.transfer[2] = input_basis.pseudoinverse[2] * cross00 +
        input_basis.pseudoinverse[3] * cross10;
    stats.transfer[3] = input_basis.pseudoinverse[2] * cross01 +
        input_basis.pseudoinverse[3] * cross11;
    double input_projector[4];
    input_projector[0] = input_basis.pseudoinverse[0] * b00 +
        input_basis.pseudoinverse[1] * b01;
    input_projector[1] = input_basis.pseudoinverse[0] * b01 +
        input_basis.pseudoinverse[1] * b11;
    input_projector[2] = input_basis.pseudoinverse[2] * b00 +
        input_basis.pseudoinverse[3] * b01;
    input_projector[3] = input_basis.pseudoinverse[2] * b01 +
        input_basis.pseudoinverse[3] * b11;
    stats.transfer_fixed_defect = sqrt(
        (stats.transfer[0] - input_projector[0]) *
            (stats.transfer[0] - input_projector[0]) +
        (stats.transfer[1] - input_projector[1]) *
            (stats.transfer[1] - input_projector[1]) +
        (stats.transfer[2] - input_projector[2]) *
            (stats.transfer[2] - input_projector[2]) +
        (stats.transfer[3] - input_projector[3]) *
            (stats.transfer[3] - input_projector[3])
    );

    double residual_square = 0.0;
    double output_square = output00 + output11;
    for (int lane = 0; lane < width; lane++) {
        double in_origin = input_origin == NULL ? 0.0 : input_origin[lane];
        double out_origin = output_origin == NULL ? 0.0 : output_origin[lane];
        double in_a = (double)input_a_endpoint[lane] - in_origin;
        double in_b = (double)input_b_endpoint[lane] - in_origin;
        double out_a = (double)output_a_endpoint[lane] - out_origin;
        double out_b = (double)output_b_endpoint[lane] - out_origin;
        double predicted_a = in_a * stats.transfer[0] +
            in_b * stats.transfer[2];
        double predicted_b = in_a * stats.transfer[1] +
            in_b * stats.transfer[3];
        double residual_a = out_a - predicted_a;
        double residual_b = out_b - predicted_b;
        residual_square += residual_a * residual_a + residual_b * residual_b;
    }
    stats.subspace_escape = output_square == 0.0 ? 0.0 :
        sqrt(residual_square / output_square);
    stats.edit_a_gain = !stats.edit_a_defined ? 0.0 : sqrt(output00 / b00);
    stats.edit_b_gain = !stats.edit_b_defined ? 0.0 : sqrt(output11 / b11);
    stats.edit_a_alignment = !stats.edit_a_defined || output00 == 0.0 ? 0.0 :
        in_a_out_a / sqrt(b00 * output00);
    stats.edit_b_alignment = !stats.edit_b_defined || output11 == 0.0 ? 0.0 :
        in_b_out_b / sqrt(b11 * output11);
    stats.edit_a_fixed_defect = !stats.edit_a_defined ? 0.0 :
        sqrt(error00 / b00);
    stats.edit_b_fixed_defect = !stats.edit_b_defined ? 0.0 :
        sqrt(error11 / b11);
    stats.closest_fixed_defect = sqrt(fixed.value);
    stats.closest_fixed_a = fixed.coefficient_a;
    stats.closest_fixed_b = fixed.coefficient_b;
    stats.closest_discarded_gain = sqrt(discarded.value);
    stats.closest_discarded_a = discarded.coefficient_a;
    stats.closest_discarded_b = discarded.coefficient_b;
    return stats;
}

static TransitionStats measure_secant_transition(
    const float *const input[4],
    const float *const output[4],
    int width
) {
    return measure_direction_map(
        input[0], input[1], input[2],
        output[0], output[1], output[2],
        width
    );
}

static TransitionStats measure_tangent_transition(
    const float *input_a,
    const float *input_b,
    const float *output_a,
    const float *output_b,
    int width
) {
    return measure_direction_map(
        NULL, input_a, input_b,
        NULL, output_a, output_b,
        width
    );
}

static void stage_corners(
    ContextCapture captures[4],
    int layer,
    Stage stage,
    const float *values[4],
    int *width
) {
    *width = captures[0].layers[layer].stages[stage].width;
    for (int context = 0; context < 4; context++) {
        StageBuffer *buffer = &captures[context].layers[layer].stages[stage];
        if (buffer->width != *width) fail("context capture width mismatch");
        values[context] = buffer->values;
    }
}

static void stage_tangents(
    ContextCapture captures[2],
    int layer,
    Stage stage,
    const float *values[2],
    int *width
) {
    *width = captures[0].layers[layer].stages[stage].width;
    for (int direction = 0; direction < 2; direction++) {
        StageBuffer *buffer = &captures[direction].layers[layer].stages[stage];
        if (buffer->width != *width) fail("tangent capture width mismatch");
        values[direction] = buffer->values;
    }
}

static void write_context_record(
    FILE *trace,
    Tokenizer *tokenizer,
    const EncodedContext *context,
    int index
) {
    if (trace == NULL) return;
    fprintf(trace, "{\"kind\":\"context\",\"index\":%d,\"text\":", index);
    fprint_json_string(trace, context->text);
    fputs(",\"tokens\":[", trace);
    for (int position = 0; position < context->count; position++) {
        if (position != 0) fputc(',', trace);
        int previous = position == 0 ? 0 : context->tokens[position - 1];
        int token = context->tokens[position];
        fprintf(trace, "{\"position\":%d,\"id\":%d,\"piece\":", position, token);
        fprint_json_string(trace, decode(tokenizer, previous, token));
        fputc('}', trace);
    }
    fputs("]}\n", trace);
    fflush(trace);
}

static void write_stage_record(
    FILE *trace,
    int layer,
    Stage stage,
    const CornerStats *stats
) {
    if (trace == NULL) return;
    fprintf(
        trace,
        "{\"kind\":\"stage\",\"layer\":%d,\"stage\":\"%s\","
        "\"edit_a_norm\":%.17g,\"edit_b_norm\":%.17g,"
        "\"edit_cosine\":%.17g,\"interaction_norm\":%.17g,"
        "\"interaction_fraction\":%.17g}\n",
        layer,
        stage_names[stage],
        stats->edit_a_norm,
        stats->edit_b_norm,
        stats->edit_cosine,
        stats->interaction_norm,
        stats->interaction_fraction
    );
    fflush(trace);
}

static void write_tangent_stage_record(
    FILE *trace,
    int layer,
    Stage stage,
    const DirectionStats *stats
) {
    if (trace == NULL) return;
    fprintf(
        trace,
        "{\"kind\":\"tangent_stage\",\"layer\":%d,\"stage\":\"%s\","
        "\"edit_a_norm\":%.17g,\"edit_b_norm\":%.17g,"
        "\"edit_cosine\":%.17g}\n",
        layer,
        stage_names[stage],
        stats->edit_a_norm,
        stats->edit_b_norm,
        stats->edit_cosine
    );
    fflush(trace);
}

static void write_transition_record(
    FILE *trace,
    const char *kind,
    int layer,
    const char *operation,
    const TransitionStats *stats
) {
    if (trace == NULL) return;
    fprintf(
        trace,
        "{\"kind\":\"%s\",\"layer\":%d,"
        "\"operation\":\"%s\",\"input_rank\":%d,"
        "\"output_rank\":%d,\"rank_change\":%d,"
        "\"edit_a_defined\":%s,\"edit_b_defined\":%s,"
        "\"modes_defined\":%s,"
        "\"edit_a_gain\":%.17g,"
        "\"edit_b_gain\":%.17g,\"edit_a_alignment\":%.17g,"
        "\"edit_b_alignment\":%.17g,\"edit_a_fixed_defect\":%.17g,"
        "\"edit_b_fixed_defect\":%.17g,"
        "\"closest_fixed_defect\":%.17g,"
        "\"closest_fixed_coefficients\":[%.17g,%.17g],"
        "\"closest_discarded_gain\":%.17g,"
        "\"closest_discarded_coefficients\":[%.17g,%.17g],"
        "\"transfer\":[[%.17g,%.17g],[%.17g,%.17g]],"
        "\"transfer_fixed_defect\":%.17g,"
        "\"subspace_escape\":%.17g}\n",
        kind,
        layer,
        operation,
        stats->input_rank,
        stats->output_rank,
        stats->output_rank - stats->input_rank,
        stats->edit_a_defined ? "true" : "false",
        stats->edit_b_defined ? "true" : "false",
        stats->modes_defined ? "true" : "false",
        stats->edit_a_gain,
        stats->edit_b_gain,
        stats->edit_a_alignment,
        stats->edit_b_alignment,
        stats->edit_a_fixed_defect,
        stats->edit_b_fixed_defect,
        stats->closest_fixed_defect,
        stats->closest_fixed_a,
        stats->closest_fixed_b,
        stats->closest_discarded_gain,
        stats->closest_discarded_a,
        stats->closest_discarded_b,
        stats->transfer[0],
        stats->transfer[1],
        stats->transfer[2],
        stats->transfer[3],
        stats->transfer_fixed_defect,
        stats->subspace_escape
    );
    fflush(trace);
}

static TransitionStats report_secant_transition(
    FILE *trace,
    ContextCapture captures[4],
    int layer,
    const char *name,
    Stage input_stage,
    Stage output_stage
) {
    const float *input[4];
    const float *output[4];
    int input_width;
    int output_width;
    stage_corners(captures, layer, input_stage, input, &input_width);
    stage_corners(captures, layer, output_stage, output, &output_width);
    if (input_width != output_width) fail("fixed transition changed width");
    TransitionStats stats = measure_secant_transition(
        input, output, input_width
    );
    write_transition_record(trace, "secant_transition", layer, name, &stats);
    printf(
        "  secant %-12s rank=%d->%d fixed=%.6g "
        "mode=(%+.4g A,%+.4g B) "
        "discard=%.6g escape=%.6g\n",
        name,
        stats.input_rank,
        stats.output_rank,
        stats.closest_fixed_defect,
        stats.closest_fixed_a,
        stats.closest_fixed_b,
        stats.closest_discarded_gain,
        stats.subspace_escape
    );
    return stats;
}

static TransitionStats report_tangent_transition(
    FILE *trace,
    ContextCapture captures[2],
    int layer,
    const char *name,
    Stage input_stage,
    Stage output_stage
) {
    const float *input[2];
    const float *output[2];
    int input_width;
    int output_width;
    stage_tangents(captures, layer, input_stage, input, &input_width);
    stage_tangents(captures, layer, output_stage, output, &output_width);
    if (input_width != output_width) fail("tangent transition changed width");
    TransitionStats stats = measure_tangent_transition(
        input[0], input[1], output[0], output[1], input_width
    );
    write_transition_record(trace, "tangent_transition", layer, name, &stats);
    printf(
        "  jvp     %-12s rank=%d->%d fixed=%.6g "
        "mode=(%+.4g A,%+.4g B) discard=%.6g escape=%.6g\n",
        name,
        stats.input_rank,
        stats.output_rank,
        stats.closest_fixed_defect,
        stats.closest_fixed_a,
        stats.closest_fixed_b,
        stats.closest_discarded_gain,
        stats.subspace_escape
    );
    return stats;
}

static void report_attention_relations(
    FILE *trace,
    ContextCapture captures[4],
    const Config *config,
    int layer,
    int observe_position,
    int factor_a_position,
    int factor_b_position
) {
    int stride = observe_position + 1;
    double largest = -1.0;
    int largest_head = -1;
    int largest_position = -1;
    double largest_score = 0.0;
    double largest_probability = 0.0;
    int positions[2] = {factor_a_position, factor_b_position};
    const char *factor_names[2] = {"A", "B"};
    for (int head = 0; head < config->n_heads; head++) {
        for (int factor = 0; factor < 2; factor++) {
            int key_position = positions[factor];
            size_t offset = (size_t)head * stride + key_position;
            double scores[4];
            double probabilities[4];
            for (int context = 0; context < 4; context++) {
                scores[context] = captures[context].layers[layer]
                    .stages[STAGE_ATTENTION_SCORES].values[offset];
                probabilities[context] = captures[context].layers[layer]
                    .stages[STAGE_ATTENTION_PROBABILITIES].values[offset];
            }
            double mixed_score = scores[3] - scores[1] - scores[2] + scores[0];
            double mixed_probability = probabilities[3] - probabilities[1] -
                probabilities[2] + probabilities[0];
            if (trace != NULL) {
                fprintf(
                    trace,
                    "{\"kind\":\"attention_relation\",\"layer\":%d,"
                    "\"head\":%d,\"query_position\":%d,"
                    "\"key_position\":%d,\"key_factor\":\"%s\","
                    "\"scores\":[%.17g,%.17g,%.17g,%.17g],"
                    "\"probabilities\":[%.17g,%.17g,%.17g,%.17g],"
                    "\"mixed_score\":%.17g,"
                    "\"mixed_probability\":%.17g}\n",
                    layer,
                    head,
                    observe_position,
                    key_position,
                    factor_names[factor],
                    scores[0], scores[1], scores[2], scores[3],
                    probabilities[0], probabilities[1],
                    probabilities[2], probabilities[3],
                    mixed_score,
                    mixed_probability
                );
                fflush(trace);
            }
            double magnitude = fabs(mixed_score);
            if (magnitude > largest) {
                largest = magnitude;
                largest_head = head;
                largest_position = key_position;
                largest_score = mixed_score;
                largest_probability = mixed_probability;
            }
        }
    }
    printf(
        "  strongest_qk_relation head=%d key_position=%d "
        "mixed_score=%+.8g mixed_probability=%+.8g\n",
        largest_head,
        largest_position,
        largest_score,
        largest_probability
    );
}

static int classify_factor_positions(
    const EncodedContext contexts[4],
    int *factor_a_position,
    int *factor_b_position
) {
    int count = contexts[0].count;
    for (int context = 1; context < 4; context++) {
        if (contexts[context].count != count) {
            for (int shown = 0; shown < 4; shown++) {
                fprintf(
                    stderr,
                    "context[%d] token_count=%d text=%s\n",
                    shown,
                    contexts[shown].count,
                    contexts[shown].text
                );
            }
            fail("semantic-square texts did not tokenize to equal lengths");
        }
    }
    int a_count = 0;
    int b_count = 0;
    for (int position = 0; position < count; position++) {
        int t00 = contexts[0].tokens[position];
        int t10 = contexts[1].tokens[position];
        int t01 = contexts[2].tokens[position];
        int t11 = contexts[3].tokens[position];
        if (t00 == t10 && t00 == t01 && t00 == t11) continue;
        if (t00 == t01 && t10 == t11 && t00 != t10) {
            *factor_a_position = position;
            a_count++;
        } else if (t00 == t10 && t01 == t11 && t00 != t01) {
            *factor_b_position = position;
            b_count++;
        } else {
            fprintf(
                stderr,
                "token position %d is not an independent A/B edit: "
                "%d %d %d %d\n",
                position,
                t00,
                t10,
                t01,
                t11
            );
            fail("semantic square does not commute at the token-constructor level");
        }
    }
    if (a_count != 1 || b_count != 1) {
        fail("semantic square needs exactly one token position for each edit");
    }
    return count;
}

static void print_factor_piece(
    Tokenizer *tokenizer,
    const EncodedContext *left,
    const EncodedContext *right,
    int position,
    const char *name
) {
    int left_previous = position == 0 ? 0 : left->tokens[position - 1];
    int right_previous = position == 0 ? 0 : right->tokens[position - 1];
    printf("edit_%s position=%d ", name, position);
    fprint_json_string(
        stdout,
        decode(tokenizer, left_previous, left->tokens[position])
    );
    fputs(" -> ", stdout);
    fprint_json_string(
        stdout,
        decode(tokenizer, right_previous, right->tokens[position])
    );
    putchar('\n');
}

int main(int argc, char **argv) {
    Options options = parse_options(argc, argv);
    Transformer transformer;
    build_transformer(&transformer, argv[1]);
    Tokenizer tokenizer;
    build_tokenizer(
        &tokenizer,
        argv[2],
        transformer.config.vocab_size
    );

    EncodedContext contexts[4];
    for (int context = 0; context < 4; context++) {
        contexts[context] = encode_context(&tokenizer, argv[3 + context]);
    }
    int factor_a_position = -1;
    int factor_b_position = -1;
    int token_count = classify_factor_positions(
        contexts,
        &factor_a_position,
        &factor_b_position
    );
    int later_factor = factor_a_position > factor_b_position ?
        factor_a_position : factor_b_position;
    int observe_position = options.observe_position < 0 ?
        later_factor : options.observe_position;
    if (observe_position < later_factor || observe_position >= token_count) {
        fail("observe position must contain both edits and remain in the context");
    }
    if (observe_position >= transformer.config.seq_len) {
        fail("observe position exceeds the model context length");
    }

    FILE *trace = NULL;
    if (options.trace_path != NULL) {
        trace = fopen(options.trace_path, "wb");
        if (trace == NULL) fail("could not create trace file");
    }
    if (trace != NULL) {
        fprintf(
            trace,
            "{\"kind\":\"meta\",\"schema_version\":1,"
            "\"tangent_base_context\":0,"
            "\"tangent_seed\":\"embedding_arrows\","
            "\"layers\":%d,\"dim\":%d,"
            "\"hidden_dim\":%d,\"heads\":%d,\"token_count\":%d,"
            "\"factor_a_position\":%d,\"factor_b_position\":%d,"
            "\"observe_position\":%d}\n",
            transformer.config.n_layers,
            transformer.config.dim,
            transformer.config.hidden_dim,
            transformer.config.n_heads,
            token_count,
            factor_a_position,
            factor_b_position,
            observe_position
        );
        fflush(trace);
        for (int context = 0; context < 4; context++) {
            write_context_record(trace, &tokenizer, &contexts[context], context);
        }
    }

    printf(
        "model layers=%d dim=%d hidden=%d heads=%d tokens=%d observe=%d\n",
        transformer.config.n_layers,
        transformer.config.dim,
        transformer.config.hidden_dim,
        transformer.config.n_heads,
        token_count,
        observe_position
    );
    print_factor_piece(
        &tokenizer,
        &contexts[0],
        &contexts[1],
        factor_a_position,
        "A"
    );
    print_factor_piece(
        &tokenizer,
        &contexts[0],
        &contexts[2],
        factor_b_position,
        "B"
    );

    ContextCapture captures[4];
    for (int context = 0; context < 4; context++) {
        captures[context] = allocate_capture(
            &transformer.config,
            observe_position
        );
    }
    ContextCapture tangent_captures[2];
    RunState tangent_states[2] = {0};
    for (int direction = 0; direction < 2; direction++) {
        tangent_captures[direction] = allocate_capture(
            &transformer.config,
            observe_position
        );
        malloc_run_state(&tangent_states[direction], &transformer.config);
    }
    capture_context_with_tangents(
        &transformer,
        tangent_states,
        &contexts[0],
        &contexts[1],
        &contexts[2],
        observe_position,
        &captures[0],
        tangent_captures
    );
    for (int context = 1; context < 4; context++) {
        capture_context(
            &transformer,
            &contexts[context],
            observe_position,
            &captures[context]
        );
    }

    for (int layer = 0; layer < transformer.config.n_layers; layer++) {
        printf("layer=%d\n", layer);
        for (int stage = 0; stage < STAGE_COUNT; stage++) {
            const float *values[4];
            int width;
            stage_corners(captures, layer, (Stage)stage, values, &width);
            CornerStats stats = measure_corners(values, width);
            write_stage_record(trace, layer, (Stage)stage, &stats);
            const float *tangent_values[2];
            int tangent_width;
            stage_tangents(
                tangent_captures,
                layer,
                (Stage)stage,
                tangent_values,
                &tangent_width
            );
            if (tangent_width != width) fail("stage tangent width changed");
            DirectionStats tangent_stats = measure_directions(
                tangent_values[0], tangent_values[1], tangent_width
            );
            write_tangent_stage_record(
                trace, layer, (Stage)stage, &tangent_stats
            );
            if (stage == STAGE_LAYER_INPUT ||
                stage == STAGE_ATTENTION_SCORES ||
                stage == STAGE_ATTENTION_PROBABILITIES ||
                stage == STAGE_POST_ATTENTION ||
                stage == STAGE_SWIGLU_PRODUCT ||
                stage == STAGE_LAYER_OUTPUT) {
                printf(
                    "  %-24s interaction=%.8g fraction=%.8g\n",
                    stage_names[stage],
                    stats.interaction_norm,
                    stats.interaction_fraction
                );
            }
        }
        report_attention_relations(
            trace,
            captures,
            &transformer.config,
            layer,
            observe_position,
            factor_a_position,
            factor_b_position
        );
        const struct {
            const char *name;
            Stage input;
            Stage output;
        } transitions[] = {
            {"attention_rms", STAGE_LAYER_INPUT, STAGE_ATTENTION_RMS},
            {"attention_residual", STAGE_LAYER_INPUT, STAGE_POST_ATTENTION},
            {"ffn_rms", STAGE_POST_ATTENTION, STAGE_FFN_RMS},
            {"swiglu_residual", STAGE_POST_ATTENTION, STAGE_LAYER_OUTPUT},
            {"whole_layer", STAGE_LAYER_INPUT, STAGE_LAYER_OUTPUT}
        };
        int transition_count =
            (int)(sizeof(transitions) / sizeof(transitions[0]));
        for (int transition = 0;
             transition < transition_count;
             transition++) {
            report_secant_transition(
                trace,
                captures,
                layer,
                transitions[transition].name,
                transitions[transition].input,
                transitions[transition].output
            );
            report_tangent_transition(
                trace,
                tangent_captures,
                layer,
                transitions[transition].name,
                transitions[transition].input,
                transitions[transition].output
            );
        }
        fflush(stdout);
    }

    if (trace != NULL && fclose(trace) != 0) fail("could not close trace file");
    for (int context = 0; context < 4; context++) {
        free_capture(&captures[context], transformer.config.n_layers);
        free_context(&contexts[context]);
    }
    for (int direction = 0; direction < 2; direction++) {
        free_capture(
            &tangent_captures[direction],
            transformer.config.n_layers
        );
        free_run_state(&tangent_states[direction]);
    }
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return EXIT_SUCCESS;
}
