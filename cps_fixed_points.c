/*
 * Exact sampled continuation pullbacks for the frozen llama2.c transformer.
 *
 * The CPS term deliberately has no classifier, logits, completion reward, or
 * linguistic parse.  Four token-constructor variants supply two commuting
 * edits.  The transformer is evaluated as a layer-frontier term over every
 * token position at once.  Its continuation IR is the literal CPS law
 *
 *     pullback(F, k)(x) = k(F(x)).
 *
 * The root continuation is identity on the complete final hidden frontier.
 * Suffix continuations are generated only by repeated pullback through the
 * actual attention, SwiGLU, and layer maps.  A final parity check invokes the
 * stock forward() schedule and reads only its post-final-RMS state; the logits
 * that forward() also computes are discarded and never enter this term.
 */

#define TESTING
#include "run.c"

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <stdbool.h>

typedef struct {
    const char *text;
    int *tokens;
    int count;
} EncodedContext;

typedef struct {
    const char *trace_path;
} Options;

typedef struct {
    int positions;
    int dim;
    int hidden_dim;
    int kv_dim;
    int heads;
    float *attention_norm;
    float *query;
    float *key;
    float *value;
    float *probability;
    float *value_sum;
    float *attention_update;
    float *post_attention;
    float *ffn_norm;
    float *w1;
    float *w3;
    float *silu;
    float *product;
    float *ffn_update;
    float *layer_output;
    float *temporary_frontier;
} LayerWorkspace;

typedef struct {
    Transformer *transformer;
    int layer;
    int positions;
    LayerWorkspace workspace;
} LayerRuntime;

typedef struct {
    Transformer *transformer;
    int positions;
} FinalRmsRuntime;

typedef void (*FrontierMapApply)(
    void *environment,
    const float *input,
    float *output
);

typedef struct {
    const char *name;
    int input_width;
    int output_width;
    FrontierMapApply apply;
    void *environment;
} FrontierMap;

typedef void (*ContinuationApply)(
    void *environment,
    const float *input,
    float *result
);

typedef struct {
    int input_width;
    int result_width;
    ContinuationApply apply;
    void *environment;
} Continuation;

typedef struct {
    FrontierMap map;
    Continuation next;
    float *mapped;
} PullbackEnvironment;

typedef struct {
    int layers;
    int width;
    float *layer_frontiers;
    float *post_attention_frontiers;
} ContextFrontiers;

typedef struct {
    double local_interaction_l2;
    double visible_interaction_l2;
    double visible_interaction_relative;
    double prior_interaction_l2;
} InteractionMeasurement;

typedef struct {
    double defects[4];
    double relative_defects[4];
    double maximum_defect;
    double maximum_relative_defect;
} FixedMeasurement;

static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

static void *checked_calloc(size_t count, size_t width) {
    void *memory = calloc(count, width);
    if (memory == NULL) fail("allocation failed");
    return memory;
}

static Options parse_options(int argc, char **argv) {
    if (argc < 7) {
        fprintf(
            stderr,
            "usage: %s CHECKPOINT TOKENIZER TEXT00 TEXT10 TEXT01 TEXT11 "
            "[--trace PATH]\n",
            argv[0]
        );
        exit(EXIT_FAILURE);
    }
    Options options = {0};
    for (int index = 7; index < argc;) {
        if (strcmp(argv[index], "--trace") == 0 && index + 1 < argc) {
            options.trace_path = argv[index + 1];
            index += 2;
        } else {
            fail("unrecognized cps_fixed_points option");
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

static EncodedContext encode_context(Tokenizer *tokenizer, const char *text) {
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

static int classify_factor_positions(
    const EncodedContext contexts[4],
    int *factor_a_position,
    int *factor_b_position
) {
    int count = contexts[0].count;
    for (int context = 1; context < 4; context++) {
        if (contexts[context].count != count) {
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
            fail("semantic square does not commute at token constructors");
        }
    }
    if (a_count != 1 || b_count != 1) {
        fail("semantic square needs exactly one token position per edit");
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

static LayerWorkspace allocate_workspace(const Config *config, int positions) {
    LayerWorkspace workspace = {
        .positions = positions,
        .dim = config->dim,
        .hidden_dim = config->hidden_dim,
        .kv_dim = config->dim * config->n_kv_heads / config->n_heads,
        .heads = config->n_heads
    };
    size_t frontier = (size_t)positions * config->dim;
    size_t kv_frontier = (size_t)positions * workspace.kv_dim;
    size_t attention = (size_t)positions * config->n_heads * positions;
    size_t hidden = (size_t)positions * config->hidden_dim;
    workspace.attention_norm = checked_calloc(frontier, sizeof(float));
    workspace.query = checked_calloc(frontier, sizeof(float));
    workspace.key = checked_calloc(kv_frontier, sizeof(float));
    workspace.value = checked_calloc(kv_frontier, sizeof(float));
    workspace.probability = checked_calloc(attention, sizeof(float));
    workspace.value_sum = checked_calloc(frontier, sizeof(float));
    workspace.attention_update = checked_calloc(frontier, sizeof(float));
    workspace.post_attention = checked_calloc(frontier, sizeof(float));
    workspace.ffn_norm = checked_calloc(frontier, sizeof(float));
    workspace.w1 = checked_calloc(hidden, sizeof(float));
    workspace.w3 = checked_calloc(hidden, sizeof(float));
    workspace.silu = checked_calloc(hidden, sizeof(float));
    workspace.product = checked_calloc(hidden, sizeof(float));
    workspace.ffn_update = checked_calloc(frontier, sizeof(float));
    workspace.layer_output = checked_calloc(frontier, sizeof(float));
    workspace.temporary_frontier = checked_calloc(frontier, sizeof(float));
    return workspace;
}

static void free_workspace(LayerWorkspace *workspace) {
    free(workspace->attention_norm);
    free(workspace->query);
    free(workspace->key);
    free(workspace->value);
    free(workspace->probability);
    free(workspace->value_sum);
    free(workspace->attention_update);
    free(workspace->post_attention);
    free(workspace->ffn_norm);
    free(workspace->w1);
    free(workspace->w3);
    free(workspace->silu);
    free(workspace->product);
    free(workspace->ffn_update);
    free(workspace->layer_output);
    free(workspace->temporary_frontier);
    memset(workspace, 0, sizeof(*workspace));
}

static void apply_rope(
    float *query,
    float *key,
    int position,
    int dim,
    int kv_dim,
    int head_size
) {
    for (int index = 0; index < dim; index += 2) {
        int head_dimension = index % head_size;
        float frequency = 1.0f /
            powf(10000.0f, head_dimension / (float)head_size);
        float angle = position * frequency;
        float real = cosf(angle);
        float imaginary = sinf(angle);
        float q0 = query[index];
        float q1 = query[index + 1];
        query[index] = q0 * real - q1 * imaginary;
        query[index + 1] = q0 * imaginary + q1 * real;
        if (index < kv_dim) {
            float k0 = key[index];
            float k1 = key[index + 1];
            key[index] = k0 * real - k1 * imaginary;
            key[index + 1] = k0 * imaginary + k1 * real;
        }
    }
}

static void attention_frontier(
    LayerRuntime *runtime,
    const float *input,
    float *output
) {
    Transformer *transformer = runtime->transformer;
    Config *config = &transformer->config;
    TransformerWeights *weights = &transformer->weights;
    LayerWorkspace *workspace = &runtime->workspace;
    int layer = runtime->layer;
    int positions = runtime->positions;
    int dim = config->dim;
    int kv_dim = workspace->kv_dim;
    int head_size = dim / config->n_heads;
    int kv_mul = config->n_heads / config->n_kv_heads;
    float *attention_weight = weights->rms_att_weight + (size_t)layer * dim;
    float *wq = weights->wq + (size_t)layer * dim * dim;
    float *wk = weights->wk + (size_t)layer * dim * kv_dim;
    float *wv = weights->wv + (size_t)layer * dim * kv_dim;
    float *wo = weights->wo + (size_t)layer * dim * dim;

    for (int position = 0; position < positions; position++) {
        const float *x = input + (size_t)position * dim;
        float *normalized = workspace->attention_norm + (size_t)position * dim;
        float *query = workspace->query + (size_t)position * dim;
        float *key = workspace->key + (size_t)position * kv_dim;
        float *value = workspace->value + (size_t)position * kv_dim;
        rmsnorm(normalized, (float *)x, attention_weight, dim);
        matmul(query, normalized, wq, dim, dim);
        matmul(key, normalized, wk, dim, kv_dim);
        matmul(value, normalized, wv, dim, kv_dim);
        apply_rope(query, key, position, dim, kv_dim, head_size);
    }

    float score_scale = 1.0f / sqrtf((float)head_size);
    for (int position = 0; position < positions; position++) {
        float *position_output =
            workspace->value_sum + (size_t)position * dim;
        memset(position_output, 0, (size_t)dim * sizeof(*position_output));
        for (int head = 0; head < config->n_heads; head++) {
            float *probability = workspace->probability +
                ((size_t)position * config->n_heads + head) * positions;
            const float *query = workspace->query +
                (size_t)position * dim + head * head_size;
            for (int key_position = 0;
                 key_position <= position;
                 key_position++) {
                const float *key = workspace->key +
                    (size_t)key_position * kv_dim +
                    (head / kv_mul) * head_size;
                float score = 0.0f;
                for (int lane = 0; lane < head_size; lane++) {
                    score += query[lane] * key[lane];
                }
                probability[key_position] = score * score_scale;
            }
            softmax(probability, position + 1);
            float *head_output = position_output + head * head_size;
            for (int key_position = 0;
                 key_position <= position;
                 key_position++) {
                const float *value = workspace->value +
                    (size_t)key_position * kv_dim +
                    (head / kv_mul) * head_size;
                float mass = probability[key_position];
                for (int lane = 0; lane < head_size; lane++) {
                    head_output[lane] += mass * value[lane];
                }
            }
        }
        float *update = workspace->attention_update + (size_t)position * dim;
        matmul(update, position_output, wo, dim, dim);
        float *post = workspace->post_attention + (size_t)position * dim;
        const float *x = input + (size_t)position * dim;
        for (int lane = 0; lane < dim; lane++) {
            post[lane] = x[lane] + update[lane];
        }
    }
    memcpy(
        output,
        workspace->post_attention,
        (size_t)positions * dim * sizeof(*output)
    );
}

static void ffn_frontier(
    LayerRuntime *runtime,
    const float *input,
    float *output
) {
    Transformer *transformer = runtime->transformer;
    Config *config = &transformer->config;
    TransformerWeights *weights = &transformer->weights;
    LayerWorkspace *workspace = &runtime->workspace;
    int layer = runtime->layer;
    int positions = runtime->positions;
    int dim = config->dim;
    int hidden_dim = config->hidden_dim;
    float *norm_weight = weights->rms_ffn_weight + (size_t)layer * dim;
    float *w1 = weights->w1 + (size_t)layer * dim * hidden_dim;
    float *w2 = weights->w2 + (size_t)layer * dim * hidden_dim;
    float *w3 = weights->w3 + (size_t)layer * dim * hidden_dim;

    for (int position = 0; position < positions; position++) {
        const float *x = input + (size_t)position * dim;
        float *normalized = workspace->ffn_norm + (size_t)position * dim;
        float *a = workspace->w1 + (size_t)position * hidden_dim;
        float *b = workspace->w3 + (size_t)position * hidden_dim;
        float *silu = workspace->silu + (size_t)position * hidden_dim;
        float *product = workspace->product + (size_t)position * hidden_dim;
        float *update = workspace->ffn_update + (size_t)position * dim;
        float *y = workspace->layer_output + (size_t)position * dim;
        rmsnorm(normalized, (float *)x, norm_weight, dim);
        matmul(a, normalized, w1, dim, hidden_dim);
        matmul(b, normalized, w3, dim, hidden_dim);
        for (int lane = 0; lane < hidden_dim; lane++) {
            float sigmoid = 1.0f / (1.0f + expf(-a[lane]));
            silu[lane] = a[lane] * sigmoid;
            product[lane] = silu[lane] * b[lane];
        }
        matmul(update, product, w2, hidden_dim, dim);
        for (int lane = 0; lane < dim; lane++) {
            y[lane] = x[lane] + update[lane];
        }
    }
    memcpy(
        output,
        workspace->layer_output,
        (size_t)positions * dim * sizeof(*output)
    );
}

static void attention_map_apply(
    void *environment,
    const float *input,
    float *output
) {
    attention_frontier(environment, input, output);
}

static void ffn_map_apply(
    void *environment,
    const float *input,
    float *output
) {
    ffn_frontier(environment, input, output);
}

static void layer_map_apply(
    void *environment,
    const float *input,
    float *output
) {
    LayerRuntime *runtime = environment;
    attention_frontier(runtime, input, runtime->workspace.temporary_frontier);
    ffn_frontier(runtime, runtime->workspace.temporary_frontier, output);
}

static int frontier_width_for(const LayerRuntime *runtime) {
    return runtime->positions * runtime->transformer->config.dim;
}

static int kv_frontier_width_for(const LayerRuntime *runtime) {
    int kv_dim = runtime->transformer->config.dim *
        runtime->transformer->config.n_kv_heads /
        runtime->transformer->config.n_heads;
    return runtime->positions * kv_dim;
}

static int attention_table_width_for(const LayerRuntime *runtime) {
    return runtime->positions * runtime->transformer->config.n_heads *
        runtime->positions;
}

static int attention_norm_state_width(const LayerRuntime *runtime) {
    return 2 * frontier_width_for(runtime);
}

static int qkv_state_width(const LayerRuntime *runtime) {
    return 2 * frontier_width_for(runtime) +
        2 * kv_frontier_width_for(runtime);
}

static int score_state_width(const LayerRuntime *runtime) {
    return frontier_width_for(runtime) + kv_frontier_width_for(runtime) +
        attention_table_width_for(runtime);
}

static int residual_pair_width(const LayerRuntime *runtime) {
    return 2 * frontier_width_for(runtime);
}

static void attention_norm_stage_apply(
    void *environment,
    const float *input,
    float *output
) {
    LayerRuntime *runtime = environment;
    Config *config = &runtime->transformer->config;
    int dim = config->dim;
    int frontier_width = frontier_width_for(runtime);
    memcpy(output, input, (size_t)frontier_width * sizeof(*output));
    float *normalized_frontier = output + frontier_width;
    float *weight = runtime->transformer->weights.rms_att_weight +
        (size_t)runtime->layer * dim;
    for (int position = 0; position < runtime->positions; position++) {
        rmsnorm(
            normalized_frontier + (size_t)position * dim,
            (float *)input + (size_t)position * dim,
            weight,
            dim
        );
    }
}

static void qkv_stage_apply(
    void *environment,
    const float *input,
    float *output
) {
    LayerRuntime *runtime = environment;
    Transformer *transformer = runtime->transformer;
    Config *config = &transformer->config;
    TransformerWeights *weights = &transformer->weights;
    int dim = config->dim;
    int kv_dim = dim * config->n_kv_heads / config->n_heads;
    int frontier_width = frontier_width_for(runtime);
    int kv_frontier_width = kv_frontier_width_for(runtime);
    int head_size = dim / config->n_heads;
    const float *x = input;
    const float *normalized = input + frontier_width;
    float *out_x = output;
    float *query = output + frontier_width;
    float *key = query + frontier_width;
    float *value = key + kv_frontier_width;
    memcpy(out_x, x, (size_t)frontier_width * sizeof(*out_x));
    float *wq = weights->wq + (size_t)runtime->layer * dim * dim;
    float *wk = weights->wk + (size_t)runtime->layer * dim * kv_dim;
    float *wv = weights->wv + (size_t)runtime->layer * dim * kv_dim;
    for (int position = 0; position < runtime->positions; position++) {
        const float *n = normalized + (size_t)position * dim;
        float *q = query + (size_t)position * dim;
        float *k = key + (size_t)position * kv_dim;
        float *v = value + (size_t)position * kv_dim;
        matmul(q, (float *)n, wq, dim, dim);
        matmul(k, (float *)n, wk, dim, kv_dim);
        matmul(v, (float *)n, wv, dim, kv_dim);
        apply_rope(q, k, position, dim, kv_dim, head_size);
    }
}

static void qk_stage_apply(
    void *environment,
    const float *input,
    float *output
) {
    LayerRuntime *runtime = environment;
    Config *config = &runtime->transformer->config;
    int positions = runtime->positions;
    int dim = config->dim;
    int kv_dim = dim * config->n_kv_heads / config->n_heads;
    int kv_mul = config->n_heads / config->n_kv_heads;
    int head_size = dim / config->n_heads;
    int frontier_width = frontier_width_for(runtime);
    int kv_frontier_width = kv_frontier_width_for(runtime);
    int table_width = attention_table_width_for(runtime);
    const float *x = input;
    const float *query = input + frontier_width;
    const float *key = query + frontier_width;
    const float *value = key + kv_frontier_width;
    float *out_x = output;
    float *out_value = output + frontier_width;
    float *scores = out_value + kv_frontier_width;
    memcpy(out_x, x, (size_t)frontier_width * sizeof(*out_x));
    memcpy(
        out_value,
        value,
        (size_t)kv_frontier_width * sizeof(*out_value)
    );
    memset(scores, 0, (size_t)table_width * sizeof(*scores));
    float scale = 1.0f / sqrtf((float)head_size);
    for (int position = 0; position < positions; position++) {
        for (int head = 0; head < config->n_heads; head++) {
            const float *q = query + (size_t)position * dim +
                head * head_size;
            float *row = scores +
                ((size_t)position * config->n_heads + head) * positions;
            for (int key_position = 0;
                 key_position <= position;
                 key_position++) {
                const float *k = key + (size_t)key_position * kv_dim +
                    (head / kv_mul) * head_size;
                float score = 0.0f;
                for (int lane = 0; lane < head_size; lane++) {
                    score += q[lane] * k[lane];
                }
                row[key_position] = score * scale;
            }
        }
    }
}

static void softmax_stage_apply(
    void *environment,
    const float *input,
    float *output
) {
    LayerRuntime *runtime = environment;
    int positions = runtime->positions;
    int heads = runtime->transformer->config.n_heads;
    int frontier_width = frontier_width_for(runtime);
    int kv_frontier_width = kv_frontier_width_for(runtime);
    int table_width = attention_table_width_for(runtime);
    int prefix_width = frontier_width + kv_frontier_width;
    memcpy(
        output,
        input,
        (size_t)(prefix_width + table_width) * sizeof(*output)
    );
    float *probabilities = output + prefix_width;
    for (int position = 0; position < positions; position++) {
        for (int head = 0; head < heads; head++) {
            float *row = probabilities +
                ((size_t)position * heads + head) * positions;
            softmax(row, position + 1);
        }
    }
}

static void value_stage_apply(
    void *environment,
    const float *input,
    float *output
) {
    LayerRuntime *runtime = environment;
    Config *config = &runtime->transformer->config;
    int positions = runtime->positions;
    int dim = config->dim;
    int kv_dim = dim * config->n_kv_heads / config->n_heads;
    int kv_mul = config->n_heads / config->n_kv_heads;
    int head_size = dim / config->n_heads;
    int frontier_width = frontier_width_for(runtime);
    int kv_frontier_width = kv_frontier_width_for(runtime);
    const float *x = input;
    const float *value = input + frontier_width;
    const float *probabilities = value + kv_frontier_width;
    float *out_x = output;
    float *value_sum = output + frontier_width;
    memcpy(out_x, x, (size_t)frontier_width * sizeof(*out_x));
    memset(value_sum, 0, (size_t)frontier_width * sizeof(*value_sum));
    for (int position = 0; position < positions; position++) {
        for (int head = 0; head < config->n_heads; head++) {
            const float *row = probabilities +
                ((size_t)position * config->n_heads + head) * positions;
            float *head_output = value_sum + (size_t)position * dim +
                head * head_size;
            for (int key_position = 0;
                 key_position <= position;
                 key_position++) {
                const float *v = value + (size_t)key_position * kv_dim +
                    (head / kv_mul) * head_size;
                for (int lane = 0; lane < head_size; lane++) {
                    head_output[lane] += row[key_position] * v[lane];
                }
            }
        }
    }
}

static void wo_stage_apply(
    void *environment,
    const float *input,
    float *output
) {
    LayerRuntime *runtime = environment;
    Config *config = &runtime->transformer->config;
    int dim = config->dim;
    int frontier_width = frontier_width_for(runtime);
    const float *x = input;
    const float *value_sum = input + frontier_width;
    float *out_x = output;
    float *update = output + frontier_width;
    memcpy(out_x, x, (size_t)frontier_width * sizeof(*out_x));
    float *wo = runtime->transformer->weights.wo +
        (size_t)runtime->layer * dim * dim;
    for (int position = 0; position < runtime->positions; position++) {
        matmul(
            update + (size_t)position * dim,
            (float *)value_sum + (size_t)position * dim,
            wo,
            dim,
            dim
        );
    }
}

static void attention_residual_stage_apply(
    void *environment,
    const float *input,
    float *output
) {
    LayerRuntime *runtime = environment;
    int frontier_width = frontier_width_for(runtime);
    const float *x = input;
    const float *update = input + frontier_width;
    for (int index = 0; index < frontier_width; index++) {
        output[index] = x[index] + update[index];
    }
}

static int ffn_norm_state_width(const LayerRuntime *runtime) {
    return 2 * frontier_width_for(runtime);
}

static int ffn_branch_state_width(const LayerRuntime *runtime) {
    return frontier_width_for(runtime) +
        2 * runtime->positions * runtime->transformer->config.hidden_dim;
}

static int ffn_product_state_width(const LayerRuntime *runtime) {
    return frontier_width_for(runtime) +
        runtime->positions * runtime->transformer->config.hidden_dim;
}

static void ffn_norm_stage_apply(
    void *environment,
    const float *input,
    float *output
) {
    LayerRuntime *runtime = environment;
    Config *config = &runtime->transformer->config;
    int dim = config->dim;
    int frontier_width = frontier_width_for(runtime);
    memcpy(output, input, (size_t)frontier_width * sizeof(*output));
    float *normalized = output + frontier_width;
    float *weight = runtime->transformer->weights.rms_ffn_weight +
        (size_t)runtime->layer * dim;
    for (int position = 0; position < runtime->positions; position++) {
        rmsnorm(
            normalized + (size_t)position * dim,
            (float *)input + (size_t)position * dim,
            weight,
            dim
        );
    }
}

static void ffn_projection_stage_apply(
    void *environment,
    const float *input,
    float *output
) {
    LayerRuntime *runtime = environment;
    Transformer *transformer = runtime->transformer;
    Config *config = &transformer->config;
    int dim = config->dim;
    int hidden_dim = config->hidden_dim;
    int frontier_width = frontier_width_for(runtime);
    int hidden_frontier = runtime->positions * hidden_dim;
    const float *x = input;
    const float *normalized = input + frontier_width;
    float *out_x = output;
    float *a = output + frontier_width;
    float *b = a + hidden_frontier;
    memcpy(out_x, x, (size_t)frontier_width * sizeof(*out_x));
    float *w1 = transformer->weights.w1 +
        (size_t)runtime->layer * dim * hidden_dim;
    float *w3 = transformer->weights.w3 +
        (size_t)runtime->layer * dim * hidden_dim;
    for (int position = 0; position < runtime->positions; position++) {
        const float *n = normalized + (size_t)position * dim;
        matmul(
            a + (size_t)position * hidden_dim,
            (float *)n,
            w1,
            dim,
            hidden_dim
        );
        matmul(
            b + (size_t)position * hidden_dim,
            (float *)n,
            w3,
            dim,
            hidden_dim
        );
    }
}

static void silu_stage_apply(
    void *environment,
    const float *input,
    float *output
) {
    LayerRuntime *runtime = environment;
    int frontier_width = frontier_width_for(runtime);
    int hidden_frontier = runtime->positions *
        runtime->transformer->config.hidden_dim;
    const float *x = input;
    const float *a = input + frontier_width;
    const float *b = a + hidden_frontier;
    float *out_x = output;
    float *silu = output + frontier_width;
    float *out_b = silu + hidden_frontier;
    memcpy(out_x, x, (size_t)frontier_width * sizeof(*out_x));
    memcpy(out_b, b, (size_t)hidden_frontier * sizeof(*out_b));
    for (int index = 0; index < hidden_frontier; index++) {
        float sigmoid = 1.0f / (1.0f + expf(-a[index]));
        silu[index] = a[index] * sigmoid;
    }
}

static void swiglu_product_stage_apply(
    void *environment,
    const float *input,
    float *output
) {
    LayerRuntime *runtime = environment;
    int frontier_width = frontier_width_for(runtime);
    int hidden_frontier = runtime->positions *
        runtime->transformer->config.hidden_dim;
    const float *x = input;
    const float *silu = input + frontier_width;
    const float *b = silu + hidden_frontier;
    float *out_x = output;
    float *product = output + frontier_width;
    memcpy(out_x, x, (size_t)frontier_width * sizeof(*out_x));
    for (int index = 0; index < hidden_frontier; index++) {
        product[index] = silu[index] * b[index];
    }
}

static void ffn_output_stage_apply(
    void *environment,
    const float *input,
    float *output
) {
    LayerRuntime *runtime = environment;
    Transformer *transformer = runtime->transformer;
    Config *config = &transformer->config;
    int dim = config->dim;
    int hidden_dim = config->hidden_dim;
    int frontier_width = frontier_width_for(runtime);
    const float *x = input;
    const float *product = input + frontier_width;
    float *out_x = output;
    float *update = output + frontier_width;
    memcpy(out_x, x, (size_t)frontier_width * sizeof(*out_x));
    float *w2 = transformer->weights.w2 +
        (size_t)runtime->layer * dim * hidden_dim;
    for (int position = 0; position < runtime->positions; position++) {
        matmul(
            update + (size_t)position * dim,
            (float *)product + (size_t)position * hidden_dim,
            w2,
            hidden_dim,
            dim
        );
    }
}

static void ffn_residual_stage_apply(
    void *environment,
    const float *input,
    float *output
) {
    LayerRuntime *runtime = environment;
    int frontier_width = frontier_width_for(runtime);
    const float *x = input;
    const float *update = input + frontier_width;
    for (int index = 0; index < frontier_width; index++) {
        output[index] = x[index] + update[index];
    }
}

static void final_rms_map_apply(
    void *environment,
    const float *input,
    float *output
) {
    FinalRmsRuntime *runtime = environment;
    Transformer *transformer = runtime->transformer;
    int dim = transformer->config.dim;
    for (int position = 0; position < runtime->positions; position++) {
        rmsnorm(
            output + (size_t)position * dim,
            (float *)input + (size_t)position * dim,
            transformer->weights.rms_final_weight,
            dim
        );
    }
}

static void identity_continuation_apply(
    void *environment,
    const float *input,
    float *result
) {
    int width = *(int *)environment;
    memcpy(result, input, (size_t)width * sizeof(*result));
}

static void pullback_apply(
    void *environment,
    const float *input,
    float *result
) {
    PullbackEnvironment *pullback = environment;
    pullback->map.apply(
        pullback->map.environment,
        input,
        pullback->mapped
    );
    pullback->next.apply(
        pullback->next.environment,
        pullback->mapped,
        result
    );
}

static Continuation make_pullback(
    PullbackEnvironment *environment,
    FrontierMap map,
    Continuation next
) {
    if (map.output_width != next.input_width) {
        fail("pullback type mismatch");
    }
    environment->map = map;
    environment->next = next;
    environment->mapped = checked_calloc(
        (size_t)map.output_width,
        sizeof(float)
    );
    return (Continuation){
        .input_width = map.input_width,
        .result_width = next.result_width,
        .apply = pullback_apply,
        .environment = environment
    };
}

static void free_pullback(PullbackEnvironment *environment) {
    free(environment->mapped);
    environment->mapped = NULL;
}

static ContextFrontiers allocate_frontiers(int layers, int width) {
    ContextFrontiers capture = {.layers = layers, .width = width};
    capture.layer_frontiers = checked_calloc(
        (size_t)(layers + 1) * width,
        sizeof(float)
    );
    capture.post_attention_frontiers = checked_calloc(
        (size_t)layers * width,
        sizeof(float)
    );
    return capture;
}

static void free_frontiers(ContextFrontiers *capture) {
    free(capture->layer_frontiers);
    free(capture->post_attention_frontiers);
    capture->layer_frontiers = NULL;
    capture->post_attention_frontiers = NULL;
}

static float *layer_frontier(ContextFrontiers *capture, int layer) {
    return capture->layer_frontiers + (size_t)layer * capture->width;
}

static float *post_attention_frontier(
    ContextFrontiers *capture,
    int layer
) {
    return capture->post_attention_frontiers +
        (size_t)layer * capture->width;
}

static void capture_context_frontiers(
    Transformer *transformer,
    const EncodedContext *context,
    LayerRuntime *runtimes,
    ContextFrontiers *capture
) {
    int dim = transformer->config.dim;
    int positions = context->count;
    float *initial = layer_frontier(capture, 0);
    for (int position = 0; position < positions; position++) {
        const float *embedding = transformer->weights.token_embedding_table +
            (size_t)context->tokens[position] * dim;
        memcpy(
            initial + (size_t)position * dim,
            embedding,
            (size_t)dim * sizeof(*initial)
        );
    }
    for (int layer = 0; layer < transformer->config.n_layers; layer++) {
        attention_frontier(
            &runtimes[layer],
            layer_frontier(capture, layer),
            post_attention_frontier(capture, layer)
        );
        ffn_frontier(
            &runtimes[layer],
            post_attention_frontier(capture, layer),
            layer_frontier(capture, layer + 1)
        );
    }
}

static double vector_l2(const float *values, int width) {
    double square = 0.0;
    for (int index = 0; index < width; index++) {
        double value = values[index];
        square += value * value;
    }
    return sqrt(square);
}

static double difference_l2(
    const float *left,
    const float *right,
    int width
) {
    double square = 0.0;
    for (int index = 0; index < width; index++) {
        double difference = (double)left[index] - right[index];
        square += difference * difference;
    }
    return sqrt(square);
}

static InteractionMeasurement measure_interaction(
    FrontierMap map,
    Continuation continuation,
    const float *input00,
    const float *input10,
    const float *input01,
    const float *input11
) {
    int input_width = map.input_width;
    int output_width = map.output_width;
    int result_width = continuation.result_width;
    float *independent11 = checked_calloc(
        (size_t)input_width,
        sizeof(float)
    );
    float *output00 = checked_calloc((size_t)output_width, sizeof(float));
    float *output10 = checked_calloc((size_t)output_width, sizeof(float));
    float *output01 = checked_calloc((size_t)output_width, sizeof(float));
    float *output11 = checked_calloc((size_t)output_width, sizeof(float));
    float *independent_output = checked_calloc(
        (size_t)output_width,
        sizeof(float)
    );
    float *root11 = checked_calloc((size_t)result_width, sizeof(float));
    float *independent_root = checked_calloc(
        (size_t)result_width,
        sizeof(float)
    );
    for (int index = 0; index < input_width; index++) {
        independent11[index] = input10[index] + input01[index] - input00[index];
    }
    map.apply(map.environment, input00, output00);
    map.apply(map.environment, input10, output10);
    map.apply(map.environment, input01, output01);
    map.apply(map.environment, independent11, output11);
    for (int index = 0; index < output_width; index++) {
        independent_output[index] = output10[index] + output01[index] -
            output00[index];
    }
    continuation.apply(continuation.environment, output11, root11);
    continuation.apply(
        continuation.environment,
        independent_output,
        independent_root
    );
    double root_scale = vector_l2(root11, result_width);
    InteractionMeasurement measurement = {
        .local_interaction_l2 = difference_l2(
            output11,
            independent_output,
            output_width
        ),
        .visible_interaction_l2 = difference_l2(
            root11,
            independent_root,
            result_width
        ),
        .prior_interaction_l2 = difference_l2(
            input11,
            independent11,
            input_width
        )
    };
    measurement.visible_interaction_relative = root_scale == 0.0 ? 0.0 :
        measurement.visible_interaction_l2 / root_scale;
    free(independent11);
    free(output00);
    free(output10);
    free(output01);
    free(output11);
    free(independent_output);
    free(root11);
    free(independent_root);
    return measurement;
}

static FixedMeasurement measure_fixed_continuation(
    FrontierMap map,
    Continuation continuation,
    const float *const inputs[4]
) {
    if (map.input_width != map.output_width) {
        fail("fixed-continuation measurement requires an endomorphism");
    }
    int width = map.input_width;
    int result_width = continuation.result_width;
    float *mapped = checked_calloc((size_t)width, sizeof(float));
    float *pulled = checked_calloc((size_t)result_width, sizeof(float));
    float *unpulled = checked_calloc((size_t)result_width, sizeof(float));
    FixedMeasurement measurement = {0};
    for (int context = 0; context < 4; context++) {
        map.apply(map.environment, inputs[context], mapped);
        continuation.apply(continuation.environment, mapped, pulled);
        continuation.apply(continuation.environment, inputs[context], unpulled);
        double defect = difference_l2(pulled, unpulled, result_width);
        double scale = vector_l2(pulled, result_width);
        measurement.defects[context] = defect;
        measurement.relative_defects[context] = scale == 0.0 ? 0.0 :
            defect / scale;
        if (defect > measurement.maximum_defect) {
            measurement.maximum_defect = defect;
        }
        if (measurement.relative_defects[context] >
            measurement.maximum_relative_defect) {
            measurement.maximum_relative_defect =
                measurement.relative_defects[context];
        }
    }
    free(mapped);
    free(pulled);
    free(unpulled);
    return measurement;
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

static void write_measurements(
    FILE *trace,
    int layer,
    const char *operation,
    const InteractionMeasurement *interaction,
    const FixedMeasurement *fixed
) {
    if (trace == NULL) return;
    fprintf(
        trace,
        "{\"kind\":\"cps_operation\",\"layer\":%d,"
        "\"operation\":\"%s\","
        "\"root_observer\":\"identity_after_final_rms_hidden_frontier\","
        "\"pullback\":\"k_after_composed_with_operation\","
        "\"prior_interaction_l2\":%.17g,"
        "\"local_interaction_l2\":%.17g,"
        "\"continuation_visible_interaction_l2\":%.17g,"
        "\"continuation_visible_interaction_relative\":%.17g,"
        "\"fixed_defects\":[%.17g,%.17g,%.17g,%.17g],"
        "\"fixed_relative_defects\":[%.17g,%.17g,%.17g,%.17g],"
        "\"maximum_fixed_defect\":%.17g,"
        "\"maximum_fixed_relative_defect\":%.17g}\n",
        layer,
        operation,
        interaction->prior_interaction_l2,
        interaction->local_interaction_l2,
        interaction->visible_interaction_l2,
        interaction->visible_interaction_relative,
        fixed->defects[0], fixed->defects[1],
        fixed->defects[2], fixed->defects[3],
        fixed->relative_defects[0], fixed->relative_defects[1],
        fixed->relative_defects[2], fixed->relative_defects[3],
        fixed->maximum_defect,
        fixed->maximum_relative_defect
    );
    fflush(trace);
}

static void write_stage_interaction(
    FILE *trace,
    int layer,
    const FrontierMap *map,
    const InteractionMeasurement *interaction
) {
    if (trace == NULL) return;
    fprintf(
        trace,
        "{\"kind\":\"cps_stage\",\"layer\":%d,"
        "\"operation\":\"%s\",\"input_width\":%d,"
        "\"output_width\":%d,"
        "\"root_observer\":\"identity_after_final_rms_hidden_frontier\","
        "\"prior_interaction_l2\":%.17g,"
        "\"local_interaction_l2\":%.17g,"
        "\"continuation_visible_interaction_l2\":%.17g,"
        "\"continuation_visible_interaction_relative\":%.17g}\n",
        layer,
        map->name,
        map->input_width,
        map->output_width,
        interaction->prior_interaction_l2,
        interaction->local_interaction_l2,
        interaction->visible_interaction_l2,
        interaction->visible_interaction_relative
    );
    fflush(trace);
}

static double check_composition(
    Continuation left,
    Continuation right,
    const float *const inputs[4]
) {
    float *left_result = checked_calloc(
        (size_t)left.result_width,
        sizeof(float)
    );
    float *right_result = checked_calloc(
        (size_t)right.result_width,
        sizeof(float)
    );
    double maximum = 0.0;
    for (int context = 0; context < 4; context++) {
        left.apply(left.environment, inputs[context], left_result);
        right.apply(right.environment, inputs[context], right_result);
        double difference = difference_l2(
            left_result,
            right_result,
            left.result_width
        );
        if (difference > maximum) maximum = difference;
    }
    free(left_result);
    free(right_result);
    return maximum;
}

static double run_attention_stage_term(
    FILE *trace,
    LayerRuntime *runtime,
    Continuation expected_attention_suffix,
    Continuation post_attention_suffix,
    const float *const inputs[4],
    const float *const expected_outputs[4]
) {
    int frontier_width = frontier_width_for(runtime);
    int norm_width = attention_norm_state_width(runtime);
    int qkv_width = qkv_state_width(runtime);
    int score_width = score_state_width(runtime);
    int pair_width = residual_pair_width(runtime);
    FrontierMap maps[7] = {
        {
            .name = "attention_rms_pair",
            .input_width = frontier_width,
            .output_width = norm_width,
            .apply = attention_norm_stage_apply,
            .environment = runtime
        },
        {
            .name = "qkv_projection",
            .input_width = norm_width,
            .output_width = qkv_width,
            .apply = qkv_stage_apply,
            .environment = runtime
        },
        {
            .name = "qk_contraction",
            .input_width = qkv_width,
            .output_width = score_width,
            .apply = qk_stage_apply,
            .environment = runtime
        },
        {
            .name = "softmax",
            .input_width = score_width,
            .output_width = score_width,
            .apply = softmax_stage_apply,
            .environment = runtime
        },
        {
            .name = "attention_value_contraction",
            .input_width = score_width,
            .output_width = pair_width,
            .apply = value_stage_apply,
            .environment = runtime
        },
        {
            .name = "attention_output_projection",
            .input_width = pair_width,
            .output_width = pair_width,
            .apply = wo_stage_apply,
            .environment = runtime
        },
        {
            .name = "attention_residual_addition",
            .input_width = pair_width,
            .output_width = frontier_width,
            .apply = attention_residual_stage_apply,
            .environment = runtime
        }
    };
    PullbackEnvironment pullbacks[7] = {0};
    Continuation suffixes[8] = {0};
    suffixes[7] = post_attention_suffix;
    for (int stage = 6; stage >= 0; stage--) {
        suffixes[stage] = make_pullback(
            &pullbacks[stage],
            maps[stage],
            suffixes[stage + 1]
        );
    }
    double composition_defect = check_composition(
        expected_attention_suffix,
        suffixes[0],
        inputs
    );

    const float *current[4] = {
        inputs[0], inputs[1], inputs[2], inputs[3]
    };
    bool current_owned = false;
    double output_defect = 0.0;
    for (int stage = 0; stage < 7; stage++) {
        float *next[4];
        for (int context = 0; context < 4; context++) {
            next[context] = checked_calloc(
                (size_t)maps[stage].output_width,
                sizeof(float)
            );
            maps[stage].apply(
                maps[stage].environment,
                current[context],
                next[context]
            );
        }
        InteractionMeasurement interaction = measure_interaction(
            maps[stage],
            suffixes[stage + 1],
            current[0],
            current[1],
            current[2],
            current[3]
        );
        write_stage_interaction(trace, runtime->layer, &maps[stage], &interaction);
        printf(
            "    %-28s local=%.8g visible=%.8g prior=%.8g\n",
            maps[stage].name,
            interaction.local_interaction_l2,
            interaction.visible_interaction_l2,
            interaction.prior_interaction_l2
        );
        if (current_owned) {
            for (int context = 0; context < 4; context++) {
                free((void *)current[context]);
            }
        }
        for (int context = 0; context < 4; context++) {
            current[context] = next[context];
        }
        current_owned = true;
    }
    for (int context = 0; context < 4; context++) {
        double defect = difference_l2(
            current[context],
            expected_outputs[context],
            frontier_width
        );
        if (defect > output_defect) output_defect = defect;
        free((void *)current[context]);
    }
    for (int stage = 0; stage < 7; stage++) {
        free_pullback(&pullbacks[stage]);
    }
    if (trace != NULL) {
        fprintf(
            trace,
            "{\"kind\":\"attention_term_check\",\"layer\":%d,"
            "\"composition_l2_defect\":%.17g,"
            "\"output_l2_defect\":%.17g}\n",
            runtime->layer,
            composition_defect,
            output_defect
        );
        fflush(trace);
    }
    return fmax(composition_defect, output_defect);
}

static double run_ffn_stage_term(
    FILE *trace,
    LayerRuntime *runtime,
    Continuation expected_ffn_suffix,
    Continuation layer_output_suffix,
    const float *const inputs[4],
    const float *const expected_outputs[4]
) {
    int frontier_width = frontier_width_for(runtime);
    int norm_width = ffn_norm_state_width(runtime);
    int branch_width = ffn_branch_state_width(runtime);
    int product_width = ffn_product_state_width(runtime);
    int pair_width = residual_pair_width(runtime);
    FrontierMap maps[6] = {
        {
            .name = "ffn_rms_pair",
            .input_width = frontier_width,
            .output_width = norm_width,
            .apply = ffn_norm_stage_apply,
            .environment = runtime
        },
        {
            .name = "w1_w3_projection",
            .input_width = norm_width,
            .output_width = branch_width,
            .apply = ffn_projection_stage_apply,
            .environment = runtime
        },
        {
            .name = "silu",
            .input_width = branch_width,
            .output_width = branch_width,
            .apply = silu_stage_apply,
            .environment = runtime
        },
        {
            .name = "swiglu_product",
            .input_width = branch_width,
            .output_width = product_width,
            .apply = swiglu_product_stage_apply,
            .environment = runtime
        },
        {
            .name = "ffn_output_projection",
            .input_width = product_width,
            .output_width = pair_width,
            .apply = ffn_output_stage_apply,
            .environment = runtime
        },
        {
            .name = "ffn_residual_addition",
            .input_width = pair_width,
            .output_width = frontier_width,
            .apply = ffn_residual_stage_apply,
            .environment = runtime
        }
    };
    PullbackEnvironment pullbacks[6] = {0};
    Continuation suffixes[7] = {0};
    suffixes[6] = layer_output_suffix;
    for (int stage = 5; stage >= 0; stage--) {
        suffixes[stage] = make_pullback(
            &pullbacks[stage],
            maps[stage],
            suffixes[stage + 1]
        );
    }
    double composition_defect = check_composition(
        expected_ffn_suffix,
        suffixes[0],
        inputs
    );

    const float *current[4] = {
        inputs[0], inputs[1], inputs[2], inputs[3]
    };
    bool current_owned = false;
    double output_defect = 0.0;
    for (int stage = 0; stage < 6; stage++) {
        float *next[4];
        for (int context = 0; context < 4; context++) {
            next[context] = checked_calloc(
                (size_t)maps[stage].output_width,
                sizeof(float)
            );
            maps[stage].apply(
                maps[stage].environment,
                current[context],
                next[context]
            );
        }
        InteractionMeasurement interaction = measure_interaction(
            maps[stage],
            suffixes[stage + 1],
            current[0],
            current[1],
            current[2],
            current[3]
        );
        write_stage_interaction(trace, runtime->layer, &maps[stage], &interaction);
        printf(
            "    %-28s local=%.8g visible=%.8g prior=%.8g\n",
            maps[stage].name,
            interaction.local_interaction_l2,
            interaction.visible_interaction_l2,
            interaction.prior_interaction_l2
        );
        if (current_owned) {
            for (int context = 0; context < 4; context++) {
                free((void *)current[context]);
            }
        }
        for (int context = 0; context < 4; context++) {
            current[context] = next[context];
        }
        current_owned = true;
    }
    for (int context = 0; context < 4; context++) {
        double defect = difference_l2(
            current[context],
            expected_outputs[context],
            frontier_width
        );
        if (defect > output_defect) output_defect = defect;
        free((void *)current[context]);
    }
    for (int stage = 0; stage < 6; stage++) {
        free_pullback(&pullbacks[stage]);
    }
    if (trace != NULL) {
        fprintf(
            trace,
            "{\"kind\":\"ffn_term_check\",\"layer\":%d,"
            "\"composition_l2_defect\":%.17g,"
            "\"output_l2_defect\":%.17g}\n",
            runtime->layer,
            composition_defect,
            output_defect
        );
        fflush(trace);
    }
    return fmax(composition_defect, output_defect);
}

static double check_reference_hidden_frontier(
    Transformer *transformer,
    const EncodedContext *context,
    const float *normalized_frontier,
    double *relative_defect
) {
    Config *config = &transformer->config;
    int dim = config->dim;
    int kv_dim = dim * config->n_kv_heads / config->n_heads;
    size_t cache_count =
        (size_t)config->n_layers * config->seq_len * kv_dim;
    memset(transformer->state.key_cache, 0, cache_count * sizeof(float));
    memset(transformer->state.value_cache, 0, cache_count * sizeof(float));
    double square = 0.0;
    double reference_square = 0.0;
    for (int position = 0; position < context->count; position++) {
        (void)forward(transformer, context->tokens[position], position);
        const float *expected = normalized_frontier + (size_t)position * dim;
        for (int lane = 0; lane < dim; lane++) {
            double difference = (double)transformer->state.x[lane] -
                expected[lane];
            square += difference * difference;
            reference_square += (double)transformer->state.x[lane] *
                transformer->state.x[lane];
        }
    }
    double defect = sqrt(square);
    double reference_norm = sqrt(reference_square);
    *relative_defect = reference_norm == 0.0 ? 0.0 :
        defect / reference_norm;
    return defect;
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
    int positions = classify_factor_positions(
        contexts,
        &factor_a_position,
        &factor_b_position
    );
    if (positions > transformer.config.seq_len) {
        fail("sampled context exceeds model sequence length");
    }
    int layers = transformer.config.n_layers;
    int frontier_width = positions * transformer.config.dim;

    FILE *trace = NULL;
    if (options.trace_path != NULL) {
        trace = fopen(options.trace_path, "wb");
        if (trace == NULL) fail("could not create CPS trace");
        fprintf(
            trace,
            "{\"kind\":\"meta\",\"schema_version\":1,"
            "\"semantics\":\"exact_sampled_cps_pullback\","
            "\"root_observer\":\"identity_after_final_rms_hidden_frontier\","
            "\"layers\":%d,\"positions\":%d,\"dim\":%d,"
            "\"frontier_width\":%d,\"factor_a_position\":%d,"
            "\"factor_b_position\":%d}\n",
            layers,
            positions,
            transformer.config.dim,
            frontier_width,
            factor_a_position,
            factor_b_position
        );
        fflush(trace);
        for (int context = 0; context < 4; context++) {
            write_context_record(trace, &tokenizer, &contexts[context], context);
        }
    }

    printf(
        "model layers=%d dim=%d positions=%d frontier=%d\n",
        layers,
        transformer.config.dim,
        positions,
        frontier_width
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

    LayerRuntime *runtimes = checked_calloc(
        (size_t)layers,
        sizeof(*runtimes)
    );
    FrontierMap *attention_maps = checked_calloc(
        (size_t)layers,
        sizeof(*attention_maps)
    );
    FrontierMap *ffn_maps = checked_calloc(
        (size_t)layers,
        sizeof(*ffn_maps)
    );
    FrontierMap *layer_maps = checked_calloc(
        (size_t)layers,
        sizeof(*layer_maps)
    );
    for (int layer = 0; layer < layers; layer++) {
        runtimes[layer] = (LayerRuntime){
            .transformer = &transformer,
            .layer = layer,
            .positions = positions,
            .workspace = allocate_workspace(&transformer.config, positions)
        };
        attention_maps[layer] = (FrontierMap){
            .name = "attention_residual",
            .input_width = frontier_width,
            .output_width = frontier_width,
            .apply = attention_map_apply,
            .environment = &runtimes[layer]
        };
        ffn_maps[layer] = (FrontierMap){
            .name = "swiglu_residual",
            .input_width = frontier_width,
            .output_width = frontier_width,
            .apply = ffn_map_apply,
            .environment = &runtimes[layer]
        };
        layer_maps[layer] = (FrontierMap){
            .name = "whole_layer",
            .input_width = frontier_width,
            .output_width = frontier_width,
            .apply = layer_map_apply,
            .environment = &runtimes[layer]
        };
    }

    FinalRmsRuntime final_rms_runtime = {
        .transformer = &transformer,
        .positions = positions
    };
    FrontierMap final_rms_map = {
        .name = "final_rms",
        .input_width = frontier_width,
        .output_width = frontier_width,
        .apply = final_rms_map_apply,
        .environment = &final_rms_runtime
    };

    ContextFrontiers captures[4];
    for (int context = 0; context < 4; context++) {
        captures[context] = allocate_frontiers(layers, frontier_width);
        capture_context_frontiers(
            &transformer,
            &contexts[context],
            runtimes,
            &captures[context]
        );
    }

    Continuation *layer_suffixes = checked_calloc(
        (size_t)layers + 1,
        sizeof(*layer_suffixes)
    );
    Continuation *post_attention_suffixes = checked_calloc(
        (size_t)layers,
        sizeof(*post_attention_suffixes)
    );
    Continuation *attention_compositions = checked_calloc(
        (size_t)layers,
        sizeof(*attention_compositions)
    );
    PullbackEnvironment *layer_pullbacks = checked_calloc(
        (size_t)layers,
        sizeof(*layer_pullbacks)
    );
    PullbackEnvironment *ffn_pullbacks = checked_calloc(
        (size_t)layers,
        sizeof(*ffn_pullbacks)
    );
    PullbackEnvironment *attention_pullbacks = checked_calloc(
        (size_t)layers,
        sizeof(*attention_pullbacks)
    );
    Continuation root_identity = (Continuation){
        .input_width = frontier_width,
        .result_width = frontier_width,
        .apply = identity_continuation_apply,
        .environment = &frontier_width
    };
    PullbackEnvironment final_rms_pullback = {0};
    layer_suffixes[layers] = make_pullback(
        &final_rms_pullback,
        final_rms_map,
        root_identity
    );
    for (int layer = layers - 1; layer >= 0; layer--) {
        post_attention_suffixes[layer] = make_pullback(
            &ffn_pullbacks[layer],
            ffn_maps[layer],
            layer_suffixes[layer + 1]
        );
        attention_compositions[layer] = make_pullback(
            &attention_pullbacks[layer],
            attention_maps[layer],
            post_attention_suffixes[layer]
        );
        layer_suffixes[layer] = make_pullback(
            &layer_pullbacks[layer],
            layer_maps[layer],
            layer_suffixes[layer + 1]
        );
    }

    double maximum_composition_defect = 0.0;

    const float *final_inputs[4];
    for (int context = 0; context < 4; context++) {
        final_inputs[context] = layer_frontier(&captures[context], layers);
    }
    InteractionMeasurement final_interaction = measure_interaction(
        final_rms_map,
        root_identity,
        final_inputs[0],
        final_inputs[1],
        final_inputs[2],
        final_inputs[3]
    );
    FixedMeasurement final_fixed = measure_fixed_continuation(
        final_rms_map,
        root_identity,
        final_inputs
    );
    write_measurements(
        trace,
        layers,
        final_rms_map.name,
        &final_interaction,
        &final_fixed
    );
    printf(
        "final_rms local_interaction=%.8g visible=%.8g "
        "fixed_relative=%.8g prior=%.8g\n",
        final_interaction.local_interaction_l2,
        final_interaction.visible_interaction_l2,
        final_fixed.maximum_relative_defect,
        final_interaction.prior_interaction_l2
    );

    for (int layer = 0; layer < layers; layer++) {
        const float *layer_inputs[4];
        const float *post_inputs[4];
        const float *layer_outputs[4];
        for (int context = 0; context < 4; context++) {
            layer_inputs[context] = layer_frontier(&captures[context], layer);
            post_inputs[context] = post_attention_frontier(
                &captures[context],
                layer
            );
            layer_outputs[context] = layer_frontier(
                &captures[context],
                layer + 1
            );
        }
        double composition_defect = check_composition(
            layer_suffixes[layer],
            attention_compositions[layer],
            layer_inputs
        );
        if (composition_defect > maximum_composition_defect) {
            maximum_composition_defect = composition_defect;
        }

        printf("layer=%d composition_defect=%.8g\n", layer, composition_defect);
        double attention_term_defect = run_attention_stage_term(
            trace,
            &runtimes[layer],
            attention_compositions[layer],
            post_attention_suffixes[layer],
            layer_inputs,
            post_inputs
        );
        if (attention_term_defect > maximum_composition_defect) {
            maximum_composition_defect = attention_term_defect;
        }
        double ffn_term_defect = run_ffn_stage_term(
            trace,
            &runtimes[layer],
            post_attention_suffixes[layer],
            layer_suffixes[layer + 1],
            post_inputs,
            layer_outputs
        );
        if (ffn_term_defect > maximum_composition_defect) {
            maximum_composition_defect = ffn_term_defect;
        }

        const struct {
            FrontierMap map;
            Continuation continuation;
            const float *const *inputs;
        } operations[] = {
            {
                attention_maps[layer],
                post_attention_suffixes[layer],
                layer_inputs
            },
            {
                ffn_maps[layer],
                layer_suffixes[layer + 1],
                post_inputs
            },
            {
                layer_maps[layer],
                layer_suffixes[layer + 1],
                layer_inputs
            }
        };
        for (size_t operation = 0;
             operation < sizeof(operations) / sizeof(operations[0]);
             operation++) {
            InteractionMeasurement interaction = measure_interaction(
                operations[operation].map,
                operations[operation].continuation,
                operations[operation].inputs[0],
                operations[operation].inputs[1],
                operations[operation].inputs[2],
                operations[operation].inputs[3]
            );
            FixedMeasurement fixed = measure_fixed_continuation(
                operations[operation].map,
                operations[operation].continuation,
                operations[operation].inputs
            );
            write_measurements(
                trace,
                layer,
                operations[operation].map.name,
                &interaction,
                &fixed
            );
            printf(
                "  %-20s local_interaction=%.8g visible=%.8g "
                "fixed_relative=%.8g prior=%.8g\n",
                operations[operation].map.name,
                interaction.local_interaction_l2,
                interaction.visible_interaction_l2,
                fixed.maximum_relative_defect,
                interaction.prior_interaction_l2
            );
        }
        fflush(stdout);
    }
    printf(
        "maximum_cps_composition_defect=%.8g "
        "root_observer=identity_after_final_rms_hidden\n",
        maximum_composition_defect
    );

    float *normalized_frontier = checked_calloc(
        (size_t)frontier_width,
        sizeof(float)
    );
    double maximum_reference_defect = 0.0;
    double maximum_reference_relative_defect = 0.0;
    for (int context = 0; context < 4; context++) {
        final_rms_map.apply(
            final_rms_map.environment,
            final_inputs[context],
            normalized_frontier
        );
        double relative_defect = 0.0;
        double defect = check_reference_hidden_frontier(
            &transformer,
            &contexts[context],
            normalized_frontier,
            &relative_defect
        );
        if (defect > maximum_reference_defect) {
            maximum_reference_defect = defect;
        }
        if (relative_defect > maximum_reference_relative_defect) {
            maximum_reference_relative_defect = relative_defect;
        }
    }
    free(normalized_frontier);
    printf(
        "maximum_llama2c_hidden_defect=%.8g relative=%.8g\n",
        maximum_reference_defect,
        maximum_reference_relative_defect
    );

    if (trace != NULL) {
        fprintf(
            trace,
            "{\"kind\":\"composition_check\","
            "\"maximum_l2_defect\":%.17g,"
            "\"maximum_llama2c_hidden_l2_defect\":%.17g,"
            "\"maximum_llama2c_hidden_relative_defect\":%.17g}\n",
            maximum_composition_defect,
            maximum_reference_defect,
            maximum_reference_relative_defect
        );
        if (fclose(trace) != 0) fail("could not close CPS trace");
    }
    for (int layer = 0; layer < layers; layer++) {
        free_pullback(&attention_pullbacks[layer]);
        free_pullback(&ffn_pullbacks[layer]);
        free_pullback(&layer_pullbacks[layer]);
        free_workspace(&runtimes[layer].workspace);
    }
    free_pullback(&final_rms_pullback);
    for (int context = 0; context < 4; context++) {
        free_frontiers(&captures[context]);
        free_context(&contexts[context]);
    }
    free(attention_pullbacks);
    free(ffn_pullbacks);
    free(layer_pullbacks);
    free(attention_compositions);
    free(post_attention_suffixes);
    free(layer_suffixes);
    free(layer_maps);
    free(ffn_maps);
    free(attention_maps);
    free(runtimes);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return EXIT_SUCCESS;
}
