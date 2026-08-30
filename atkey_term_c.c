/* Keep the embedded llama2.c reference helpers private to this runtime. */
#define malloc_run_state atkey_reference_malloc_run_state
#define free_run_state atkey_reference_free_run_state
#define memory_map_weights atkey_reference_memory_map_weights
#define read_checkpoint atkey_reference_read_checkpoint
#define build_transformer atkey_reference_build_transformer
#define free_transformer atkey_reference_free_transformer
#define rmsnorm atkey_reference_rmsnorm
#define softmax atkey_reference_softmax
#define matmul atkey_reference_matmul
#define forward atkey_reference_forward
#define compare_tokens atkey_reference_compare_tokens
#define build_tokenizer atkey_reference_build_tokenizer
#define decode atkey_reference_decode
#define str_lookup atkey_reference_str_lookup
#define encode atkey_reference_encode
#define free_tokenizer atkey_reference_free_tokenizer
#define safe_printf atkey_reference_safe_printf
#define sample_argmax atkey_reference_sample_argmax
#define sample_mult atkey_reference_sample_mult
#define compare atkey_reference_compare
#define sample_topp atkey_reference_sample_topp
#define random_u32 atkey_reference_random_u32
#define random_f32 atkey_reference_random_f32
#define build_sampler atkey_reference_build_sampler
#define free_sampler atkey_reference_free_sampler
#define sample atkey_reference_sample
#define time_in_ms atkey_reference_time_in_ms
#define read_stdin atkey_reference_read_stdin
#define chat atkey_reference_chat

#define TESTING
#include "run.c"
#include "atkey_term_c.h"

#include <stdint.h>

#ifdef ATKEY_METAL
#include "metal_backend.h"
#endif

/*
 * Numerical leaves for the functional Atkey/Escardo term.
 *
 * There is deliberately no exported forward/model/layer operation here.
 * atkey_term.c receives the learned maps themselves and composes these
 * primitive leaves into its field/continuation term. Token selection is not
 * implemented in this file.
 */

struct AtkeyRuntime {
    Transformer transformer;
    Tokenizer tokenizer;
    size_t filler_count;
    size_t *filler_calls;
    size_t *filler_scalar_reads;
#ifdef ATKEY_METAL
    AtkeyMetalBackend *metal;
#endif
};

enum {
    ATKEY_LAYER_FILLERS = 9,
    ATKEY_ATTN_RMS = 0,
    ATKEY_QUERY = 1,
    ATKEY_KEY = 2,
    ATKEY_VALUE = 3,
    ATKEY_ATTN_OUTPUT = 4,
    ATKEY_FFN_RMS = 5,
    ATKEY_FFN_GATE = 6,
    ATKEY_FFN_UP = 7,
    ATKEY_FFN_DOWN = 8,
};

static size_t atkey_layer_filler(int layer, int stage) {
    return 1 + (size_t)layer * ATKEY_LAYER_FILLERS + (size_t)stage;
}

static size_t atkey_final_rms_filler(const Config *config) {
    return 1 + (size_t)config->n_layers * ATKEY_LAYER_FILLERS;
}

static size_t atkey_output_filler(const Config *config) {
    return atkey_final_rms_filler(config) + 1;
}

static void atkey_note_filler(
    AtkeyRuntime *runtime,
    int filler,
    size_t scalar_reads
) {
    if (runtime == NULL || filler < 0 || (size_t)filler >= runtime->filler_count) {
        fprintf(stderr, "atkey term: invalid learned filler\n");
        abort();
    }
    runtime->filler_calls[filler]++;
    runtime->filler_scalar_reads[filler] += scalar_reads;
}

AtkeyRuntime *atkey_runtime_new(
    const char *checkpoint_path,
    const char *tokenizer_path
) {
    if (checkpoint_path == NULL || tokenizer_path == NULL) return NULL;
    AtkeyRuntime *runtime = calloc(1, sizeof(*runtime));
    if (runtime == NULL) return NULL;
    build_transformer(&runtime->transformer, (char *)checkpoint_path);
    build_tokenizer(
        &runtime->tokenizer,
        (char *)tokenizer_path,
        runtime->transformer.config.vocab_size
    );
    runtime->filler_count = 3 +
        (size_t)runtime->transformer.config.n_layers * ATKEY_LAYER_FILLERS;
    runtime->filler_calls = calloc(runtime->filler_count, sizeof(size_t));
    runtime->filler_scalar_reads = calloc(
        runtime->filler_count,
        sizeof(size_t)
    );
    if (runtime->filler_calls == NULL || runtime->filler_scalar_reads == NULL) {
        free(runtime->filler_calls);
        free(runtime->filler_scalar_reads);
        free_tokenizer(&runtime->tokenizer);
        free_transformer(&runtime->transformer);
        free(runtime);
        return NULL;
    }
    return runtime;
}

void atkey_runtime_free(AtkeyRuntime *runtime) {
    if (runtime == NULL) return;
#ifdef ATKEY_METAL
    atkey_metal_backend_free(runtime->metal);
#endif
    free(runtime->filler_calls);
    free(runtime->filler_scalar_reads);
    free_tokenizer(&runtime->tokenizer);
    free_transformer(&runtime->transformer);
    free(runtime);
}

bool atkey_enable_metal(AtkeyRuntime *runtime, const char *library_path) {
    if (runtime == NULL || library_path == NULL) return false;
#ifdef ATKEY_METAL
    if (runtime->metal != NULL) return true;
    runtime->metal = atkey_metal_backend_new(library_path);
    return runtime->metal != NULL;
#else
    (void)runtime;
    (void)library_path;
    fprintf(stderr, "atkey term: Metal support was not compiled in\n");
    return false;
#endif
}

const char *atkey_backend_name(AtkeyRuntime *runtime) {
#ifdef ATKEY_METAL
    if (runtime != NULL && runtime->metal != NULL) return "metal";
#else
    (void)runtime;
#endif
    return "cpu";
}

const char *atkey_backend_device_name(AtkeyRuntime *runtime) {
#ifdef ATKEY_METAL
    if (runtime != NULL && runtime->metal != NULL) {
        return atkey_metal_backend_device_name(runtime->metal);
    }
#else
    (void)runtime;
#endif
    return "host CPU";
}

uint64_t atkey_backend_dispatch_count(AtkeyRuntime *runtime) {
#ifdef ATKEY_METAL
    if (runtime != NULL && runtime->metal != NULL) {
        return atkey_metal_backend_dispatch_count(runtime->metal);
    }
#else
    (void)runtime;
#endif
    return 0;
}

uint64_t atkey_backend_weight_upload_count(AtkeyRuntime *runtime) {
#ifdef ATKEY_METAL
    if (runtime != NULL && runtime->metal != NULL) {
        return atkey_metal_backend_weight_upload_count(runtime->metal);
    }
#else
    (void)runtime;
#endif
    return 0;
}

uint64_t atkey_backend_weight_upload_bytes(AtkeyRuntime *runtime) {
#ifdef ATKEY_METAL
    if (runtime != NULL && runtime->metal != NULL) {
        return atkey_metal_backend_weight_upload_bytes(runtime->metal);
    }
#else
    (void)runtime;
#endif
    return 0;
}

int atkey_dim(AtkeyRuntime *runtime) {
    return runtime->transformer.config.dim;
}

int atkey_hidden_dim(AtkeyRuntime *runtime) {
    return runtime->transformer.config.hidden_dim;
}

int atkey_layer_count(AtkeyRuntime *runtime) {
    return runtime->transformer.config.n_layers;
}

int atkey_head_count(AtkeyRuntime *runtime) {
    return runtime->transformer.config.n_heads;
}

int atkey_kv_head_count(AtkeyRuntime *runtime) {
    return runtime->transformer.config.n_kv_heads;
}

int atkey_vocab_size(AtkeyRuntime *runtime) {
    return runtime->transformer.config.vocab_size;
}

int atkey_sequence_length(AtkeyRuntime *runtime) {
    return runtime->transformer.config.seq_len;
}

int atkey_filler_count(AtkeyRuntime *runtime) {
    return (int)runtime->filler_count;
}

const float *atkey_embedding_weight(AtkeyRuntime *runtime) {
    return runtime->transformer.weights.token_embedding_table;
}

const float *atkey_attention_rms_weight(AtkeyRuntime *runtime, int layer) {
    int dim = runtime->transformer.config.dim;
    return runtime->transformer.weights.rms_att_weight + (size_t)layer * dim;
}

const float *atkey_query_weight(AtkeyRuntime *runtime, int layer) {
    int dim = runtime->transformer.config.dim;
    return runtime->transformer.weights.wq + (size_t)layer * dim * dim;
}

const float *atkey_key_weight(AtkeyRuntime *runtime, int layer) {
    Config *config = &runtime->transformer.config;
    int dim = config->dim;
    int kv_dim = dim * config->n_kv_heads / config->n_heads;
    return runtime->transformer.weights.wk + (size_t)layer * dim * kv_dim;
}

const float *atkey_value_weight(AtkeyRuntime *runtime, int layer) {
    Config *config = &runtime->transformer.config;
    int dim = config->dim;
    int kv_dim = dim * config->n_kv_heads / config->n_heads;
    return runtime->transformer.weights.wv + (size_t)layer * dim * kv_dim;
}

const float *atkey_attention_output_weight(AtkeyRuntime *runtime, int layer) {
    int dim = runtime->transformer.config.dim;
    return runtime->transformer.weights.wo + (size_t)layer * dim * dim;
}

const float *atkey_ffn_rms_weight(AtkeyRuntime *runtime, int layer) {
    int dim = runtime->transformer.config.dim;
    return runtime->transformer.weights.rms_ffn_weight + (size_t)layer * dim;
}

const float *atkey_ffn_gate_weight(AtkeyRuntime *runtime, int layer) {
    Config *config = &runtime->transformer.config;
    return runtime->transformer.weights.w1 +
        (size_t)layer * config->dim * config->hidden_dim;
}

const float *atkey_ffn_up_weight(AtkeyRuntime *runtime, int layer) {
    Config *config = &runtime->transformer.config;
    return runtime->transformer.weights.w3 +
        (size_t)layer * config->dim * config->hidden_dim;
}

const float *atkey_ffn_down_weight(AtkeyRuntime *runtime, int layer) {
    Config *config = &runtime->transformer.config;
    return runtime->transformer.weights.w2 +
        (size_t)layer * config->hidden_dim * config->dim;
}

const float *atkey_final_rms_weight(AtkeyRuntime *runtime) {
    return runtime->transformer.weights.rms_final_weight;
}

const float *atkey_output_weight(AtkeyRuntime *runtime) {
    return runtime->transformer.weights.wcls;
}

int atkey_embedding_filler_id(void) {
    return 0;
}

int atkey_layer_filler_id(int layer, int stage) {
    return (int)atkey_layer_filler(layer, stage);
}

int atkey_final_rms_filler_id(AtkeyRuntime *runtime) {
    return (int)atkey_final_rms_filler(&runtime->transformer.config);
}

int atkey_output_filler_id(AtkeyRuntime *runtime) {
    return (int)atkey_output_filler(&runtime->transformer.config);
}

size_t atkey_filler_calls(AtkeyRuntime *runtime, int filler) {
    if (filler < 0 || (size_t)filler >= runtime->filler_count) return 0;
    return runtime->filler_calls[filler];
}

size_t atkey_filler_scalar_reads(AtkeyRuntime *runtime, int filler) {
    if (filler < 0 || (size_t)filler >= runtime->filler_count) return 0;
    return runtime->filler_scalar_reads[filler];
}

void atkey_embedding_apply(
    AtkeyRuntime *runtime,
    int filler,
    float *output,
    int token,
    const float *weights,
    int vocab_size,
    int dim
) {
    if (token < 0 || token >= vocab_size) abort();
    memcpy(
        output,
        weights + (size_t)token * dim,
        (size_t)dim * sizeof(float)
    );
    atkey_note_filler(runtime, filler, (size_t)dim);
}

void atkey_embedding_family_apply(
    AtkeyRuntime *runtime,
    int filler,
    float *outputs,
    const int *tokens,
    int count,
    const float *weights,
    int vocab_size,
    int dim
) {
    if (count < 0 || vocab_size <= 0 || dim <= 0) abort();
    unsigned char *present = calloc((size_t)vocab_size, sizeof(*present));
    if (present == NULL) abort();
    size_t scalar_reads = 0;
    for (int occurrence = 0; occurrence < count; occurrence++) {
        int token = tokens[occurrence];
        if (token < 0 || token >= vocab_size) abort();
        present[token] = 1;
    }
#ifdef ATKEY_METAL
    if (runtime->metal != NULL) {
        size_t scalar_reads = 0;
        for (int token = 0; token < vocab_size; token++) {
            if (present[token]) scalar_reads += (size_t)dim;
        }
        if (!atkey_metal_embedding_family(
                runtime->metal,
                outputs,
                tokens,
                count,
                weights,
                vocab_size,
                dim)) {
            fprintf(stderr, "atkey term: Metal embedding family failed\n");
            abort();
        }
        free(present);
        atkey_note_filler(runtime, filler, scalar_reads);
        return;
    }
#endif
    for (int token = 0; token < vocab_size; token++) {
        if (!present[token]) continue;
        for (int lane = 0; lane < dim; lane++) {
            float weight = weights[(size_t)token * dim + lane];
            scalar_reads++;
            for (int occurrence = 0; occurrence < count; occurrence++) {
                if (tokens[occurrence] == token) {
                    outputs[(size_t)occurrence * dim + lane] = weight;
                }
            }
        }
    }
    free(present);
    atkey_note_filler(runtime, filler, scalar_reads);
}

void atkey_rms_apply(
    AtkeyRuntime *runtime,
    int filler,
    float *output,
    const float *input,
    const float *weights,
    int width
) {
    rmsnorm(output, (float *)input, (float *)weights, width);
    atkey_note_filler(runtime, filler, (size_t)width);
}

void atkey_rms_family_apply(
    AtkeyRuntime *runtime,
    int filler,
    float *outputs,
    const float *inputs,
    int count,
    const float *weights,
    int width
) {
    if (count < 0 || width <= 0) abort();
#ifdef ATKEY_METAL
    if (runtime->metal != NULL) {
        if (!atkey_metal_rms_family(
                runtime->metal,
                outputs,
                inputs,
                count,
                weights,
                width)) {
            fprintf(stderr, "atkey term: Metal RMS family failed\n");
            abort();
        }
        atkey_note_filler(runtime, filler, (size_t)width);
        return;
    }
#endif
    float *scales = malloc((size_t)count * sizeof(*scales));
    if (scales == NULL) abort();
    for (int occurrence = 0; occurrence < count; occurrence++) {
        const float *input = inputs + (size_t)occurrence * width;
        float sum_squares = 0.0f;
        for (int lane = 0; lane < width; lane++) {
            sum_squares += input[lane] * input[lane];
        }
        sum_squares /= width;
        sum_squares += 1e-5f;
        scales[occurrence] = 1.0f / sqrtf(sum_squares);
    }
    for (int lane = 0; lane < width; lane++) {
        float weight = weights[lane];
        for (int occurrence = 0; occurrence < count; occurrence++) {
            const float *input = inputs + (size_t)occurrence * width;
            float *output = outputs + (size_t)occurrence * width;
            output[lane] = weight * (scales[occurrence] * input[lane]);
        }
    }
    free(scales);
    atkey_note_filler(runtime, filler, (size_t)width);
}

void atkey_matmul_apply(
    AtkeyRuntime *runtime,
    int filler,
    float *output,
    const float *input,
    const float *weights,
    int input_width,
    int output_width
) {
    matmul(
        output,
        (float *)input,
        (float *)weights,
        input_width,
        output_width
    );
    atkey_note_filler(
        runtime,
        filler,
        (size_t)input_width * output_width
    );
}

void atkey_matmul_family_apply(
    AtkeyRuntime *runtime,
    int filler,
    float *outputs,
    const float *inputs,
    int count,
    const float *weights,
    int input_width,
    int output_width
) {
    if (count < 0 || input_width <= 0 || output_width <= 0) abort();
#ifdef ATKEY_METAL
    if (runtime->metal != NULL) {
        if (!atkey_metal_matmul_family(
                runtime->metal,
                outputs,
                inputs,
                count,
                weights,
                input_width,
                output_width)) {
            fprintf(stderr, "atkey term: Metal matrix family failed\n");
            abort();
        }
        atkey_note_filler(
            runtime,
            filler,
            (size_t)input_width * output_width
        );
        return;
    }
#endif
    memset(
        outputs,
        0,
        (size_t)count * output_width * sizeof(*outputs)
    );
    for (int output_lane = 0; output_lane < output_width; output_lane++) {
        for (int input_lane = 0; input_lane < input_width; input_lane++) {
            float weight =
                weights[(size_t)output_lane * input_width + input_lane];
            for (int occurrence = 0; occurrence < count; occurrence++) {
                outputs[(size_t)occurrence * output_width + output_lane] +=
                    weight *
                    inputs[(size_t)occurrence * input_width + input_lane];
            }
        }
    }
    atkey_note_filler(
        runtime,
        filler,
        (size_t)input_width * output_width
    );
}

void atkey_add(float *output, const float *left, const float *right, int width) {
    for (int lane = 0; lane < width; lane++) {
        output[lane] = left[lane] + right[lane];
    }
}

void atkey_swiglu(
    float *output,
    const float *raw_gate,
    const float *sigmoid_input,
    const float *up,
    int width
) {
    for (int lane = 0; lane < width; lane++) {
        float value = raw_gate[lane];
        value *= 1.0f / (1.0f + expf(-sigmoid_input[lane]));
        output[lane] = value * up[lane];
    }
}

void atkey_rope(
    float *query_output,
    float *key_output,
    const float *query,
    const float *key,
    int position,
    int dim,
    int kv_dim,
    int head_size
) {
    if (query_output != query) {
        memcpy(query_output, query, (size_t)dim * sizeof(float));
    }
    if (key_output != key) {
        memcpy(key_output, key, (size_t)kv_dim * sizeof(float));
    }
    for (int lane = 0; lane < dim; lane += 2) {
        int head_dim = lane % head_size;
        float frequency = 1.0f /
            powf(10000.0f, head_dim / (float)head_size);
        float angle = position * frequency;
        float real = cosf(angle);
        float imaginary = sinf(angle);
        float q0 = query_output[lane];
        float q1 = query_output[lane + 1];
        query_output[lane] = q0 * real - q1 * imaginary;
        query_output[lane + 1] = q0 * imaginary + q1 * real;
        if (lane < kv_dim) {
            float k0 = key_output[lane];
            float k1 = key_output[lane + 1];
            key_output[lane] = k0 * real - k1 * imaginary;
            key_output[lane + 1] = k0 * imaginary + k1 * real;
        }
    }
}

void atkey_attention(
    float *output,
    const float *query,
    const float *const *keys,
    const float *const *values,
    int count,
    int dim,
    int n_heads,
    int n_kv_heads
) {
    if (count <= 0) abort();
    int head_size = dim / n_heads;
    int kv_mul = n_heads / n_kv_heads;
    float *scores = malloc((size_t)count * sizeof(float));
    if (scores == NULL) abort();
    for (int head = 0; head < n_heads; head++) {
        const float *head_query = query + head * head_size;
        for (int timestep = 0; timestep < count; timestep++) {
            const float *key = keys[timestep] +
                (head / kv_mul) * head_size;
            float score = 0.0f;
            for (int lane = 0; lane < head_size; lane++) {
                score += head_query[lane] * key[lane];
            }
            scores[timestep] = score / sqrtf(head_size);
        }
        softmax(scores, count);
        float *head_output = output + head * head_size;
        memset(head_output, 0, (size_t)head_size * sizeof(float));
        for (int timestep = 0; timestep < count; timestep++) {
            const float *value = values[timestep] +
                (head / kv_mul) * head_size;
            float amount = scores[timestep];
            for (int lane = 0; lane < head_size; lane++) {
                head_output[lane] += amount * value[lane];
            }
        }
    }
    free(scores);
}

int *atkey_encode(
    AtkeyRuntime *runtime,
    const char *text,
    int *count
) {
    if (runtime == NULL || text == NULL || count == NULL) return NULL;
    size_t capacity = strlen(text) + 3;
    int *tokens = malloc(capacity * sizeof(int));
    if (tokens == NULL) return NULL;
    encode(&runtime->tokenizer, (char *)text, 1, 0, tokens, count);
    return tokens;
}

void atkey_free_tokens(int *tokens) {
    free(tokens);
}

const char *atkey_decode(
    AtkeyRuntime *runtime,
    int previous,
    int token
) {
    return decode(&runtime->tokenizer, previous, token);
}

void atkey_print_piece(
    AtkeyRuntime *runtime,
    int previous,
    int token
) {
    safe_printf(decode(&runtime->tokenizer, previous, token));
    fflush(stdout);
}

#ifdef ATKEY_REFERENCE_TEST_API
void atkey_reference_sequence_logits(
    AtkeyRuntime *runtime,
    const int *tokens,
    int count,
    float *logits
) {
    if (runtime == NULL || tokens == NULL || count <= 0 || logits == NULL) {
        abort();
    }
    float *current = NULL;
    for (int position = 0; position < count; position++) {
        current = forward(
            &runtime->transformer,
            tokens[position],
            position
        );
    }
    memcpy(
        logits,
        current,
        (size_t)runtime->transformer.config.vocab_size * sizeof(*logits)
    );
}
#endif
