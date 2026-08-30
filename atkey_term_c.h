#ifndef ATKEY_TERM_C_H
#define ATKEY_TERM_C_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct AtkeyRuntime AtkeyRuntime;

AtkeyRuntime *atkey_runtime_new(
    const char *checkpoint_path,
    const char *tokenizer_path
);
void atkey_runtime_free(AtkeyRuntime *runtime);
bool atkey_enable_metal(AtkeyRuntime *runtime, const char *library_path);
const char *atkey_backend_name(AtkeyRuntime *runtime);
const char *atkey_backend_device_name(AtkeyRuntime *runtime);
uint64_t atkey_backend_dispatch_count(AtkeyRuntime *runtime);
uint64_t atkey_backend_weight_upload_count(AtkeyRuntime *runtime);
uint64_t atkey_backend_weight_upload_bytes(AtkeyRuntime *runtime);

int atkey_dim(AtkeyRuntime *runtime);
int atkey_hidden_dim(AtkeyRuntime *runtime);
int atkey_layer_count(AtkeyRuntime *runtime);
int atkey_head_count(AtkeyRuntime *runtime);
int atkey_kv_head_count(AtkeyRuntime *runtime);
int atkey_vocab_size(AtkeyRuntime *runtime);
int atkey_sequence_length(AtkeyRuntime *runtime);
int atkey_filler_count(AtkeyRuntime *runtime);

const float *atkey_embedding_weight(AtkeyRuntime *runtime);
const float *atkey_attention_rms_weight(AtkeyRuntime *runtime, int layer);
const float *atkey_query_weight(AtkeyRuntime *runtime, int layer);
const float *atkey_key_weight(AtkeyRuntime *runtime, int layer);
const float *atkey_value_weight(AtkeyRuntime *runtime, int layer);
const float *atkey_attention_output_weight(AtkeyRuntime *runtime, int layer);
const float *atkey_ffn_rms_weight(AtkeyRuntime *runtime, int layer);
const float *atkey_ffn_gate_weight(AtkeyRuntime *runtime, int layer);
const float *atkey_ffn_up_weight(AtkeyRuntime *runtime, int layer);
const float *atkey_ffn_down_weight(AtkeyRuntime *runtime, int layer);
const float *atkey_final_rms_weight(AtkeyRuntime *runtime);
const float *atkey_output_weight(AtkeyRuntime *runtime);

int atkey_embedding_filler_id(void);
int atkey_layer_filler_id(int layer, int stage);
int atkey_final_rms_filler_id(AtkeyRuntime *runtime);
int atkey_output_filler_id(AtkeyRuntime *runtime);
size_t atkey_filler_calls(AtkeyRuntime *runtime, int filler);
size_t atkey_filler_scalar_reads(AtkeyRuntime *runtime, int filler);

void atkey_embedding_apply(
    AtkeyRuntime *runtime,
    int filler,
    float *output,
    int token,
    const float *weights,
    int vocab_size,
    int dim
);
void atkey_embedding_family_apply(
    AtkeyRuntime *runtime,
    int filler,
    float *outputs,
    const int *tokens,
    int count,
    const float *weights,
    int vocab_size,
    int dim
);
void atkey_rms_apply(
    AtkeyRuntime *runtime,
    int filler,
    float *output,
    const float *input,
    const float *weights,
    int width
);
void atkey_rms_family_apply(
    AtkeyRuntime *runtime,
    int filler,
    float *outputs,
    const float *inputs,
    int count,
    const float *weights,
    int width
);
void atkey_matmul_apply(
    AtkeyRuntime *runtime,
    int filler,
    float *output,
    const float *input,
    const float *weights,
    int input_width,
    int output_width
);
void atkey_matmul_family_apply(
    AtkeyRuntime *runtime,
    int filler,
    float *outputs,
    const float *inputs,
    int count,
    const float *weights,
    int input_width,
    int output_width
);
void atkey_add(
    float *output,
    const float *left,
    const float *right,
    int width
);
void atkey_swiglu(
    float *output,
    const float *raw_gate,
    const float *sigmoid_input,
    const float *up,
    int width
);
void atkey_rope(
    float *query_output,
    float *key_output,
    const float *query,
    const float *key,
    int position,
    int dim,
    int kv_dim,
    int head_size
);
void atkey_attention(
    float *output,
    const float *query,
    const float *const *keys,
    const float *const *values,
    int count,
    int dim,
    int n_heads,
    int n_kv_heads
);

int *atkey_encode(AtkeyRuntime *runtime, const char *text, int *count);
void atkey_free_tokens(int *tokens);
const char *atkey_decode(AtkeyRuntime *runtime, int previous, int token);
void atkey_print_piece(AtkeyRuntime *runtime, int previous, int token);

#ifdef ATKEY_REFERENCE_TEST_API
void atkey_reference_sequence_logits(
    AtkeyRuntime *runtime,
    const int *tokens,
    int count,
    float *logits
);
#endif

#endif
