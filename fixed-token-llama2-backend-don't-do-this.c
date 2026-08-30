/*
 * DO NOT USE AS THE LLAMA INFERENCE ADAPTER.
 *
 * This adapter lowers the rejected fixed-token family interface to llama2.c
 * kernels.  It cannot place model-produced token selection inside the
 * recursively composed term and is retained only for audit history.
 */

#include "llama2_backend.h"

#include "atkey_term_c.h"

#include <stdlib.h>

typedef struct {
    AtkeyRuntime *runtime;
} Llama2BackendState;

static TermBackendConfig llama2_config(Llama2BackendState *state) {
    return (TermBackendConfig){
        .dim = atkey_dim(state->runtime),
        .hidden_dim = atkey_hidden_dim(state->runtime),
        .layers = atkey_layer_count(state->runtime),
        .heads = atkey_head_count(state->runtime),
        .kv_heads = atkey_kv_head_count(state->runtime),
        .vocab = atkey_vocab_size(state->runtime),
        .sequence_length = atkey_sequence_length(state->runtime),
    };
}

static void llama2_destroy(void *raw_state) {
    Llama2BackendState *state = raw_state;
    if (state == NULL) return;
    atkey_runtime_free(state->runtime);
    free(state);
}

static bool llama2_filler(
    void *raw_state,
    TermFillerSlot slot,
    int layer,
    TermFiller *result
) {
    Llama2BackendState *state = raw_state;
    AtkeyRuntime *runtime = state->runtime;
    TermBackendConfig config = llama2_config(state);
    if (layer < 0 || layer >= config.layers) {
        if (slot != TERM_SLOT_TOKEN_EMBEDDING &&
            slot != TERM_SLOT_FINAL_RMS && slot != TERM_SLOT_OUTPUT) {
            return false;
        }
    }
    *result = (TermFiller){
        .slot = slot,
        .layer = layer,
        .vocab = config.vocab,
    };
    switch (slot) {
        case TERM_SLOT_TOKEN_EMBEDDING:
            result->kind = TERM_FILLER_EMBEDDING;
            result->id = atkey_embedding_filler_id();
            result->input_width = 1;
            result->output_width = config.dim;
            result->handle = atkey_embedding_weight(runtime);
            return true;
        case TERM_SLOT_ATTENTION_RMS:
            result->kind = TERM_FILLER_RMS;
            result->id = atkey_layer_filler_id(layer, 0);
            result->input_width = config.dim;
            result->output_width = config.dim;
            result->handle = atkey_attention_rms_weight(runtime, layer);
            return true;
        case TERM_SLOT_QUERY:
            result->kind = TERM_FILLER_LINEAR;
            result->id = atkey_layer_filler_id(layer, 1);
            result->input_width = config.dim;
            result->output_width = config.dim;
            result->handle = atkey_query_weight(runtime, layer);
            return true;
        case TERM_SLOT_KEY:
            result->kind = TERM_FILLER_LINEAR;
            result->id = atkey_layer_filler_id(layer, 2);
            result->input_width = config.dim;
            result->output_width =
                config.dim * config.kv_heads / config.heads;
            result->handle = atkey_key_weight(runtime, layer);
            return true;
        case TERM_SLOT_VALUE:
            result->kind = TERM_FILLER_LINEAR;
            result->id = atkey_layer_filler_id(layer, 3);
            result->input_width = config.dim;
            result->output_width =
                config.dim * config.kv_heads / config.heads;
            result->handle = atkey_value_weight(runtime, layer);
            return true;
        case TERM_SLOT_ATTENTION_OUTPUT:
            result->kind = TERM_FILLER_LINEAR;
            result->id = atkey_layer_filler_id(layer, 4);
            result->input_width = config.dim;
            result->output_width = config.dim;
            result->handle = atkey_attention_output_weight(runtime, layer);
            return true;
        case TERM_SLOT_FFN_RMS:
            result->kind = TERM_FILLER_RMS;
            result->id = atkey_layer_filler_id(layer, 5);
            result->input_width = config.dim;
            result->output_width = config.dim;
            result->handle = atkey_ffn_rms_weight(runtime, layer);
            return true;
        case TERM_SLOT_FFN_GATE:
            result->kind = TERM_FILLER_LINEAR;
            result->id = atkey_layer_filler_id(layer, 6);
            result->input_width = config.dim;
            result->output_width = config.hidden_dim;
            result->handle = atkey_ffn_gate_weight(runtime, layer);
            return true;
        case TERM_SLOT_FFN_UP:
            result->kind = TERM_FILLER_LINEAR;
            result->id = atkey_layer_filler_id(layer, 7);
            result->input_width = config.dim;
            result->output_width = config.hidden_dim;
            result->handle = atkey_ffn_up_weight(runtime, layer);
            return true;
        case TERM_SLOT_FFN_DOWN:
            result->kind = TERM_FILLER_LINEAR;
            result->id = atkey_layer_filler_id(layer, 8);
            result->input_width = config.hidden_dim;
            result->output_width = config.dim;
            result->handle = atkey_ffn_down_weight(runtime, layer);
            return true;
        case TERM_SLOT_FINAL_RMS:
            result->kind = TERM_FILLER_RMS;
            result->id = atkey_final_rms_filler_id(runtime);
            result->input_width = config.dim;
            result->output_width = config.dim;
            result->handle = atkey_final_rms_weight(runtime);
            return true;
        case TERM_SLOT_OUTPUT:
            result->kind = TERM_FILLER_LINEAR;
            result->id = atkey_output_filler_id(runtime);
            result->input_width = config.dim;
            result->output_width = config.vocab;
            result->handle = atkey_output_weight(runtime);
            return true;
    }
    return false;
}

static void llama2_embedding_family(
    void *raw_state,
    const TermFiller *filler,
    float *outputs,
    const int *tokens,
    int count
) {
    Llama2BackendState *state = raw_state;
    atkey_embedding_family_apply(
        state->runtime,
        filler->id,
        outputs,
        tokens,
        count,
        filler->handle,
        filler->vocab,
        filler->output_width
    );
}

static void llama2_hidden_family(
    void *raw_state,
    const TermFiller *filler,
    float *outputs,
    const float *inputs,
    int count
) {
    Llama2BackendState *state = raw_state;
    if (filler->kind == TERM_FILLER_RMS) {
        atkey_rms_family_apply(
            state->runtime,
            filler->id,
            outputs,
            inputs,
            count,
            filler->handle,
            filler->input_width
        );
        return;
    }
    atkey_matmul_family_apply(
        state->runtime,
        filler->id,
        outputs,
        inputs,
        count,
        filler->handle,
        filler->input_width,
        filler->output_width
    );
}

static void llama2_add(
    void *raw_state,
    float *output,
    const float *left,
    const float *right,
    int width
) {
    (void)raw_state;
    atkey_add(output, left, right, width);
}

static void llama2_swiglu(
    void *raw_state,
    float *output,
    const float *gate,
    const float *up,
    int width
) {
    (void)raw_state;
    atkey_swiglu(output, gate, gate, up, width);
}

static void llama2_rope(
    void *raw_state,
    float *query_output,
    float *key_output,
    const float *query,
    const float *key,
    int position,
    int dim,
    int kv_dim,
    int head_size
) {
    (void)raw_state;
    atkey_rope(
        query_output,
        key_output,
        query,
        key,
        position,
        dim,
        kv_dim,
        head_size
    );
}

static void llama2_attention(
    void *raw_state,
    float *output,
    const float *query,
    const float *const *keys,
    const float *const *values,
    int count,
    int dim,
    int heads,
    int kv_heads
) {
    (void)raw_state;
    atkey_attention(
        output,
        query,
        keys,
        values,
        count,
        dim,
        heads,
        kv_heads
    );
}

static int *llama2_encode(void *raw_state, const char *text, int *count) {
    Llama2BackendState *state = raw_state;
    return atkey_encode(state->runtime, text, count);
}

static void llama2_free_tokens(void *raw_state, int *tokens) {
    (void)raw_state;
    atkey_free_tokens(tokens);
}

static const char *llama2_decode(void *raw_state, int previous, int token) {
    Llama2BackendState *state = raw_state;
    return atkey_decode(state->runtime, previous, token);
}

static int llama2_filler_count(void *raw_state) {
    Llama2BackendState *state = raw_state;
    return atkey_filler_count(state->runtime);
}

static size_t llama2_filler_crossings(void *raw_state, int filler_id) {
    Llama2BackendState *state = raw_state;
    return atkey_filler_calls(state->runtime, filler_id);
}

static size_t llama2_filler_scalar_reads(void *raw_state, int filler_id) {
    Llama2BackendState *state = raw_state;
    return atkey_filler_scalar_reads(state->runtime, filler_id);
}

static const TermBackendOps LLAMA2_OPS = {
    .name = "llama2.c-f32",
    .destroy = llama2_destroy,
    .filler = llama2_filler,
    .embedding_family = llama2_embedding_family,
    .hidden_family = llama2_hidden_family,
    .add = llama2_add,
    .swiglu = llama2_swiglu,
    .rope = llama2_rope,
    .attention = llama2_attention,
    .encode = llama2_encode,
    .free_tokens = llama2_free_tokens,
    .decode = llama2_decode,
    .filler_count = llama2_filler_count,
    .filler_crossings = llama2_filler_crossings,
    .filler_scalar_reads = llama2_filler_scalar_reads,
};

TermBackend *llama2_backend_new(
    const char *checkpoint_path,
    const char *tokenizer_path
) {
    Llama2BackendState *state = calloc(1, sizeof(*state));
    TermBackend *backend = calloc(1, sizeof(*backend));
    if (state == NULL || backend == NULL) {
        free(state);
        free(backend);
        return NULL;
    }
    state->runtime = atkey_runtime_new(checkpoint_path, tokenizer_path);
    if (state->runtime == NULL) {
        free(state);
        free(backend);
        return NULL;
    }
    backend->ops = &LLAMA2_OPS;
    backend->state = state;
    backend->config = llama2_config(state);
    return backend;
}
