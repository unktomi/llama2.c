/*
 * DO NOT USE AS THE INFERENCE BACKEND.
 *
 * The interface consumes concrete token and activation families, so selection
 * has already been externalized before it is entered.  That is the wrong
 * boundary for the requested recursively composed inference term.
 */

#ifndef TERM_BACKEND_H
#define TERM_BACKEND_H

#include <stdbool.h>
#include <stddef.h>

/*
 * Numerical boundary for the unevaluated inference term.
 *
 * The term knows Llama's learned holes and their shapes, but it does not know
 * whether a filler is backed by a llama2.c float array, a GGUF/ggml tensor, or
 * another kernel implementation.  In particular, this interface deliberately
 * has no whole-model or forward(prefix) operation.
 */

typedef struct TermBackend TermBackend;

typedef struct {
    int dim;
    int hidden_dim;
    int layers;
    int heads;
    int kv_heads;
    int vocab;
    int sequence_length;
} TermBackendConfig;

typedef enum {
    TERM_FILLER_EMBEDDING,
    TERM_FILLER_RMS,
    TERM_FILLER_LINEAR,
} TermFillerKind;

typedef enum {
    TERM_SLOT_TOKEN_EMBEDDING,
    TERM_SLOT_ATTENTION_RMS,
    TERM_SLOT_QUERY,
    TERM_SLOT_KEY,
    TERM_SLOT_VALUE,
    TERM_SLOT_ATTENTION_OUTPUT,
    TERM_SLOT_FFN_RMS,
    TERM_SLOT_FFN_GATE,
    TERM_SLOT_FFN_UP,
    TERM_SLOT_FFN_DOWN,
    TERM_SLOT_FINAL_RMS,
    TERM_SLOT_OUTPUT,
} TermFillerSlot;

typedef struct {
    TermFillerKind kind;
    TermFillerSlot slot;
    int layer;
    int id;
    int input_width;
    int output_width;
    int vocab;
    const void *handle;
} TermFiller;

typedef struct {
    const char *name;
    void (*destroy)(void *state);
    bool (*filler)(
        void *state,
        TermFillerSlot slot,
        int layer,
        TermFiller *result
    );
    void (*embedding_family)(
        void *state,
        const TermFiller *filler,
        float *outputs,
        const int *tokens,
        int count
    );
    void (*hidden_family)(
        void *state,
        const TermFiller *filler,
        float *outputs,
        const float *inputs,
        int count
    );
    void (*add)(
        void *state,
        float *output,
        const float *left,
        const float *right,
        int width
    );
    void (*swiglu)(
        void *state,
        float *output,
        const float *gate,
        const float *up,
        int width
    );
    void (*rope)(
        void *state,
        float *query_output,
        float *key_output,
        const float *query,
        const float *key,
        int position,
        int dim,
        int kv_dim,
        int head_size
    );
    void (*attention)(
        void *state,
        float *output,
        const float *query,
        const float *const *keys,
        const float *const *values,
        int count,
        int dim,
        int heads,
        int kv_heads
    );
    int *(*encode)(void *state, const char *text, int *count);
    void (*free_tokens)(void *state, int *tokens);
    const char *(*decode)(void *state, int previous, int token);
    int (*filler_count)(void *state);
    size_t (*filler_crossings)(void *state, int filler_id);
    size_t (*filler_scalar_reads)(void *state, int filler_id);
} TermBackendOps;

struct TermBackend {
    const TermBackendOps *ops;
    void *state;
    TermBackendConfig config;
};

const char *term_backend_name(const TermBackend *backend);
const TermBackendConfig *term_backend_config(const TermBackend *backend);
void term_backend_free(TermBackend *backend);

bool term_backend_filler(
    TermBackend *backend,
    TermFillerSlot slot,
    int layer,
    TermFiller *result
);
void term_backend_embedding_family(
    TermBackend *backend,
    const TermFiller *filler,
    float *outputs,
    const int *tokens,
    int count
);
void term_backend_hidden_family(
    TermBackend *backend,
    const TermFiller *filler,
    float *outputs,
    const float *inputs,
    int count
);
void term_backend_add(
    TermBackend *backend,
    float *output,
    const float *left,
    const float *right,
    int width
);
void term_backend_swiglu(
    TermBackend *backend,
    float *output,
    const float *gate,
    const float *up,
    int width
);
void term_backend_rope(
    TermBackend *backend,
    float *query_output,
    float *key_output,
    const float *query,
    const float *key,
    int position,
    int dim,
    int kv_dim,
    int head_size
);
void term_backend_attention(
    TermBackend *backend,
    float *output,
    const float *query,
    const float *const *keys,
    const float *const *values,
    int count,
    int dim,
    int heads,
    int kv_heads
);

int *term_backend_encode(TermBackend *backend, const char *text, int *count);
void term_backend_free_tokens(TermBackend *backend, int *tokens);
const char *term_backend_decode(TermBackend *backend, int previous, int token);

int term_backend_filler_count(TermBackend *backend);
size_t term_backend_filler_crossings(TermBackend *backend, int filler_id);
size_t term_backend_filler_scalar_reads(TermBackend *backend, int filler_id);

#endif
