/*
 * DO NOT USE AS THE INFERENCE BACKEND.
 *
 * This adapter was designed around already-materialized float/token families.
 * It consequently supports the rejected fixed-token evaluator rather than a
 * higher-order Select/Cont term.  It is retained only for audit history.
 */

#include "term_backend.h"

#include <stdio.h>
#include <stdlib.h>

static void backend_error(const char *message) {
    fprintf(stderr, "term backend: %s\n", message);
    abort();
}

static void require_backend(const TermBackend *backend) {
    if (backend == NULL || backend->ops == NULL || backend->state == NULL) {
        backend_error("invalid backend");
    }
}

const char *term_backend_name(const TermBackend *backend) {
    require_backend(backend);
    return backend->ops->name;
}

const TermBackendConfig *term_backend_config(const TermBackend *backend) {
    require_backend(backend);
    return &backend->config;
}

void term_backend_free(TermBackend *backend) {
    if (backend == NULL) return;
    if (backend->ops != NULL && backend->ops->destroy != NULL &&
        backend->state != NULL) {
        backend->ops->destroy(backend->state);
    }
    free(backend);
}

bool term_backend_filler(
    TermBackend *backend,
    TermFillerSlot slot,
    int layer,
    TermFiller *result
) {
    require_backend(backend);
    if (result == NULL || backend->ops->filler == NULL) {
        backend_error("invalid filler request");
    }
    return backend->ops->filler(backend->state, slot, layer, result);
}

void term_backend_embedding_family(
    TermBackend *backend,
    const TermFiller *filler,
    float *outputs,
    const int *tokens,
    int count
) {
    require_backend(backend);
    if (filler == NULL || filler->kind != TERM_FILLER_EMBEDDING ||
        backend->ops->embedding_family == NULL) {
        backend_error("invalid embedding family");
    }
    backend->ops->embedding_family(
        backend->state,
        filler,
        outputs,
        tokens,
        count
    );
}

void term_backend_hidden_family(
    TermBackend *backend,
    const TermFiller *filler,
    float *outputs,
    const float *inputs,
    int count
) {
    require_backend(backend);
    if (filler == NULL || filler->kind == TERM_FILLER_EMBEDDING ||
        backend->ops->hidden_family == NULL) {
        backend_error("invalid hidden family");
    }
    backend->ops->hidden_family(
        backend->state,
        filler,
        outputs,
        inputs,
        count
    );
}

void term_backend_add(
    TermBackend *backend,
    float *output,
    const float *left,
    const float *right,
    int width
) {
    require_backend(backend);
    backend->ops->add(backend->state, output, left, right, width);
}

void term_backend_swiglu(
    TermBackend *backend,
    float *output,
    const float *gate,
    const float *up,
    int width
) {
    require_backend(backend);
    backend->ops->swiglu(backend->state, output, gate, up, width);
}

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
) {
    require_backend(backend);
    backend->ops->rope(
        backend->state,
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
) {
    require_backend(backend);
    backend->ops->attention(
        backend->state,
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

int *term_backend_encode(TermBackend *backend, const char *text, int *count) {
    require_backend(backend);
    return backend->ops->encode(backend->state, text, count);
}

void term_backend_free_tokens(TermBackend *backend, int *tokens) {
    require_backend(backend);
    backend->ops->free_tokens(backend->state, tokens);
}

const char *term_backend_decode(TermBackend *backend, int previous, int token) {
    require_backend(backend);
    return backend->ops->decode(backend->state, previous, token);
}

int term_backend_filler_count(TermBackend *backend) {
    require_backend(backend);
    return backend->ops->filler_count(backend->state);
}

size_t term_backend_filler_crossings(TermBackend *backend, int filler_id) {
    require_backend(backend);
    return backend->ops->filler_crossings(backend->state, filler_id);
}

size_t term_backend_filler_scalar_reads(TermBackend *backend, int filler_id) {
    require_backend(backend);
    return backend->ops->filler_scalar_reads(backend->state, filler_id);
}
