/*
 * Finite open-transformer term with Escardo-style selection strength.
 *
 * A hypothetical token is never emitted by a branch. Every demanded token
 * constructor is embedded and traverses the original RMSNorm, attention,
 * residual, SwiGLU, and output kernels inside its suspended continuation.
 * Ready applications of the same learned filler are batched by WeightScope.
 * No call to llama2.c's eager forward() occurs in this file.
 *
 * The continuation result is structured rather than scalar:
 *
 *     SelectionOutcome =
 *         (token path, layerwise K/V summary, final hidden, covector family).
 *
 * Once all child continuations of one Select are ready, the same root
 * observer is restricted to each bound continuation.  For fixed left company
 * L and the selected suffix s = b(x), it returns one causal-posterior frame
 *
 *     observe(L, s)[x'] = log P_model(x' ++ s | L).
 *
 * Every coordinate in a frame therefore has the same left and right company.
 * Coordinates from different suffix frames are never compared or added.  A
 * branch's model-native rating is its own coordinate after normalizing that
 * frame: log P(x | L, s, finite support).  Finite Select maximizes this one
 * returned observation for every x; it does not fold ratings across token
 * positions.  Attainment (whether x is the maximum in its own frame) remains
 * a fixed-point diagnostic.  The selected outcome retains the complete
 * position-indexed family of frames, and only the root emits its token tuple.
 *
 * This executable currently exposes only exact evaluation of an explicitly
 * requested finite local support. The former wall-clock implementation was
 * removed from this path because it backed a scalar path likelihood; its
 * provenance remains in git history and the named *-don't-do-this sources.
 */

#include "atkey_term_c.h"

#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { ESCARDO_SEQUENCE_DELIMITER = 1 };

_Noreturn static void escardo_fail(const char *message) {
    fprintf(stderr, "escardo: %s\n", message);
    exit(EXIT_FAILURE);
}

static void *escardo_calloc(size_t count, size_t width) {
    if (width != 0 && count > SIZE_MAX / width) {
        escardo_fail("allocation size overflow");
    }
    void *memory = calloc(count == 0 ? 1 : count, width == 0 ? 1 : width);
    if (memory == NULL) escardo_fail("out of memory");
    return memory;
}

typedef struct ArenaBlock ArenaBlock;
struct ArenaBlock {
    ArenaBlock *next;
    size_t used;
    size_t capacity;
    max_align_t alignment;
    unsigned char bytes[];
};

typedef struct {
    ArenaBlock *head;
    size_t block_size;
} Arena;

static size_t arena_aligned_size(size_t size) {
    size_t alignment = _Alignof(max_align_t);
    if (size > SIZE_MAX - alignment + 1) escardo_fail("arena size overflow");
    return (size + alignment - 1) & ~(alignment - 1);
}

static void *arena_allocate(Arena *arena, size_t size) {
    size = arena_aligned_size(size == 0 ? 1 : size);
    ArenaBlock *block = arena->head;
    if (block == NULL || block->capacity - block->used < size) {
        size_t capacity = arena->block_size;
        if (capacity < size) capacity = size;
        if (capacity > SIZE_MAX - sizeof(*block)) {
            escardo_fail("arena block size overflow");
        }
        block = escardo_calloc(1, sizeof(*block) + capacity);
        block->capacity = capacity;
        block->next = arena->head;
        arena->head = block;
    }
    void *result = block->bytes + block->used;
    block->used += size;
    return result;
}

static void arena_release(Arena *arena) {
    ArenaBlock *block = arena->head;
    while (block != NULL) {
        ArenaBlock *next = block->next;
        free(block);
        block = next;
    }
    arena->head = NULL;
}

typedef struct {
    int width;
    float values[];
} Tensor;

static Tensor *tensor_new(Arena *arena, int width) {
    if (width <= 0) escardo_fail("nonpositive tensor width");
    if ((size_t)width > (SIZE_MAX - sizeof(Tensor)) / sizeof(float)) {
        escardo_fail("tensor size overflow");
    }
    Tensor *tensor = arena_allocate(
        arena,
        sizeof(*tensor) + (size_t)width * sizeof(*tensor->values)
    );
    tensor->width = width;
    return tensor;
}

typedef struct Scheduler Scheduler;
typedef struct WeightScope WeightScope;
typedef struct WeightRequest WeightRequest;
typedef struct WeightBinding WeightBinding;

typedef void (*TensorContinuation)(void *environment, Tensor *value);

typedef enum {
    WEIGHT_EMBEDDING,
    WEIGHT_RMS,
    WEIGHT_MATMUL,
} WeightOperation;

struct WeightBinding {
    const float *weights;
    size_t scalar_count;
    bool plugged;
    uint64_t batch_count;
    uint64_t request_count;
};

struct WeightRequest {
    WeightRequest *next;
    int token;
    Tensor *input;
    TensorContinuation continuation;
    void *continuation_environment;
};

struct WeightScope {
    Scheduler *scheduler;
    WeightBinding *binding;
    WeightOperation operation;
    int filler_id;
    int input_width;
    int output_width;
    int vocab_size;
    WeightRequest *request_head;
    WeightRequest *request_tail;
    WeightScope *ready_next;
    bool ready;
    const char *name;
};

struct Scheduler {
    AtkeyRuntime *runtime;
    Arena *arena;
    WeightScope *ready_head;
    WeightScope *ready_tail;
    bool failed;
};

static void binding_plug(
    WeightBinding *binding,
    const float *weights,
    size_t scalar_count
) {
    if (binding->plugged) escardo_fail("learned binding plugged twice");
    if (weights == NULL || scalar_count == 0) {
        escardo_fail("invalid learned binding");
    }
    binding->weights = weights;
    binding->scalar_count = scalar_count;
    binding->plugged = true;
}

static void scheduler_mark_ready(WeightScope *scope) {
    if (scope->ready) return;
    scope->ready = true;
    scope->ready_next = NULL;
    Scheduler *scheduler = scope->scheduler;
    if (scheduler->ready_tail == NULL) {
        scheduler->ready_head = scope;
    } else {
        scheduler->ready_tail->ready_next = scope;
    }
    scheduler->ready_tail = scope;
}

static void scope_request_tensor(
    WeightScope *scope,
    Tensor *input,
    TensorContinuation continuation,
    void *continuation_environment
) {
    if (scope->operation == WEIGHT_EMBEDDING) {
        escardo_fail("tensor sent to embedding scope");
    }
    if (input == NULL || input->width != scope->input_width ||
        continuation == NULL) {
        escardo_fail("invalid learned tensor request");
    }
    WeightRequest *request = arena_allocate(
        scope->scheduler->arena,
        sizeof(*request)
    );
    *request = (WeightRequest){
        .input = input,
        .continuation = continuation,
        .continuation_environment = continuation_environment,
    };
    if (scope->request_tail == NULL) {
        scope->request_head = request;
    } else {
        scope->request_tail->next = request;
    }
    scope->request_tail = request;
    scheduler_mark_ready(scope);
}

static void scope_request_token(
    WeightScope *scope,
    int token,
    TensorContinuation continuation,
    void *continuation_environment
) {
    if (scope->operation != WEIGHT_EMBEDDING || token < 0 ||
        token >= scope->vocab_size || continuation == NULL) {
        escardo_fail("invalid learned embedding request");
    }
    WeightRequest *request = arena_allocate(
        scope->scheduler->arena,
        sizeof(*request)
    );
    *request = (WeightRequest){
        .token = token,
        .continuation = continuation,
        .continuation_environment = continuation_environment,
    };
    if (scope->request_tail == NULL) {
        scope->request_head = request;
    } else {
        scope->request_tail->next = request;
    }
    scope->request_tail = request;
    scheduler_mark_ready(scope);
}

static int request_list_count(WeightRequest *request) {
    int count = 0;
    for (; request != NULL; request = request->next) {
        if (count == INT_MAX) escardo_fail("too many learned requests");
        count++;
    }
    return count;
}

static void scheduler_apply_scope(Scheduler *scheduler, WeightScope *scope) {
    WeightRequest *requests = scope->request_head;
    scope->request_head = NULL;
    scope->request_tail = NULL;
    scope->ready = false;
    scope->ready_next = NULL;
    if (requests == NULL) return;
    if (scope->binding == NULL || !scope->binding->plugged) {
        escardo_fail("unplugged learned scope observed");
    }

    int count = request_list_count(requests);
    size_t output_elements = (size_t)count * (size_t)scope->output_width;
    float *outputs = escardo_calloc(output_elements, sizeof(*outputs));

    if (scope->operation == WEIGHT_EMBEDDING) {
        int *tokens = escardo_calloc((size_t)count, sizeof(*tokens));
        int index = 0;
        for (WeightRequest *request = requests; request != NULL;
             request = request->next) {
            tokens[index++] = request->token;
        }
        atkey_embedding_family_apply(
            scheduler->runtime,
            scope->filler_id,
            outputs,
            tokens,
            count,
            scope->binding->weights,
            scope->vocab_size,
            scope->output_width
        );
        free(tokens);
    } else {
        size_t input_elements = (size_t)count * (size_t)scope->input_width;
        float *inputs = escardo_calloc(input_elements, sizeof(*inputs));
        int index = 0;
        for (WeightRequest *request = requests; request != NULL;
             request = request->next) {
            memcpy(
                inputs + (size_t)index * scope->input_width,
                request->input->values,
                (size_t)scope->input_width * sizeof(*inputs)
            );
            index++;
        }
        if (scope->operation == WEIGHT_RMS) {
            atkey_rms_family_apply(
                scheduler->runtime,
                scope->filler_id,
                outputs,
                inputs,
                count,
                scope->binding->weights,
                scope->input_width
            );
        } else {
            atkey_matmul_family_apply(
                scheduler->runtime,
                scope->filler_id,
                outputs,
                inputs,
                count,
                scope->binding->weights,
                scope->input_width,
                scope->output_width
            );
        }
        free(inputs);
    }

    scope->binding->batch_count++;
    scope->binding->request_count += (uint64_t)count;

    int index = 0;
    for (WeightRequest *request = requests; request != NULL;
         request = request->next) {
        Tensor *output = tensor_new(scheduler->arena, scope->output_width);
        memcpy(
            output->values,
            outputs + (size_t)index * scope->output_width,
            (size_t)scope->output_width * sizeof(*outputs)
        );
        index++;
        request->continuation(request->continuation_environment, output);
    }
    free(outputs);
}

static bool scheduler_step(Scheduler *scheduler) {
    WeightScope *scope = scheduler->ready_head;
    if (scope == NULL) return false;
    scheduler->ready_head = scope->ready_next;
    if (scheduler->ready_head == NULL) scheduler->ready_tail = NULL;
    scheduler_apply_scope(scheduler, scope);
    return true;
}

typedef struct ModelTerm ModelTerm;
typedef struct ModelNode ModelNode;
typedef struct ModelChild ModelChild;
typedef struct NodeWaiter NodeWaiter;
typedef struct ModelStep ModelStep;

typedef void (*NodeContinuation)(void *environment, ModelNode *node);

struct NodeWaiter {
    NodeWaiter *next;
    NodeContinuation continuation;
    void *environment;
};

struct ModelChild {
    ModelChild *next;
    int token;
    ModelNode *node;
};

struct ModelNode {
    uint64_t id;
    ModelNode *parent;
    ModelChild *children;
    int token;
    int position;
    bool ready;
    Tensor **keys;
    Tensor **values;
    Tensor **scales;
    Tensor *final_hidden;
    Tensor *logits;
    double log_partition;
    bool log_partition_ready;
    NodeWaiter *waiter_head;
    NodeWaiter *waiter_tail;
};

typedef struct {
    WeightBinding attention_norm_binding;
    WeightBinding query_binding;
    WeightBinding key_binding;
    WeightBinding value_binding;
    WeightBinding attention_output_binding;
    WeightBinding ffn_norm_binding;
    WeightBinding ffn_gate_binding;
    WeightBinding ffn_up_binding;
    WeightBinding ffn_down_binding;
    WeightScope attention_norm;
    WeightScope query;
    WeightScope key;
    WeightScope value;
    WeightScope attention_output;
    WeightScope ffn_norm;
    WeightScope ffn_gate;
    WeightScope ffn_up;
    WeightScope ffn_down;
} LayerTerm;

struct ModelTerm {
    AtkeyRuntime *runtime;
    Arena arena;
    Scheduler scheduler;
    int dim;
    int hidden_dim;
    int layer_count;
    int head_count;
    int kv_head_count;
    int kv_dim;
    int head_size;
    int vocab_size;
    int sequence_length;
    WeightBinding embedding_output_binding;
    WeightBinding output_binding;
    WeightBinding final_norm_binding;
    WeightScope embedding;
    LayerTerm *layers;
    WeightScope final_norm;
    WeightScope output;
    uint64_t next_node_id;
    uint64_t model_steps;
};

static void init_scope(
    ModelTerm *model,
    WeightScope *scope,
    WeightBinding *binding,
    WeightOperation operation,
    int filler_id,
    int input_width,
    int output_width,
    int vocab_size,
    const char *name
) {
    *scope = (WeightScope){
        .scheduler = &model->scheduler,
        .binding = binding,
        .operation = operation,
        .filler_id = filler_id,
        .input_width = input_width,
        .output_width = output_width,
        .vocab_size = vocab_size,
        .name = name,
    };
}

static void plug_layer(ModelTerm *model, LayerTerm *layer, int index) {
    int dim = model->dim;
    int hidden = model->hidden_dim;
    int kv = model->kv_dim;
    binding_plug(
        &layer->attention_norm_binding,
        atkey_attention_rms_weight(model->runtime, index),
        (size_t)dim
    );
    binding_plug(
        &layer->query_binding,
        atkey_query_weight(model->runtime, index),
        (size_t)dim * dim
    );
    binding_plug(
        &layer->key_binding,
        atkey_key_weight(model->runtime, index),
        (size_t)dim * kv
    );
    binding_plug(
        &layer->value_binding,
        atkey_value_weight(model->runtime, index),
        (size_t)dim * kv
    );
    binding_plug(
        &layer->attention_output_binding,
        atkey_attention_output_weight(model->runtime, index),
        (size_t)dim * dim
    );
    binding_plug(
        &layer->ffn_norm_binding,
        atkey_ffn_rms_weight(model->runtime, index),
        (size_t)dim
    );
    binding_plug(
        &layer->ffn_gate_binding,
        atkey_ffn_gate_weight(model->runtime, index),
        (size_t)dim * hidden
    );
    binding_plug(
        &layer->ffn_up_binding,
        atkey_ffn_up_weight(model->runtime, index),
        (size_t)dim * hidden
    );
    binding_plug(
        &layer->ffn_down_binding,
        atkey_ffn_down_weight(model->runtime, index),
        (size_t)hidden * dim
    );

    init_scope(model, &layer->attention_norm,
        &layer->attention_norm_binding, WEIGHT_RMS,
        atkey_layer_filler_id(index, 0), dim, dim, 0, "attention_norm");
    init_scope(model, &layer->query,
        &layer->query_binding, WEIGHT_MATMUL,
        atkey_layer_filler_id(index, 1), dim, dim, 0, "query");
    init_scope(model, &layer->key,
        &layer->key_binding, WEIGHT_MATMUL,
        atkey_layer_filler_id(index, 2), dim, kv, 0, "key");
    init_scope(model, &layer->value,
        &layer->value_binding, WEIGHT_MATMUL,
        atkey_layer_filler_id(index, 3), dim, kv, 0, "value");
    init_scope(model, &layer->attention_output,
        &layer->attention_output_binding, WEIGHT_MATMUL,
        atkey_layer_filler_id(index, 4), dim, dim, 0, "attention_output");
    init_scope(model, &layer->ffn_norm,
        &layer->ffn_norm_binding, WEIGHT_RMS,
        atkey_layer_filler_id(index, 5), dim, dim, 0, "ffn_norm");
    init_scope(model, &layer->ffn_gate,
        &layer->ffn_gate_binding, WEIGHT_MATMUL,
        atkey_layer_filler_id(index, 6), dim, hidden, 0, "ffn_gate");
    init_scope(model, &layer->ffn_up,
        &layer->ffn_up_binding, WEIGHT_MATMUL,
        atkey_layer_filler_id(index, 7), dim, hidden, 0, "ffn_up");
    init_scope(model, &layer->ffn_down,
        &layer->ffn_down_binding, WEIGHT_MATMUL,
        atkey_layer_filler_id(index, 8), hidden, dim, 0, "ffn_down");
}

/* Install checkpoint pointers without performing numerical evaluation. */
static void model_term_init(ModelTerm *model, AtkeyRuntime *runtime) {
    memset(model, 0, sizeof(*model));
    model->runtime = runtime;
    model->arena.block_size = 1U << 20;
    model->scheduler = (Scheduler){
        .runtime = runtime,
        .arena = &model->arena,
    };
    model->dim = atkey_dim(runtime);
    model->hidden_dim = atkey_hidden_dim(runtime);
    model->layer_count = atkey_layer_count(runtime);
    model->head_count = atkey_head_count(runtime);
    model->kv_head_count = atkey_kv_head_count(runtime);
    model->vocab_size = atkey_vocab_size(runtime);
    model->sequence_length = atkey_sequence_length(runtime);
    if (model->dim <= 0 || model->hidden_dim <= 0 ||
        model->layer_count <= 0 || model->head_count <= 0 ||
        model->kv_head_count <= 0 ||
        model->dim % model->head_count != 0 ||
        model->head_count % model->kv_head_count != 0) {
        escardo_fail("unsupported model dimensions");
    }
    model->kv_dim = model->dim * model->kv_head_count / model->head_count;
    model->head_size = model->dim / model->head_count;

    const float *embedding = atkey_embedding_weight(runtime);
    const float *output = atkey_output_weight(runtime);
    size_t embedding_scalars = (size_t)model->vocab_size * model->dim;
    binding_plug(
        &model->embedding_output_binding,
        embedding,
        embedding_scalars
    );
    WeightBinding *output_binding = &model->embedding_output_binding;
    if (output != embedding) {
        binding_plug(&model->output_binding, output, embedding_scalars);
        output_binding = &model->output_binding;
    }
    binding_plug(
        &model->final_norm_binding,
        atkey_final_rms_weight(runtime),
        (size_t)model->dim
    );

    init_scope(model, &model->embedding,
        &model->embedding_output_binding, WEIGHT_EMBEDDING,
        atkey_embedding_filler_id(), 0, model->dim, model->vocab_size,
        "embedding");
    init_scope(model, &model->final_norm,
        &model->final_norm_binding, WEIGHT_RMS,
        atkey_final_rms_filler_id(runtime), model->dim, model->dim, 0,
        "final_norm");
    init_scope(model, &model->output,
        output_binding, WEIGHT_MATMUL,
        atkey_output_filler_id(runtime), model->dim, model->vocab_size, 0,
        "output");

    model->layers = escardo_calloc(
        (size_t)model->layer_count,
        sizeof(*model->layers)
    );
    for (int layer = 0; layer < model->layer_count; layer++) {
        plug_layer(model, &model->layers[layer], layer);
    }
}

static Tensor *tensor_add(
    ModelTerm *model,
    const Tensor *left,
    const Tensor *right
) {
    if (left == NULL || right == NULL || left->width != right->width) {
        escardo_fail("residual tensor mismatch");
    }
    Tensor *output = tensor_new(&model->arena, left->width);
    atkey_add(output->values, left->values, right->values, left->width);
    return output;
}

static Tensor *tensor_swiglu(
    ModelTerm *model,
    const Tensor *gate,
    const Tensor *up
) {
    if (gate == NULL || up == NULL || gate->width != up->width) {
        escardo_fail("SwiGLU tensor mismatch");
    }
    Tensor *output = tensor_new(&model->arena, gate->width);
    atkey_swiglu(
        output->values,
        gate->values,
        gate->values,
        up->values,
        gate->width
    );
    return output;
}

struct ModelStep {
    ModelTerm *model;
    ModelNode *node;
    int layer;
    Tensor *hidden;
    Tensor *normalized;
    Tensor *query;
    Tensor *key;
    Tensor *value;
    Tensor *gate;
    Tensor *up;
    int join_count;
};

static void model_step_start_layer(ModelStep *step);

static void model_node_finish(ModelStep *step, Tensor *logits) {
    ModelNode *node = step->node;
    node->logits = logits;
    node->ready = true;
    step->model->model_steps++;
    NodeWaiter *waiter = node->waiter_head;
    node->waiter_head = NULL;
    node->waiter_tail = NULL;
    while (waiter != NULL) {
        NodeWaiter *next = waiter->next;
        waiter->continuation(waiter->environment, node);
        waiter = next;
    }
}

static void model_step_on_output(void *environment, Tensor *logits) {
    model_node_finish(environment, logits);
}

static void model_step_on_final_norm(void *environment, Tensor *normalized) {
    ModelStep *step = environment;
    step->node->final_hidden = normalized;
    scope_request_tensor(
        &step->model->output,
        normalized,
        model_step_on_output,
        step
    );
}

static void model_step_on_down(void *environment, Tensor *down) {
    ModelStep *step = environment;
    step->hidden = tensor_add(step->model, step->hidden, down);
    step->node->scales[step->layer + 1] = step->hidden;
    step->layer++;
    model_step_start_layer(step);
}

static void model_step_maybe_join_gate(ModelStep *step) {
    step->join_count++;
    if (step->join_count != 2) return;
    Tensor *gated = tensor_swiglu(step->model, step->gate, step->up);
    scope_request_tensor(
        &step->model->layers[step->layer].ffn_down,
        gated,
        model_step_on_down,
        step
    );
}

static void model_step_on_gate(void *environment, Tensor *gate) {
    ModelStep *step = environment;
    step->gate = gate;
    model_step_maybe_join_gate(step);
}

static void model_step_on_up(void *environment, Tensor *up) {
    ModelStep *step = environment;
    step->up = up;
    model_step_maybe_join_gate(step);
}

static void model_step_on_ffn_norm(void *environment, Tensor *normalized) {
    ModelStep *step = environment;
    LayerTerm *layer = &step->model->layers[step->layer];
    step->join_count = 0;
    step->gate = NULL;
    step->up = NULL;
    scope_request_tensor(&layer->ffn_gate, normalized, model_step_on_gate, step);
    scope_request_tensor(&layer->ffn_up, normalized, model_step_on_up, step);
}

static void model_step_on_attention_output(
    void *environment,
    Tensor *projected
) {
    ModelStep *step = environment;
    step->hidden = tensor_add(step->model, step->hidden, projected);
    scope_request_tensor(
        &step->model->layers[step->layer].ffn_norm,
        step->hidden,
        model_step_on_ffn_norm,
        step
    );
}

static void model_step_after_qkv(ModelStep *step) {
    ModelTerm *model = step->model;
    ModelNode *node = step->node;
    Tensor *rotated_query = tensor_new(&model->arena, model->dim);
    Tensor *rotated_key = tensor_new(&model->arena, model->kv_dim);
    atkey_rope(
        rotated_query->values,
        rotated_key->values,
        step->query->values,
        step->key->values,
        node->position,
        model->dim,
        model->kv_dim,
        model->head_size
    );
    node->keys[step->layer] = rotated_key;
    node->values[step->layer] = step->value;

    int context_count = node->position + 1;
    const float **keys = arena_allocate(
        &model->arena,
        (size_t)context_count * sizeof(*keys)
    );
    const float **values = arena_allocate(
        &model->arena,
        (size_t)context_count * sizeof(*values)
    );
    ModelNode *member = node;
    for (int index = context_count - 1; index >= 0; index--) {
        if (member == NULL || member->keys[step->layer] == NULL ||
            member->values[step->layer] == NULL) {
            escardo_fail("broken causal attention telescope");
        }
        keys[index] = member->keys[step->layer]->values;
        values[index] = member->values[step->layer]->values;
        member = member->parent;
    }
    if (member != NULL) escardo_fail("causal context length mismatch");

    Tensor *attended = tensor_new(&model->arena, model->dim);
    atkey_attention(
        attended->values,
        rotated_query->values,
        keys,
        values,
        context_count,
        model->dim,
        model->head_count,
        model->kv_head_count
    );
    scope_request_tensor(
        &model->layers[step->layer].attention_output,
        attended,
        model_step_on_attention_output,
        step
    );
}

static void model_step_maybe_join_qkv(ModelStep *step) {
    step->join_count++;
    if (step->join_count == 3) model_step_after_qkv(step);
}

static void model_step_on_query(void *environment, Tensor *query) {
    ModelStep *step = environment;
    step->query = query;
    model_step_maybe_join_qkv(step);
}

static void model_step_on_key(void *environment, Tensor *key) {
    ModelStep *step = environment;
    step->key = key;
    model_step_maybe_join_qkv(step);
}

static void model_step_on_value(void *environment, Tensor *value) {
    ModelStep *step = environment;
    step->value = value;
    model_step_maybe_join_qkv(step);
}

static void model_step_on_attention_norm(
    void *environment,
    Tensor *normalized
) {
    ModelStep *step = environment;
    LayerTerm *layer = &step->model->layers[step->layer];
    step->join_count = 0;
    step->query = NULL;
    step->key = NULL;
    step->value = NULL;
    scope_request_tensor(&layer->query, normalized, model_step_on_query, step);
    scope_request_tensor(&layer->key, normalized, model_step_on_key, step);
    scope_request_tensor(&layer->value, normalized, model_step_on_value, step);
}

static void model_step_start_layer(ModelStep *step) {
    if (step->layer == step->model->layer_count) {
        scope_request_tensor(
            &step->model->final_norm,
            step->hidden,
            model_step_on_final_norm,
            step
        );
        return;
    }
    scope_request_tensor(
        &step->model->layers[step->layer].attention_norm,
        step->hidden,
        model_step_on_attention_norm,
        step
    );
}

static void model_step_on_embedding(void *environment, Tensor *embedding) {
    ModelStep *step = environment;
    step->hidden = embedding;
    step->node->scales[0] = embedding;
    model_step_start_layer(step);
}

static ModelNode *model_new_node(
    ModelTerm *model,
    ModelNode *parent,
    int token
) {
    int position = parent == NULL ? 0 : parent->position + 1;
    if (position >= model->sequence_length) {
        escardo_fail("requested token exceeds checkpoint context length");
    }
    ModelNode *node = arena_allocate(&model->arena, sizeof(*node));
    memset(node, 0, sizeof(*node));
    node->id = model->next_node_id++;
    node->parent = parent;
    node->token = token;
    node->position = position;
    node->keys = arena_allocate(
        &model->arena,
        (size_t)model->layer_count * sizeof(*node->keys)
    );
    node->values = arena_allocate(
        &model->arena,
        (size_t)model->layer_count * sizeof(*node->values)
    );
    node->scales = arena_allocate(
        &model->arena,
        (size_t)(model->layer_count + 1) * sizeof(*node->scales)
    );
    memset(node->keys, 0,
        (size_t)model->layer_count * sizeof(*node->keys));
    memset(node->values, 0,
        (size_t)model->layer_count * sizeof(*node->values));
    memset(node->scales, 0,
        (size_t)(model->layer_count + 1) * sizeof(*node->scales));
    return node;
}

static void model_add_waiter(
    ModelTerm *model,
    ModelNode *node,
    NodeContinuation continuation,
    void *environment
) {
    if (node->ready) {
        continuation(environment, node);
        return;
    }
    NodeWaiter *waiter = arena_allocate(&model->arena, sizeof(*waiter));
    *waiter = (NodeWaiter){
        .continuation = continuation,
        .environment = environment,
    };
    if (node->waiter_tail == NULL) {
        node->waiter_head = waiter;
    } else {
        node->waiter_tail->next = waiter;
    }
    node->waiter_tail = waiter;
}

static ModelNode *model_find_child(ModelNode *parent, int token) {
    if (parent == NULL) return NULL;
    for (ModelChild *child = parent->children; child != NULL;
         child = child->next) {
        if (child->token == token) return child->node;
    }
    return NULL;
}

static void model_request_node(
    ModelTerm *model,
    ModelNode *parent,
    int token,
    NodeContinuation continuation,
    void *environment
) {
    if (token < 0 || token >= model->vocab_size) {
        escardo_fail("token outside vocabulary");
    }
    ModelNode *node = model_find_child(parent, token);
    if (node == NULL) {
        node = model_new_node(model, parent, token);
        if (parent != NULL) {
            ModelChild *child = arena_allocate(&model->arena, sizeof(*child));
            *child = (ModelChild){
                .next = parent->children,
                .token = token,
                .node = node,
            };
            parent->children = child;
        }
        ModelStep *step = arena_allocate(&model->arena, sizeof(*step));
        memset(step, 0, sizeof(*step));
        step->model = model;
        step->node = node;
        model_add_waiter(model, node, continuation, environment);
        scope_request_token(
            &model->embedding,
            token,
            model_step_on_embedding,
            step
        );
        return;
    }
    model_add_waiter(model, node, continuation, environment);
}

typedef struct ModelLogit ModelLogit;
typedef struct LogitPath LogitPath;

struct ModelLogit {
    ModelNode *context;
    int token;
    int local_rank;
    float logit;
    double log_probability;
};

struct LogitPath {
    ModelLogit head;
    LogitPath *tail;
};

typedef struct {
    LogitPath *path;
    ModelNode *terminal;
    Tensor **keys;
    Tensor **values;
    Tensor *final_hidden;
    struct CovectorFrame *observations;
} SelectionOutcome;

typedef void (*OutcomeContinuation)(
    void *environment,
    SelectionOutcome *outcome
);

typedef struct Search Search;
typedef struct SelectJob SelectJob;
typedef struct SelectBranch SelectBranch;
typedef struct CovectorFrame CovectorFrame;
typedef struct PosteriorCoordinate PosteriorCoordinate;

struct CovectorFrame {
    uint64_t id;
    int count;
    int *tokens;
    double *coordinates;
    int maximum_rank;
    CovectorFrame *tail;
};

struct Search {
    ModelTerm *model;
    const char *prompt_text;
    int prompt_count;
    int prompt_last_token;
    int horizon;
    int top_k;
    bool allow_delimiter;
    FILE *trace;
    uint64_t next_frame_id;
    uint64_t next_observer_frame_id;
    uint64_t candidate_observations;
    uint64_t continuation_demands;
    uint64_t posterior_coordinates;
    uint64_t attaining_alternatives;
    uint64_t ambiguous_selection_nodes;
    uint64_t zero_attainer_nodes;
    uint64_t strength_nodes;
};

struct SelectBranch {
    SelectJob *job;
    ModelLogit value;
    SelectionOutcome *outcome;
    CovectorFrame *observation;
    double company_log_probability;
    bool attains;
    bool ready;
};

struct SelectJob {
    Search *search;
    uint64_t frame_id;
    ModelNode *history;
    int remaining;
    OutcomeContinuation continuation;
    void *continuation_environment;
    SelectBranch *branches;
    int branch_count;
    int started_count;
    int ready_count;
    int observation_expected_count;
    int observation_ready_count;
    int attainer_count;
    bool observation_started;
};

struct PosteriorCoordinate {
    SelectJob *job;
    CovectorFrame *frame;
    int coordinate_rank;
    LogitPath *remaining_suffix;
};

static void json_string(FILE *stream, const char *text) {
    fputc('"', stream);
    for (const unsigned char *cursor = (const unsigned char *)text;
         *cursor != '\0'; cursor++) {
        unsigned char byte = *cursor;
        switch (byte) {
            case '"': fputs("\\\"", stream); break;
            case '\\': fputs("\\\\", stream); break;
            case '\b': fputs("\\b", stream); break;
            case '\f': fputs("\\f", stream); break;
            case '\n': fputs("\\n", stream); break;
            case '\r': fputs("\\r", stream); break;
            case '\t': fputs("\\t", stream); break;
            default:
                if (byte < 0x20) {
                    fprintf(stream, "\\u%04x", byte);
                } else {
                    fputc(byte, stream);
                }
        }
    }
    fputc('"', stream);
}

static void trace_decoded_path(
    Search *search,
    ModelNode *history,
    LogitPath *suffix
) {
    FILE *stream = search->trace;
    fputc('"', stream);
    for (const unsigned char *cursor =
             (const unsigned char *)search->prompt_text;
         *cursor != '\0'; cursor++) {
        unsigned char byte = *cursor;
        if (byte == '"') fputs("\\\"", stream);
        else if (byte == '\\') fputs("\\\\", stream);
        else if (byte == '\n') fputs("\\n", stream);
        else if (byte == '\r') fputs("\\r", stream);
        else if (byte == '\t') fputs("\\t", stream);
        else if (byte < 0x20) fprintf(stream, "\\u%04x", byte);
        else fputc(byte, stream);
    }

    int completion_prefix_count = history->position - search->prompt_count + 1;
    if (completion_prefix_count < 0) completion_prefix_count = 0;
    int *prefix_tokens = NULL;
    if (completion_prefix_count > 0) {
        prefix_tokens = escardo_calloc(
            (size_t)completion_prefix_count,
            sizeof(*prefix_tokens)
        );
        ModelNode *node = history;
        for (int index = completion_prefix_count - 1; index >= 0; index--) {
            prefix_tokens[index] = node->token;
            node = node->parent;
        }
    }
    int previous = search->prompt_last_token;
    for (int index = 0; index < completion_prefix_count; index++) {
        const char *piece = atkey_decode(
            search->model->runtime,
            previous,
            prefix_tokens[index]
        );
        for (const unsigned char *cursor = (const unsigned char *)piece;
             *cursor != '\0'; cursor++) {
            unsigned char byte = *cursor;
            if (byte == '"') fputs("\\\"", stream);
            else if (byte == '\\') fputs("\\\\", stream);
            else if (byte == '\n') fputs("\\n", stream);
            else if (byte == '\r') fputs("\\r", stream);
            else if (byte == '\t') fputs("\\t", stream);
            else if (byte < 0x20) fprintf(stream, "\\u%04x", byte);
            else fputc(byte, stream);
        }
        previous = prefix_tokens[index];
    }
    free(prefix_tokens);
    for (LogitPath *cell = suffix; cell != NULL; cell = cell->tail) {
        const char *piece = atkey_decode(
            search->model->runtime,
            previous,
            cell->head.token
        );
        for (const unsigned char *cursor = (const unsigned char *)piece;
             *cursor != '\0'; cursor++) {
            unsigned char byte = *cursor;
            if (byte == '"') fputs("\\\"", stream);
            else if (byte == '\\') fputs("\\\\", stream);
            else if (byte == '\n') fputs("\\n", stream);
            else if (byte == '\r') fputs("\\r", stream);
            else if (byte == '\t') fputs("\\t", stream);
            else if (byte < 0x20) fprintf(stream, "\\u%04x", byte);
            else fputc(byte, stream);
        }
        previous = cell->head.token;
    }
    fputc('"', stream);
}

static void trace_candidate(SelectBranch *branch) {
    Search *search = branch->job->search;
    if (search->trace == NULL) return;
    FILE *stream = search->trace;
    fprintf(
        stream,
        "{\"event\":\"candidate\",\"frame\":%" PRIu64
        ",\"depth\":%d,\"remaining\":%d,\"token\":%d"
        ",\"local_rank\":%d,\"logit\":%.9g"
        ",\"proposal_log_probability\":%.17g"
        ",\"observer_frame\":%" PRIu64 ",\"attains\":%s"
        ",\"observer_max_rank\":%d,\"observer_max_token\":%d"
        ",\"candidate_covector_logit\":%.17g"
        ",\"company_log_probability\":%.17g"
        ",\"text\":",
        branch->job->frame_id,
        branch->job->history->position - search->prompt_count + 1,
        branch->job->remaining,
        branch->value.token,
        branch->value.local_rank,
        branch->value.logit,
        branch->value.log_probability,
        branch->observation->id,
        branch->attains ? "true" : "false",
        branch->observation->maximum_rank + 1,
        branch->observation->tokens[branch->observation->maximum_rank],
        branch->observation->coordinates[branch->value.local_rank - 1],
        branch->company_log_probability
    );
    trace_decoded_path(
        search,
        branch->job->history,
        branch->outcome->path
    );
    fputs(",\"observer_support\":[", stream);
    for (int index = 0; index < branch->observation->count; index++) {
        if (index != 0) fputc(',', stream);
        fprintf(
            stream,
            "{\"rank\":%d,\"token\":%d,\"piece\":",
            index + 1,
            branch->observation->tokens[index]
        );
        json_string(
            stream,
            atkey_decode(
                search->model->runtime,
                branch->job->history->token,
                branch->observation->tokens[index]
            )
        );
        fprintf(
            stream,
            ",\"logit\":%.17g}",
            branch->observation->coordinates[index]
        );
    }
    fputs("]}\n", stream);
    fflush(stream);
}

static void trace_choice(SelectJob *job, const SelectBranch *branch) {
    Search *search = job->search;
    if (search->trace == NULL) return;
    fprintf(
        search->trace,
        "{\"event\":\"select\",\"frame\":%" PRIu64
        ",\"depth\":%d,\"remaining\":%d,\"token\":%d"
        ",\"local_rank\":%d,\"observer_frame\":%" PRIu64
        ",\"attains\":%s"
        ",\"attainer_count\":%d"
        ",\"company_log_probability\":%.17g"
        ",\"selection_rule\":\"max_normalized_self_company\""
        ",\"propagated\":\"complete_covector_family\",\"text\":",
        job->frame_id,
        job->history->position - search->prompt_count + 1,
        job->remaining,
        branch->value.token,
        branch->value.local_rank,
        branch->observation->id,
        branch->attains ? "true" : "false",
        job->attainer_count,
        branch->company_log_probability
    );
    trace_decoded_path(search, job->history, branch->outcome->path);
    fputs("}\n", search->trace);
    fflush(search->trace);
}

static void trace_continuation_demand(
    SelectJob *job,
    int ordinal,
    const ModelLogit *value
) {
    Search *search = job->search;
    search->continuation_demands++;
    if (search->trace == NULL) return;
    fprintf(
        search->trace,
        "{\"event\":\"continuation_demand\",\"frame\":%" PRIu64
        ",\"depth\":%d,\"remaining\":%d,\"ordinal\":%d"
        ",\"token\":%d,\"local_rank\":%d}\n",
        job->frame_id,
        job->history->position - search->prompt_count + 1,
        job->remaining,
        ordinal,
        value->token,
        value->local_rank
    );
    fflush(search->trace);
}

static int logit_precedes(const Tensor *logits, int left, int right) {
    if (logits->values[left] > logits->values[right]) return 1;
    if (logits->values[left] < logits->values[right]) return 0;
    return left < right;
}

static int *top_tokens(
    Search *search,
    const Tensor *logits,
    int count
) {
    int *tokens = escardo_calloc((size_t)count, sizeof(*tokens));
    int filled = 0;
    for (int token = 0; token < search->model->vocab_size; token++) {
        if (!search->allow_delimiter && token == ESCARDO_SEQUENCE_DELIMITER) {
            continue;
        }
        int insertion = filled;
        while (insertion > 0 && logit_precedes(
                logits,
                token,
                tokens[insertion - 1]
            )) {
            insertion--;
        }
        if (insertion >= count) continue;
        int upper = filled < count ? filled : count - 1;
        for (int index = upper; index > insertion; index--) {
            tokens[index] = tokens[index - 1];
        }
        tokens[insertion] = token;
        if (filled < count) filled++;
    }
    if (filled != count) escardo_fail("could not construct local support");
    return tokens;
}

static double log_partition(const Tensor *logits) {
    float maximum = -FLT_MAX;
    for (int token = 0; token < logits->width; token++) {
        if (logits->values[token] > maximum) maximum = logits->values[token];
    }
    double total = 0.0;
    for (int token = 0; token < logits->width; token++) {
        total += exp((double)logits->values[token] - maximum);
    }
    return (double)maximum + log(total);
}

static double model_node_log_probability(ModelNode *node, int token) {
    if (node == NULL || !node->ready || node->logits == NULL) {
        escardo_fail("posterior observer received an unavailable model node");
    }
    if (token < 0 || token >= node->logits->width) {
        escardo_fail("posterior observer token lies outside the vocabulary");
    }
    if (!node->log_partition_ready) {
        node->log_partition = log_partition(node->logits);
        node->log_partition_ready = true;
    }
    return (double)node->logits->values[token] - node->log_partition;
}

static LogitPath *path_cons(
    Search *search,
    ModelLogit head,
    LogitPath *tail
) {
    LogitPath *path = arena_allocate(&search->model->arena, sizeof(*path));
    *path = (LogitPath){.head = head, .tail = tail};
    return path;
}

static void select_job_after_branch(SelectJob *job);

static void select_path(
    Search *search,
    ModelNode *history,
    int remaining,
    OutcomeContinuation continuation,
    void *continuation_environment
);

static void select_job_finish_branch(
    SelectBranch *branch,
    SelectionOutcome *suffix
) {
    SelectJob *job = branch->job;
    if (suffix == NULL || suffix->terminal == NULL) {
        escardo_fail("selection branch returned no terminal outcome");
    }
    SelectionOutcome *outcome = arena_allocate(
        &job->search->model->arena,
        sizeof(*outcome)
    );
    *outcome = *suffix;
    outcome->path = path_cons(job->search, branch->value, suffix->path);
    branch->outcome = outcome;
    branch->ready = true;
    job->ready_count++;
    select_job_after_branch(job);
}

static void select_branch_tail_ready(
    void *environment,
    SelectionOutcome *suffix
) {
    select_job_finish_branch(environment, suffix);
}

static void select_branch_history_ready(
    void *environment,
    ModelNode *child
) {
    SelectBranch *branch = environment;
    SelectJob *job = branch->job;
    select_path(
        job->search,
        child,
        job->remaining - 1,
        select_branch_tail_ready,
        branch
    );
}

static void select_start_branch(SelectBranch *branch) {
    SelectJob *job = branch->job;
    if (job->remaining == 1) {
        /* Unit of the finite product: the empty suffix observes the proposal
         * already present at `history`.  Decoding the selected constructor
         * here would manufacture a child state no continuation can inspect. */
        SelectionOutcome *suffix = arena_allocate(
            &job->search->model->arena,
            sizeof(*suffix)
        );
        *suffix = (SelectionOutcome){
            .terminal = job->history,
            .keys = job->history->keys,
            .values = job->history->values,
            .final_hidden = job->history->final_hidden,
        };
        select_job_finish_branch(branch, suffix);
        return;
    }
    model_request_node(
        job->search->model,
        job->history,
        branch->value.token,
        select_branch_history_ready,
        branch
    );
}

static void select_admit_branch(SelectJob *job, int index) {
    if (index != job->started_count || index < 0 ||
        index >= job->branch_count) {
        escardo_fail("invalid continuation admission");
    }
    SelectBranch *branch = &job->branches[index];
    job->started_count++;
    trace_continuation_demand(job, index, &branch->value);
    select_start_branch(branch);
}

static void select_job_finish_observation(SelectJob *job);

static int path_length(const LogitPath *path) {
    int count = 0;
    for (const LogitPath *cell = path; cell != NULL; cell = cell->tail) {
        if (count == INT_MAX) escardo_fail("token path is too long");
        count++;
    }
    return count;
}

static CovectorFrame *covector_frame_new(SelectJob *job) {
    Search *search = job->search;
    ModelTerm *model = search->model;
    CovectorFrame *frame = arena_allocate(&model->arena, sizeof(*frame));
    memset(frame, 0, sizeof(*frame));
    frame->id = search->next_observer_frame_id++;
    frame->count = job->branch_count;
    frame->tokens = arena_allocate(
        &model->arena,
        (size_t)frame->count * sizeof(*frame->tokens)
    );
    frame->coordinates = arena_allocate(
        &model->arena,
        (size_t)frame->count * sizeof(*frame->coordinates)
    );
    for (int index = 0; index < frame->count; index++) {
        frame->tokens[index] = job->branches[index].value.token;
        frame->coordinates[index] =
            job->branches[index].value.log_probability;
    }
    search->posterior_coordinates += (uint64_t)frame->count;
    return frame;
}

static double covector_log_partition(const CovectorFrame *frame) {
    double maximum = -DBL_MAX;
    for (int index = 0; index < frame->count; index++) {
        if (frame->coordinates[index] > maximum) {
            maximum = frame->coordinates[index];
        }
    }
    double mass = 0.0;
    for (int index = 0; index < frame->count; index++) {
        mass += exp(frame->coordinates[index] - maximum);
    }
    if (!(mass > 0.0) || !isfinite(mass)) {
        escardo_fail("invalid causal-posterior covector");
    }
    return maximum + log(mass);
}

static void posterior_coordinate_finish(PosteriorCoordinate *coordinate) {
    SelectJob *job = coordinate->job;
    job->observation_ready_count++;
    if (job->observation_ready_count > job->observation_expected_count) {
        escardo_fail("posterior observer completed a coordinate twice");
    }
    if (job->observation_ready_count == job->observation_expected_count) {
        select_job_finish_observation(job);
    }
}

static void posterior_coordinate_advance(
    void *environment,
    ModelNode *node
) {
    PosteriorCoordinate *coordinate = environment;
    LogitPath *suffix = coordinate->remaining_suffix;
    if (suffix == NULL) {
        escardo_fail("posterior observer advanced beyond its fixed suffix");
    }

    const int token = suffix->head.token;
    coordinate->frame->coordinates[coordinate->coordinate_rank] +=
        model_node_log_probability(node, token);
    coordinate->remaining_suffix = suffix->tail;
    if (coordinate->remaining_suffix == NULL) {
        posterior_coordinate_finish(coordinate);
        return;
    }

    model_request_node(
        coordinate->job->search->model,
        node,
        token,
        posterior_coordinate_advance,
        coordinate
    );
}

static void select_job_finish_observation(SelectJob *job) {
    if (job->observation_ready_count != job->observation_expected_count) {
        escardo_fail("selection observer terminated with missing coordinates");
    }

    int selected_index = 0;
    for (int demand = 0; demand < job->branch_count; demand++) {
        SelectBranch *branch = &job->branches[demand];
        CovectorFrame *frame = branch->observation;
        if (frame == NULL || frame->count != job->branch_count) {
            escardo_fail("selection observer returned an invalid frame");
        }

        frame->maximum_rank = 0;
        for (int coordinate = 1; coordinate < frame->count; coordinate++) {
            if (frame->coordinates[coordinate] >
                    frame->coordinates[frame->maximum_rank]) {
                frame->maximum_rank = coordinate;
            }
        }
        branch->attains =
            frame->maximum_rank == branch->value.local_rank - 1;
        branch->company_log_probability =
            frame->coordinates[branch->value.local_rank - 1] -
            covector_log_partition(frame);
        if (branch->attains) job->attainer_count++;
        frame->tail = branch->outcome->observations;
        job->search->candidate_observations++;
        trace_candidate(branch);
        SelectBranch *incumbent = &job->branches[selected_index];
        if (demand == 0 ||
            branch->company_log_probability >
                incumbent->company_log_probability ||
            (branch->company_log_probability ==
                 incumbent->company_log_probability &&
             branch->value.local_rank < incumbent->value.local_rank)) {
            selected_index = demand;
        }
    }

    job->search->attaining_alternatives += (uint64_t)job->attainer_count;
    if (job->attainer_count == 0) {
        job->search->zero_attainer_nodes++;
    } else if (job->attainer_count > 1) {
        job->search->ambiguous_selection_nodes++;
    }

    SelectBranch *selected = &job->branches[selected_index];
    trace_choice(job, selected);
    SelectionOutcome *propagated = arena_allocate(
        &job->search->model->arena,
        sizeof(*propagated)
    );
    *propagated = *selected->outcome;
    propagated->observations = selected->observation;
    job->continuation(job->continuation_environment, propagated);
}

static void select_job_start_observation(SelectJob *job) {
    if (job->ready_count != job->branch_count || job->observation_started) {
        return;
    }
    job->observation_started = true;

    for (int demand = 0; demand < job->branch_count; demand++) {
        SelectBranch *branch = &job->branches[demand];
        if (branch->outcome == NULL ||
            path_length(branch->outcome->path) != job->remaining) {
            escardo_fail("bound continuation has the wrong horizon");
        }
        branch->observation = covector_frame_new(job);
    }

    const int suffix_length = job->remaining - 1;
    if (suffix_length == 0) {
        job->observation_expected_count = 0;
        select_job_finish_observation(job);
        return;
    }

    if (job->branch_count > INT_MAX / job->branch_count) {
        escardo_fail("posterior coordinate count overflow");
    }
    job->observation_expected_count =
        job->branch_count * job->branch_count;
    for (int demand = 0; demand < job->branch_count; demand++) {
        SelectBranch *branch = &job->branches[demand];
        LogitPath *suffix = branch->outcome->path->tail;
        if (path_length(suffix) != suffix_length) {
            escardo_fail("observer restriction lost its selected suffix");
        }

        for (int coordinate_rank = 0;
             coordinate_rank < job->branch_count;
             coordinate_rank++) {
            PosteriorCoordinate *coordinate = arena_allocate(
                &job->search->model->arena,
                sizeof(*coordinate)
            );
            *coordinate = (PosteriorCoordinate){
                .job = job,
                .frame = branch->observation,
                .coordinate_rank = coordinate_rank,
                .remaining_suffix = suffix,
            };
            model_request_node(
                job->search->model,
                job->history,
                job->branches[coordinate_rank].value.token,
                posterior_coordinate_advance,
                coordinate
            );
        }
    }
}

static void select_job_after_branch(SelectJob *job) {
    select_job_start_observation(job);
}

/* Mechanical asynchronous transcription of Escardo's dependent product:
 *
 *     b(x) = delta(x)(xs -> p(x : xs))
 *     a    = epsilon(x -> p(x : b(x)))
 *     result = a : b(a)
 *
 * Each branch stores b(x), so the chosen b(a) is returned rather than run a
 * second time. Every demanded x receives the recursively composed suffix
 * before the root observer is restricted to x : b(x). */
static void select_path(
    Search *search,
    ModelNode *history,
    int remaining,
    OutcomeContinuation continuation,
    void *continuation_environment
) {
    if (remaining <= 0) {
        SelectionOutcome *outcome = arena_allocate(
            &search->model->arena,
            sizeof(*outcome)
        );
        *outcome = (SelectionOutcome){
            .terminal = history,
            .keys = history->keys,
            .values = history->values,
            .final_hidden = history->final_hidden,
        };
        continuation(continuation_environment, outcome);
        return;
    }
    if (history == NULL || !history->ready || history->logits == NULL) {
        escardo_fail("selection observed an unavailable model logit");
    }
    SelectJob *job = arena_allocate(&search->model->arena, sizeof(*job));
    memset(job, 0, sizeof(*job));
    job->search = search;
    job->frame_id = search->next_frame_id++;
    job->history = history;
    job->remaining = remaining;
    job->continuation = continuation;
    job->continuation_environment = continuation_environment;
    job->branch_count = search->top_k;
    job->branches = arena_allocate(
        &search->model->arena,
        (size_t)job->branch_count * sizeof(*job->branches)
    );
    memset(job->branches, 0,
        (size_t)job->branch_count * sizeof(*job->branches));
    search->strength_nodes++;

    int *tokens = top_tokens(search, history->logits, job->branch_count);
    double partition = log_partition(history->logits);
    for (int index = 0; index < job->branch_count; index++) {
        int token = tokens[index];
        SelectBranch *branch = &job->branches[index];
        *branch = (SelectBranch){
            .job = job,
            .value = {
                .context = history,
                .token = token,
                .local_rank = index + 1,
                .logit = history->logits->values[token],
                .log_probability =
                    (double)history->logits->values[token] - partition,
            },
        };
    }
    free(tokens);
    for (int index = 0; index < job->branch_count; index++) {
        select_admit_branch(job, index);
    }
}

typedef struct {
    Search *search;
    int *tokens;
    int count;
    int index;
    ModelNode *last;
    NodeContinuation continuation;
    void *continuation_environment;
} Prefill;

static void prefill_step(void *environment, ModelNode *node) {
    Prefill *prefill = environment;
    prefill->last = node;
    prefill->index++;
    if (prefill->index == prefill->count) {
        prefill->continuation(prefill->continuation_environment, node);
        return;
    }
    model_request_node(
        prefill->search->model,
        node,
        prefill->tokens[prefill->index],
        prefill_step,
        prefill
    );
}

static void prefill_start(
    Search *search,
    int *tokens,
    int count,
    NodeContinuation continuation,
    void *continuation_environment
) {
    if (count <= 0) escardo_fail("empty prompt token sequence");
    Prefill *prefill = arena_allocate(
        &search->model->arena,
        sizeof(*prefill)
    );
    *prefill = (Prefill){
        .search = search,
        .tokens = tokens,
        .count = count,
        .continuation = continuation,
        .continuation_environment = continuation_environment,
    };
    model_request_node(
        search->model,
        NULL,
        tokens[0],
        prefill_step,
        prefill
    );
}

typedef struct {
    Search *search;
    bool done;
    LogitPath *selected;
} RootRun;

static void tau_selection_ready(
    void *environment,
    SelectionOutcome *selected
) {
    RootRun *root = environment;
    /* The root is the only place where the selected token tuple is emitted. */
    root->selected = selected->path;
    root->done = true;
}

static void exact_prompt_ready(void *environment, ModelNode *prompt) {
    RootRun *root = environment;
    Search *search = root->search;
    select_path(
        search,
        prompt,
        search->horizon,
        tau_selection_ready,
        root
    );
}

static RootRun escardo_exact_run(
    Search *search,
    int *prompt_tokens,
    int prompt_count
) {
    RootRun root = {.search = search};
    prefill_start(
        search,
        prompt_tokens,
        prompt_count,
        exact_prompt_ready,
        &root
    );
    while (!root.done) {
        if (!scheduler_step(&search->model->scheduler)) {
            escardo_fail("continuation scheduler reached a deadlock");
        }
    }
    return root;
}


static void print_selected(
    Search *search,
    LogitPath *path
) {
    fputs(search->prompt_text, stdout);
    int previous = search->prompt_last_token;
    for (LogitPath *cell = path; cell != NULL; cell = cell->tail) {
        atkey_print_piece(search->model->runtime, previous, cell->head.token);
        previous = cell->head.token;
    }
    fputc('\n', stdout);
    fflush(stdout);
}

typedef struct {
    const char *checkpoint;
    const char *tokenizer;
    const char *prompt;
    const char *trace_path;
    int length;
    int top_k;
    const char *metal_library;
    bool allow_delimiter;
    bool exact;
    bool use_metal;
} Options;

static long parse_long(const char *text, const char *name) {
    errno = 0;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        fprintf(stderr, "escardo: %s must be an integer\n", name);
        exit(EXIT_FAILURE);
    }
    return value;
}

_Noreturn static void usage(const char *program) {
    fprintf(
        stderr,
        "usage: %s CHECKPOINT TOKENIZER --prompt TEXT --length N "
        "--exact --top-k K "
        "[--trace FILE] [--allow-delimiter] "
        "[--metal [--metal-library FILE] | --cpu]\n",
        program
    );
    exit(EXIT_FAILURE);
}

static Options parse_options(int argc, char **argv) {
    if (argc < 3) usage(argv[0]);
    Options options = {
        .checkpoint = argv[1],
        .tokenizer = argv[2],
        .length = -1,
        .top_k = 0,
        .metal_library = "metal_kernels.metallib",
#ifdef ATKEY_METAL
        .use_metal = true,
#endif
    };
    for (int index = 3; index < argc; index++) {
        const char *flag = argv[index];
        if (strcmp(flag, "--allow-delimiter") == 0) {
            options.allow_delimiter = true;
            continue;
        }
        if (strcmp(flag, "--exact") == 0) {
            options.exact = true;
            continue;
        }
        if (strcmp(flag, "--metal") == 0) {
            options.use_metal = true;
            continue;
        }
        if (strcmp(flag, "--cpu") == 0) {
            options.use_metal = false;
            continue;
        }
        if (index + 1 >= argc) usage(argv[0]);
        const char *value = argv[++index];
        if (strcmp(flag, "--prompt") == 0) options.prompt = value;
        else if (strcmp(flag, "--length") == 0) {
            long parsed = parse_long(value, "length");
            if (parsed <= 0 || parsed > INT_MAX) usage(argv[0]);
            options.length = (int)parsed;
        } else if (strcmp(flag, "--top-k") == 0) {
            long parsed = parse_long(value, "top-k");
            if (parsed <= 0 || parsed > INT_MAX) usage(argv[0]);
            options.top_k = (int)parsed;
        } else if (strcmp(flag, "--trace") == 0) {
            options.trace_path = value;
        } else if (strcmp(flag, "--metal-library") == 0) {
            options.metal_library = value;
        } else {
            usage(argv[0]);
        }
    }
    if (options.prompt == NULL || options.length <= 0 || !options.exact ||
        options.top_k <= 0) {
        usage(argv[0]);
    }
    return options;
}

static void print_kernel_summary(
    const char *name,
    int layer,
    const WeightBinding *binding
) {
    printf(
        "kernel_scope name=%s layer=%d numerical_applications=%" PRIu64
        " continuation_uses=%" PRIu64 "\n",
        name,
        layer,
        binding->batch_count,
        binding->request_count
    );
}

static void print_model_summary(ModelTerm *model) {
    print_kernel_summary(
        "embedding_output",
        -1,
        &model->embedding_output_binding
    );
    if (model->output_binding.plugged) {
        print_kernel_summary("output", -1, &model->output_binding);
    }
    for (int index = 0; index < model->layer_count; index++) {
        LayerTerm *layer = &model->layers[index];
        print_kernel_summary("attention_norm", index,
            &layer->attention_norm_binding);
        print_kernel_summary("query", index, &layer->query_binding);
        print_kernel_summary("key", index, &layer->key_binding);
        print_kernel_summary("value", index, &layer->value_binding);
        print_kernel_summary("attention_output", index,
            &layer->attention_output_binding);
        print_kernel_summary("ffn_norm", index, &layer->ffn_norm_binding);
        print_kernel_summary("ffn_gate", index, &layer->ffn_gate_binding);
        print_kernel_summary("ffn_up", index, &layer->ffn_up_binding);
        print_kernel_summary("ffn_down", index, &layer->ffn_down_binding);
    }
    print_kernel_summary("final_norm", -1, &model->final_norm_binding);
}

int main(int argc, char **argv) {
    Options options = parse_options(argc, argv);
    AtkeyRuntime *runtime = atkey_runtime_new(
        options.checkpoint,
        options.tokenizer
    );
    if (runtime == NULL) escardo_fail("could not initialize llama2 backend");
    if (options.use_metal &&
        !atkey_enable_metal(runtime, options.metal_library)) {
        escardo_fail("could not initialize Metal backend");
    }

    ModelTerm model;
    model_term_init(&model, runtime);
    int prompt_count = 0;
    int *prompt_tokens = atkey_encode(runtime, options.prompt, &prompt_count);
    if (prompt_tokens == NULL || prompt_count <= 0) {
        escardo_fail("could not encode prompt");
    }
    if (prompt_count + options.length > model.sequence_length) {
        escardo_fail("prompt and completion exceed checkpoint context length");
    }
    if (options.top_k > model.vocab_size -
            (options.allow_delimiter ? 0 : 1)) {
        escardo_fail("top-k exceeds selectable vocabulary");
    }

    FILE *trace = NULL;
    if (options.trace_path != NULL) {
        trace = fopen(options.trace_path, "w");
        if (trace == NULL) escardo_fail("could not open trace output");
        fputs("{\"event\":\"run\",\"prompt\":", trace);
        json_string(trace, options.prompt);
        fprintf(
            trace,
            ",\"length\":%d,\"proposal_top_k\":",
            options.length
        );
        fprintf(trace, "%d", options.top_k);
        fprintf(
            trace,
            ",\"backend\":\"%s\",\"mode\":\"%s\""
            ",\"selection_rule\":\"max_normalized_self_company\"}\n",
            atkey_backend_name(runtime),
            "exact_select_product_causal_posterior"
        );
        fflush(trace);
    }

    Search search = {
        .model = &model,
        .prompt_text = options.prompt,
        .prompt_count = prompt_count,
        .prompt_last_token = prompt_tokens[prompt_count - 1],
        .horizon = options.length,
        .top_k = options.top_k,
        .allow_delimiter = options.allow_delimiter,
        .trace = trace,
        .next_frame_id = 1,
        .next_observer_frame_id = 1,
    };

    RootRun result = escardo_exact_run(
        &search,
        prompt_tokens,
        prompt_count
    );
    puts("completion:");
    print_selected(&search, result.selected);
    printf(
        "score_kind=normalized_self_company_log_probability\n"
        "selection_carrier=token_path_with_covector_family\n"
        "selection_observer=causal_posterior_root_callback\n"
        "observer_attention=causal_posterior_over_fixed_bound_suffix\n"
        "aggregate_path_score=none\n"
        "root_terminalizations=1\n"
        "strength_nodes=%" PRIu64 "\n"
        "candidate_observations=%" PRIu64 "\n"
        "continuation_demands=%" PRIu64 "\n"
        "posterior_coordinates=%" PRIu64 "\n"
        "attaining_alternatives=%" PRIu64 "\n"
        "ambiguous_selection_nodes=%" PRIu64 "\n"
        "zero_attainer_nodes=%" PRIu64 "\n"
        "model_token_terms=%" PRIu64 "\n",
        search.strength_nodes,
        search.candidate_observations,
        search.continuation_demands,
        search.posterior_coordinates,
        search.attaining_alternatives,
        search.ambiguous_selection_nodes,
        search.zero_attainer_nodes,
        model.model_steps
    );
    printf(
        "numerical_backend=%s\n"
        "backend_device=%s\n"
        "backend_dispatches=%" PRIu64 "\n"
        "backend_weight_uploads=%" PRIu64 "\n"
        "backend_weight_upload_bytes=%" PRIu64 "\n",
        atkey_backend_name(runtime),
        atkey_backend_device_name(runtime),
        atkey_backend_dispatch_count(runtime),
        atkey_backend_weight_upload_count(runtime),
        atkey_backend_weight_upload_bytes(runtime)
    );
    print_model_summary(&model);

    if (trace != NULL) {
        fprintf(
            trace,
            "{\"event\":\"run_end\",\"strength_nodes\":%" PRIu64
            ",\"candidate_observations\":%" PRIu64 "}\n",
            search.strength_nodes,
            search.candidate_observations
        );
        fflush(trace);
        fclose(trace);
    }

    atkey_free_tokens(prompt_tokens);
    free(model.layers);
    arena_release(&model.arena);
    atkey_runtime_free(runtime);
    return 0;
}
