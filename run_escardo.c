/*
 * Unoptimized non-greedy inference by the dependent product of selection
 * functions.  Physical kernel/weight reuse is deliberately not claimed here;
 * that is a later lowering and performance problem.
 *
 * The selection carrier is a hypothetical token constructor and the outcome
 * is the frozen model trace at the endpoint of the completed branch:
 *
 *     R = (selected suffix, terminal model node, terminal logit covector)
 *     J_R ModelLogit = (ModelLogit -> R) -> ModelLogit
 *
 * Token positions are composed with Escardo's dependent product.  The code
 * below is a defunctionalized, asynchronous transcription of
 *
 *     b x = delta x (prefix x p)
 *     a   = epsilon (\x -> p (x : b x))
 *     result = a : b a
 *
 * The branch for b(a) and its attained outcome are shared; neither is
 * evaluated again after epsilon has selected a. Only the root forgets the
 * retained outcome and emits the token witness.
 *
 * For a demanded constructor x, the local Select first forces and memoizes
 * b(x), then compares
 *
 *     ev(x, root_covector(x : b(x))).
 *
 * No path likelihood is formed.  Exact mode enumerates an explicitly
 * requested finite local support.  Timed mode has no leaf-count bound: until
 * a wall-clock deadline it samples which memo cells are demanded, then
 * re-forces this same product over the retained finite function tree.  A
 * positive --top-k changes the sampling proposal only; the default proposal
 * is the full selectable vocabulary.
 *
 * The model evaluator below is CPS and batches currently ready kernel calls,
 * but it may evaluate a learned tensor many times while the selection product
 * demands continuations.  Its counters report that cost without treating a
 * retained pointer or scope installation as a one-shot numerical application.
 * No call to llama2.c's eager forward() occurs in this file.
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
typedef struct TimedSelectFrame TimedSelectFrame;

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
    Tensor *logits;
    TimedSelectFrame *timed_selection;
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
typedef struct SelectionOutcome SelectionOutcome;

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

struct SelectionOutcome {
    LogitPath *path;
    ModelNode *terminal;
};

typedef void (*OutcomeContinuation)(
    void *environment,
    SelectionOutcome *outcome,
    double selection_score
);

typedef struct Search Search;
typedef struct SelectJob SelectJob;
typedef struct SelectBranch SelectBranch;
typedef struct TimedSelectEdge TimedSelectEdge;

struct Search {
    ModelTerm *model;
    const char *prompt_text;
    int prompt_count;
    int prompt_last_token;
    int horizon;
    /* Zero means that timed sampling uses the full selectable vocabulary.
     * A positive value is only a proposal-distribution truncation; it is not
     * a bound on the number of continuations the term may demand. */
    int top_k;
    int sample_milliseconds;
    int batch_size;
    uint64_t sample_seed;
    uint64_t sample_random_state;
    bool sampling_enabled;
    bool deadline_armed;
    struct timespec deadline;
    bool allow_delimiter;
    FILE *trace;
    uint64_t next_frame_id;
    uint64_t candidate_observations;
    uint64_t payoff_observations;
    uint64_t strength_nodes;
    uint64_t sampled_candidate_demands;
    uint64_t completed_samples;
};

struct SelectBranch {
    SelectJob *job;
    ModelLogit value;
    uint64_t child_budget;
    bool sampled;
    double support_probability;
    double draw;
    SelectionOutcome *outcome;
    double score;
    bool ready;
};

struct SelectJob {
    Search *search;
    uint64_t frame_id;
    ModelNode *history;
    int remaining;
    uint64_t budget;
    OutcomeContinuation continuation;
    void *continuation_environment;
    SelectBranch *branches;
    int branch_count;
    int started_count;
    int ready_count;
};

/* A memoized local Select in the wall-clock-driven product. Sampling may
 * demand the same outgoing continuation repeatedly; the edge is stored once
 * while multiplicity records every demand. The edge retains b(x)'s terminal
 * outcome and its ev(x, root_covector(x : b(x))) coordinate. */
struct TimedSelectEdge {
    TimedSelectEdge *next;
    TimedSelectFrame *owner;
    TimedSelectFrame *suffix;
    int token;
    int local_rank;
    float logit;
    double log_probability;
    uint64_t demand_count;
    uint64_t observation_count;
    bool observed;
    double selected_rating;
    LogitPath *selected_path;
    ModelNode *selected_terminal;
    ModelNode *terminal;
};

struct TimedSelectFrame {
    Search *search;
    ModelNode *history;
    TimedSelectEdge *edges;
    TimedSelectEdge *selected;
    int remaining;
    uint64_t demand_count;
    uint64_t observation_count;
};

static struct timespec monotonic_now(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        escardo_fail("could not read monotonic clock");
    }
    return value;
}

static struct timespec add_milliseconds(
    struct timespec value,
    int milliseconds
) {
    value.tv_sec += milliseconds / 1000;
    value.tv_nsec += (long)(milliseconds % 1000) * 1000000L;
    if (value.tv_nsec >= 1000000000L) {
        value.tv_sec++;
        value.tv_nsec -= 1000000000L;
    }
    return value;
}

static bool deadline_reached(const Search *search) {
    if (!search->deadline_armed) return false;
    struct timespec now = monotonic_now();
    if (now.tv_sec != search->deadline.tv_sec) {
        return now.tv_sec > search->deadline.tv_sec;
    }
    return now.tv_nsec >= search->deadline.tv_nsec;
}

/* The sampler chooses which arguments a local selection function is allowed
 * to ask its observer about.  It never constructs a complete path.  State is
 * derived from the causal history so enlarging a budget does not reshuffle
 * already-demanded continuations. */
static uint64_t sample_random_mix(uint64_t value) {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31;
    return value;
}

static uint64_t sample_history_state(
    const Search *search,
    const ModelNode *history
) {
    uint64_t hash = search->sample_seed ^ UINT64_C(0xcbf29ce484222325);
    for (const unsigned char *byte =
             (const unsigned char *)search->prompt_text;
         *byte != '\0'; byte++) {
        hash ^= (uint64_t)*byte;
        hash *= UINT64_C(0x100000001b3);
    }
    for (const ModelNode *node = history; node != NULL; node = node->parent) {
        hash ^= (uint64_t)(uint32_t)node->token;
        hash *= UINT64_C(0x100000001b3);
    }
    hash ^= (uint64_t)(uint32_t)history->position;
    hash = sample_random_mix(hash);
    return hash == 0 ? UINT64_C(0x4d595df4d0f33173) : hash;
}

static uint64_t sample_random_next(uint64_t *state) {
    uint64_t value = *state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    return value * UINT64_C(2685821657736338717);
}

static double sample_random_unit(uint64_t *state) {
    return (double)(sample_random_next(state) >> 11) *
        (1.0 / 9007199254740992.0);
}

static double log_partition(const Tensor *logits);

/* The sole constructor/codata elimination used by selection.  The covector
 * belongs to the endpoint of the already completed branch x : b(x), not to
 * x's eager prefix row. */
static double terminal_root_coordinate(
    Search *search,
    int constructor,
    ModelNode *terminal
) {
    if (terminal == NULL || !terminal->ready || terminal->logits == NULL ||
        constructor < 0 || constructor >= search->model->vocab_size) {
        escardo_fail("selection received an incomplete terminal outcome");
    }
    search->payoff_observations++;
    double coordinate =
        (double)terminal->logits->values[constructor] -
        log_partition(terminal->logits);
    if (!isfinite(coordinate)) {
        escardo_fail("terminal root coordinate is not finite");
    }
    return coordinate;
}

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
        ",\"terminal_node\":%" PRIu64
        ",\"root_coordinate\":%.17g"
        ",\"text\":",
        branch->job->frame_id,
        branch->job->history->position - search->prompt_count + 1,
        branch->job->remaining,
        branch->value.token,
        branch->value.local_rank,
        branch->value.logit,
        branch->value.log_probability,
        branch->outcome->terminal->id,
        branch->score
    );
    trace_decoded_path(
        search,
        branch->job->history,
        branch->outcome->path
    );
    fputs("}\n", stream);
    fflush(stream);
}

static void trace_choice(SelectJob *job, const SelectBranch *branch) {
    Search *search = job->search;
    if (search->trace == NULL) return;
    fprintf(
        search->trace,
        "{\"event\":\"select\",\"frame\":%" PRIu64
        ",\"depth\":%d,\"remaining\":%d,\"token\":%d"
        ",\"local_rank\":%d,\"terminal_node\":%" PRIu64
        ",\"root_coordinate\":%.17g,\"text\":",
        job->frame_id,
        job->history->position - search->prompt_count + 1,
        job->remaining,
        branch->value.token,
        branch->value.local_rank,
        branch->outcome->terminal->id,
        branch->score
    );
    trace_decoded_path(search, job->history, branch->outcome->path);
    fputs("}\n", search->trace);
    fflush(search->trace);
}

static void trace_continuation_demand(
    SelectJob *job,
    int ordinal,
    const ModelLogit *value,
    bool sampled,
    double support_probability,
    double draw,
    uint64_t child_budget
) {
    Search *search = job->search;
    if (search->trace == NULL) return;
    fprintf(
        search->trace,
        "{\"event\":\"continuation_demand\",\"frame\":%" PRIu64
        ",\"depth\":%d,\"remaining\":%d,\"ordinal\":%d"
        ",\"token\":%d,\"local_rank\":%d,\"sampled\":%s"
        ",\"support_probability\":",
        job->frame_id,
        job->history->position - search->prompt_count + 1,
        job->remaining,
        ordinal,
        value->token,
        value->local_rank,
        sampled ? "true" : "false"
    );
    if (sampled) fprintf(search->trace, "%.17g", support_probability);
    else fputs("null", search->trace);
    fputs(",\"draw\":", search->trace);
    if (sampled) fprintf(search->trace, "%.17g", draw);
    else fputs("null", search->trace);
    fputs(",\"child_leaf_budget\":", search->trace);
    if (child_budget == UINT64_MAX) fputs("null", search->trace);
    else fprintf(search->trace, "%" PRIu64, child_budget);
    fputs("}\n", search->trace);
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

/* Draw one not-yet-demanded continuation from the model's top-k support.
 * The returned probability is conditional on the remaining sampled support;
 * ModelLogit's reward remains its probability under the full vocabulary. */
static int sample_next_local_rank(
    const Tensor *logits,
    const int *ranked_support,
    int support_count,
    const bool *already_sampled,
    uint64_t *random_state,
    double *support_probability,
    double *draw
) {
    double maximum = -DBL_MAX;
    int remaining = 0;
    for (int rank = 0; rank < support_count; rank++) {
        if (already_sampled[rank]) continue;
        double value = logits->values[ranked_support[rank]];
        if (value > maximum) maximum = value;
        remaining++;
    }
    if (remaining == 0) escardo_fail("sampled empty local support");

    double mass = 0.0;
    for (int rank = 0; rank < support_count; rank++) {
        if (already_sampled[rank]) continue;
        mass += exp(
            (double)logits->values[ranked_support[rank]] - maximum
        );
    }
    if (!(mass > 0.0) || !isfinite(mass)) {
        escardo_fail("invalid sampled local support mass");
    }

    *draw = sample_random_unit(random_state);
    double target = *draw * mass;
    double cumulative = 0.0;
    int selected = -1;
    for (int rank = 0; rank < support_count; rank++) {
        if (already_sampled[rank]) continue;
        cumulative += exp(
            (double)logits->values[ranked_support[rank]] - maximum
        );
        selected = rank;
        if (target < cumulative) break;
    }
    if (selected < 0) escardo_fail("could not sample local continuation");
    *support_probability = exp(
        (double)logits->values[ranked_support[selected]] - maximum
    ) / mass;
    return selected;
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

static bool token_is_selectable(const Search *search, int token) {
    return search->allow_delimiter || token != ESCARDO_SEQUENCE_DELIMITER;
}

static int timed_local_rank(
    const Search *search,
    const Tensor *logits,
    int token
) {
    int rank = 1;
    for (int candidate = 0; candidate < search->model->vocab_size;
         candidate++) {
        if (!token_is_selectable(search, candidate)) continue;
        if (logit_precedes(logits, candidate, token)) rank++;
    }
    return rank;
}

static TimedSelectEdge *timed_find_edge(
    const TimedSelectFrame *frame,
    int token
) {
    for (TimedSelectEdge *edge = frame->edges; edge != NULL;
         edge = edge->next) {
        if (edge->token == token) return edge;
    }
    return NULL;
}

static bool timed_argument_is_eligible(
    const TimedSelectFrame *frame,
    int token,
    bool demand_unseen,
    uint64_t minimum_demand
) {
    TimedSelectEdge *edge = timed_find_edge(frame, token);
    if (demand_unseen) return edge == NULL;
    return edge != NULL && edge->demand_count == minimum_demand;
}

/* Draw a memo cell for one local Select. A cell not yet present in this frame
 * is demanded before an existing cell is repeated. Once the finite proposal
 * support is present, only least-demanded cells are eligible for the next
 * draw. The model logits choose probabilistically among eligible cells; demand
 * counts never enter constructor/codata comparison. */
static ModelLogit timed_sample_argument(
    Search *search,
    TimedSelectFrame *frame
) {
    ModelNode *history = frame->history;
    Tensor *logits = history->logits;
    int *support = NULL;
    int support_count = search->top_k;
    if (support_count > 0) {
        support = top_tokens(search, logits, support_count);
    } else {
        support_count = search->model->vocab_size -
            (search->allow_delimiter ? 0 : 1);
    }
    if (support_count <= 0) escardo_fail("empty timed sampling support");

    int existing_count = 0;
    uint64_t minimum_demand = UINT64_MAX;
    if (support != NULL) {
        for (int index = 0; index < support_count; index++) {
            TimedSelectEdge *edge = timed_find_edge(frame, support[index]);
            if (edge == NULL) continue;
            existing_count++;
            if (edge->demand_count < minimum_demand) {
                minimum_demand = edge->demand_count;
            }
        }
    } else {
        for (int token = 0; token < search->model->vocab_size; token++) {
            if (!token_is_selectable(search, token)) continue;
            TimedSelectEdge *edge = timed_find_edge(frame, token);
            if (edge == NULL) continue;
            existing_count++;
            if (edge->demand_count < minimum_demand) {
                minimum_demand = edge->demand_count;
            }
        }
    }
    bool demand_unseen = existing_count < support_count;
    if (!demand_unseen && minimum_demand == UINT64_MAX) {
        escardo_fail("timed sampling lost its demanded support");
    }

    double maximum = -DBL_MAX;
    if (support != NULL) {
        for (int index = 0; index < support_count; index++) {
            int token = support[index];
            if (!timed_argument_is_eligible(
                    frame, token, demand_unseen, minimum_demand
                )) {
                continue;
            }
            double value = logits->values[token];
            if (value > maximum) maximum = value;
        }
    } else {
        for (int token = 0; token < search->model->vocab_size; token++) {
            if (!token_is_selectable(search, token)) continue;
            if (!timed_argument_is_eligible(
                    frame, token, demand_unseen, minimum_demand
                )) {
                continue;
            }
            double value = logits->values[token];
            if (value > maximum) maximum = value;
        }
    }

    double mass = 0.0;
    if (support != NULL) {
        for (int index = 0; index < support_count; index++) {
            int token = support[index];
            if (!timed_argument_is_eligible(
                    frame, token, demand_unseen, minimum_demand
                )) {
                continue;
            }
            mass += exp((double)logits->values[token] - maximum);
        }
    } else {
        for (int token = 0; token < search->model->vocab_size; token++) {
            if (!token_is_selectable(search, token)) continue;
            if (!timed_argument_is_eligible(
                    frame, token, demand_unseen, minimum_demand
                )) {
                continue;
            }
            mass += exp((double)logits->values[token] - maximum);
        }
    }
    if (!(mass > 0.0) || !isfinite(mass)) {
        escardo_fail("invalid timed sampling mass");
    }

    double target = sample_random_unit(&search->sample_random_state) * mass;
    double cumulative = 0.0;
    int selected = -1;
    if (support != NULL) {
        for (int index = 0; index < support_count; index++) {
            int token = support[index];
            if (!timed_argument_is_eligible(
                    frame, token, demand_unseen, minimum_demand
                )) {
                continue;
            }
            cumulative += exp((double)logits->values[token] - maximum);
            selected = token;
            if (target < cumulative) break;
        }
    } else {
        for (int token = 0; token < search->model->vocab_size; token++) {
            if (!token_is_selectable(search, token)) continue;
            if (!timed_argument_is_eligible(
                    frame, token, demand_unseen, minimum_demand
                )) {
                continue;
            }
            cumulative += exp((double)logits->values[token] - maximum);
            selected = token;
            if (target < cumulative) break;
        }
    }
    free(support);
    if (selected < 0) escardo_fail("timed sampler selected no argument");

    double partition = log_partition(logits);
    return (ModelLogit){
        .context = history,
        .token = selected,
        .local_rank = timed_local_rank(search, logits, selected),
        .logit = logits->values[selected],
        .log_probability = (double)logits->values[selected] - partition,
    };
}

static TimedSelectFrame *timed_frame_for(
    Search *search,
    ModelNode *history,
    int remaining
) {
    if (history == NULL || !history->ready || history->logits == NULL ||
        remaining <= 0) {
        escardo_fail("invalid timed selection frame");
    }
    TimedSelectFrame *frame = history->timed_selection;
    if (frame != NULL) {
        if (frame->search != search || frame->remaining != remaining) {
            escardo_fail("memoized timed selection frame changed meaning");
        }
        return frame;
    }
    frame = arena_allocate(&search->model->arena, sizeof(*frame));
    memset(frame, 0, sizeof(*frame));
    frame->search = search;
    frame->history = history;
    frame->remaining = remaining;
    history->timed_selection = frame;
    search->strength_nodes++;
    return frame;
}

static TimedSelectEdge *timed_edge_for(
    TimedSelectFrame *frame,
    ModelLogit value
) {
    for (TimedSelectEdge *edge = frame->edges; edge != NULL;
         edge = edge->next) {
        if (edge->token != value.token) continue;
        if (edge->local_rank != value.local_rank ||
            edge->logit != value.logit ||
            fabs(edge->log_probability - value.log_probability) > 1e-12) {
            escardo_fail("memoized timed continuation changed model value");
        }
        return edge;
    }
    TimedSelectEdge *edge = arena_allocate(
        &frame->search->model->arena,
        sizeof(*edge)
    );
    memset(edge, 0, sizeof(*edge));
    edge->owner = frame;
    edge->token = value.token;
    edge->local_rank = value.local_rank;
    edge->logit = value.logit;
    edge->log_probability = value.log_probability;
    edge->next = frame->edges;
    frame->edges = edge;
    return edge;
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
    uint64_t budget,
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
    branch->score = terminal_root_coordinate(
        job->search,
        branch->value.token,
        outcome->terminal
    );
    branch->ready = true;
    job->ready_count++;
    job->search->candidate_observations++;
    trace_candidate(branch);
    select_job_after_branch(job);
}

static void select_branch_tail_ready(
    void *environment,
    SelectionOutcome *suffix,
    double selection_score
) {
    (void)selection_score;
    select_job_finish_branch(environment, suffix);
}

static void select_branch_history_ready(
    void *environment,
    ModelNode *child
) {
    SelectBranch *branch = environment;
    SelectJob *job = branch->job;
    if (job->remaining == 1) {
        SelectionOutcome *suffix = arena_allocate(
            &job->search->model->arena,
            sizeof(*suffix)
        );
        *suffix = (SelectionOutcome){
            .terminal = child,
        };
        select_job_finish_branch(branch, suffix);
        return;
    }
    select_path(
        job->search,
        child,
        job->remaining - 1,
        branch->child_budget,
        select_branch_tail_ready,
        branch
    );
}

static void select_start_branch(SelectBranch *branch) {
    SelectJob *job = branch->job;
    model_request_node(
        job->search->model,
        job->history,
        branch->value.token,
        select_branch_history_ready,
        branch
    );
}

static void trace_deadline_cutoff(SelectJob *job, int planned_count) {
    Search *search = job->search;
    if (search->trace == NULL) return;
    fprintf(
        search->trace,
        "{\"event\":\"deadline_cutoff\",\"frame\":%" PRIu64
        ",\"depth\":%d,\"remaining\":%d,\"admitted\":%d"
        ",\"planned\":%d}\n",
        job->frame_id,
        job->history->position - search->prompt_count + 1,
        job->remaining,
        job->started_count,
        planned_count
    );
    fflush(search->trace);
}

static void select_admit_branch(SelectJob *job, int index) {
    if (index != job->started_count || index < 0 ||
        index >= job->branch_count) {
        escardo_fail("invalid continuation admission");
    }
    SelectBranch *branch = &job->branches[index];
    job->started_count++;
    if (branch->sampled) job->search->sampled_candidate_demands++;
    trace_continuation_demand(
        job,
        index,
        &branch->value,
        branch->sampled,
        branch->support_probability,
        branch->draw,
        branch->child_budget
    );
    select_start_branch(branch);
}

static void select_job_after_branch(SelectJob *job) {
    bool timed = job->search->sample_milliseconds > 0;
    if (timed) {
        if (job->ready_count != job->started_count) return;
        if (job->started_count < job->branch_count &&
            !deadline_reached(job->search)) {
            select_admit_branch(job, job->started_count);
            return;
        }
        if (job->started_count < job->branch_count) {
            int planned_count = job->branch_count;
            trace_deadline_cutoff(job, planned_count);
            job->branch_count = job->started_count;
        }
    } else if (job->ready_count != job->branch_count) {
        return;
    }

    int best = 0;
    for (int index = 1; index < job->branch_count; index++) {
        if (job->branches[index].score > job->branches[best].score ||
            (job->branches[index].score == job->branches[best].score &&
             job->branches[index].value.local_rank <
                job->branches[best].value.local_rank)) {
            best = index;
        }
    }
    SelectBranch *selected = &job->branches[best];
    trace_choice(job, selected);
    job->continuation(
        job->continuation_environment,
        selected->outcome,
        selected->score
    );
}

static int demanded_branch_count(
    Search *search,
    uint64_t budget
) {
    int count = search->top_k;
    if (budget != UINT64_MAX && budget < (uint64_t)count) {
        count = (int)budget;
    }
    if (deadline_reached(search)) count = 1;
    if (count < 1) count = 1;
    return count;
}

static uint64_t branch_budget(
    uint64_t budget,
    int branch_count,
    int branch_index
) {
    if (budget == UINT64_MAX) return UINT64_MAX;
    uint64_t quotient = budget / (uint64_t)branch_count;
    uint64_t remainder = budget % (uint64_t)branch_count;
    uint64_t result = quotient + ((uint64_t)branch_index < remainder ? 1 : 0);
    return result == 0 ? 1 : result;
}

/* Mechanical asynchronous transcription of Escardo's dependent product:
 *
 *     b(x) = delta(x)(xs -> p(x : xs))
 *     a    = epsilon(x -> p(x : b(x)))
 *     result = a : b(a)
 *
 * Each branch stores b(x), so the chosen b(a) is returned rather than run a
 * second time.  Sampling changes only the finite set of x arguments demanded
 * from epsilon; every demanded x still receives the recursively composed
 * suffix observer before epsilon chooses a. */
static void select_path(
    Search *search,
    ModelNode *history,
    int remaining,
    uint64_t budget,
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
        };
        continuation(continuation_environment, outcome, NAN);
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
    job->budget = budget;
    job->continuation = continuation;
    job->continuation_environment = continuation_environment;
    job->branch_count = demanded_branch_count(search, budget);
    job->branches = arena_allocate(
        &search->model->arena,
        (size_t)job->branch_count * sizeof(*job->branches)
    );
    memset(job->branches, 0,
        (size_t)job->branch_count * sizeof(*job->branches));
    search->strength_nodes++;

    int support_count = search->top_k;
    int *tokens = top_tokens(search, history->logits, support_count);
    bool *sampled_ranks = escardo_calloc(
        (size_t)support_count,
        sizeof(*sampled_ranks)
    );
    uint64_t random_state = sample_history_state(search, history);
    double partition = log_partition(history->logits);
    for (int index = 0; index < job->branch_count; index++) {
        int candidate_rank = index;
        double support_probability = 0.0;
        double draw = 0.0;
        if (search->sampling_enabled) {
            candidate_rank = sample_next_local_rank(
                history->logits,
                tokens,
                support_count,
                sampled_ranks,
                &random_state,
                &support_probability,
                &draw
            );
            sampled_ranks[candidate_rank] = true;
        }
        int token = tokens[candidate_rank];
        uint64_t child_budget = branch_budget(
            budget,
            job->branch_count,
            index
        );
        SelectBranch *branch = &job->branches[index];
        *branch = (SelectBranch){
            .job = job,
            .value = {
                .context = history,
                .token = token,
                .local_rank = candidate_rank + 1,
                .logit = history->logits->values[token],
                .log_probability =
                    (double)history->logits->values[token] - partition,
            },
            .child_budget = child_budget,
            .sampled = search->sampling_enabled,
            .support_probability = support_probability,
            .draw = draw,
        };
    }
    free(sampled_ranks);
    free(tokens);
    if (search->sample_milliseconds > 0) {
        select_admit_branch(job, 0);
    } else {
        for (int index = 0; index < job->branch_count; index++) {
            select_admit_branch(job, index);
        }
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
    ModelNode *terminal;
    double score;
} RootRun;

static void tau_selection_ready(
    void *environment,
    SelectionOutcome *selected,
    double selection_score
) {
    RootRun *root = environment;
    if (selected == NULL || selected->terminal == NULL ||
        selected->path == NULL || !isfinite(selection_score)) {
        escardo_fail("root received an incomplete selection outcome");
    }
    /* Only the root forgets the structured outcome and emits its witness. */
    root->selected = selected->path;
    root->terminal = selected->terminal;
    root->score = selection_score;
    root->done = true;
}

static void exact_prompt_ready(void *environment, ModelNode *prompt) {
    RootRun *root = environment;
    Search *search = root->search;
    select_path(
        search,
        prompt,
        search->horizon,
        UINT64_MAX,
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

typedef struct {
    Search *search;
    TimedSelectFrame *root;
    TimedSelectFrame *frame;
    TimedSelectEdge **edges;
    ModelLogit *values;
    uint64_t sample_id;
    int depth;
    int count;
    bool done;
    LogitPath *path;
    double score;
} TimedSample;

static void timed_trace_demand(
    TimedSample *sample,
    TimedSelectEdge *edge
) {
    Search *search = sample->search;
    if (search->trace == NULL) return;
    fprintf(
        search->trace,
        "{\"event\":\"continuation_demand\",\"sample\":%" PRIu64
        ",\"frame\":%" PRIu64 ",\"depth\":%d,\"remaining\":%d"
        ",\"token\":%d,\"local_rank\":%d,\"logit\":%.9g"
        ",\"proposal_log_probability\":%.17g"
        ",\"multiplicity\":%" PRIu64 "}\n",
        sample->sample_id,
        edge->owner->history->id,
        sample->depth,
        edge->owner->remaining,
        edge->token,
        edge->local_rank,
        edge->logit,
        edge->log_probability,
        edge->demand_count
    );
    fflush(search->trace);
}

static bool timed_edge_improves(
    const TimedSelectEdge *candidate,
    const TimedSelectEdge *selected
) {
    if (selected == NULL) return true;
    if (candidate->selected_rating > selected->selected_rating) return true;
    if (candidate->selected_rating < selected->selected_rating) return false;
    if (candidate->local_rank < selected->local_rank) return true;
    if (candidate->local_rank > selected->local_rank) return false;
    return candidate->token < selected->token;
}

static void timed_trace_observation(TimedSample *sample) {
    Search *search = sample->search;
    if (search->trace == NULL) return;
    fprintf(
        search->trace,
        "{\"event\":\"strength_observation\",\"sample\":%" PRIu64
        ",\"token_count\":%d,\"terminal_node\":%" PRIu64
        ",\"root_coordinate\":%.17g,\"text\":",
        sample->sample_id,
        sample->count,
        sample->root->selected->selected_terminal->id,
        sample->score
    );
    trace_decoded_path(search, sample->root->history, sample->path);
    fputs("}\n", search->trace);
    fflush(search->trace);
}

static void timed_trace_candidate(TimedSelectEdge *edge) {
    Search *search = edge->owner->search;
    if (search->trace == NULL) return;
    fprintf(
        search->trace,
        "{\"event\":\"candidate\",\"frame\":%" PRIu64
        ",\"depth\":%d,\"remaining\":%d,\"token\":%d"
        ",\"local_rank\":%d,\"multiplicity\":%" PRIu64
        ",\"terminal_node\":%" PRIu64
        ",\"frame_observations\":%" PRIu64
        ",\"root_coordinate\":%.17g,\"text\":",
        edge->owner->history->id,
        edge->owner->history->position - search->prompt_count + 1,
        edge->owner->remaining,
        edge->token,
        edge->local_rank,
        edge->demand_count,
        edge->selected_terminal->id,
        edge->owner->observation_count,
        edge->selected_rating
    );
    trace_decoded_path(search, edge->owner->history, edge->selected_path);
    fputs("}\n", search->trace);
    fflush(search->trace);
}

static ModelLogit timed_edge_value(const TimedSelectEdge *edge) {
    return (ModelLogit){
        .context = edge->owner->history,
        .token = edge->token,
        .local_rank = edge->local_rank,
        .logit = edge->logit,
        .log_probability = edge->log_probability,
    };
}

/* Force Escardo's product over exactly the memo cells sampled so far. Each
 * edge is re-observed only when its recursively selected terminal outcome
 * changes. This is b(x) memoization; there is no rollout reward or path
 * backup. */
static bool timed_strength_force(TimedSelectFrame *frame) {
    TimedSelectEdge *best = NULL;
    for (TimedSelectEdge *edge = frame->edges; edge != NULL;
         edge = edge->next) {
        ModelNode *terminal = NULL;
        LogitPath *tail = NULL;
        if (edge->suffix != NULL) {
            if (!timed_strength_force(edge->suffix)) continue;
            TimedSelectEdge *suffix = edge->suffix->selected;
            terminal = suffix->selected_terminal;
            tail = suffix->selected_path;
        } else {
            terminal = edge->terminal;
        }
        if (terminal == NULL) continue;

        if (!edge->observed || edge->selected_terminal != terminal) {
            edge->selected_terminal = terminal;
            edge->selected_path = path_cons(
                frame->search,
                timed_edge_value(edge),
                tail
            );
            edge->selected_rating = terminal_root_coordinate(
                frame->search,
                edge->token,
                terminal
            );
            edge->observed = true;
            edge->observation_count++;
            frame->observation_count++;
            frame->search->candidate_observations++;
            timed_trace_candidate(edge);
        }
        if (timed_edge_improves(edge, best)) best = edge;
    }
    frame->selected = best;
    return best != NULL;
}

static void timed_sample_finish(TimedSample *sample) {
    if (!timed_strength_force(sample->root) ||
        sample->root->selected->selected_path == NULL ||
        sample->root->selected->selected_terminal == NULL) {
        escardo_fail("sampled strength has no completed outcome");
    }
    sample->path = sample->root->selected->selected_path;
    sample->score = sample->root->selected->selected_rating;
    timed_trace_observation(sample);
    sample->search->completed_samples++;
    sample->done = true;
}

static void timed_sample_step(TimedSample *sample);

static void timed_sample_child_ready(void *environment, ModelNode *child) {
    TimedSample *sample = environment;
    TimedSelectEdge *edge = sample->edges[sample->depth];
    if (edge->owner->remaining == 1 ||
        (sample->search->allow_delimiter &&
         edge->token == ESCARDO_SEQUENCE_DELIMITER)) {
        if (edge->terminal != NULL && edge->terminal != child) {
            escardo_fail("memoized terminal continuation changed node");
        }
        edge->terminal = child;
        timed_sample_finish(sample);
        return;
    }
    TimedSelectFrame *suffix = timed_frame_for(
        sample->search,
        child,
        edge->owner->remaining - 1
    );
    if (edge->suffix != NULL && edge->suffix != suffix) {
        escardo_fail("timed continuation changed suffix frame");
    }
    edge->suffix = suffix;
    sample->frame = suffix;
    sample->depth++;
    timed_sample_step(sample);
}

/* One stochastic invocation of the recursively composed product.  Each local
 * Select samples one observer argument, records it immediately in its
 * persistent frame, and passes the still-open observation to the suffix
 * Select.  The wall-clock loop invokes the same memoized product repeatedly;
 * it never predetermines a leaf count or divides one across early branches. */
static void timed_sample_step(TimedSample *sample) {
    TimedSelectFrame *frame = sample->frame;
    ModelLogit value = timed_sample_argument(sample->search, frame);
    TimedSelectEdge *edge = timed_edge_for(frame, value);
    frame->demand_count++;
    edge->demand_count++;
    sample->search->sampled_candidate_demands++;
    sample->edges[sample->depth] = edge;
    sample->values[sample->depth] = value;
    sample->count = sample->depth + 1;
    timed_trace_demand(sample, edge);

    model_request_node(
        sample->search->model,
        frame->history,
        value.token,
        timed_sample_child_ready,
        sample
    );
}

typedef struct {
    bool ready;
    ModelNode *node;
} TimedPrompt;

static void timed_prompt_ready(void *environment, ModelNode *node) {
    TimedPrompt *prompt = environment;
    prompt->node = node;
    prompt->ready = true;
}

static void timed_trace_selected(Search *search, TimedSelectFrame *root) {
    if (search->trace == NULL || root == NULL || root->selected == NULL) return;
    TimedSelectFrame *frame = root;
    int depth = 0;
    while (frame != NULL && frame->selected != NULL) {
        TimedSelectEdge *edge = frame->selected;
        int alternatives = 0;
        for (TimedSelectEdge *member = frame->edges; member != NULL;
             member = member->next) {
            if (member->observed) alternatives++;
        }
        fprintf(
            search->trace,
            "{\"event\":\"select\",\"frame\":%" PRIu64
            ",\"depth\":%d,\"token\":%d,\"local_rank\":%d"
            ",\"alternatives\":%d,\"multiplicity\":%" PRIu64
            ",\"terminal_node\":%" PRIu64
            ",\"root_coordinate\":%.17g}\n",
            frame->history->id,
            depth,
            edge->token,
            edge->local_rank,
            alternatives,
            edge->demand_count,
            edge->selected_terminal->id,
            edge->selected_rating
        );
        fflush(search->trace);
        frame = edge->suffix;
        depth++;
    }
}

static RootRun escardo_timed_run(
    Search *search,
    int *prompt_tokens,
    int prompt_count
) {
    TimedPrompt prompt = {0};
    prefill_start(
        search,
        prompt_tokens,
        prompt_count,
        timed_prompt_ready,
        &prompt
    );
    while (!prompt.ready) {
        if (!scheduler_step(&search->model->scheduler)) {
            escardo_fail("timed prefill scheduler reached a deadlock");
        }
    }

    TimedSelectFrame *root_frame = timed_frame_for(
        search,
        prompt.node,
        search->horizon
    );
    search->deadline = add_milliseconds(
        monotonic_now(),
        search->sample_milliseconds
    );
    search->deadline_armed = true;

    TimedSample *samples = escardo_calloc(
        (size_t)search->batch_size,
        sizeof(*samples)
    );
    bool first = true;
    while (first || !deadline_reached(search)) {
        first = false;
        uint64_t first_sample_id = search->completed_samples;
        for (int index = 0; index < search->batch_size; index++) {
            samples[index] = (TimedSample){
                .search = search,
                .root = root_frame,
                .frame = root_frame,
                .sample_id = first_sample_id + (uint64_t)index,
            };
            samples[index].edges = escardo_calloc(
                (size_t)search->horizon,
                sizeof(*samples[index].edges)
            );
            samples[index].values = escardo_calloc(
                (size_t)search->horizon,
                sizeof(*samples[index].values)
            );
        }

        /* All product invocations enter their local Select before the
         * scheduler is drained.  Consequently equal learned scopes receive
         * the whole ready family in one numerical application. */
        for (int index = 0; index < search->batch_size; index++) {
            timed_sample_step(&samples[index]);
        }
        for (;;) {
            bool all_done = true;
            for (int index = 0; index < search->batch_size; index++) {
                if (!samples[index].done) {
                    all_done = false;
                    break;
                }
            }
            if (all_done) break;
            if (!scheduler_step(&search->model->scheduler)) {
                escardo_fail("timed continuation scheduler reached a deadlock");
            }
        }
        for (int index = 0; index < search->batch_size; index++) {
            free(samples[index].values);
            free(samples[index].edges);
        }
    }
    free(samples);

    if (root_frame->selected == NULL ||
        root_frame->selected->selected_path == NULL ||
        root_frame->selected->selected_terminal == NULL) {
        escardo_fail("timed selection produced no complete observation");
    }
    timed_trace_selected(search, root_frame);
    RootRun root = {
        .search = search,
        .done = true,
        .selected = root_frame->selected->selected_path,
        .terminal = root_frame->selected->selected_terminal,
        .score = root_frame->selected->selected_rating,
    };
    /* The root alone forgets the selected structured outcome and emits the
     * token witness. It does not run another model observation. */
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
    int sample_milliseconds;
    int batch_size;
    uint64_t sample_seed;
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

static uint64_t parse_u64(const char *text, const char *name) {
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        fprintf(stderr, "escardo: %s must be an unsigned integer\n", name);
        exit(EXIT_FAILURE);
    }
    return (uint64_t)value;
}

_Noreturn static void usage(const char *program) {
    fprintf(
        stderr,
        "usage: %s CHECKPOINT TOKENIZER --prompt TEXT --length N "
        "(--sample-ms MS [--top-k K] | --exact --top-k K) "
        "[--batch-size N] [--seed N] [--trace FILE] [--allow-delimiter] "
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
        .sample_milliseconds = -1,
        .batch_size = 1,
        .sample_seed = UINT64_C(42),
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
        } else if (strcmp(flag, "--sample-ms") == 0) {
            long parsed = parse_long(value, "sample-ms");
            if (parsed <= 0 || parsed > INT_MAX) usage(argv[0]);
            options.sample_milliseconds = (int)parsed;
        } else if (strcmp(flag, "--batch-size") == 0) {
            long parsed = parse_long(value, "batch-size");
            if (parsed <= 0 || parsed > INT_MAX) usage(argv[0]);
            options.batch_size = (int)parsed;
        } else if (strcmp(flag, "--seed") == 0) {
            options.sample_seed = parse_u64(value, "seed");
        } else if (strcmp(flag, "--trace") == 0) {
            options.trace_path = value;
        } else if (strcmp(flag, "--metal-library") == 0) {
            options.metal_library = value;
        } else {
            usage(argv[0]);
        }
    }
    if (options.prompt == NULL || options.length <= 0 ||
        (options.exact && options.sample_milliseconds > 0) ||
        (options.exact && options.top_k <= 0) ||
        (!options.exact && options.sample_milliseconds <= 0)) {
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
        if (options.top_k == 0) fputs("null", trace);
        else fprintf(trace, "%d", options.top_k);
        fputs(",\"sample_ms\":", trace);
        if (options.exact) fputs("null", trace);
        else fprintf(trace, "%d", options.sample_milliseconds);
        fprintf(
            trace,
            ",\"seed\":%" PRIu64 ",\"batch_size\":%d"
            ",\"backend\":\"%s\",\"mode\":\"%s\"}\n",
            options.sample_seed,
            options.batch_size,
            atkey_backend_name(runtime),
            options.exact ? "exact_product" : "timed_sampled_product"
        );
        fflush(trace);
    }

    uint64_t random_state = sample_random_mix(
        options.sample_seed ^ UINT64_C(0x6a09e667f3bcc909)
    );
    if (random_state == 0) random_state = UINT64_C(0x4d595df4d0f33173);
    Search search = {
        .model = &model,
        .prompt_text = options.prompt,
        .prompt_count = prompt_count,
        .prompt_last_token = prompt_tokens[prompt_count - 1],
        .horizon = options.length,
        .top_k = options.top_k,
        .sample_milliseconds = options.exact ? 0 :
            options.sample_milliseconds,
        .batch_size = options.batch_size,
        .sample_seed = options.sample_seed,
        .sample_random_state = random_state,
        .sampling_enabled = !options.exact,
        .allow_delimiter = options.allow_delimiter,
        .trace = trace,
    };

    RootRun result = options.exact ?
        escardo_exact_run(&search, prompt_tokens, prompt_count) :
        escardo_timed_run(&search, prompt_tokens, prompt_count);
    puts("completion:");
    print_selected(&search, result.selected);
    printf(
        "selected_root_coordinate=%.17g\n"
        "score_kind=terminal_root_covector_coordinate\n"
        "selection_carrier=ModelLogit_with_terminal_SelectionOutcome\n"
        "root_terminalizations=1\n"
        "strength_nodes=%" PRIu64 "\n"
        "candidate_observations=%" PRIu64 "\n"
        "sampled_candidate_demands=%" PRIu64 "\n"
        "completed_samples=%" PRIu64 "\n"
        "constructor_codata_evaluations=%" PRIu64 "\n"
        "model_token_terms=%" PRIu64 "\n",
        result.score,
        search.strength_nodes,
        search.candidate_observations,
        search.sampled_candidate_demands,
        search.completed_samples,
        search.payoff_observations,
        model.model_steps
    );
    printf(
        "numerical_backend=%s\n"
        "backend_device=%s\n"
        "backend_dispatches=%" PRIu64 "\n"
        "backend_weight_uploads=%" PRIu64 "\n"
        "backend_weight_upload_bytes=%" PRIu64 "\n"
        "search_batch_size=%d\n",
        atkey_backend_name(runtime),
        atkey_backend_device_name(runtime),
        atkey_backend_dispatch_count(runtime),
        atkey_backend_weight_upload_count(runtime),
        atkey_backend_weight_upload_bytes(runtime),
        options.batch_size
    );
    print_model_summary(&model);

    if (trace != NULL) {
        fprintf(
            trace,
            "{\"event\":\"run_end\",\"selected_root_coordinate\":%.17g"
            ",\"strength_nodes\":%" PRIu64
            ",\"candidate_observations\":%" PRIu64
            ",\"completed_samples\":%" PRIu64 "}\n",
            result.score,
            search.strength_nodes,
            search.candidate_observations,
            search.completed_samples
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
