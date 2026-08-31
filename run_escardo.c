/*
 * Unoptimized non-greedy inference by the dependent product of selection
 * functions.  Physical kernel/weight reuse is deliberately not claimed here;
 * that is a later lowering and performance problem.
 *
 * The selection carrier is a hypothetical token constructor and the outcome
 * is the frozen model trace of the completed hypothetical company:
 *
 *     R = (selected suffix, terminal node, token-indexed company covectors)
 *     J_R ModelLogit = (ModelLogit -> R) -> ModelLogit
 *
 * Before selection, the prompt is evaluated once and its final normalized
 * hidden state is fed back through the unchanged transformer for the requested
 * horizon without projecting or embedding generated tokens.  The resulting
 * fixed logit tape proposes constructor arguments at each position.  It never
 * rates or emits them, and it is independent of every hypothetical branch.
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
 * b(x), then compares the whole outcome x : b(x) with the same leximin order
 * used by every other local Select. Each constructor is paired with its own
 * incoming model covector; the coordinates are retained and ordered from
 * worst company opinion to best. They are never summed. Exact mode enumerates
 * an explicitly requested finite local support from the fixed tape. Timed
 * mode uses the same recursive interpreter: epsilon demands one x, the
 * composed observer returns p(x : b(x)), and only then may epsilon demand its
 * next x. The deadline only stops a returned epsilon from making another
 * demand; it never separates candidate construction from selection. No
 * AR-prefix covector participates in proposal or selection.
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
typedef struct CompanyOutcome CompanyOutcome;

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
    CompanyOutcome *company_outcome;
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

/* Hidden-feedback proposal nodes use the preceding final normalized hidden
 * state as the next residual-stream input.  They are not token branches and
 * are therefore never inserted into ModelNode.children.  The resulting tape
 * is constructed once, before Select is run, and can only propose arguments;
 * completed token branches are still observed through model_request_node. */
static void model_request_feedback_node(
    ModelTerm *model,
    ModelNode *parent,
    Tensor *input,
    NodeContinuation continuation,
    void *environment
) {
    if (parent == NULL || !parent->ready || parent->final_hidden == NULL ||
        input == NULL || input->width != model->dim) {
        escardo_fail("invalid hidden-feedback proposal request");
    }
    ModelNode *node = model_new_node(model, parent, -1);
    ModelStep *step = arena_allocate(&model->arena, sizeof(*step));
    memset(step, 0, sizeof(*step));
    step->model = model;
    step->node = node;
    model_add_waiter(model, node, continuation, environment);
    model_step_on_embedding(step, input);
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
    CompanyOutcome *company;
};

struct CompanyOutcome {
    int count;
    double *coordinates;
    int *leximin_positions;
    double *leximin_coordinates;
};

typedef void (*OutcomeContinuation)(
    void *environment,
    SelectionOutcome *outcome,
    double selection_score
);

typedef struct Search Search;
typedef struct SelectJob SelectJob;
typedef struct SelectBranch SelectBranch;

struct Search {
    ModelTerm *model;
    const char *prompt_text;
    int prompt_count;
    int prompt_last_token;
    int horizon;
    /* Candidate-independent codata used only to choose which constructor
     * arguments a local Select demands.  Entry zero is the final prompt
     * covector; later entries come from hidden-state feedback without any
     * intervening token projection. */
    ModelNode **proposal_nodes;
    Tensor **proposal_logits;
    /* Zero means that timed sampling uses the full selectable vocabulary.
     * A positive value is only a proposal-distribution truncation; it is not
     * a bound on the number of continuations the term may demand. */
    int top_k;
    int sample_milliseconds;
    uint64_t sample_seed;
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

/* The sampler chooses which arguments a local selection function asks its
 * observer about.  Proposal state is indexed only by the fixed prompt and
 * completion position.  It must not depend on a hypothetical token prefix:
 * branch-dependent AR logits belong to the continuation being observed, not
 * to the local selector's proposal tape. */
static uint64_t sample_random_mix(uint64_t value) {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31;
    return value;
}

static int selection_depth(
    const Search *search,
    const ModelNode *history
) {
    if (history == NULL) escardo_fail("selection has no model history");
    int depth = history->position - search->prompt_count + 1;
    if (depth < 0 || depth >= search->horizon) {
        escardo_fail("selection escaped the hidden-feedback proposal tape");
    }
    return depth;
}

static Tensor *selection_proposal_logits(
    const Search *search,
    const ModelNode *history
) {
    int depth = selection_depth(search, history);
    Tensor *logits = search->proposal_logits[depth];
    if (logits == NULL || logits->width != search->model->vocab_size) {
        escardo_fail("selection observed an incomplete proposal covector");
    }
    return logits;
}

static uint64_t sample_position_state(
    const Search *search,
    int depth
) {
    uint64_t hash = search->sample_seed ^ UINT64_C(0xcbf29ce484222325);
    for (const unsigned char *byte =
             (const unsigned char *)search->prompt_text;
         *byte != '\0'; byte++) {
        hash ^= (uint64_t)*byte;
        hash *= UINT64_C(0x100000001b3);
    }
    hash ^= (uint64_t)(uint32_t)depth;
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

typedef struct {
    double coordinate;
    int position;
} OrderedCompanyCoordinate;

static int ordered_company_coordinate_compare(
    const void *left_value,
    const void *right_value
) {
    const OrderedCompanyCoordinate *left = left_value;
    const OrderedCompanyCoordinate *right = right_value;
    if (left->coordinate < right->coordinate) return -1;
    if (left->coordinate > right->coordinate) return 1;
    if (left->position < right->position) return -1;
    if (left->position > right->position) return 1;
    return 0;
}

/* Retain one model-native opinion for every constructor in a completed
 * hypothetical company. The covector is the constructor's incoming causal
 * context. Sorting does not add or average coordinates from different
 * contexts: it gives one shared leximin order to every local Select. */
static CompanyOutcome *company_outcome_for_terminal(
    Search *search,
    ModelNode *terminal
) {
    if (terminal == NULL || !terminal->ready) {
        escardo_fail("selection received an incomplete company outcome");
    }
    if (terminal->company_outcome != NULL) {
        return terminal->company_outcome;
    }

    int count = terminal->position - search->prompt_count + 1;
    if (count <= 0 || count > search->horizon) {
        escardo_fail("terminal company has an invalid completion length");
    }
    if (count != search->horizon) {
        escardo_fail("leximin company outcomes require a fixed completion length");
    }

    CompanyOutcome *outcome = arena_allocate(
        &search->model->arena,
        sizeof(*outcome)
    );
    outcome->count = count;
    outcome->coordinates = arena_allocate(
        &search->model->arena,
        (size_t)count * sizeof(*outcome->coordinates)
    );
    outcome->leximin_positions = arena_allocate(
        &search->model->arena,
        (size_t)count * sizeof(*outcome->leximin_positions)
    );
    outcome->leximin_coordinates = arena_allocate(
        &search->model->arena,
        (size_t)count * sizeof(*outcome->leximin_coordinates)
    );
    OrderedCompanyCoordinate *ordered = escardo_calloc(
        (size_t)count,
        sizeof(*ordered)
    );

    ModelNode *node = terminal;
    for (int position = count - 1; position >= 0; position--) {
        ModelNode *context = node->parent;
        if (context == NULL || !context->ready || context->logits == NULL ||
            node->token < 0 || node->token >= search->model->vocab_size) {
            escardo_fail("company constructor lacks its incoming covector");
        }
        if (!context->log_partition_ready) {
            context->log_partition = log_partition(context->logits);
            context->log_partition_ready = true;
        }
        double coordinate =
            (double)context->logits->values[node->token] -
            context->log_partition;
        if (!isfinite(coordinate)) {
            escardo_fail("company coordinate is not finite");
        }
        outcome->coordinates[position] = coordinate;
        ordered[position] = (OrderedCompanyCoordinate){
            .coordinate = coordinate,
            .position = position,
        };
        search->payoff_observations++;
        node = context;
    }
    if (node->position != search->prompt_count - 1) {
        escardo_fail("company outcome did not return to the prompt boundary");
    }

    qsort(
        ordered,
        (size_t)count,
        sizeof(*ordered),
        ordered_company_coordinate_compare
    );
    for (int rank = 0; rank < count; rank++) {
        outcome->leximin_positions[rank] = ordered[rank].position;
        outcome->leximin_coordinates[rank] = ordered[rank].coordinate;
    }
    free(ordered);
    terminal->company_outcome = outcome;
    return outcome;
}

static int company_outcome_compare(
    const CompanyOutcome *left,
    const CompanyOutcome *right
) {
    if (left == NULL || right == NULL || left->count != right->count) {
        escardo_fail("cannot compare incompatible company outcomes");
    }
    for (int rank = 0; rank < left->count; rank++) {
        if (left->leximin_coordinates[rank] >
            right->leximin_coordinates[rank]) return 1;
        if (left->leximin_coordinates[rank] <
            right->leximin_coordinates[rank]) return -1;
    }
    return 0;
}

static double company_outcome_diagnostic(const CompanyOutcome *outcome) {
    if (outcome == NULL || outcome->count <= 0) {
        escardo_fail("company outcome has no coordinate");
    }
    return outcome->leximin_coordinates[0];
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

static void trace_company_outcome(
    FILE *stream,
    const CompanyOutcome *outcome
) {
    if (outcome == NULL) escardo_fail("trace received no company outcome");
    fputs(",\"company_coordinates\":[", stream);
    for (int position = 0; position < outcome->count; position++) {
        if (position != 0) fputc(',', stream);
        fprintf(stream, "%.17g", outcome->coordinates[position]);
    }
    fputs("],\"leximin\":[", stream);
    for (int rank = 0; rank < outcome->count; rank++) {
        if (rank != 0) fputc(',', stream);
        fprintf(
            stream,
            "{\"rank\":%d,\"position\":%d,\"opinion\":%.17g}",
            rank,
            outcome->leximin_positions[rank],
            outcome->leximin_coordinates[rank]
        );
    }
    fputc(']', stream);
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
        ",\"worst_company_coordinate\":%.17g",
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
    trace_company_outcome(stream, branch->outcome->company);
    fputs(",\"text\":", stream);
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
        ",\"worst_company_coordinate\":%.17g",
        job->frame_id,
        job->history->position - search->prompt_count + 1,
        job->remaining,
        branch->value.token,
        branch->value.local_rank,
        branch->outcome->terminal->id,
        branch->score
    );
    trace_company_outcome(search->trace, branch->outcome->company);
    fputs(",\"text\":", search->trace);
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

static bool token_is_selectable(const Search *search, int token) {
    return search->allow_delimiter || token != ESCARDO_SEQUENCE_DELIMITER;
}

static int selectable_token_count(const Search *search) {
    return search->model->vocab_size -
        (search->allow_delimiter ? 0 : 1);
}

static int local_logit_rank(
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

typedef struct {
    int token;
    int local_rank;
    double key;
    double draw;
    double probability;
} SampledArgument;

static int sampled_argument_compare(
    const void *left_value,
    const void *right_value
) {
    const SampledArgument *left = left_value;
    const SampledArgument *right = right_value;
    if (left->key > right->key) return -1;
    if (left->key < right->key) return 1;
    if (left->token < right->token) return -1;
    if (left->token > right->token) return 1;
    return 0;
}

/* A Gumbel ordering is a complete, without-replacement sampling order for
 * epsilon's finite support. Constructing this order samples only arguments of
 * the local selection; it does not evaluate any suffix or complete path. */
static SampledArgument *sample_argument_order(
    const Tensor *logits,
    const int *support,
    int support_count,
    bool support_is_ranked,
    uint64_t *random_state
) {
    SampledArgument *arguments = escardo_calloc(
        (size_t)support_count,
        sizeof(*arguments)
    );
    double maximum = -DBL_MAX;
    for (int index = 0; index < support_count; index++) {
        double value = logits->values[support[index]];
        if (value > maximum) maximum = value;
    }
    double mass = 0.0;
    for (int index = 0; index < support_count; index++) {
        mass += exp((double)logits->values[support[index]] - maximum);
    }
    if (!(mass > 0.0) || !isfinite(mass)) {
        escardo_fail("invalid sampled local support mass");
    }
    for (int index = 0; index < support_count; index++) {
        int token = support[index];
        double draw = sample_random_unit(random_state);
        if (draw <= 0.0) draw = 1.0 / 9007199254740992.0;
        arguments[index] = (SampledArgument){
            .token = token,
            .local_rank = support_is_ranked ? index + 1 : 0,
            .key = (double)logits->values[token] - log(-log(draw)),
            .draw = draw,
            .probability = exp((double)logits->values[token] - maximum) /
                mass,
        };
    }
    qsort(
        arguments,
        (size_t)support_count,
        sizeof(*arguments),
        sampled_argument_compare
    );
    return arguments;
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
    if (suffix == NULL || suffix->terminal == NULL ||
        suffix->company == NULL) {
        escardo_fail("selection branch returned no terminal outcome");
    }
    SelectionOutcome *outcome = arena_allocate(
        &job->search->model->arena,
        sizeof(*outcome)
    );
    *outcome = *suffix;
    outcome->path = path_cons(job->search, branch->value, suffix->path);
    branch->outcome = outcome;
    branch->score = company_outcome_diagnostic(outcome->company);
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
            .company = company_outcome_for_terminal(job->search, child),
        };
        job->search->completed_samples++;
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
    if (branch->value.local_rank == 0) {
        branch->value.local_rank = local_logit_rank(
            job->search,
            selection_proposal_logits(job->search, job->history),
            branch->value.token
        );
    }
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
        int order = company_outcome_compare(
            job->branches[index].outcome->company,
            job->branches[best].outcome->company
        );
        if (order > 0 ||
            (order == 0 &&
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
    int count = search->top_k > 0 ?
        search->top_k : selectable_token_count(search);
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
            .company = company_outcome_for_terminal(search, history),
        };
        continuation(continuation_environment, outcome, NAN);
        return;
    }
    if (history == NULL || !history->ready || history->logits == NULL) {
        escardo_fail("selection observed an unavailable model logit");
    }
    Tensor *proposal_logits = selection_proposal_logits(search, history);
    int depth = selection_depth(search, history);
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

    int support_count = search->top_k > 0 ?
        search->top_k : selectable_token_count(search);
    bool support_is_ranked = search->top_k > 0;
    int *tokens;
    if (support_is_ranked) {
        tokens = top_tokens(search, proposal_logits, support_count);
    } else {
        tokens = escardo_calloc((size_t)support_count, sizeof(*tokens));
        int index = 0;
        for (int token = 0; token < search->model->vocab_size; token++) {
            if (!token_is_selectable(search, token)) continue;
            tokens[index++] = token;
        }
        if (index != support_count) {
            escardo_fail("could not construct full local support");
        }
    }
    uint64_t random_state = sample_position_state(search, depth);
    SampledArgument *sampled_arguments = NULL;
    if (search->sampling_enabled) {
        sampled_arguments = sample_argument_order(
            proposal_logits,
            tokens,
            support_count,
            support_is_ranked,
            &random_state
        );
    }
    double partition = log_partition(proposal_logits);
    for (int index = 0; index < job->branch_count; index++) {
        int token = tokens[index];
        int local_rank = index + 1;
        double support_probability = 0.0;
        double draw = 0.0;
        if (search->sampling_enabled) {
            SampledArgument sampled = sampled_arguments[index];
            token = sampled.token;
            local_rank = sampled.local_rank;
            support_probability = sampled.probability;
            draw = sampled.draw;
        }
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
                .local_rank = local_rank,
                .logit = proposal_logits->values[token],
                .log_probability =
                    (double)proposal_logits->values[token] - partition,
            },
            .child_budget = child_budget,
            .sampled = search->sampling_enabled,
            .support_probability = support_probability,
            .draw = draw,
        };
    }
    free(sampled_arguments);
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
    ModelNode *prompt;
    int next_index;
    bool done;
} ProposalBuild;

static void proposal_feedback_ready(void *environment, ModelNode *node) {
    ProposalBuild *build = environment;
    Search *search = build->search;
    int index = build->next_index;
    if (index <= 0 || index >= search->horizon || node == NULL ||
        !node->ready || node->final_hidden == NULL || node->logits == NULL) {
        escardo_fail("hidden-feedback proposal tape returned an invalid node");
    }
    search->proposal_nodes[index] = node;
    search->proposal_logits[index] = node->logits;
    build->next_index++;
    if (build->next_index == search->horizon) {
        build->done = true;
        return;
    }
    model_request_feedback_node(
        search->model,
        node,
        node->final_hidden,
        proposal_feedback_ready,
        build
    );
}

static void proposal_prompt_ready(void *environment, ModelNode *prompt) {
    ProposalBuild *build = environment;
    Search *search = build->search;
    if (prompt == NULL || !prompt->ready || prompt->final_hidden == NULL ||
        prompt->logits == NULL) {
        escardo_fail("prompt did not produce hidden-feedback codata");
    }
    build->prompt = prompt;
    search->proposal_nodes[0] = prompt;
    search->proposal_logits[0] = prompt->logits;
    build->next_index = 1;
    if (search->horizon == 1) {
        build->done = true;
        return;
    }
    model_request_feedback_node(
        search->model,
        prompt,
        prompt->final_hidden,
        proposal_feedback_ready,
        build
    );
}

static void trace_proposal_tape(Search *search) {
    if (search->trace == NULL) return;
    for (int index = 0; index < search->horizon; index++) {
        Tensor *logits = search->proposal_logits[index];
        int *top = top_tokens(search, logits, 1);
        int token = top[0];
        free(top);
        double partition = log_partition(logits);
        fprintf(
            search->trace,
            "{\"event\":\"proposal_covector\",\"position\":%d,"
            "\"node\":%" PRIu64 ",\"top_token\":%d,\"top_logit\":%.9g,"
            "\"top_log_probability\":%.17g,\"piece\":",
            index,
            search->proposal_nodes[index]->id,
            token,
            logits->values[token],
            (double)logits->values[token] - partition
        );
        json_string(
            search->trace,
            atkey_decode(search->model->runtime, 0, token)
        );
        fputs("}\n", search->trace);
        fflush(search->trace);
    }
}

static ModelNode *prepare_prompt_and_proposals(
    Search *search,
    int *prompt_tokens,
    int prompt_count
) {
    search->proposal_nodes = arena_allocate(
        &search->model->arena,
        (size_t)search->horizon * sizeof(*search->proposal_nodes)
    );
    search->proposal_logits = arena_allocate(
        &search->model->arena,
        (size_t)search->horizon * sizeof(*search->proposal_logits)
    );
    memset(
        search->proposal_nodes,
        0,
        (size_t)search->horizon * sizeof(*search->proposal_nodes)
    );
    memset(
        search->proposal_logits,
        0,
        (size_t)search->horizon * sizeof(*search->proposal_logits)
    );
    ProposalBuild build = {.search = search};
    prefill_start(
        search,
        prompt_tokens,
        prompt_count,
        proposal_prompt_ready,
        &build
    );
    while (!build.done) {
        if (!scheduler_step(&search->model->scheduler)) {
            escardo_fail("hidden-feedback proposal scheduler deadlocked");
        }
    }
    trace_proposal_tape(search);
    return build.prompt;
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
        selected->path == NULL || selected->company == NULL ||
        !isfinite(selection_score)) {
        escardo_fail("root received an incomplete selection outcome");
    }
    /* Only the root forgets the structured outcome and emits its witness. */
    root->selected = selected->path;
    root->terminal = selected->terminal;
    root->score = selection_score;
    root->done = true;
}

static RootRun escardo_exact_run(
    Search *search,
    ModelNode *prompt
) {
    RootRun root = {.search = search};
    select_path(
        search,
        prompt,
        search->horizon,
        UINT64_MAX,
        tau_selection_ready,
        &root
    );
    while (!root.done) {
        if (!scheduler_step(&search->model->scheduler)) {
            escardo_fail("continuation scheduler reached a deadlock");
        }
    }
    return root;
}

static RootRun escardo_timed_run(
    Search *search,
    ModelNode *prompt
) {
    RootRun root = {.search = search};
    search->deadline = add_milliseconds(
        monotonic_now(),
        search->sample_milliseconds
    );
    search->deadline_armed = true;
    select_path(
        search,
        prompt,
        search->horizon,
        UINT64_MAX,
        tau_selection_ready,
        &root
    );
    while (!root.done) {
        if (!scheduler_step(&search->model->scheduler)) {
            escardo_fail("timed continuation scheduler reached a deadlock");
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
    int sample_milliseconds;
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
        "[--seed N] [--trace FILE] [--allow-delimiter] "
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
            ",\"seed\":%" PRIu64
            ",\"backend\":\"%s\",\"mode\":\"%s\""
            ",\"proposal\":\"candidate_independent_hidden_feedback\"}\n",
            options.sample_seed,
            atkey_backend_name(runtime),
            options.exact ? "exact_product" : "timed_sampled_product"
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
        .sample_milliseconds = options.exact ? 0 :
            options.sample_milliseconds,
        .sample_seed = options.sample_seed,
        .sampling_enabled = !options.exact,
        .allow_delimiter = options.allow_delimiter,
        .trace = trace,
    };

    ModelNode *prompt_node = prepare_prompt_and_proposals(
        &search,
        prompt_tokens,
        prompt_count
    );
    RootRun result = options.exact ?
        escardo_exact_run(&search, prompt_node) :
        escardo_timed_run(&search, prompt_node);
    puts("completion:");
    print_selected(&search, result.selected);
    printf(
        "selected_worst_company_coordinate=%.17g\n"
        "score_kind=leximin_complete_company_outcome\n"
        "proposal_kind=candidate_independent_hidden_feedback_tape\n"
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
            "{\"event\":\"run_end\",\"selected_worst_company_coordinate\":%.17g"
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
