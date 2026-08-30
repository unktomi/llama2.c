/*
 * DO NOT USE AS THE INFERENCE TERM.
 *
 * This fixed-token IR requires a concrete completion family before model
 * composition.  That places selection outside the term and recreates the
 * pre-materialization obstruction already rejected by the exhaustive-prefix
 * experiments.  It is retained only to make that failed direction auditable.
 */

#include "term_program.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    TERM_EXPR_TOKENS,
    TERM_EXPR_EMBEDDING,
    TERM_EXPR_HIDDEN,
    TERM_EXPR_ADD,
    TERM_EXPR_SWIGLU,
    TERM_EXPR_ROPE_QUERY,
    TERM_EXPR_ROPE_KEY,
    TERM_EXPR_CAUSAL_ATTENTION,
    TERM_EXPR_GATHER_POSITIONS,
} TermExprKind;

typedef struct {
    TermExprKind kind;
    int batch;
    int positions;
    int width;
    TermExpr first;
    TermExpr second;
    TermExpr third;
    TermFiller filler;
    int parameter0;
    int parameter1;
    int *tokens;
} TermExpression;

struct TermProgram {
    TermBackend *backend;
    TermExpression *expressions;
    int expression_count;
    int expression_capacity;
    int *filler_occurrences;
    int filler_count;
    bool has_run;
};

typedef struct {
    void *data;
    int remaining_uses;
    bool retained;
} Evaluation;

static const TermExpr TERM_EXPR_NONE = { UINT32_MAX };

static void fail(const char *message) {
    fprintf(stderr, "term program: %s\n", message);
    abort();
}

static void *allocate(size_t count, size_t width) {
    if (width != 0 && count > SIZE_MAX / width) fail("allocation overflow");
    void *memory = calloc(count, width);
    if (memory == NULL) fail("allocation failed");
    return memory;
}

static size_t checked_elements(int batch, int positions, int width) {
    if (batch <= 0 || positions <= 0 || width <= 0) fail("invalid shape");
    size_t count = (size_t)batch;
    if (count > SIZE_MAX / (size_t)positions) fail("shape overflow");
    count *= (size_t)positions;
    if (count > SIZE_MAX / (size_t)width) fail("shape overflow");
    return count * (size_t)width;
}

static TermExpression *expression(
    const TermProgram *program,
    TermExpr value
) {
    if (program == NULL || value.id >= (uint32_t)program->expression_count) {
        fail("invalid expression");
    }
    return &program->expressions[value.id];
}

static void reserve_expression(TermProgram *program) {
    if (program->expression_count < program->expression_capacity) return;
    int capacity = program->expression_capacity == 0 ?
        64 : program->expression_capacity * 2;
    if (capacity < program->expression_capacity) fail("too many expressions");
    TermExpression *resized = realloc(
        program->expressions,
        (size_t)capacity * sizeof(*resized)
    );
    if (resized == NULL) fail("expression allocation failed");
    program->expressions = resized;
    program->expression_capacity = capacity;
}

static TermExpr append_expression(
    TermProgram *program,
    TermExpression value
) {
    if (program == NULL || program->has_run) {
        fail("cannot extend a missing or already interpreted term");
    }
    reserve_expression(program);
    TermExpr result = { (uint32_t)program->expression_count };
    program->expressions[program->expression_count++] = value;
    return result;
}

static void require_same_shape(
    const TermExpression *left,
    const TermExpression *right
) {
    if (left->batch != right->batch ||
        left->positions != right->positions ||
        left->width != right->width) {
        fail("shape mismatch");
    }
}

static void note_filler(TermProgram *program, const TermFiller *filler) {
    if (filler == NULL || filler->id < 0 || filler->id >= program->filler_count) {
        fail("invalid learned filler");
    }
    program->filler_occurrences[filler->id]++;
}

TermProgram *term_program_new(TermBackend *backend) {
    if (backend == NULL) return NULL;
    int filler_count = term_backend_filler_count(backend);
    if (filler_count <= 0) return NULL;
    TermProgram *program = allocate(1, sizeof(*program));
    program->backend = backend;
    program->filler_count = filler_count;
    program->filler_occurrences = allocate(
        (size_t)filler_count,
        sizeof(*program->filler_occurrences)
    );
    return program;
}

void term_program_free(TermProgram *program) {
    if (program == NULL) return;
    for (int index = 0; index < program->expression_count; index++) {
        free(program->expressions[index].tokens);
    }
    free(program->filler_occurrences);
    free(program->expressions);
    free(program);
}

TermExpr term_program_tokens(
    TermProgram *program,
    const int *tokens,
    int batch,
    int positions
) {
    if (tokens == NULL) fail("missing token family");
    size_t count = checked_elements(batch, positions, 1);
    int *copy = allocate(count, sizeof(*copy));
    memcpy(copy, tokens, count * sizeof(*copy));
    return append_expression(program, (TermExpression){
        .kind = TERM_EXPR_TOKENS,
        .batch = batch,
        .positions = positions,
        .width = 1,
        .first = TERM_EXPR_NONE,
        .second = TERM_EXPR_NONE,
        .third = TERM_EXPR_NONE,
        .tokens = copy,
    });
}

TermExpr term_program_embedding(
    TermProgram *program,
    const TermFiller *filler,
    TermExpr tokens
) {
    TermExpression *input = expression(program, tokens);
    if (input->kind != TERM_EXPR_TOKENS ||
        filler == NULL || filler->kind != TERM_FILLER_EMBEDDING) {
        fail("invalid embedding term");
    }
    note_filler(program, filler);
    return append_expression(program, (TermExpression){
        .kind = TERM_EXPR_EMBEDDING,
        .batch = input->batch,
        .positions = input->positions,
        .width = filler->output_width,
        .first = tokens,
        .second = TERM_EXPR_NONE,
        .third = TERM_EXPR_NONE,
        .filler = *filler,
    });
}

TermExpr term_program_hidden(
    TermProgram *program,
    const TermFiller *filler,
    TermExpr input_value
) {
    TermExpression *input = expression(program, input_value);
    if (input->kind == TERM_EXPR_TOKENS || filler == NULL ||
        filler->kind == TERM_FILLER_EMBEDDING ||
        input->width != filler->input_width) {
        fail("invalid hidden filler term");
    }
    note_filler(program, filler);
    return append_expression(program, (TermExpression){
        .kind = TERM_EXPR_HIDDEN,
        .batch = input->batch,
        .positions = input->positions,
        .width = filler->output_width,
        .first = input_value,
        .second = TERM_EXPR_NONE,
        .third = TERM_EXPR_NONE,
        .filler = *filler,
    });
}

TermExpr term_program_add(
    TermProgram *program,
    TermExpr left_value,
    TermExpr right_value
) {
    TermExpression *left = expression(program, left_value);
    TermExpression *right = expression(program, right_value);
    require_same_shape(left, right);
    return append_expression(program, (TermExpression){
        .kind = TERM_EXPR_ADD,
        .batch = left->batch,
        .positions = left->positions,
        .width = left->width,
        .first = left_value,
        .second = right_value,
        .third = TERM_EXPR_NONE,
    });
}

TermExpr term_program_swiglu(
    TermProgram *program,
    TermExpr gate_value,
    TermExpr up_value
) {
    TermExpression *gate = expression(program, gate_value);
    TermExpression *up = expression(program, up_value);
    require_same_shape(gate, up);
    return append_expression(program, (TermExpression){
        .kind = TERM_EXPR_SWIGLU,
        .batch = gate->batch,
        .positions = gate->positions,
        .width = gate->width,
        .first = gate_value,
        .second = up_value,
        .third = TERM_EXPR_NONE,
    });
}

static TermExpr rope_term(
    TermProgram *program,
    TermExpr input_value,
    int head_size,
    TermExprKind kind
) {
    TermExpression *input = expression(program, input_value);
    if (input->kind == TERM_EXPR_TOKENS || head_size <= 0 ||
        input->width % 2 != 0) {
        fail("invalid RoPE term");
    }
    return append_expression(program, (TermExpression){
        .kind = kind,
        .batch = input->batch,
        .positions = input->positions,
        .width = input->width,
        .first = input_value,
        .second = TERM_EXPR_NONE,
        .third = TERM_EXPR_NONE,
        .parameter0 = head_size,
    });
}

TermExpr term_program_rope_query(
    TermProgram *program,
    TermExpr query,
    int head_size
) {
    return rope_term(program, query, head_size, TERM_EXPR_ROPE_QUERY);
}

TermExpr term_program_rope_key(
    TermProgram *program,
    TermExpr key,
    int head_size
) {
    return rope_term(program, key, head_size, TERM_EXPR_ROPE_KEY);
}

TermExpr term_program_causal_attention(
    TermProgram *program,
    TermExpr query_value,
    TermExpr key_value,
    TermExpr value_value,
    int heads,
    int kv_heads
) {
    TermExpression *query = expression(program, query_value);
    TermExpression *key = expression(program, key_value);
    TermExpression *value = expression(program, value_value);
    if (query->batch != key->batch || query->batch != value->batch ||
        query->positions != key->positions ||
        query->positions != value->positions ||
        key->width != value->width || heads <= 0 || kv_heads <= 0 ||
        query->width % heads != 0 ||
        key->width != query->width * kv_heads / heads) {
        fail("invalid causal attention term");
    }
    return append_expression(program, (TermExpression){
        .kind = TERM_EXPR_CAUSAL_ATTENTION,
        .batch = query->batch,
        .positions = query->positions,
        .width = query->width,
        .first = query_value,
        .second = key_value,
        .third = value_value,
        .parameter0 = heads,
        .parameter1 = kv_heads,
    });
}

TermExpr term_program_gather_positions(
    TermProgram *program,
    TermExpr input_value,
    int first,
    int count
) {
    TermExpression *input = expression(program, input_value);
    if (input->kind == TERM_EXPR_TOKENS || first < 0 || count <= 0 ||
        first > input->positions - count) {
        fail("invalid position gather");
    }
    return append_expression(program, (TermExpression){
        .kind = TERM_EXPR_GATHER_POSITIONS,
        .batch = input->batch,
        .positions = count,
        .width = input->width,
        .first = input_value,
        .second = TERM_EXPR_NONE,
        .third = TERM_EXPR_NONE,
        .parameter0 = first,
    });
}

static void increment_use(Evaluation *values, TermExpr input) {
    if (input.id != UINT32_MAX) values[input.id].remaining_uses++;
}

static void release_input(Evaluation *values, TermExpr input) {
    if (input.id == UINT32_MAX) return;
    Evaluation *value = &values[input.id];
    if (value->remaining_uses <= 0) fail("expression use underflow");
    value->remaining_uses--;
    if (value->remaining_uses == 0 && !value->retained) {
        free(value->data);
        value->data = NULL;
    }
}

static void evaluate_rope(
    TermProgram *program,
    const TermExpression *term,
    float *output,
    const float *input,
    bool query
) {
    int dim = query ? term->width : term->width;
    int kv_dim = query ? term->width : term->width;
    float *other_input = allocate((size_t)(query ? term->width : dim), sizeof(float));
    float *other_output = allocate((size_t)(query ? kv_dim : dim), sizeof(float));
    for (int batch = 0; batch < term->batch; batch++) {
        for (int position = 0; position < term->positions; position++) {
            int occurrence = batch * term->positions + position;
            const float *source = input + (size_t)occurrence * term->width;
            float *destination = output + (size_t)occurrence * term->width;
            if (query) {
                memset(other_input, 0, (size_t)kv_dim * sizeof(float));
                term_backend_rope(
                    program->backend,
                    destination,
                    other_output,
                    source,
                    other_input,
                    position,
                    dim,
                    kv_dim,
                    term->parameter0
                );
            } else {
                memset(other_input, 0, (size_t)dim * sizeof(float));
                term_backend_rope(
                    program->backend,
                    other_output,
                    destination,
                    other_input,
                    source,
                    position,
                    dim,
                    kv_dim,
                    term->parameter0
                );
            }
        }
    }
    free(other_output);
    free(other_input);
}

static void evaluate_attention(
    TermProgram *program,
    const TermExpression *term,
    float *output,
    const float *queries,
    const float *keys,
    const float *values
) {
    const float **key_pointers = allocate(
        (size_t)term->positions,
        sizeof(*key_pointers)
    );
    const float **value_pointers = allocate(
        (size_t)term->positions,
        sizeof(*value_pointers)
    );
    int kv_width = term->width * term->parameter1 / term->parameter0;
    for (int batch = 0; batch < term->batch; batch++) {
        for (int position = 0; position < term->positions; position++) {
            for (int source = 0; source <= position; source++) {
                int source_occurrence = batch * term->positions + source;
                key_pointers[source] =
                    keys + (size_t)source_occurrence * kv_width;
                value_pointers[source] =
                    values + (size_t)source_occurrence * kv_width;
            }
            int occurrence = batch * term->positions + position;
            term_backend_attention(
                program->backend,
                output + (size_t)occurrence * term->width,
                queries + (size_t)occurrence * term->width,
                key_pointers,
                value_pointers,
                position + 1,
                term->width,
                term->parameter0,
                term->parameter1
            );
        }
    }
    free(value_pointers);
    free(key_pointers);
}

static void evaluate_expression(
    TermProgram *program,
    int index,
    Evaluation *values
) {
    TermExpression *term = &program->expressions[index];
    size_t elements = checked_elements(term->batch, term->positions, term->width);
    if (term->kind == TERM_EXPR_TOKENS) {
        values[index].data = allocate(elements, sizeof(int));
        memcpy(values[index].data, term->tokens, elements * sizeof(int));
        return;
    }
    float *output = allocate(elements, sizeof(*output));
    values[index].data = output;
    const void *first = values[term->first.id].data;
    if (first == NULL) fail("missing first operand");
    int occurrence_count = term->batch * term->positions;
    switch (term->kind) {
        case TERM_EXPR_TOKENS:
            fail("unreachable token evaluation");
            break;
        case TERM_EXPR_EMBEDDING:
            term_backend_embedding_family(
                program->backend,
                &term->filler,
                output,
                first,
                occurrence_count
            );
            break;
        case TERM_EXPR_HIDDEN:
            term_backend_hidden_family(
                program->backend,
                &term->filler,
                output,
                first,
                occurrence_count
            );
            break;
        case TERM_EXPR_ADD: {
            const float *right = values[term->second.id].data;
            if (right == NULL) fail("missing add operand");
            for (int occurrence = 0; occurrence < occurrence_count; occurrence++) {
                term_backend_add(
                    program->backend,
                    output + (size_t)occurrence * term->width,
                    (const float *)first + (size_t)occurrence * term->width,
                    right + (size_t)occurrence * term->width,
                    term->width
                );
            }
            break;
        }
        case TERM_EXPR_SWIGLU: {
            const float *up = values[term->second.id].data;
            if (up == NULL) fail("missing SwiGLU operand");
            for (int occurrence = 0; occurrence < occurrence_count; occurrence++) {
                term_backend_swiglu(
                    program->backend,
                    output + (size_t)occurrence * term->width,
                    (const float *)first + (size_t)occurrence * term->width,
                    up + (size_t)occurrence * term->width,
                    term->width
                );
            }
            break;
        }
        case TERM_EXPR_ROPE_QUERY:
            evaluate_rope(program, term, output, first, true);
            break;
        case TERM_EXPR_ROPE_KEY:
            evaluate_rope(program, term, output, first, false);
            break;
        case TERM_EXPR_CAUSAL_ATTENTION: {
            const float *keys = values[term->second.id].data;
            const float *attention_values = values[term->third.id].data;
            if (keys == NULL || attention_values == NULL) {
                fail("missing attention operand");
            }
            evaluate_attention(
                program,
                term,
                output,
                first,
                keys,
                attention_values
            );
            break;
        }
        case TERM_EXPR_GATHER_POSITIONS: {
            const float *source = first;
            TermExpression *input = expression(program, term->first);
            for (int batch = 0; batch < term->batch; batch++) {
                for (int position = 0; position < term->positions; position++) {
                    size_t source_index =
                        ((size_t)batch * input->positions +
                         term->parameter0 + position) * term->width;
                    size_t output_index =
                        ((size_t)batch * term->positions + position) *
                        term->width;
                    memcpy(
                        output + output_index,
                        source + source_index,
                        (size_t)term->width * sizeof(*output)
                    );
                }
            }
            break;
        }
    }
}

bool term_program_run(
    TermProgram *program,
    const TermProgramOutput *outputs,
    int output_count
) {
    if (program == NULL || outputs == NULL || output_count <= 0 ||
        program->has_run) {
        return false;
    }
    program->has_run = true;
    Evaluation *values = allocate(
        (size_t)program->expression_count,
        sizeof(*values)
    );
    for (int index = 0; index < program->expression_count; index++) {
        TermExpression *term = &program->expressions[index];
        increment_use(values, term->first);
        increment_use(values, term->second);
        increment_use(values, term->third);
    }
    for (int output = 0; output < output_count; output++) {
        TermExpression *term = expression(program, outputs[output].expression);
        if (outputs[output].destination == NULL ||
            outputs[output].element_count !=
                checked_elements(term->batch, term->positions, term->width)) {
            free(values);
            return false;
        }
        values[outputs[output].expression.id].retained = true;
    }

    for (int index = 0; index < program->expression_count; index++) {
        evaluate_expression(program, index, values);
        TermExpression *term = &program->expressions[index];
        release_input(values, term->first);
        release_input(values, term->second);
        release_input(values, term->third);
    }
    for (int output = 0; output < output_count; output++) {
        TermExpr value = outputs[output].expression;
        if (values[value.id].data == NULL) fail("missing requested observation");
        memcpy(
            outputs[output].destination,
            values[value.id].data,
            outputs[output].element_count * sizeof(float)
        );
    }
    for (int index = 0; index < program->expression_count; index++) {
        free(values[index].data);
    }
    free(values);
    return true;
}

int term_program_expression_count(const TermProgram *program) {
    return program == NULL ? 0 : program->expression_count;
}

int term_program_filler_occurrences(const TermProgram *program, int filler_id) {
    if (program == NULL || filler_id < 0 || filler_id >= program->filler_count) {
        return 0;
    }
    return program->filler_occurrences[filler_id];
}

bool term_program_each_filler_at_most_once(const TermProgram *program) {
    if (program == NULL) return false;
    for (int filler = 0; filler < program->filler_count; filler++) {
        if (program->filler_occurrences[filler] > 1) return false;
    }
    return true;
}

int term_expr_batch(const TermProgram *program, TermExpr value) {
    return expression(program, value)->batch;
}

int term_expr_positions(const TermProgram *program, TermExpr value) {
    return expression(program, value)->positions;
}

int term_expr_width(const TermProgram *program, TermExpr value) {
    return expression(program, value)->width;
}

size_t term_expr_element_count(const TermProgram *program, TermExpr value) {
    TermExpression *term = expression(program, value);
    return checked_elements(term->batch, term->positions, term->width);
}
