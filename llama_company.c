#include "llama_company.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
    LLAMA_COMPANY_ATTN_RMS = 0,
    LLAMA_COMPANY_QUERY = 1,
    LLAMA_COMPANY_KEY = 2,
    LLAMA_COMPANY_VALUE = 3,
    LLAMA_COMPANY_ATTN_OUTPUT = 4,
    LLAMA_COMPANY_FFN_RMS = 5,
    LLAMA_COMPANY_FFN_GATE = 6,
    LLAMA_COMPANY_FFN_UP = 7,
    LLAMA_COMPANY_FFN_DOWN = 8,
};

static bool checked_product(size_t left, size_t right, size_t *product) {
    if (right != 0 && left > SIZE_MAX / right) return false;
    *product = left * right;
    return true;
}

static void *company_calloc(size_t count, size_t width) {
    size_t bytes = 0;
    if (!checked_product(count, width, &bytes)) return NULL;
    if (bytes == 0) bytes = 1;
    return calloc(1, bytes);
}

static bool shape_is_valid(
    AtkeyRuntime *runtime,
    const LlamaCompanyShape *shape,
    int *maximum_context,
    size_t *context_members
) {
    if (runtime == NULL || shape == NULL || shape->row_count <= 0 ||
        shape->tokens == NULL || shape->positions == NULL ||
        shape->parents == NULL) {
        return false;
    }
    int vocab_size = atkey_vocab_size(runtime);
    int sequence_length = atkey_sequence_length(runtime);
    int max_context = 0;
    size_t members = 0;
    for (int row = 0; row < shape->row_count; row++) {
        int token = shape->tokens[row];
        int position = shape->positions[row];
        int parent = shape->parents[row];
        if (token < 0 || token >= vocab_size || position < 0 ||
            position >= sequence_length || parent >= row || parent < -1) {
            return false;
        }
        if (parent == -1) {
            if (position != 0) return false;
        } else if (shape->positions[parent] + 1 != position) {
            return false;
        }
        int context = 1;
        for (int ancestor = parent; ancestor != -1;
             ancestor = shape->parents[ancestor]) {
            if (context == INT_MAX) return false;
            context++;
        }
        if (context > max_context) max_context = context;
        if (members > SIZE_MAX - (size_t)context) return false;
        members += (size_t)context;
    }
    *maximum_context = max_context;
    *context_members = members;
    return true;
}

static bool build_context_index(
    const LlamaCompanyShape *shape,
    size_t member_count,
    size_t **raw_offsets,
    int **raw_members
) {
    size_t *offsets = company_calloc(
        (size_t)shape->row_count + 1,
        sizeof(*offsets)
    );
    int *members = company_calloc(member_count, sizeof(*members));
    if (offsets == NULL || members == NULL) {
        free(offsets);
        free(members);
        return false;
    }
    size_t cursor = 0;
    for (int row = 0; row < shape->row_count; row++) {
        offsets[row] = cursor;
        int count = 0;
        for (int ancestor = row; ancestor != -1;
             ancestor = shape->parents[ancestor]) {
            count++;
        }
        size_t finish = cursor + (size_t)count;
        int ancestor = row;
        for (int index = count - 1; index >= 0; index--) {
            members[cursor + (size_t)index] = ancestor;
            ancestor = shape->parents[ancestor];
        }
        cursor = finish;
    }
    offsets[shape->row_count] = cursor;
    if (cursor != member_count) {
        free(offsets);
        free(members);
        return false;
    }
    *raw_offsets = offsets;
    *raw_members = members;
    return true;
}

static float *new_family(int rows, int width) {
    size_t elements = 0;
    if (rows <= 0 || width <= 0 ||
        !checked_product((size_t)rows, (size_t)width, &elements)) {
        return NULL;
    }
    return company_calloc(elements, sizeof(float));
}

static bool copy_scale(
    LlamaCompanyResult *result,
    int scale,
    const float *hidden
) {
    if (result->scales == NULL) return true;
    if (scale < 0 || scale >= result->scale_count) return false;
    size_t rows_dim = 0;
    if (!checked_product(
            (size_t)result->row_count,
            (size_t)result->dim,
            &rows_dim
        )) {
        return false;
    }
    memcpy(
        result->scales + (size_t)scale * rows_dim,
        hidden,
        rows_dim * sizeof(float)
    );
    return true;
}

void llama_company_result_free(LlamaCompanyResult *result) {
    if (result == NULL) return;
    free(result->logits);
    free(result->scales);
    memset(result, 0, sizeof(*result));
}

bool llama_company_evaluate(
    AtkeyRuntime *runtime,
    const LlamaCompanyShape *shape,
    bool retain_scales,
    LlamaCompanyResult *result
) {
    if (result == NULL) return false;
    memset(result, 0, sizeof(*result));

    int maximum_context = 0;
    size_t context_member_count = 0;
    if (!shape_is_valid(
            runtime,
            shape,
            &maximum_context,
            &context_member_count
        )) {
        return false;
    }

    int rows = shape->row_count;
    int dim = atkey_dim(runtime);
    int hidden_dim = atkey_hidden_dim(runtime);
    int layers = atkey_layer_count(runtime);
    int heads = atkey_head_count(runtime);
    int kv_heads = atkey_kv_head_count(runtime);
    int vocab_size = atkey_vocab_size(runtime);
    if (dim <= 0 || hidden_dim <= 0 || layers <= 0 || heads <= 0 ||
        kv_heads <= 0 || dim % heads != 0 || heads % kv_heads != 0) {
        return false;
    }
    int kv_dim = dim * kv_heads / heads;
    int head_size = dim / heads;

    size_t *context_offsets = NULL;
    int *context_members = NULL;
    if (!build_context_index(
            shape,
            context_member_count,
            &context_offsets,
            &context_members
        )) {
        return false;
    }

    result->row_count = rows;
    result->dim = dim;
    result->vocab_size = vocab_size;
    result->scale_count = layers + 1;

    if (retain_scales) {
        size_t scale_rows = 0;
        size_t scale_elements = 0;
        if (!checked_product((size_t)(layers + 1), (size_t)rows, &scale_rows) ||
            !checked_product(scale_rows, (size_t)dim, &scale_elements)) {
            free(context_offsets);
            free(context_members);
            return false;
        }
        result->scales = company_calloc(scale_elements, sizeof(float));
        if (result->scales == NULL) {
            free(context_offsets);
            free(context_members);
            return false;
        }
    }

    float *hidden = new_family(rows, dim);
    float *normalized = new_family(rows, dim);
    float *query = new_family(rows, dim);
    float *key = new_family(rows, kv_dim);
    float *value = new_family(rows, kv_dim);
    float *attended = new_family(rows, dim);
    float *projected = new_family(rows, dim);
    float *gate = new_family(rows, hidden_dim);
    float *up = new_family(rows, hidden_dim);
    float *down = new_family(rows, dim);
    const float **context_keys = company_calloc(
        (size_t)maximum_context,
        sizeof(*context_keys)
    );
    const float **context_values = company_calloc(
        (size_t)maximum_context,
        sizeof(*context_values)
    );

    if (hidden == NULL || normalized == NULL || query == NULL || key == NULL ||
        value == NULL || attended == NULL || projected == NULL || gate == NULL ||
        up == NULL || down == NULL || context_keys == NULL ||
        context_values == NULL) {
        goto failure;
    }

    atkey_embedding_family_apply(
        runtime,
        atkey_embedding_filler_id(),
        hidden,
        shape->tokens,
        rows,
        atkey_embedding_weight(runtime),
        vocab_size,
        dim
    );
    if (!copy_scale(result, 0, hidden)) goto failure;

    for (int layer = 0; layer < layers; layer++) {
        atkey_rms_family_apply(
            runtime,
            atkey_layer_filler_id(layer, LLAMA_COMPANY_ATTN_RMS),
            normalized,
            hidden,
            rows,
            atkey_attention_rms_weight(runtime, layer),
            dim
        );
        atkey_matmul_family_apply(
            runtime,
            atkey_layer_filler_id(layer, LLAMA_COMPANY_QUERY),
            query,
            normalized,
            rows,
            atkey_query_weight(runtime, layer),
            dim,
            dim
        );
        atkey_matmul_family_apply(
            runtime,
            atkey_layer_filler_id(layer, LLAMA_COMPANY_KEY),
            key,
            normalized,
            rows,
            atkey_key_weight(runtime, layer),
            dim,
            kv_dim
        );
        atkey_matmul_family_apply(
            runtime,
            atkey_layer_filler_id(layer, LLAMA_COMPANY_VALUE),
            value,
            normalized,
            rows,
            atkey_value_weight(runtime, layer),
            dim,
            kv_dim
        );

        for (int row = 0; row < rows; row++) {
            atkey_rope(
                query + (size_t)row * dim,
                key + (size_t)row * kv_dim,
                query + (size_t)row * dim,
                key + (size_t)row * kv_dim,
                shape->positions[row],
                dim,
                kv_dim,
                head_size
            );
        }

        for (int row = 0; row < rows; row++) {
            size_t begin = context_offsets[row];
            size_t finish = context_offsets[row + 1];
            int count = (int)(finish - begin);
            for (int index = 0; index < count; index++) {
                int member = context_members[begin + (size_t)index];
                context_keys[index] = key + (size_t)member * kv_dim;
                context_values[index] = value + (size_t)member * kv_dim;
            }
            atkey_attention(
                attended + (size_t)row * dim,
                query + (size_t)row * dim,
                context_keys,
                context_values,
                count,
                dim,
                heads,
                kv_heads
            );
        }

        atkey_matmul_family_apply(
            runtime,
            atkey_layer_filler_id(layer, LLAMA_COMPANY_ATTN_OUTPUT),
            projected,
            attended,
            rows,
            atkey_attention_output_weight(runtime, layer),
            dim,
            dim
        );
        for (int row = 0; row < rows; row++) {
            atkey_add(
                hidden + (size_t)row * dim,
                hidden + (size_t)row * dim,
                projected + (size_t)row * dim,
                dim
            );
        }

        atkey_rms_family_apply(
            runtime,
            atkey_layer_filler_id(layer, LLAMA_COMPANY_FFN_RMS),
            normalized,
            hidden,
            rows,
            atkey_ffn_rms_weight(runtime, layer),
            dim
        );
        atkey_matmul_family_apply(
            runtime,
            atkey_layer_filler_id(layer, LLAMA_COMPANY_FFN_GATE),
            gate,
            normalized,
            rows,
            atkey_ffn_gate_weight(runtime, layer),
            dim,
            hidden_dim
        );
        atkey_matmul_family_apply(
            runtime,
            atkey_layer_filler_id(layer, LLAMA_COMPANY_FFN_UP),
            up,
            normalized,
            rows,
            atkey_ffn_up_weight(runtime, layer),
            dim,
            hidden_dim
        );
        atkey_swiglu(gate, gate, gate, up, rows * hidden_dim);
        atkey_matmul_family_apply(
            runtime,
            atkey_layer_filler_id(layer, LLAMA_COMPANY_FFN_DOWN),
            down,
            gate,
            rows,
            atkey_ffn_down_weight(runtime, layer),
            hidden_dim,
            dim
        );
        for (int row = 0; row < rows; row++) {
            atkey_add(
                hidden + (size_t)row * dim,
                hidden + (size_t)row * dim,
                down + (size_t)row * dim,
                dim
            );
        }
        if (!copy_scale(result, layer + 1, hidden)) goto failure;
    }

    atkey_rms_family_apply(
        runtime,
        atkey_final_rms_filler_id(runtime),
        normalized,
        hidden,
        rows,
        atkey_final_rms_weight(runtime),
        dim
    );
    result->logits = new_family(rows, vocab_size);
    if (result->logits == NULL) goto failure;
    atkey_matmul_family_apply(
        runtime,
        atkey_output_filler_id(runtime),
        result->logits,
        normalized,
        rows,
        atkey_output_weight(runtime),
        dim,
        vocab_size
    );

    free(context_values);
    free(context_keys);
    free(down);
    free(up);
    free(gate);
    free(projected);
    free(attended);
    free(value);
    free(key);
    free(query);
    free(normalized);
    free(hidden);
    free(context_members);
    free(context_offsets);
    return true;

failure:
    free(context_values);
    free(context_keys);
    free(down);
    free(up);
    free(gate);
    free(projected);
    free(attended);
    free(value);
    free(key);
    free(query);
    free(normalized);
    free(hidden);
    free(context_members);
    free(context_offsets);
    llama_company_result_free(result);
    return false;
}
