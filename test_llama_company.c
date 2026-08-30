#define ATKEY_REFERENCE_TEST_API

#include "atkey_term_c.h"
#include "llama_company.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void require(int condition, const char *message) {
    if (condition) return;
    fprintf(stderr, "company system test: %s\n", message);
    exit(EXIT_FAILURE);
}

static int argmax(const float *values, int count) {
    int best = 0;
    for (int index = 1; index < count; index++) {
        if (values[index] > values[best]) best = index;
    }
    return best;
}

static int collect_history(
    const LlamaCompanyShape *shape,
    int row,
    int *tokens
) {
    int count = 0;
    for (int cursor = row; cursor != -1; cursor = shape->parents[cursor]) {
        tokens[count++] = shape->tokens[cursor];
    }
    for (int left = 0, right = count - 1; left < right; left++, right--) {
        int temporary = tokens[left];
        tokens[left] = tokens[right];
        tokens[right] = temporary;
    }
    return count;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s CHECKPOINT TOKENIZER\n", argv[0]);
        return EXIT_FAILURE;
    }
    AtkeyRuntime *runtime = atkey_runtime_new(argv[1], argv[2]);
    require(runtime != NULL, "could not load runtime");

    int prompt_count = 0;
    int *prompt = atkey_encode(runtime, "Lily was", &prompt_count);
    require(prompt != NULL && prompt_count > 0, "could not encode prompt");
    int extra = 6;
    int rows = prompt_count + extra;
    int *tokens = calloc((size_t)rows, sizeof(*tokens));
    int *positions = calloc((size_t)rows, sizeof(*positions));
    int *parents = calloc((size_t)rows, sizeof(*parents));
    require(tokens != NULL && positions != NULL && parents != NULL,
            "allocation failed");

    for (int row = 0; row < prompt_count; row++) {
        tokens[row] = prompt[row];
        positions[row] = row;
        parents[row] = row == 0 ? -1 : row - 1;
    }
    int root = prompt_count - 1;
    int first_a = prompt_count;
    int first_b = prompt_count + 1;
    tokens[first_a] = 261;
    tokens[first_b] = 376;
    positions[first_a] = prompt_count;
    positions[first_b] = prompt_count;
    parents[first_a] = root;
    parents[first_b] = root;
    const int second_tokens[4] = {376, 261, 261, 376};
    const int second_parents[4] = {first_a, first_a, first_b, first_b};
    for (int index = 0; index < 4; index++) {
        int row = prompt_count + 2 + index;
        tokens[row] = second_tokens[index];
        positions[row] = prompt_count + 1;
        parents[row] = second_parents[index];
    }

    LlamaCompanyShape shape = {
        .row_count = rows,
        .tokens = tokens,
        .positions = positions,
        .parents = parents,
    };
    LlamaCompanyResult result;
    require(
        llama_company_evaluate(runtime, &shape, true, &result),
        "company evaluation failed"
    );

    int filler_count = atkey_filler_count(runtime);
    require(filler_count == 48, "unexpected Stories260K filler count");
    for (int filler = 0; filler < filler_count; filler++) {
        require(
            atkey_filler_calls(runtime, filler) == 1,
            "a learned filler did not receive its whole company once"
        );
    }

    int vocab = atkey_vocab_size(runtime);
    int sequence_length = atkey_sequence_length(runtime);
    int *history = calloc((size_t)sequence_length, sizeof(*history));
    float *reference = calloc((size_t)vocab, sizeof(*reference));
    require(history != NULL && reference != NULL, "reference allocation failed");
    double maximum_error = 0.0;
    int bit_mismatches = 0;
    int argmax_mismatches = 0;
    for (int row = 0; row < rows; row++) {
        int count = collect_history(&shape, row, history);
        atkey_reference_sequence_logits(runtime, history, count, reference);
        const float *actual = result.logits + (size_t)row * vocab;
        if (argmax(actual, vocab) != argmax(reference, vocab)) {
            argmax_mismatches++;
        }
        for (int token = 0; token < vocab; token++) {
            if (actual[token] != reference[token]) bit_mismatches++;
            double error = fabs((double)actual[token] - reference[token]);
            if (error > maximum_error) maximum_error = error;
        }
    }
    require(maximum_error <= 5e-5, "company logits exceed numerical tolerance");
    require(argmax_mismatches == 0, "company changed a row argmax");
    require(result.scales != NULL, "layer companies were not retained");
    require(
        result.scale_count == atkey_layer_count(runtime) + 1,
        "wrong retained scale count"
    );

    printf(
        "Stories260K causal company: rows=%d fillers=%d max_error=%.9g "
        "bit_mismatches=%d argmax_mismatches=%d scales=%d\n",
        rows,
        filler_count,
        maximum_error,
        bit_mismatches,
        argmax_mismatches,
        result.scale_count
    );

    free(reference);
    free(history);
    llama_company_result_free(&result);
    free(parents);
    free(positions);
    free(tokens);
    atkey_free_tokens(prompt);
    atkey_runtime_free(runtime);
    return EXIT_SUCCESS;
}
