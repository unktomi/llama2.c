/*
 * DO NOT USE AS INFERENCE VALIDATION.
 *
 * This test checks numerical parity only after concrete completion families
 * have been supplied.  Its passing result says nothing about recursive
 * selection composition and must not be presented as inferencer progress.
 */

#define ATKEY_REFERENCE_TEST_API

#include "atkey_term_c.h"
#include "escardo_model.h"
#include "llama2_backend.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *message) {
    fprintf(stderr, "escardo model parity: %s\n", message);
    exit(EXIT_FAILURE);
}

static void *allocate(size_t count, size_t width) {
    void *memory = calloc(count, width);
    if (memory == NULL) fail("allocation failed");
    return memory;
}

static void top_tokens(const float *logits, int count, int *tokens, int top_k) {
    for (int rank = 0; rank < top_k; rank++) tokens[rank] = -1;
    for (int token = 0; token < count; token++) {
        int insertion = top_k;
        for (int rank = 0; rank < top_k; rank++) {
            int incumbent = tokens[rank];
            if (incumbent < 0 || logits[token] > logits[incumbent] ||
                (logits[token] == logits[incumbent] && token < incumbent)) {
                insertion = rank;
                break;
            }
        }
        if (insertion == top_k) continue;
        for (int rank = top_k - 1; rank > insertion; rank--) {
            tokens[rank] = tokens[rank - 1];
        }
        tokens[insertion] = token;
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fail("usage: escardo_model_parity checkpoint tokenizer prompt");
    }
    const char *checkpoint = argv[1];
    const char *tokenizer = argv[2];
    const char *prompt_text = argv[3];
    TermBackend *backend = llama2_backend_new(checkpoint, tokenizer);
    if (backend == NULL) fail("could not load family backend");
    EscardoModel *model = escardo_model_new(backend);
    int prompt_count = 0;
    int *prompt = term_backend_encode(backend, prompt_text, &prompt_count);
    if (prompt == NULL || prompt_count <= 0) fail("could not encode prompt");

    enum { BATCH = 3, HORIZON = 2 };
    int vocab = escardo_model_vocab(model);
    int completions[BATCH * HORIZON];
    for (int batch = 0; batch < BATCH; batch++) {
        for (int position = 0; position < HORIZON; position++) {
            completions[batch * HORIZON + position] =
                3 + (batch * HORIZON + position) % (vocab - 3);
        }
    }
    float *actual = allocate(
        (size_t)BATCH * HORIZON * vocab,
        sizeof(float)
    );
    EscardoLogitsTerm *term = escardo_model_compose_logits_family(
        model,
        prompt,
        prompt_count,
        completions,
        BATCH,
        HORIZON
    );
    if (term == NULL) fail("could not compose logits term");
    if (escardo_logits_term_element_count(term) !=
        (size_t)BATCH * HORIZON * vocab) {
        fail("composed logits term has wrong shape");
    }

    size_t compose_crossing_mismatches = 0;
    int filler_count = term_backend_filler_count(backend);
    for (int filler = 0; filler < filler_count; filler++) {
        if (term_backend_filler_crossings(backend, filler) != 0) {
            compose_crossing_mismatches++;
        }
    }
    if (!escardo_logits_term_run(term, actual)) {
        fail("could not run logits term");
    }
    escardo_logits_term_free(term);

    size_t filler_crossing_mismatches = 0;
    for (int filler = 0; filler < filler_count; filler++) {
        if (term_backend_filler_crossings(backend, filler) != 1) {
            filler_crossing_mismatches++;
        }
    }

    size_t bit_mismatches = 0;
    size_t tolerance_mismatches = 0;
    size_t argmax_mismatches = 0;
    size_t top4_mismatches = 0;
    double maximum_absolute_error = 0.0;
    int *sequence = allocate(
        (size_t)prompt_count + HORIZON,
        sizeof(*sequence)
    );
    float *expected = allocate((size_t)vocab, sizeof(float));
    for (int batch = 0; batch < BATCH; batch++) {
        for (int position = 0; position < HORIZON; position++) {
            AtkeyRuntime *reference = atkey_runtime_new(checkpoint, tokenizer);
            if (reference == NULL) fail("could not load reference runtime");
            memcpy(sequence, prompt, (size_t)prompt_count * sizeof(*sequence));
            memcpy(
                sequence + prompt_count,
                completions + batch * HORIZON,
                (size_t)position * sizeof(*sequence)
            );
            atkey_reference_sequence_logits(
                reference,
                sequence,
                prompt_count + position,
                expected
            );
            size_t row = (size_t)batch * HORIZON + position;
            int observed_argmax = 0;
            int expected_argmax = 0;
            for (int token = 0; token < vocab; token++) {
                float observed = actual[row * vocab + token];
                float wanted = expected[token];
                if (memcmp(&observed, &wanted, sizeof(float)) != 0) {
                    bit_mismatches++;
                }
                double error = fabs((double)observed - wanted);
                if (error > 2e-5) tolerance_mismatches++;
                if (error > maximum_absolute_error) {
                    maximum_absolute_error = error;
                }
                if (observed > actual[row * vocab + observed_argmax]) {
                    observed_argmax = token;
                }
                if (wanted > expected[expected_argmax]) expected_argmax = token;
            }
            if (observed_argmax != expected_argmax) argmax_mismatches++;
            int observed_top4[4];
            int expected_top4[4];
            top_tokens(actual + row * vocab, vocab, observed_top4, 4);
            top_tokens(expected, vocab, expected_top4, 4);
            if (memcmp(observed_top4, expected_top4, sizeof(observed_top4)) != 0) {
                top4_mismatches++;
            }
            atkey_runtime_free(reference);
        }
    }
    printf(
        "backend=%s batch=%d horizon=%d logits=%d bit_mismatches=%zu "
        "tolerance_mismatches=%zu argmax_mismatches=%zu "
        "top4_mismatches=%zu compose_crossing_mismatches=%zu "
        "filler_crossing_mismatches=%zu "
        "max_abs_error=%.17g\n",
        term_backend_name(backend),
        BATCH,
        HORIZON,
        BATCH * HORIZON * vocab,
        bit_mismatches,
        tolerance_mismatches,
        argmax_mismatches,
        top4_mismatches,
        compose_crossing_mismatches,
        filler_crossing_mismatches,
        maximum_absolute_error
    );

    free(expected);
    free(sequence);
    free(actual);
    term_backend_free_tokens(backend, prompt);
    escardo_model_free(model);
    term_backend_free(backend);
    return tolerance_mismatches == 0 && argmax_mismatches == 0 &&
        top4_mismatches == 0 && compose_crossing_mismatches == 0 &&
        filler_crossing_mismatches == 0 ?
        EXIT_SUCCESS : EXIT_FAILURE;
}
