/*
 * DO NOT USE AS ESCARDO INFERENCE.
 *
 * Despite its former name, this code receives complete token candidates from
 * its caller and only then composes the numerical model.  Selection is absent
 * from the term, so this is a fixed-family evaluator, not Escardo inference.
 */

#include "escardo_model.h"

#include "term_program.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    TermFiller attention_norm;
    TermFiller query;
    TermFiller key;
    TermFiller value;
    TermFiller attention_output;
    TermFiller ffn_norm;
    TermFiller ffn_gate;
    TermFiller ffn_up;
    TermFiller ffn_down;
} LayerTerm;

struct EscardoModel {
    TermBackend *backend;
    int dim;
    int hidden_dim;
    int layers;
    int heads;
    int kv_heads;
    int head_size;
    int vocab;
    int sequence_length;
    TermFiller embedding;
    TermFiller final_norm;
    TermFiller output;
    LayerTerm *layer_terms;
};

struct EscardoLogitsTerm {
    TermProgram *program;
    TermExpr output;
    size_t element_count;
};

static void die(const char *message) {
    fprintf(stderr, "escardo model: %s\n", message);
    exit(EXIT_FAILURE);
}

static void *allocate(size_t count, size_t width) {
    if (width != 0 && count > SIZE_MAX / width) die("allocation overflow");
    void *memory = calloc(count, width);
    if (memory == NULL) die("allocation failed");
    return memory;
}

static TermFiller require_filler(
    TermBackend *backend,
    TermFillerSlot slot,
    int layer
) {
    TermFiller result;
    if (!term_backend_filler(backend, slot, layer, &result)) {
        die("backend is missing a required Llama filler");
    }
    return result;
}

EscardoModel *escardo_model_new(TermBackend *backend) {
    if (backend == NULL) return NULL;
    const TermBackendConfig *config = term_backend_config(backend);
    if (config->heads <= 0 || config->dim % config->heads != 0) return NULL;
    EscardoModel *model = allocate(1, sizeof(*model));
    model->backend = backend;
    model->dim = config->dim;
    model->hidden_dim = config->hidden_dim;
    model->layers = config->layers;
    model->heads = config->heads;
    model->kv_heads = config->kv_heads;
    model->head_size = model->dim / model->heads;
    model->vocab = config->vocab;
    model->sequence_length = config->sequence_length;
    model->embedding = require_filler(
        backend,
        TERM_SLOT_TOKEN_EMBEDDING,
        -1
    );
    model->final_norm = require_filler(backend, TERM_SLOT_FINAL_RMS, -1);
    model->output = require_filler(backend, TERM_SLOT_OUTPUT, -1);
    model->layer_terms = allocate(
        (size_t)model->layers,
        sizeof(*model->layer_terms)
    );
    for (int layer = 0; layer < model->layers; layer++) {
        LayerTerm *term = &model->layer_terms[layer];
        term->attention_norm = require_filler(
            backend,
            TERM_SLOT_ATTENTION_RMS,
            layer
        );
        term->query = require_filler(backend, TERM_SLOT_QUERY, layer);
        term->key = require_filler(backend, TERM_SLOT_KEY, layer);
        term->value = require_filler(backend, TERM_SLOT_VALUE, layer);
        term->attention_output = require_filler(
            backend,
            TERM_SLOT_ATTENTION_OUTPUT,
            layer
        );
        term->ffn_norm = require_filler(
            backend,
            TERM_SLOT_FFN_RMS,
            layer
        );
        term->ffn_gate = require_filler(
            backend,
            TERM_SLOT_FFN_GATE,
            layer
        );
        term->ffn_up = require_filler(backend, TERM_SLOT_FFN_UP, layer);
        term->ffn_down = require_filler(
            backend,
            TERM_SLOT_FFN_DOWN,
            layer
        );
    }
    return model;
}

void escardo_model_free(EscardoModel *model) {
    if (model == NULL) return;
    free(model->layer_terms);
    free(model);
}

int escardo_model_dim(const EscardoModel *model) {
    return model->dim;
}

int escardo_model_layers(const EscardoModel *model) {
    return model->layers;
}

int escardo_model_vocab(const EscardoModel *model) {
    return model->vocab;
}

static int *whole_context_tokens(
    EscardoModel *model,
    const int *prompt,
    int prompt_count,
    const int *completions,
    int batch_count,
    int horizon,
    int *positions_result
) {
    if (model == NULL || prompt == NULL || prompt_count <= 0 ||
        completions == NULL || batch_count <= 0 || horizon <= 0) {
        die("invalid whole-context family");
    }
    int positions = prompt_count + horizon;
    if (positions > model->sequence_length) die("whole context exceeds model");
    if (batch_count > INT32_MAX / positions) die("whole-context family too large");
    int *tokens = allocate((size_t)batch_count * positions, sizeof(*tokens));
    for (int batch = 0; batch < batch_count; batch++) {
        int *sequence = tokens + (size_t)batch * positions;
        memcpy(sequence, prompt, (size_t)prompt_count * sizeof(*sequence));
        memcpy(
            sequence + prompt_count,
            completions + (size_t)batch * horizon,
            (size_t)horizon * sizeof(*sequence)
        );
    }
    *positions_result = positions;
    return tokens;
}

/*
 * Compose the Llama architecture without interpreting any numerical leaf.
 * Every learned filler is appended once. Residuals and SwiGLU share TermExpr
 * values; they do not append another occurrence of the producing filler.
 */
static TermExpr build_hidden_term(
    EscardoModel *model,
    TermProgram *program,
    const int *tokens,
    int batch_count,
    int positions
) {
    TermExpr token_family = term_program_tokens(
        program,
        tokens,
        batch_count,
        positions
    );
    TermExpr hidden = term_program_embedding(
        program,
        &model->embedding,
        token_family
    );
    for (int layer = 0; layer < model->layers; layer++) {
        LayerTerm *term = &model->layer_terms[layer];
        TermExpr normalized = term_program_hidden(
            program,
            &term->attention_norm,
            hidden
        );
        TermExpr query = term_program_hidden(program, &term->query, normalized);
        TermExpr key = term_program_hidden(program, &term->key, normalized);
        TermExpr value = term_program_hidden(program, &term->value, normalized);
        query = term_program_rope_query(program, query, model->head_size);
        key = term_program_rope_key(program, key, model->head_size);
        TermExpr attention = term_program_causal_attention(
            program,
            query,
            key,
            value,
            model->heads,
            model->kv_heads
        );
        TermExpr attention_output = term_program_hidden(
            program,
            &term->attention_output,
            attention
        );
        hidden = term_program_add(program, hidden, attention_output);

        normalized = term_program_hidden(program, &term->ffn_norm, hidden);
        TermExpr gate = term_program_hidden(
            program,
            &term->ffn_gate,
            normalized
        );
        TermExpr up = term_program_hidden(program, &term->ffn_up, normalized);
        TermExpr gated = term_program_swiglu(program, gate, up);
        TermExpr down = term_program_hidden(program, &term->ffn_down, gated);
        hidden = term_program_add(program, hidden, down);
    }
    if (!term_program_each_filler_at_most_once(program)) {
        die("a learned filler occurs more than once in the architecture term");
    }
    return hidden;
}

void escardo_model_apply_whole_context(
    EscardoModel *model,
    const int *prompt,
    int prompt_count,
    const int *completions,
    int batch_count,
    int horizon,
    float *final_hidden,
    float *predictor_hidden
) {
    if (final_hidden == NULL || predictor_hidden == NULL) {
        die("missing whole-context observations");
    }
    int positions = 0;
    int *tokens = whole_context_tokens(
        model,
        prompt,
        prompt_count,
        completions,
        batch_count,
        horizon,
        &positions
    );
    TermProgram *program = term_program_new(model->backend);
    if (program == NULL) die("could not allocate numerical term");
    TermExpr hidden = build_hidden_term(
        model,
        program,
        tokens,
        batch_count,
        positions
    );
    TermExpr final = term_program_gather_positions(
        program,
        hidden,
        positions - 1,
        1
    );
    TermExpr predictors = term_program_gather_positions(
        program,
        hidden,
        prompt_count - 1,
        horizon
    );
    TermProgramOutput outputs[] = {
        {
            .expression = final,
            .destination = final_hidden,
            .element_count = (size_t)batch_count * model->dim,
        },
        {
            .expression = predictors,
            .destination = predictor_hidden,
            .element_count = (size_t)batch_count * horizon * model->dim,
        },
    };
    if (!term_program_run(program, outputs, 2)) {
        die("could not interpret whole-context term");
    }
    term_program_free(program);
    free(tokens);
}

void escardo_model_apply_logits_family(
    EscardoModel *model,
    const int *prompt,
    int prompt_count,
    const int *completions,
    int batch_count,
    int horizon,
    float *logits
) {
    if (logits == NULL) die("missing logits observation");
    EscardoLogitsTerm *term = escardo_model_compose_logits_family(
        model,
        prompt,
        prompt_count,
        completions,
        batch_count,
        horizon
    );
    if (term == NULL || !escardo_logits_term_run(term, logits)) {
        escardo_logits_term_free(term);
        die("could not interpret logits term");
    }
    escardo_logits_term_free(term);
}

EscardoLogitsTerm *escardo_model_compose_logits_family(
    EscardoModel *model,
    const int *prompt,
    int prompt_count,
    const int *completions,
    int batch_count,
    int horizon
) {
    if (model == NULL || batch_count <= 0 || horizon <= 0) return NULL;
    int positions = 0;
    int *tokens = whole_context_tokens(
        model,
        prompt,
        prompt_count,
        completions,
        batch_count,
        horizon,
        &positions
    );
    TermProgram *program = term_program_new(model->backend);
    if (program == NULL) die("could not allocate numerical term");
    TermExpr hidden = build_hidden_term(
        model,
        program,
        tokens,
        batch_count,
        positions
    );
    TermExpr predictors = term_program_gather_positions(
        program,
        hidden,
        prompt_count - 1,
        horizon
    );
    TermExpr normalized = term_program_hidden(
        program,
        &model->final_norm,
        predictors
    );
    TermExpr output = term_program_hidden(program, &model->output, normalized);
    if (!term_program_each_filler_at_most_once(program)) {
        die("a learned filler occurs more than once in the logits term");
    }
    EscardoLogitsTerm *term = allocate(1, sizeof(*term));
    term->program = program;
    term->output = output;
    term->element_count = term_expr_element_count(program, output);
    free(tokens);
    return term;
}

size_t escardo_logits_term_element_count(const EscardoLogitsTerm *term) {
    return term == NULL ? 0 : term->element_count;
}

bool escardo_logits_term_run(EscardoLogitsTerm *term, float *logits) {
    if (term == NULL || logits == NULL) return false;
    TermProgramOutput observation = {
        .expression = term->output,
        .destination = logits,
        .element_count = term->element_count,
    };
    return term_program_run(term->program, &observation, 1);
}

void escardo_logits_term_free(EscardoLogitsTerm *term) {
    if (term == NULL) return;
    term_program_free(term->program);
    free(term);
}
