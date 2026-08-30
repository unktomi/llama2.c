/*
 * DO NOT USE AS ESCARDO INFERENCE.
 *
 * This API requires completions to be supplied before composition.  It thus
 * encodes the exact architectural assumption the quarantine was meant to
 * reject and is retained only as an auditable failed interface.
 */

#ifndef ESCARDO_MODEL_H
#define ESCARDO_MODEL_H

#include "term_backend.h"

typedef struct EscardoModel EscardoModel;
typedef struct EscardoLogitsTerm EscardoLogitsTerm;

EscardoModel *escardo_model_new(TermBackend *backend);
void escardo_model_free(EscardoModel *model);

int escardo_model_dim(const EscardoModel *model);
int escardo_model_layers(const EscardoModel *model);
int escardo_model_vocab(const EscardoModel *model);

/*
 * Numerical interpretation of one already-formed family of complete token
 * contexts.  This function makes no token choice.  All batch members share a
 * prompt/horizon, and every learned filler is lifted over the complete
 * (batch x position) family.
 */
void escardo_model_apply_whole_context(
    EscardoModel *model,
    const int *prompt,
    int prompt_count,
    const int *completions,
    int batch_count,
    int horizon,
    float *final_hidden,
    float *predictor_hidden
);

/* Logits for every supplied completion coordinate, laid out as
 * [batch][horizon][vocabulary]. */
void escardo_model_apply_logits_family(
    EscardoModel *model,
    const int *prompt,
    int prompt_count,
    const int *completions,
    int batch_count,
    int horizon,
    float *logits
);

/* Compose first, interpret later.  The returned term owns its copied token
 * family and contains one occurrence of every learned filler. */
EscardoLogitsTerm *escardo_model_compose_logits_family(
    EscardoModel *model,
    const int *prompt,
    int prompt_count,
    const int *completions,
    int batch_count,
    int horizon
);

size_t escardo_logits_term_element_count(const EscardoLogitsTerm *term);
bool escardo_logits_term_run(EscardoLogitsTerm *term, float *logits);
void escardo_logits_term_free(EscardoLogitsTerm *term);

#endif
