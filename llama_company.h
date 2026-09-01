#ifndef LLAMA_COMPANY_H
#define LLAMA_COMPANY_H

#include "atkey_term_c.h"

#include <stdbool.h>

/*
 * A finite causal company.  Every row is one token occurrence.  `parent[row]`
 * points to the preceding occurrence in the same causal history, or -1 for
 * the first prompt token.  Parents must precede children in storage order.
 * This is a position/use shape, not a list of completed candidates.
 */
typedef struct {
    int row_count;
    const int *tokens;
    const int *positions;
    const int *parents;
} LlamaCompanyShape;

typedef struct {
    int row_count;
    int dim;
    int vocab_size;
    int scale_count;
    /* [row_count][vocab_size] */
    float *logits;
    /* [scale_count][row_count][dim], embedding scale followed by each layer. */
    float *scales;
} LlamaCompanyResult;

/*
 * The causal company before its token codata is observed.  The transformer
 * body and final RMS have already been composed, but the output filler has
 * not been applied.  An observation consumes the complete row family in one
 * call; the codata is intentionally one-shot.
 */
typedef struct {
    AtkeyRuntime *runtime;
    int row_count;
    int dim;
    int vocab_size;
    int scale_count;
    /* [row_count][dim], after final RMSNorm. */
    float *final_hidden;
    /* [scale_count][row_count][dim], optional. */
    float *scales;
    bool observed;
} LlamaCompanyCodata;

typedef bool (*LlamaCompanyObservationApply)(
    void *environment,
    int row_count,
    int vocab_size,
    const float *logits
);

bool llama_company_codata_construct(
    AtkeyRuntime *runtime,
    const LlamaCompanyShape *shape,
    bool retain_scales,
    LlamaCompanyCodata *codata
);

bool llama_company_codata_observe(
    LlamaCompanyCodata *codata,
    LlamaCompanyObservationApply observation,
    void *environment
);

void llama_company_codata_free(LlamaCompanyCodata *codata);

bool llama_company_evaluate(
    AtkeyRuntime *runtime,
    const LlamaCompanyShape *shape,
    bool retain_scales,
    LlamaCompanyResult *result
);

void llama_company_result_free(LlamaCompanyResult *result);

#endif
