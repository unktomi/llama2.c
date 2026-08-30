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

bool llama_company_evaluate(
    AtkeyRuntime *runtime,
    const LlamaCompanyShape *shape,
    bool retain_scales,
    LlamaCompanyResult *result
);

void llama_company_result_free(LlamaCompanyResult *result);

#endif
