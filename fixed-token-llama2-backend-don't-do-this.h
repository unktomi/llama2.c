/*
 * DO NOT USE AS THE LLAMA INFERENCE ADAPTER.
 *
 * This declaration belongs to the rejected concrete-family backend.  Keeping
 * it active would encourage preserving that evaluator instead of beginning
 * with Escardo's selection term.
 */

#ifndef LLAMA2_BACKEND_H
#define LLAMA2_BACKEND_H

#include "term_backend.h"

TermBackend *llama2_backend_new(
    const char *checkpoint_path,
    const char *tokenizer_path
);

#endif
