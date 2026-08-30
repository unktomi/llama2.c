/*
 * DO NOT USE AS THE INFERENCE TERM.
 *
 * The API accepts concrete token arrays before composition and therefore
 * cannot express Escardo's recursively composed token selections.  It is
 * quarantined with its implementation as a record of the rejected design.
 */

#ifndef TERM_PROGRAM_H
#define TERM_PROGRAM_H

#include "term_backend.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*
 * Defunctionalized numerical syntax for a composed family context.
 *
 * Construction records applications; it does not execute them.  Learned
 * maps are represented by TermFiller handles and may occur once in a term
 * while their activation argument carries an arbitrary batch/position
 * family.  This is also the portability boundary expected by a future
 * llama.cpp lowering: TermExpr values can be lowered to ggml_tensor values,
 * and TermFiller handles can refer to GGUF-backed model tensors.
 */

typedef struct TermProgram TermProgram;

typedef struct {
    uint32_t id;
} TermExpr;

typedef struct {
    TermExpr expression;
    float *destination;
    size_t element_count;
} TermProgramOutput;

TermProgram *term_program_new(TermBackend *backend);
void term_program_free(TermProgram *program);

TermExpr term_program_tokens(
    TermProgram *program,
    const int *tokens,
    int batch,
    int positions
);

TermExpr term_program_embedding(
    TermProgram *program,
    const TermFiller *filler,
    TermExpr tokens
);

TermExpr term_program_hidden(
    TermProgram *program,
    const TermFiller *filler,
    TermExpr input
);

TermExpr term_program_add(
    TermProgram *program,
    TermExpr left,
    TermExpr right
);

TermExpr term_program_swiglu(
    TermProgram *program,
    TermExpr gate,
    TermExpr up
);

TermExpr term_program_rope_query(
    TermProgram *program,
    TermExpr query,
    int head_size
);

TermExpr term_program_rope_key(
    TermProgram *program,
    TermExpr key,
    int head_size
);

TermExpr term_program_causal_attention(
    TermProgram *program,
    TermExpr query,
    TermExpr key,
    TermExpr value,
    int heads,
    int kv_heads
);

/* Select `count` consecutive positions, beginning at `first`, from every
 * batch member.  No numerical observation occurs while this is constructed. */
TermExpr term_program_gather_positions(
    TermProgram *program,
    TermExpr input,
    int first,
    int count
);

/* Interpret all requested observations in one run of the composed term. */
bool term_program_run(
    TermProgram *program,
    const TermProgramOutput *outputs,
    int output_count
);

int term_program_expression_count(const TermProgram *program);
int term_program_filler_occurrences(const TermProgram *program, int filler_id);
bool term_program_each_filler_at_most_once(const TermProgram *program);

int term_expr_batch(const TermProgram *program, TermExpr expression);
int term_expr_positions(const TermProgram *program, TermExpr expression);
int term_expr_width(const TermProgram *program, TermExpr expression);
size_t term_expr_element_count(const TermProgram *program, TermExpr expression);

#endif
