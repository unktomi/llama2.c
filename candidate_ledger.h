#ifndef CANDIDATE_LEDGER_H
#define CANDIDATE_LEDGER_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    FILE *stream;
    uint64_t next_event_id;
    bool durable;
} CandidateLedger;

typedef struct {
    const char *kind;
    uint64_t frame_id;
    uint64_t parent_frame_id;
    uint64_t demand_id;
    uint64_t candidate_id;
    uint64_t source_candidate_id;
    uint64_t multiplicity;
    int depth;
    int position;
    int rank;
    int token_id;
    const char *piece;
    const char *prefix;
    const char *context;
    const char *completion;
    const char *status;
    const char *reason;
    double local_logit;
    double local_log_probability;
    double observer_score;
    double backed_score;
    double aggregate_before;
    double aggregate_after;
    const double *scale_scores;
    int scale_score_count;
} CandidateLedgerEvent;

bool candidate_ledger_open(
    CandidateLedger *ledger,
    const char *path,
    bool durable,
    const char *algorithm,
    const char *checkpoint,
    const char *tokenizer,
    const char *prompt,
    int horizon,
    int top_k,
    uint64_t seed
);

void candidate_ledger_write(
    CandidateLedger *ledger,
    const CandidateLedgerEvent *event
);

void candidate_ledger_close(CandidateLedger *ledger);

#define CANDIDATE_LEDGER_NONE_U64 UINT64_MAX
#define CANDIDATE_LEDGER_NONE_INT (-1)

#endif
