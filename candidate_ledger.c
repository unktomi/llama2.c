#define _POSIX_C_SOURCE 200809L

#include "candidate_ledger.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

static void write_json_string(FILE *stream, const char *text) {
    if (text == NULL) {
        fputs("null", stream);
        return;
    }
    fputc('"', stream);
    const unsigned char *cursor = (const unsigned char *)text;
    while (*cursor != '\0') {
        unsigned char byte = *cursor++;
        switch (byte) {
            case '"': fputs("\\\"", stream); break;
            case '\\': fputs("\\\\", stream); break;
            case '\b': fputs("\\b", stream); break;
            case '\f': fputs("\\f", stream); break;
            case '\n': fputs("\\n", stream); break;
            case '\r': fputs("\\r", stream); break;
            case '\t': fputs("\\t", stream); break;
            default:
                if (byte < 0x20) {
                    fprintf(stream, "\\u%04x", (unsigned int)byte);
                } else {
                    fputc(byte, stream);
                }
        }
    }
    fputc('"', stream);
}

static void write_u64(FILE *stream, uint64_t value) {
    if (value == CANDIDATE_LEDGER_NONE_U64) {
        fputs("null", stream);
    } else {
        fprintf(stream, "%" PRIu64, value);
    }
}

static void write_int(FILE *stream, int value) {
    if (value == CANDIDATE_LEDGER_NONE_INT) {
        fputs("null", stream);
    } else {
        fprintf(stream, "%d", value);
    }
}

static void write_double(FILE *stream, double value) {
    if (isnan(value)) {
        fputs("null", stream);
    } else if (isinf(value)) {
        write_json_string(stream, value > 0.0 ? "+inf" : "-inf");
    } else {
        fprintf(stream, "%.17g", value);
    }
}

static bool flush_ledger(CandidateLedger *ledger) {
    if (fflush(ledger->stream) != 0) return false;
    if (!ledger->durable) return true;
    return fsync(fileno(ledger->stream)) == 0;
}

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
) {
    if (ledger == NULL || path == NULL) return false;
    memset(ledger, 0, sizeof(*ledger));
    ledger->stream = fopen(path, "w");
    if (ledger->stream == NULL) return false;
    ledger->durable = durable;
    fputs("{\"schema\":\"llama2.selection-ledger\",\"version\":1", ledger->stream);
    fputs(",\"kind\":\"run_start\",\"event_id\":0,\"algorithm\":", ledger->stream);
    write_json_string(ledger->stream, algorithm);
    fputs(",\"checkpoint\":", ledger->stream);
    write_json_string(ledger->stream, checkpoint);
    fputs(",\"tokenizer\":", ledger->stream);
    write_json_string(ledger->stream, tokenizer);
    fputs(",\"prompt\":", ledger->stream);
    write_json_string(ledger->stream, prompt);
    fprintf(
        ledger->stream,
        ",\"horizon\":%d,\"top_k\":%d,\"seed\":%" PRIu64 "}\n",
        horizon,
        top_k,
        seed
    );
    ledger->next_event_id = 1;
    if (!flush_ledger(ledger)) {
        fclose(ledger->stream);
        ledger->stream = NULL;
        return false;
    }
    return true;
}

void candidate_ledger_write(
    CandidateLedger *ledger,
    const CandidateLedgerEvent *event
) {
    if (ledger == NULL || ledger->stream == NULL || event == NULL ||
        event->kind == NULL) {
        return;
    }
    FILE *stream = ledger->stream;
    fprintf(
        stream,
        "{\"schema\":\"llama2.selection-ledger\",\"version\":1,"
        "\"event_id\":%" PRIu64 ",\"kind\":",
        ledger->next_event_id++
    );
    write_json_string(stream, event->kind);
    fputs(",\"frame_id\":", stream);
    write_u64(stream, event->frame_id);
    fputs(",\"parent_frame_id\":", stream);
    write_u64(stream, event->parent_frame_id);
    fputs(",\"demand_id\":", stream);
    write_u64(stream, event->demand_id);
    fputs(",\"candidate_id\":", stream);
    write_u64(stream, event->candidate_id);
    fputs(",\"source_candidate_id\":", stream);
    write_u64(stream, event->source_candidate_id);
    fputs(",\"multiplicity\":", stream);
    write_u64(stream, event->multiplicity);
    fputs(",\"depth\":", stream);
    write_int(stream, event->depth);
    fputs(",\"position\":", stream);
    write_int(stream, event->position);
    fputs(",\"rank\":", stream);
    write_int(stream, event->rank);
    fputs(",\"token_id\":", stream);
    write_int(stream, event->token_id);
    fputs(",\"piece\":", stream);
    write_json_string(stream, event->piece);
    fputs(",\"prefix\":", stream);
    write_json_string(stream, event->prefix);
    fputs(",\"context\":", stream);
    write_json_string(stream, event->context);
    fputs(",\"completion\":", stream);
    write_json_string(stream, event->completion);
    fputs(",\"status\":", stream);
    write_json_string(stream, event->status);
    fputs(",\"reason\":", stream);
    write_json_string(stream, event->reason);
    fputs(",\"local_logit\":", stream);
    write_double(stream, event->local_logit);
    fputs(",\"local_log_probability\":", stream);
    write_double(stream, event->local_log_probability);
    fputs(",\"observer_score\":", stream);
    write_double(stream, event->observer_score);
    fputs(",\"backed_score\":", stream);
    write_double(stream, event->backed_score);
    fputs(",\"aggregate_before\":", stream);
    write_double(stream, event->aggregate_before);
    fputs(",\"aggregate_after\":", stream);
    write_double(stream, event->aggregate_after);
    fputs(",\"scale_scores\":[", stream);
    for (int index = 0; index < event->scale_score_count; index++) {
        if (index != 0) fputc(',', stream);
        write_double(stream, event->scale_scores[index]);
    }
    fputs("]}\n", stream);
    if (!flush_ledger(ledger)) {
        fprintf(stderr, "candidate ledger flush failed: %s\n", strerror(errno));
    }
}

void candidate_ledger_close(CandidateLedger *ledger) {
    if (ledger == NULL || ledger->stream == NULL) return;
    CandidateLedgerEvent event = {
        .kind = "run_end",
        .frame_id = CANDIDATE_LEDGER_NONE_U64,
        .parent_frame_id = CANDIDATE_LEDGER_NONE_U64,
        .demand_id = CANDIDATE_LEDGER_NONE_U64,
        .candidate_id = CANDIDATE_LEDGER_NONE_U64,
        .source_candidate_id = CANDIDATE_LEDGER_NONE_U64,
        .multiplicity = CANDIDATE_LEDGER_NONE_U64,
        .depth = CANDIDATE_LEDGER_NONE_INT,
        .position = CANDIDATE_LEDGER_NONE_INT,
        .rank = CANDIDATE_LEDGER_NONE_INT,
        .token_id = CANDIDATE_LEDGER_NONE_INT,
        .local_logit = NAN,
        .local_log_probability = NAN,
        .observer_score = NAN,
        .backed_score = NAN,
        .aggregate_before = NAN,
        .aggregate_after = NAN,
    };
    candidate_ledger_write(ledger, &event);
    fclose(ledger->stream);
    ledger->stream = NULL;
}
