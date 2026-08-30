/*
 * DO NOT USE AS AN INFERENCE REWARD.
 *
 * I should never have invented this entropy-distance score.  The expression
 * -abs(observed surprisal - expected entropy) / span length is neither the
 * model's observer nor a consequence of the stated Firth/selection calculus.
 * It silently replaces the problem with a heuristic objective and can make a
 * bad inference procedure look better without repairing composition.  It is
 * kept only as a reminder of the rejected approach.
 */

#define CANDIDATE_PROBE_LIBRARY
#include "candidate_probe.c"

#include <limits.h>

typedef struct {
    int first_token;
    int token_count;
    double surprisal_sum;
    double expected_entropy_sum;
} ZipSpan;

typedef struct {
    int scale;
    int count;
    ZipSpan *spans;
} ZipRow;

static void zip_usage(const char *program) {
    fprintf(
        stderr,
        "usage: %s CHECKPOINT TOKENIZER HORIZON PROMPT COMPLETION\n",
        program
    );
    exit(EXIT_FAILURE);
}

static double distribution_entropy(const float *logits, int count) {
    double maximum = -DBL_MAX;
    for (int index = 0; index < count; index++) {
        if ((double)logits[index] > maximum) maximum = logits[index];
    }
    double partition = 0.0;
    double weighted_centered_logit = 0.0;
    for (int index = 0; index < count; index++) {
        double centered = (double)logits[index] - maximum;
        double weight = exp(centered);
        partition += weight;
        weighted_centered_logit += weight * centered;
    }
    return log(partition) - weighted_centered_logit / partition;
}

static double zip_span_score(ZipSpan span) {
    return -fabs(span.surprisal_sum - span.expected_entropy_sum) /
        span.token_count;
}

static void print_zip_span(
    Tokenizer *tokenizer,
    int prompt_last_token,
    const int *tokens,
    int scale,
    int index,
    ZipSpan span
) {
    printf(
        "  scale=%d index=%d first_token=%d tokens=%d "
        "mean_surprisal=%.17g mean_expected_entropy=%.17g "
        "score=%.17g text=",
        scale,
        index,
        span.first_token,
        span.token_count,
        span.surprisal_sum / span.token_count,
        span.expected_entropy_sum / span.token_count,
        zip_span_score(span)
    );
    print_generated_span(
        tokenizer,
        prompt_last_token,
        tokens,
        span.first_token,
        span.token_count
    );
    putchar('\n');
}

int main(int argc, char **argv) {
    if (argc != 7) zip_usage(argv[0]);
    const char *checkpoint_path = argv[1];
    const char *tokenizer_path = argv[2];
    long parsed_horizon = parse_long(argv[3], "horizon");
    char *prompt = argv[4];
    const char *completion_text = argv[5];
    const char *label = argv[6];
    if (parsed_horizon <= 0 || parsed_horizon > INT_MAX) zip_usage(argv[0]);
    int horizon = (int)parsed_horizon;

    Transformer transformer;
    build_transformer(&transformer, (char *)checkpoint_path);
    Tokenizer tokenizer;
    build_tokenizer(
        &tokenizer,
        (char *)tokenizer_path,
        transformer.config.vocab_size
    );

    size_t prompt_bytes = strlen(prompt);
    size_t completion_bytes = strlen(completion_text);
    if (prompt_bytes > SIZE_MAX - completion_bytes - 1) {
        fprintf(stderr, "prompt and completion are too long\n");
        exit(EXIT_FAILURE);
    }
    char *full_text = malloc(prompt_bytes + completion_bytes + 1);
    int *prompt_tokens = malloc((prompt_bytes + 3) * sizeof(*prompt_tokens));
    int *full_tokens = malloc(
        (prompt_bytes + completion_bytes + 3) * sizeof(*full_tokens)
    );
    if (full_text == NULL || prompt_tokens == NULL || full_tokens == NULL) {
        fprintf(stderr, "could not allocate encoded zip input\n");
        exit(EXIT_FAILURE);
    }
    memcpy(full_text, prompt, prompt_bytes);
    memcpy(full_text + prompt_bytes, completion_text, completion_bytes + 1);

    int prompt_count = 0;
    int full_count = 0;
    encode(&tokenizer, prompt, 1, 0, prompt_tokens, &prompt_count);
    encode(&tokenizer, full_text, 1, 0, full_tokens, &full_count);
    if (prompt_count < 1 || full_count < prompt_count + horizon ||
        prompt_count + horizon > transformer.config.seq_len) {
        fprintf(stderr, "completion does not provide the requested horizon\n");
        exit(EXIT_FAILURE);
    }
    for (int index = 0; index < prompt_count; index++) {
        if (prompt_tokens[index] != full_tokens[index]) {
            fprintf(stderr, "prompt is not a token prefix of completion\n");
            exit(EXIT_FAILURE);
        }
    }

    int *completion_tokens = malloc((size_t)horizon * sizeof(*completion_tokens));
    double *surprisals = malloc((size_t)horizon * sizeof(*surprisals));
    double *entropies = malloc((size_t)horizon * sizeof(*entropies));
    if (completion_tokens == NULL || surprisals == NULL || entropies == NULL) {
        fprintf(stderr, "could not allocate zip trace\n");
        exit(EXIT_FAILURE);
    }

    int token = full_tokens[0];
    int generated = 0;
    for (int position = 0;
         position < prompt_count + horizon - 1;
         position++) {
        float *logits = forward(&transformer, token, position);
        int next = full_tokens[position + 1];
        if (position >= prompt_count - 1) {
            completion_tokens[generated] = next;
            surprisals[generated] = -token_log_probability(
                logits,
                transformer.config.vocab_size,
                next
            );
            entropies[generated] = distribution_entropy(
                logits,
                transformer.config.vocab_size
            );
            generated++;
        }
        token = next;
    }
    if (generated != horizon) {
        fprintf(stderr, "zip trace horizon mismatch\n");
        exit(EXIT_FAILURE);
    }

    int row_capacity = 1;
    for (int count = horizon; count > 1; count = (count + 1) / 2) {
        row_capacity++;
    }
    ZipRow *rows = calloc((size_t)row_capacity, sizeof(*rows));
    if (rows == NULL) {
        fprintf(stderr, "could not allocate zip rows\n");
        exit(EXIT_FAILURE);
    }
    rows[0].scale = 0;
    rows[0].count = horizon;
    rows[0].spans = calloc((size_t)horizon, sizeof(*rows[0].spans));
    if (rows[0].spans == NULL) {
        fprintf(stderr, "could not allocate zip leaves\n");
        exit(EXIT_FAILURE);
    }
    for (int index = 0; index < horizon; index++) {
        rows[0].spans[index] = (ZipSpan){
            .first_token = index,
            .token_count = 1,
            .surprisal_sum = surprisals[index],
            .expected_entropy_sum = entropies[index],
        };
    }

    int row_count = 1;
    while (rows[row_count - 1].count > 1) {
        ZipRow *previous = &rows[row_count - 1];
        ZipRow *row = &rows[row_count];
        row->scale = row_count;
        row->count = (previous->count + 1) / 2;
        row->spans = calloc((size_t)row->count, sizeof(*row->spans));
        if (row->spans == NULL) {
            fprintf(stderr, "could not allocate zipped row\n");
            exit(EXIT_FAILURE);
        }
        for (int index = 0; index < previous->count; index += 2) {
            ZipSpan parent = previous->spans[index];
            if (index + 1 < previous->count) {
                ZipSpan right = previous->spans[index + 1];
                parent.token_count += right.token_count;
                parent.surprisal_sum += right.surprisal_sum;
                parent.expected_entropy_sum += right.expected_entropy_sum;
            }
            row->spans[index / 2] = parent;
        }
        row_count++;
    }

    printf("candidate=%s horizon=%d\n", label, horizon);
    for (int row = 0; row < row_count; row++) {
        printf("row scale=%d spans=%d\n", rows[row].scale, rows[row].count);
        for (int index = 0; index < rows[row].count; index++) {
            print_zip_span(
                &tokenizer,
                prompt_tokens[prompt_count - 1],
                completion_tokens,
                rows[row].scale,
                index,
                rows[row].spans[index]
            );
        }
    }
    ZipSpan root = rows[row_count - 1].spans[0];
    printf(
        "root_score=%.17g root_mean_surprisal=%.17g "
        "root_mean_expected_entropy=%.17g\n",
        zip_span_score(root),
        root.surprisal_sum / root.token_count,
        root.expected_entropy_sum / root.token_count
    );

    for (int row = 0; row < row_count; row++) free(rows[row].spans);
    free(rows);
    free(entropies);
    free(surprisals);
    free(completion_tokens);
    free(full_tokens);
    free(prompt_tokens);
    free(full_text);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
