#define main exhaustive_scale_probe_program_main
#include "exhaustive_scale_probe.c"
#undef main

/*
 * Post-run audit for the reward carrier before any cross-scale reduction.
 * The original final RMS/classifier observes the embedding and every
 * completed transformer layer.  Scores are retained as a vector indexed by
 * model scale; this program deliberately does not sum or weight that vector.
 */

static void scale_reward_usage(const char *program) {
    fprintf(
        stderr,
        "usage: %s CHECKPOINT TOKENIZER PROMPT COMPLETION [TOKENS]\n",
        program
    );
    exit(EXIT_FAILURE);
}

static double scale_token_log_probability(
    const float *logits,
    int count,
    int token
) {
    double maximum = -DBL_MAX;
    for (int candidate = 0; candidate < count; candidate++) {
        if ((double)logits[candidate] > maximum) maximum = logits[candidate];
    }
    double partition = 0.0;
    for (int candidate = 0; candidate < count; candidate++) {
        partition += exp((double)logits[candidate] - maximum);
    }
    return (double)logits[token] - maximum - log(partition);
}

static void print_scale_vector(const double *values, int count) {
    putchar('[');
    for (int scale = 0; scale < count; scale++) {
        if (scale != 0) putchar(',');
        printf("%.17g", values[scale]);
    }
    putchar(']');
}

int main(int argc, char **argv) {
    if (argc != 5 && argc != 6) scale_reward_usage(argv[0]);

    Transformer transformer;
    build_transformer(&transformer, argv[1]);
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, argv[2], transformer.config.vocab_size);

    size_t prompt_bytes = strlen(argv[3]);
    size_t completion_bytes = strlen(argv[4]);
    if (prompt_bytes > SIZE_MAX - completion_bytes - 1) {
        fprintf(stderr, "prompt and completion are too long\n");
        exit(EXIT_FAILURE);
    }
    size_t full_bytes = prompt_bytes + completion_bytes;
    char *full_text = malloc(full_bytes + 1);
    int *prompt_tokens = malloc((prompt_bytes + 3) * sizeof(*prompt_tokens));
    int *full_tokens = malloc((full_bytes + 3) * sizeof(*full_tokens));
    if (full_text == NULL || prompt_tokens == NULL || full_tokens == NULL) {
        fprintf(stderr, "could not allocate scale reward input\n");
        exit(EXIT_FAILURE);
    }
    memcpy(full_text, argv[3], prompt_bytes);
    memcpy(full_text + prompt_bytes, argv[4], completion_bytes + 1);

    int prompt_count = 0;
    int full_count = 0;
    encode(&tokenizer, argv[3], 1, 0, prompt_tokens, &prompt_count);
    encode(&tokenizer, full_text, 1, 0, full_tokens, &full_count);
    if (prompt_count < 1 || full_count <= prompt_count ||
        full_count > transformer.config.seq_len) {
        fprintf(stderr, "invalid encoded scale reward context\n");
        exit(EXIT_FAILURE);
    }
    int available = full_count - prompt_count;
    int horizon = available;
    if (argc == 6) {
        long parsed = parse_long(argv[5], "tokens");
        if (parsed <= 0 || parsed > available || parsed > INT_MAX) {
            scale_reward_usage(argv[0]);
        }
        horizon = (int)parsed;
    }
    for (int index = 0; index < prompt_count; index++) {
        if (prompt_tokens[index] != full_tokens[index]) {
            fprintf(stderr, "prompt is not a token prefix of full text\n");
            exit(EXIT_FAILURE);
        }
    }

    int scale_count = transformer.config.n_layers + 1;
    int vocab_size = transformer.config.vocab_size;
    float *scale_logits = malloc(
        (size_t)scale_count * vocab_size * sizeof(*scale_logits)
    );
    double *edge_scores = malloc(
        (size_t)scale_count * sizeof(*edge_scores)
    );
    double *path_scores = calloc(
        (size_t)scale_count,
        sizeof(*path_scores)
    );
    if (scale_logits == NULL || edge_scores == NULL || path_scores == NULL) {
        fprintf(stderr, "could not allocate scale reward carrier\n");
        exit(EXIT_FAILURE);
    }

    int token = full_tokens[0];
    int generated = 0;
    printf(
        "prompt=%s completion=%s prompt_tokens=%d scales=%d\n",
        argv[3],
        argv[4],
        prompt_count,
        scale_count
    );
    puts("scale_kinds=[embedding,completed_layer_1,...,final_model]");
    int final_position = prompt_count - 1 + horizon;
    for (int position = 0; position < final_position; position++) {
        forward_scales(&transformer, token, position, scale_logits);
        int next = full_tokens[position + 1];
        if (position >= prompt_count - 1) {
            for (int scale = 0; scale < scale_count; scale++) {
                edge_scores[scale] = scale_token_log_probability(
                    scale_logits + (size_t)scale * vocab_size,
                    vocab_size,
                    next
                );
                path_scores[scale] += edge_scores[scale];
            }
            printf(
                "token_index=%d token=%d piece=",
                generated,
                next
            );
            fprint_json_piece(stdout, decode(&tokenizer, token, next));
            fputs(" edge_scale_log_probabilities=", stdout);
            print_scale_vector(edge_scores, scale_count);
            fputs(" path_scale_log_probabilities=", stdout);
            print_scale_vector(path_scores, scale_count);
            putchar('\n');
            generated++;
        }
        token = next;
    }
    fputs("completion_scale_log_probabilities=", stdout);
    print_scale_vector(path_scores, scale_count);
    putchar('\n');
    fputs("completion_scale_mean_log_probabilities=", stdout);
    for (int scale = 0; scale < scale_count; scale++) {
        edge_scores[scale] = path_scores[scale] / generated;
    }
    print_scale_vector(edge_scores, scale_count);
    putchar('\n');
    printf("completion_tokens=%d reduction=none\n", generated);

    free(path_scores);
    free(edge_scores);
    free(scale_logits);
    free(full_tokens);
    free(prompt_tokens);
    free(full_text);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return EXIT_SUCCESS;
}
