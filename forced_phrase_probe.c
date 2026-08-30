#define CANDIDATE_PROBE_LIBRARY
#include "candidate_probe.c"

static void forced_usage(const char *program) {
    fprintf(
        stderr,
        "usage: %s CHECKPOINT TOKENIZER PROMPT COMPLETION [SHOW_TOP_K]\n",
        program
    );
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
    if (argc < 5 || argc > 6) forced_usage(argv[0]);
    const char *checkpoint_path = argv[1];
    const char *tokenizer_path = argv[2];
    char *prompt = argv[3];
    const char *completion_text = argv[4];
    long parsed_show_top_k = argc == 6 ?
        parse_long(argv[5], "show top-k") : 0;
    if (parsed_show_top_k < 0) forced_usage(argv[0]);

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
    if (full_text == NULL) {
        fprintf(stderr, "could not allocate combined text\n");
        exit(EXIT_FAILURE);
    }
    memcpy(full_text, prompt, prompt_bytes);
    memcpy(full_text + prompt_bytes, completion_text, completion_bytes + 1);

    int *prompt_tokens = malloc((prompt_bytes + 3) * sizeof(*prompt_tokens));
    int *full_tokens = malloc(
        (prompt_bytes + completion_bytes + 3) * sizeof(*full_tokens)
    );
    if (prompt_tokens == NULL || full_tokens == NULL) {
        fprintf(stderr, "could not allocate encoded text\n");
        exit(EXIT_FAILURE);
    }
    int prompt_count = 0;
    int full_count = 0;
    encode(&tokenizer, prompt, 1, 0, prompt_tokens, &prompt_count);
    encode(&tokenizer, full_text, 1, 0, full_tokens, &full_count);
    if (prompt_count < 1 || full_count <= prompt_count) {
        fprintf(stderr, "completion did not add tokens to the prompt\n");
        exit(EXIT_FAILURE);
    }
    for (int index = 0; index < prompt_count; index++) {
        if (prompt_tokens[index] != full_tokens[index]) {
            fprintf(
                stderr,
                "prompt tokenization is not a prefix of the combined text "
                "at token %d\n",
                index
            );
            exit(EXIT_FAILURE);
        }
    }

    int completion_count = full_count - prompt_count;
    if (full_count > transformer.config.seq_len) {
        fprintf(stderr, "combined text exceeds model context\n");
        exit(EXIT_FAILURE);
    }
    int *completion_tokens = malloc(
        (size_t)completion_count * sizeof(*completion_tokens)
    );
    double *log_probabilities = malloc(
        (size_t)completion_count * sizeof(*log_probabilities)
    );
    if (completion_tokens == NULL || log_probabilities == NULL) {
        fprintf(stderr, "could not allocate forced-completion trace\n");
        exit(EXIT_FAILURE);
    }

    printf("prompt=%s\n", prompt);
    printf("completion: %s\n", completion_text);
    double total = 0.0;
    int maximum_rank = 0;
    int generated = 0;
    int token = full_tokens[0];
    for (int position = 0; position < full_count - 1; position++) {
        float *logits = forward(&transformer, token, position);
        int next = full_tokens[position + 1];
        if (position >= prompt_count - 1) {
            int rank = local_rank(logits, transformer.config.vocab_size, next);
            double log_probability = token_log_probability(
                logits,
                transformer.config.vocab_size,
                next
            );
            completion_tokens[generated] = next;
            log_probabilities[generated] = log_probability;
            if (rank > maximum_rank) maximum_rank = rank;
            total += log_probability;
            int show_top_k = (int)parsed_show_top_k;
            if (show_top_k > transformer.config.vocab_size) {
                show_top_k = transformer.config.vocab_size;
            }
            if (show_top_k > 0) {
                printf("token=%d id=%d piece=", generated, next);
                print_quoted_piece(decode(&tokenizer, token, next));
                printf(
                    " local_rank=%d log_probability=%.17g",
                    rank,
                    log_probability
                );
                print_top_candidates(
                    &tokenizer,
                    token,
                    logits,
                    transformer.config.vocab_size,
                    show_top_k
                );
                putchar('\n');
            }
            generated++;
        }
        token = next;
    }
    printf(
        "generated_tokens=%d max_local_rank=%d "
        "log_probability_sum=%.17g\n",
        generated,
        maximum_rank,
        total
    );
    print_phrase_aggregates(
        &tokenizer,
        prompt_tokens[prompt_count - 1],
        completion_tokens,
        log_probabilities,
        generated
    );

    free(log_probabilities);
    free(completion_tokens);
    free(full_tokens);
    free(prompt_tokens);
    free(full_text);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
