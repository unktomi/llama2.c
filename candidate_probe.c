#define TESTING
#include "run.c"

#include <errno.h>
#include <float.h>

static void probe_usage(const char *program) {
    fprintf(
        stderr,
        "usage: %s CHECKPOINT TOKENIZER PROMPT SEED COMPLETION_TOKENS "
        "[TEMPERATURE] [TOP_P] [SHOW_TOP_K]\n",
        program
    );
    exit(EXIT_FAILURE);
}

static long parse_long(const char *text, const char *name) {
    errno = 0;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        fprintf(stderr, "%s must be an integer\n", name);
        exit(EXIT_FAILURE);
    }
    return value;
}

static float parse_float(const char *text, const char *name) {
    errno = 0;
    char *end = NULL;
    float value = strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value)) {
        fprintf(stderr, "%s must be a finite number\n", name);
        exit(EXIT_FAILURE);
    }
    return value;
}

static int local_rank(const float *logits, int count, int token) {
    int rank = 1;
    for (int candidate = 0; candidate < count; candidate++) {
        if (logits[candidate] > logits[token] ||
            (logits[candidate] == logits[token] && candidate < token)) {
            rank++;
        }
    }
    return rank;
}

static double token_log_probability(
    const float *logits,
    int count,
    int token
) {
    double maximum = -DBL_MAX;
    for (int index = 0; index < count; index++) {
        if ((double)logits[index] > maximum) maximum = logits[index];
    }
    double normalizer = 0.0;
    for (int index = 0; index < count; index++) {
        normalizer += exp((double)logits[index] - maximum);
    }
    return (double)logits[token] - maximum - log(normalizer);
}

static void print_escaped_text(const char *text) {
    for (const unsigned char *cursor = (const unsigned char *)text;
         *cursor != '\0'; cursor++) {
        unsigned char byte = *cursor;
        if (byte == '\\' || byte == '"') {
            putchar('\\');
            putchar(byte);
        } else if (byte == '\n') {
            fputs("\\n", stdout);
        } else if (byte == '\r') {
            fputs("\\r", stdout);
        } else if (byte == '\t') {
            fputs("\\t", stdout);
        } else if (byte >= 0x20 && byte != 0x7f) {
            putchar(byte);
        } else {
            printf("\\x%02x", byte);
        }
    }
}

static void print_quoted_piece(const char *piece) {
    putchar('"');
    print_escaped_text(piece);
    putchar('"');
}

static int candidate_precedes(
    const float *logits,
    int left,
    int right
) {
    if (logits[left] > logits[right]) return 1;
    if (logits[left] < logits[right]) return 0;
    return left < right;
}

static void print_top_candidates(
    Tokenizer *tokenizer,
    int previous_token,
    const float *logits,
    int vocab_size,
    int count
) {
    if (count <= 0) return;
    int *tokens = malloc((size_t)count * sizeof(*tokens));
    if (tokens == NULL) {
        fprintf(stderr, "could not allocate top-k support\n");
        exit(EXIT_FAILURE);
    }
    int filled = 0;
    for (int token = 0; token < vocab_size; token++) {
        int insertion = filled;
        while (insertion > 0 && candidate_precedes(
                logits,
                token,
                tokens[insertion - 1]
            )) {
            insertion--;
        }
        if (insertion < count) {
            int last = filled < count ? filled : count - 1;
            for (int index = last; index > insertion; index--) {
                tokens[index] = tokens[index - 1];
            }
            tokens[insertion] = token;
            if (filled < count) filled++;
        }
    }
    fputs(" top_candidates=[", stdout);
    for (int index = 0; index < filled; index++) {
        int token = tokens[index];
        if (index != 0) fputs(",", stdout);
        printf("{rank=%d,id=%d,piece=", index + 1, token);
        print_quoted_piece(decode(tokenizer, previous_token, token));
        printf(
            ",log_probability=%.17g}",
            token_log_probability(logits, vocab_size, token)
        );
    }
    putchar(']');
    free(tokens);
}

typedef struct {
    int first_token;
    int token_count;
    int word_count;
    double log_probability_sum;
} PhraseAggregate;

static int phrase_ends_in_piece(const char *piece) {
    return strchr(piece, '.') != NULL ||
        strchr(piece, '!') != NULL ||
        strchr(piece, '?') != NULL;
}

static void print_generated_span(
    Tokenizer *tokenizer,
    int prompt_last_token,
    const int *tokens,
    int first,
    int count
) {
    putchar('"');
    for (int index = first; index < first + count; index++) {
        int previous = index == 0 ? prompt_last_token : tokens[index - 1];
        print_escaped_text(decode(tokenizer, previous, tokens[index]));
    }
    putchar('"');
}

static void print_phrase_aggregates(
    Tokenizer *tokenizer,
    int prompt_last_token,
    const int *tokens,
    const double *log_probabilities,
    int token_count
) {
    if (token_count == 0) return;
    PhraseAggregate *phrases = calloc(
        (size_t)token_count,
        sizeof(*phrases)
    );
    if (phrases == NULL) {
        fprintf(stderr, "could not allocate phrase aggregates\n");
        exit(EXIT_FAILURE);
    }

    int phrase_count = 0;
    PhraseAggregate current = {.first_token = 0};
    for (int index = 0; index < token_count; index++) {
        int previous = index == 0 ? prompt_last_token : tokens[index - 1];
        const char *piece = decode(tokenizer, previous, tokens[index]);
        if (current.token_count == 0) current.first_token = index;
        if (piece[0] == ' ' || current.word_count == 0) current.word_count++;
        current.token_count++;
        current.log_probability_sum += log_probabilities[index];
        if (phrase_ends_in_piece(piece)) {
            phrases[phrase_count++] = current;
            current = (PhraseAggregate){0};
        }
    }
    if (current.token_count != 0) phrases[phrase_count++] = current;

    puts("phrase_aggregates:");
    for (int index = 0; index < phrase_count; index++) {
        PhraseAggregate phrase = phrases[index];
        printf(
            "  scale=phrase index=%d tokens=%d words=%d "
            "log_probability_sum=%.17g mean_log_probability_per_word=%.17g "
            "text=",
            index,
            phrase.token_count,
            phrase.word_count,
            phrase.log_probability_sum,
            phrase.log_probability_sum / phrase.word_count
        );
        print_generated_span(
            tokenizer,
            prompt_last_token,
            tokens,
            phrase.first_token,
            phrase.token_count
        );
        putchar('\n');
    }

    PhraseAggregate *level = phrases;
    int level_count = phrase_count;
    int level_index = 1;
    while (level_count > 1) {
        int next_count = (level_count + 1) / 2;
        PhraseAggregate *next = calloc((size_t)next_count, sizeof(*next));
        if (next == NULL) {
            fprintf(stderr, "could not allocate phrase hierarchy\n");
            exit(EXIT_FAILURE);
        }
        for (int index = 0; index < level_count; index += 2) {
            PhraseAggregate combined = level[index];
            if (index + 1 < level_count) {
                combined.token_count += level[index + 1].token_count;
                combined.word_count += level[index + 1].word_count;
                combined.log_probability_sum +=
                    level[index + 1].log_probability_sum;
            }
            next[index / 2] = combined;
            if (index + 1 < level_count) {
                printf(
                    "  scale=adjacent_phrase_span level=%d index=%d "
                    "tokens=%d words=%d log_probability_sum=%.17g "
                    "mean_log_probability_per_word=%.17g text=",
                    level_index,
                    index / 2,
                    combined.token_count,
                    combined.word_count,
                    combined.log_probability_sum,
                    combined.log_probability_sum / combined.word_count
                );
                print_generated_span(
                    tokenizer,
                    prompt_last_token,
                    tokens,
                    combined.first_token,
                    combined.token_count
                );
                putchar('\n');
            }
        }
        if (level != phrases) free(level);
        level = next;
        level_count = next_count;
        level_index++;
    }
    if (level != phrases) free(level);
    free(phrases);
}

#ifndef CANDIDATE_PROBE_LIBRARY
int main(int argc, char **argv) {
    if (argc < 6 || argc > 9) probe_usage(argv[0]);

    const char *checkpoint_path = argv[1];
    const char *tokenizer_path = argv[2];
    char *prompt = argv[3];
    long parsed_seed = parse_long(argv[4], "seed");
    long parsed_completion_tokens = parse_long(argv[5], "completion tokens");
    float temperature = argc >= 7 ? parse_float(argv[6], "temperature") : 1.0f;
    float top_p = argc >= 8 ? parse_float(argv[7], "top-p") : 0.9f;
    long parsed_show_top_k = argc >= 9 ?
        parse_long(argv[8], "show top-k") : 0;
    if (parsed_seed <= 0 || parsed_completion_tokens <= 0 ||
        temperature < 0.0f || top_p < 0.0f || top_p > 1.0f ||
        parsed_show_top_k < 0) {
        probe_usage(argv[0]);
    }

    Transformer transformer;
    build_transformer(&transformer, (char *)checkpoint_path);
    Tokenizer tokenizer;
    build_tokenizer(
        &tokenizer,
        (char *)tokenizer_path,
        transformer.config.vocab_size
    );
    Sampler sampler;
    build_sampler(
        &sampler,
        transformer.config.vocab_size,
        temperature,
        top_p,
        (unsigned long long)parsed_seed
    );

    int prompt_count = 0;
    int *prompt_tokens = malloc((strlen(prompt) + 3) * sizeof(*prompt_tokens));
    if (prompt_tokens == NULL) {
        fprintf(stderr, "could not allocate prompt tokens\n");
        exit(EXIT_FAILURE);
    }
    encode(&tokenizer, prompt, 1, 0, prompt_tokens, &prompt_count);
    if (prompt_count < 1) {
        fprintf(stderr, "tokenizer produced an empty prompt\n");
        exit(EXIT_FAILURE);
    }

    float *sampling_logits = malloc(
        (size_t)transformer.config.vocab_size * sizeof(*sampling_logits)
    );
    if (sampling_logits == NULL) {
        fprintf(stderr, "could not allocate sampling logits\n");
        exit(EXIT_FAILURE);
    }
    int *generated_tokens = malloc(
        (size_t)parsed_completion_tokens * sizeof(*generated_tokens)
    );
    double *generated_log_probabilities = malloc(
        (size_t)parsed_completion_tokens *
        sizeof(*generated_log_probabilities)
    );
    if (generated_tokens == NULL || generated_log_probabilities == NULL) {
        fprintf(stderr, "could not allocate generated-token trace\n");
        exit(EXIT_FAILURE);
    }

    printf(
        "prompt=%s seed=%ld temperature=%.9g top_p=%.9g\n",
        prompt,
        parsed_seed,
        temperature,
        top_p
    );
    fputs("completion: ", stdout);
    fflush(stdout);

    int token = prompt_tokens[0];
    int generated = 0;
    int maximum_rank = 0;
    double total_log_probability = 0.0;
    for (int position = 0;
         position < transformer.config.seq_len &&
         generated < parsed_completion_tokens;
         position++) {
        float *logits = forward(&transformer, token, position);
        int next;
        if (position < prompt_count - 1) {
            next = prompt_tokens[position + 1];
        } else {
            memcpy(
                sampling_logits,
                logits,
                (size_t)transformer.config.vocab_size * sizeof(*logits)
            );
            next = sample(&sampler, sampling_logits);
            if (next == 1) break;
            const char *piece = decode(&tokenizer, token, next);
            safe_printf((char *)piece);
            int rank = local_rank(logits, transformer.config.vocab_size, next);
            double log_probability = token_log_probability(
                logits,
                transformer.config.vocab_size,
                next
            );
            if (rank > maximum_rank) maximum_rank = rank;
            total_log_probability += log_probability;
            generated++;
        }
        token = next;
    }
    putchar('\n');

    token = prompt_tokens[0];
    sampler.rng_state = (unsigned long long)parsed_seed;
    generated = 0;
    for (int position = 0;
         position < transformer.config.seq_len &&
         generated < parsed_completion_tokens;
         position++) {
        float *logits = forward(&transformer, token, position);
        int next;
        if (position < prompt_count - 1) {
            next = prompt_tokens[position + 1];
        } else {
            memcpy(
                sampling_logits,
                logits,
                (size_t)transformer.config.vocab_size * sizeof(*logits)
            );
            next = sample(&sampler, sampling_logits);
            if (next == 1) break;
            const char *piece = decode(&tokenizer, token, next);
            int rank = local_rank(logits, transformer.config.vocab_size, next);
            double log_probability = token_log_probability(
                logits,
                transformer.config.vocab_size,
                next
            );
            generated_tokens[generated] = next;
            generated_log_probabilities[generated] = log_probability;
            int show_top_k = (int)parsed_show_top_k;
            if (show_top_k > transformer.config.vocab_size) {
                show_top_k = transformer.config.vocab_size;
            }
            if (show_top_k > 0) {
                printf(
                    "token=%d id=%d piece=",
                    generated,
                    next
                );
                print_quoted_piece(piece);
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
        total_log_probability
    );
    print_phrase_aggregates(
        &tokenizer,
        prompt_tokens[prompt_count - 1],
        generated_tokens,
        generated_log_probabilities,
        generated
    );

    free(generated_log_probabilities);
    free(generated_tokens);
    free(sampling_logits);
    free(prompt_tokens);
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif
