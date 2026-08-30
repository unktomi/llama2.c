#define CANDIDATE_PROBE_LIBRARY
#include "candidate_probe.c"

static void dump_usage(const char *program) {
    fprintf(
        stderr,
        "usage: %s CHECKPOINT TOKENIZER PROMPT_FILE SEED MIN_TOKENS MAX_TOKENS "
        "TEMPERATURE TOP_P COMPLETION_FILE TRACE_FILE\n",
        program
    );
    exit(EXIT_FAILURE);
}

static char *read_prompt_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
        fprintf(stderr, "could not open prompt file %s\n", path);
        exit(EXIT_FAILURE);
    }
    long length = ftell(file);
    if (length < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "could not measure prompt file %s\n", path);
        exit(EXIT_FAILURE);
    }
    char *text = malloc((size_t)length + 1);
    if (text == NULL || fread(text, 1, (size_t)length, file) !=
            (size_t)length || fclose(file) != 0) {
        fprintf(stderr, "could not read prompt file %s\n", path);
        exit(EXIT_FAILURE);
    }
    text[length] = '\0';
    return text;
}

static void fprint_json_text(FILE *file, const char *text) {
    fputc('"', file);
    for (const unsigned char *cursor = (const unsigned char *)text;
         *cursor != '\0'; cursor++) {
        unsigned char byte = *cursor;
        if (byte == '"' || byte == '\\') {
            fputc('\\', file);
            fputc(byte, file);
        } else if (byte == '\n') {
            fputs("\\n", file);
        } else if (byte == '\r') {
            fputs("\\r", file);
        } else if (byte == '\t') {
            fputs("\\t", file);
        } else if (byte >= 0x20) {
            fputc(byte, file);
        } else {
            fprintf(file, "\\u%04x", byte);
        }
    }
    fputc('"', file);
}

static int best_token(const float *logits, int count) {
    int best = 0;
    for (int token = 1; token < count; token++) {
        if (logits[token] > logits[best]) best = token;
    }
    return best;
}

static void write_terminal(
    FILE *trace,
    const float *logits,
    int vocab_size,
    int generated,
    int selected
) {
    fprintf(
        trace,
        "{\"kind\":\"terminal\",\"generated_tokens\":%d,"
        "\"selected_delimiter\":%s,"
        "\"delimiter_log_probability\":%.17g,"
        "\"delimiter_rank\":%d}\n",
        generated,
        selected ? "true" : "false",
        token_log_probability(logits, vocab_size, 1),
        local_rank(logits, vocab_size, 1)
    );
    fflush(trace);
}

int main(int argc, char **argv) {
    if (argc != 11) dump_usage(argv[0]);
    long seed = parse_long(argv[4], "seed");
    long minimum_tokens = parse_long(argv[5], "minimum tokens");
    long maximum_tokens = parse_long(argv[6], "maximum tokens");
    float temperature = parse_float(argv[7], "temperature");
    float top_p = parse_float(argv[8], "top-p");
    if (seed <= 0 || minimum_tokens < 0 || maximum_tokens <= 0 ||
        minimum_tokens > maximum_tokens || temperature < 0.0f ||
        top_p < 0.0f || top_p > 1.0f) {
        dump_usage(argv[0]);
    }

    Transformer transformer;
    build_transformer(&transformer, argv[1]);
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, argv[2], transformer.config.vocab_size);
    char *prompt = read_prompt_file(argv[3]);
    int *prompt_tokens = malloc((strlen(prompt) + 3) * sizeof(*prompt_tokens));
    int prompt_count = 0;
    if (prompt_tokens == NULL) {
        fprintf(stderr, "could not allocate prompt tokens\n");
        exit(EXIT_FAILURE);
    }
    encode(&tokenizer, prompt, 1, 0, prompt_tokens, &prompt_count);
    int available = transformer.config.seq_len - prompt_count;
    if (prompt_count <= 0 || available <= 0 || maximum_tokens > available) {
        fprintf(
            stderr,
            "prompt_tokens=%d requested_tokens=%ld available_tokens=%d\n",
            prompt_count,
            maximum_tokens,
            available
        );
        exit(EXIT_FAILURE);
    }

    FILE *completion = fopen(argv[9], "wb");
    FILE *trace = fopen(argv[10], "w");
    if (completion == NULL || trace == NULL) {
        fprintf(stderr, "could not open candidate output files\n");
        exit(EXIT_FAILURE);
    }
    setvbuf(completion, NULL, _IONBF, 0);
    setvbuf(trace, NULL, _IOLBF, 0);

    Sampler sampler;
    build_sampler(
        &sampler,
        transformer.config.vocab_size,
        temperature,
        top_p,
        (unsigned long long)seed
    );
    float *sampling_logits = malloc(
        (size_t)transformer.config.vocab_size * sizeof(*sampling_logits)
    );
    if (sampling_logits == NULL) {
        fprintf(stderr, "could not allocate sampling logits\n");
        exit(EXIT_FAILURE);
    }

    fprintf(
        trace,
        "{\"kind\":\"meta\",\"prompt_tokens\":%d,"
        "\"minimum_tokens\":%ld,\"maximum_tokens\":%ld,\"seed\":%ld,"
        "\"temperature\":%.9g,\"top_p\":%.9g}\n",
        prompt_count,
        minimum_tokens,
        maximum_tokens,
        seed,
        temperature,
        top_p
    );
    fflush(trace);

    int token = prompt_tokens[0];
    int generated = 0;
    double score = 0.0;
    float *logits = NULL;
    int terminal_written = 0;
    for (int position = 0;
         position < transformer.config.seq_len && generated < maximum_tokens;
         position++) {
        logits = forward(&transformer, token, position);
        int next;
        if (position < prompt_count - 1) {
            next = prompt_tokens[position + 1];
        } else {
            memcpy(
                sampling_logits,
                logits,
                (size_t)transformer.config.vocab_size * sizeof(*logits)
            );
            if (generated < minimum_tokens) {
                sampling_logits[1] = -INFINITY;
            }
            next = sample(&sampler, sampling_logits);
            if (next == 1) {
                write_terminal(
                    trace,
                    logits,
                    transformer.config.vocab_size,
                    generated,
                    1
                );
                terminal_written = 1;
                break;
            }
            const char *piece = decode(&tokenizer, token, next);
            fputs(piece, completion);
            fflush(completion);
            double log_probability = token_log_probability(
                logits,
                transformer.config.vocab_size,
                next
            );
            score += log_probability;
            int top = best_token(logits, transformer.config.vocab_size);
            fprintf(
                trace,
                "{\"kind\":\"token\",\"index\":%d,\"token_id\":%d,"
                "\"piece\":",
                generated,
                next
            );
            fprint_json_text(trace, piece);
            fprintf(
                trace,
                ",\"log_probability\":%.17g,\"local_rank\":%d,"
                "\"top_token_id\":%d,\"top_piece\":",
                log_probability,
                local_rank(logits, transformer.config.vocab_size, next),
                top
            );
            fprint_json_text(trace, decode(&tokenizer, token, top));
            fputs("}\n", trace);
            fflush(trace);
            generated++;
        }
        token = next;
    }

    if (!terminal_written) {
        int last_position = prompt_count + generated - 1;
        if (last_position < transformer.config.seq_len) {
            logits = forward(&transformer, token, last_position);
            write_terminal(
                trace,
                logits,
                transformer.config.vocab_size,
                generated,
                0
            );
        }
    }
    printf(
        "prompt_tokens=%d generated_tokens=%d "
        "completion_log_probability=%.17g terminated=%d\n",
        prompt_count,
        generated,
        score,
        terminal_written
    );

    fclose(trace);
    fclose(completion);
    free(sampling_logits);
    free_sampler(&sampler);
    free(prompt_tokens);
    free(prompt);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
