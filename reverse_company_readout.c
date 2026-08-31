/*
 * Reverse-causal company readout for a frozen llama2.c transformer.
 *
 * This is deliberately not a sequence scorer.  A training observation is a
 * pair of complete, equal-length token companies that differ at one position:
 * the dataset filler and a locally plausible replacement proposed by the
 * frozen model.  The learned result is the relative preference between those
 * two fillers at that one hole.  Preferences at different holes are never
 * added, averaged, or treated as an absolute path reward.
 *
 * The frozen transformer is run on both complete companies while every
 * residual-stream state is retained.  A small reverse-causal scan reads the
 * downstream states that have already incorporated the consequences of the
 * tested filler and carries a message back to its position.  Two controls
 * use the same head width and allocated layout: a left-only head receives no
 * suffix, while an embedding-suffix head receives the future token identities
 * but none of their deeper frozen states.  The executable reports the active
 * coefficient count for each mode; equal allocation must not be mistaken for
 * equal effective input dimension.
 *
 * Evaluation is support-wide even though the current training loss is not:
 * every retained complete-word alternative is instantiated, run through its
 * own frozen continuation, ranked under every readout, decoded, and flushed to
 * JSONL.  This deliberately exposes orderings that a one-negative accuracy
 * can hide.
 */

#define TESTING
#include "run.c"

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>

typedef struct {
    int token_count;
    int target;
    int positive_token;
    int negative_token;
    int candidate_count;
    int *candidate_tokens;
    float *candidate_ar_logits;
    int *positive_tokens;
    int *negative_tokens;
    float *positive_features;
    float *negative_features;
    double ar_margin;
} CompanyPair;

typedef struct {
    int feature_dim;
    int token_dim;
    int hidden_dim;
    size_t parameter_count;
    double *parameters;
} ReverseHead;

typedef struct {
    double *states;
    double *context;
    int capacity_tokens;
    int hidden_dim;
} ScoreCache;

typedef struct {
    int train_count;
    int validation_count;
    int sequence_tokens;
    int min_suffix;
    int top_k;
    int epochs;
    int batch_size;
    int head_dim;
    int shown;
    unsigned long long seed;
    double learning_rate;
    const char *trace_path;
    const char *save_path;
} Options;

typedef struct {
    size_t w_input;
    size_t w_recurrent;
    size_t w_left;
    size_t w_token;
    size_t recurrent_bias;
    size_t context_bias;
    size_t output;
    size_t output_bias;
    size_t count;
} HeadLayout;

typedef struct {
    double loss;
    double accuracy;
    double mean_margin;
    double ar_accuracy;
} Metrics;

typedef struct {
    int support_index;
    int token;
    int observed;
    int selected_negative;
    double ar_score;
    double reverse_score;
    double embedding_score;
    double left_score;
    int ar_rank;
    int reverse_rank;
    int embedding_rank;
    int left_rank;
    char *text;
} CandidateObservation;

typedef enum {
    CANDIDATE_AR = 0,
    CANDIDATE_REVERSE = 1,
    CANDIDATE_EMBEDDING = 2,
    CANDIDATE_LEFT = 3,
} CandidateCoordinate;

typedef enum {
    READOUT_LEFT_ONLY = 0,
    READOUT_EMBEDDING_SUFFIX = 1,
    READOUT_ALL_LAYERS_SUFFIX = 2,
} ReadoutMode;

static _Noreturn void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

static void *checked_calloc(size_t count, size_t size) {
    if (count == 0) count = 1;
    if (size == 0) size = 1;
    if (count != 0 && size > SIZE_MAX / count) fail("allocation overflow");
    void *memory = calloc(count, size);
    if (memory == NULL) fail("allocation failed");
    return memory;
}

static int parse_positive(const char *text, const char *name) {
    errno = 0;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value <= 0 ||
        value > INT_MAX) {
        fprintf(stderr, "%s must be a positive integer\n", name);
        exit(EXIT_FAILURE);
    }
    return (int)value;
}

static unsigned long long parse_seed(const char *text) {
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0) {
        fail("seed must be a positive integer");
    }
    return value;
}

static double parse_rate(const char *text) {
    errno = 0;
    char *end = NULL;
    double value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value) ||
        value <= 0.0) {
        fail("learning rate must be positive and finite");
    }
    return value;
}

static void usage(const char *program) {
    fprintf(
        stderr,
        "usage: %s CHECKPOINT TOKENIZER STORIES [options]\n"
        "  --train N       training stories (default 192)\n"
        "  --validation N  validation stories (default 48)\n"
        "  --tokens N      maximum tokens per story (default 32)\n"
        "  --min-suffix N  required tokens after the tested filler (default 1)\n"
        "  --top-k N       plausible replacement support (default 8)\n"
        "  --epochs N      epochs per head (default 20)\n"
        "  --batch N       minibatch size (default 8)\n"
        "  --head-dim N    reverse message width (default 32)\n"
        "  --rate X        Adam learning rate (default 0.001)\n"
        "  --seed N        deterministic seed (default 42)\n"
        "  --shown N       decoded validation pairs (default 12)\n"
        "  --trace PATH    flushed JSONL validation trace\n"
        "  --save PATH     save the trained reverse head\n",
        program
    );
    exit(EXIT_FAILURE);
}

static Options parse_options(int argc, char **argv) {
    if (argc < 4) usage(argv[0]);
    Options options = {
        .train_count = 192,
        .validation_count = 48,
        .sequence_tokens = 32,
        .min_suffix = 1,
        .top_k = 8,
        .epochs = 20,
        .batch_size = 8,
        .head_dim = 32,
        .shown = 12,
        .seed = 42,
        .learning_rate = 0.001,
        .trace_path = NULL,
        .save_path = NULL,
    };
    for (int index = 4; index < argc; index++) {
        if (index + 1 >= argc) usage(argv[0]);
        const char *flag = argv[index++];
        const char *value = argv[index];
        if (strcmp(flag, "--train") == 0) {
            options.train_count = parse_positive(value, "train count");
        } else if (strcmp(flag, "--validation") == 0) {
            options.validation_count = parse_positive(value, "validation count");
        } else if (strcmp(flag, "--tokens") == 0) {
            options.sequence_tokens = parse_positive(value, "token count");
        } else if (strcmp(flag, "--min-suffix") == 0) {
            options.min_suffix = parse_positive(value, "minimum suffix");
        } else if (strcmp(flag, "--top-k") == 0) {
            options.top_k = parse_positive(value, "top-k");
        } else if (strcmp(flag, "--epochs") == 0) {
            options.epochs = parse_positive(value, "epochs");
        } else if (strcmp(flag, "--batch") == 0) {
            options.batch_size = parse_positive(value, "batch size");
        } else if (strcmp(flag, "--head-dim") == 0) {
            options.head_dim = parse_positive(value, "head dimension");
        } else if (strcmp(flag, "--rate") == 0) {
            options.learning_rate = parse_rate(value);
        } else if (strcmp(flag, "--seed") == 0) {
            options.seed = parse_seed(value);
        } else if (strcmp(flag, "--shown") == 0) {
            options.shown = parse_positive(value, "shown count");
        } else if (strcmp(flag, "--trace") == 0) {
            options.trace_path = value;
        } else if (strcmp(flag, "--save") == 0) {
            options.save_path = value;
        } else {
            usage(argv[0]);
        }
    }
    if (options.sequence_tokens < 8) fail("--tokens must be at least 8");
    if (options.min_suffix >= options.sequence_tokens - 2) {
        fail("--min-suffix leaves no eligible filler position");
    }
    if (options.top_k < 2) fail("--top-k must be at least 2");
    return options;
}

static void copy_unit_rms(float *destination, const float *source, int size) {
    double square_sum = 0.0;
    for (int index = 0; index < size; index++) {
        square_sum += (double)source[index] * source[index];
    }
    double scale = 1.0 / sqrt(square_sum / size + 1e-12);
    for (int index = 0; index < size; index++) {
        destination[index] = (float)(source[index] * scale);
    }
}

/*
 * This is run.c's frozen forward kernel with one observational addition:
 * the residual stream is copied after the embedding and after every layer.
 * The numerical model operations and their ordering are unchanged.
 */
static float *forward_capture(
    Transformer *transformer,
    int token,
    int pos,
    float *features
) {
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    RunState *s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    memcpy(x, w->token_embedding_table + token * dim, (size_t)dim * sizeof(*x));
    copy_unit_rms(features, x, dim);

    for (unsigned long long layer = 0; layer < p->n_layers; layer++) {
        rmsnorm(s->xb, x, w->rms_att_weight + layer * dim, dim);
        int layer_offset = (int)layer * p->seq_len * kv_dim;
        s->k = s->key_cache + layer_offset + pos * kv_dim;
        s->v = s->value_cache + layer_offset + pos * kv_dim;

        matmul(s->q, s->xb, w->wq + layer * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + layer * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + layer * dim * kv_dim, dim, kv_dim);

        for (int index = 0; index < dim; index += 2) {
            int head_dim = index % head_size;
            float frequency = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float angle = pos * frequency;
            float cosine = cosf(angle);
            float sine = sinf(angle);
            int rotations = index < kv_dim ? 2 : 1;
            for (int vector = 0; vector < rotations; vector++) {
                float *values = vector == 0 ? s->q : s->k;
                float first = values[index];
                float second = values[index + 1];
                values[index] = first * cosine - second * sine;
                values[index + 1] = first * sine + second * cosine;
            }
        }

        for (int head = 0; head < p->n_heads; head++) {
            float *query = s->q + head * head_size;
            float *attention = s->att + head * p->seq_len;
            for (int timestep = 0; timestep <= pos; timestep++) {
                float *key = s->key_cache + layer_offset + timestep * kv_dim +
                    (head / kv_mul) * head_size;
                float score = 0.0f;
                for (int index = 0; index < head_size; index++) {
                    score += query[index] * key[index];
                }
                attention[timestep] = score / sqrtf((float)head_size);
            }
            softmax(attention, pos + 1);
            float *attention_output = s->xb + head * head_size;
            memset(attention_output, 0, (size_t)head_size * sizeof(*attention_output));
            for (int timestep = 0; timestep <= pos; timestep++) {
                float *value = s->value_cache + layer_offset + timestep * kv_dim +
                    (head / kv_mul) * head_size;
                float coefficient = attention[timestep];
                for (int index = 0; index < head_size; index++) {
                    attention_output[index] += coefficient * value[index];
                }
            }
        }

        matmul(s->xb2, s->xb, w->wo + layer * dim * dim, dim, dim);
        for (int index = 0; index < dim; index++) x[index] += s->xb2[index];

        rmsnorm(s->xb, x, w->rms_ffn_weight + layer * dim, dim);
        matmul(s->hb, s->xb, w->w1 + layer * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + layer * dim * hidden_dim, dim, hidden_dim);
        for (int index = 0; index < hidden_dim; index++) {
            float value = s->hb[index];
            value *= 1.0f / (1.0f + expf(-value));
            s->hb[index] = value * s->hb2[index];
        }
        matmul(s->xb, s->hb, w->w2 + layer * dim * hidden_dim, hidden_dim, dim);
        for (int index = 0; index < dim; index++) x[index] += s->xb[index];

        copy_unit_rms(features + (layer + 1U) * dim, x, dim);
    }

    rmsnorm(x, x, w->rms_final_weight, dim);
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

static void capture_sequence(
    Transformer *transformer,
    const int *tokens,
    int token_count,
    float *features,
    float *logits
) {
    Config *config = &transformer->config;
    int feature_dim = (config->n_layers + 1) * config->dim;
    int kv_dim = (config->dim * config->n_kv_heads) / config->n_heads;
    memset(
        transformer->state.key_cache,
        0,
        (size_t)config->n_layers * config->seq_len * kv_dim * sizeof(float)
    );
    memset(
        transformer->state.value_cache,
        0,
        (size_t)config->n_layers * config->seq_len * kv_dim * sizeof(float)
    );
    for (int position = 0; position < token_count; position++) {
        float *row_logits = forward_capture(
            transformer,
            tokens[position],
            position,
            features + (size_t)position * feature_dim
        );
        if (logits != NULL) {
            memcpy(
                logits + (size_t)position * config->vocab_size,
                row_logits,
                (size_t)config->vocab_size * sizeof(*logits)
            );
        }
    }
}

static char *read_file(const char *path, size_t *size) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "could not open %s\n", path);
        exit(EXIT_FAILURE);
    }
    if (fseek(file, 0, SEEK_END) != 0) fail("could not seek story file");
    long length = ftell(file);
    if (length < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fail("could not size story file");
    }
    char *data = checked_calloc((size_t)length + 1U, 1);
    if (fread(data, 1, (size_t)length, file) != (size_t)length) {
        fail("could not read story file");
    }
    fclose(file);
    *size = (size_t)length;
    return data;
}

static int candidate_better(const float *logits, int left, int right) {
    if (logits[left] > logits[right]) return 1;
    if (logits[left] < logits[right]) return 0;
    return left < right;
}

static int is_complete_word_piece(const Tokenizer *tokenizer, int token) {
    const unsigned char *piece =
        (const unsigned char *)tokenizer->vocab[token];
    if (piece[0] != ' ' || piece[1] == '\0') return 0;
    for (int index = 1; piece[index] != '\0'; index++) {
        if (!(isalpha(piece[index]) || piece[index] == '\'' ||
              piece[index] == '-')) {
            return 0;
        }
    }
    size_t length = strlen((const char *)piece + 1);
    if (length < 3) {
        static const char *const short_words[] = {
            "a", "I", "am", "an", "as", "at", "be", "by", "do", "go",
            "he", "if", "in", "is", "it", "me", "my", "no", "of", "oh",
            "on", "or", "so", "to", "up", "us", "we",
        };
        int recognized = 0;
        for (size_t index = 0;
             index < sizeof(short_words) / sizeof(short_words[0]); index++) {
            if (strcmp((const char *)piece + 1, short_words[index]) == 0) {
                recognized = 1;
                break;
            }
        }
        if (!recognized) return 0;
    }
    return 1;
}

static int begins_next_word_or_punctuation(
    const Tokenizer *tokenizer,
    int token
) {
    const unsigned char *piece =
        (const unsigned char *)tokenizer->vocab[token];
    if (piece[0] == ' ') return 1;
    return piece[0] == '.' || piece[0] == ',' || piece[0] == '!' ||
        piece[0] == '?' || piece[0] == ';' || piece[0] == ':' ||
        piece[0] == '"';
}

static int collect_word_replacements(
    const Tokenizer *tokenizer,
    const unsigned char *word_lexicon,
    const float *logits,
    int original,
    int top_k,
    int *best
) {
    int count = 0;
    for (int token = 3; token < tokenizer->vocab_size; token++) {
        if (token == original || !word_lexicon[token]) {
            continue;
        }
        int insertion = count;
        while (insertion > 0 && candidate_better(
                logits,
                token,
                best[insertion - 1]
            )) {
            insertion--;
        }
        if (insertion < top_k) {
            int final = count < top_k ? count : top_k - 1;
            for (int index = final; index > insertion; index--) {
                best[index] = best[index - 1];
            }
            best[insertion] = token;
            if (count < top_k) count++;
        }
    }
    return count;
}

static int choose_challenging_word_replacement(
    const Tokenizer *tokenizer,
    const unsigned char *word_lexicon,
    const int *tokens,
    int token_count,
    const float *all_logits,
    int top_k,
    int min_suffix,
    unsigned long long *rng,
    int *target,
    int *replacement,
    double *margin
) {
    int *positions = checked_calloc((size_t)token_count, sizeof(*positions));
    int *replacements = checked_calloc(
        (size_t)token_count,
        sizeof(*replacements)
    );
    double *margins = checked_calloc((size_t)token_count, sizeof(*margins));
    int *best = checked_calloc((size_t)top_k, sizeof(*best));
    int viable_count = 0;
    int fallback_position = -1;
    int fallback_replacement = -1;
    double fallback_margin = DBL_MAX;

    for (int position = 2; position < token_count - min_suffix; position++) {
        int original = tokens[position];
        if (!word_lexicon[original] ||
            !begins_next_word_or_punctuation(tokenizer, tokens[position + 1])) {
            continue;
        }
        const float *logits = all_logits +
            (size_t)(position - 1) * tokenizer->vocab_size;
        int replacement_count = collect_word_replacements(
            tokenizer,
            word_lexicon,
            logits,
            original,
            top_k,
            best
        );
        for (int index = 0; index < replacement_count; index++) {
            int candidate = best[index];
            double candidate_margin =
                (double)logits[original] - logits[candidate];
            if (candidate_margin < fallback_margin) {
                fallback_margin = candidate_margin;
                fallback_position = position;
                fallback_replacement = candidate;
            }
            if (candidate_margin <= 0.0) {
                positions[viable_count] = position;
                replacements[viable_count] = candidate;
                margins[viable_count] = candidate_margin;
                viable_count++;
            }
        }
    }

    int found = 0;
    if (viable_count > 0) {
        int selected = (int)(random_u32(rng) % (unsigned int)viable_count);
        *target = positions[selected];
        *replacement = replacements[selected];
        *margin = margins[selected];
        found = 1;
    } else if (fallback_position >= 0) {
        *target = fallback_position;
        *replacement = fallback_replacement;
        *margin = fallback_margin;
        found = 1;
    }
    free(best);
    free(margins);
    free(replacements);
    free(positions);
    return found;
}

static char *next_story(char **cursor) {
    const char separator[] = "<|endoftext|>";
    while (**cursor == '\n' || **cursor == '\r') (*cursor)++;
    if (**cursor == '\0') return NULL;
    char *start = *cursor;
    char *end = strstr(start, separator);
    if (end == NULL) {
        *cursor = start + strlen(start);
        end = *cursor;
    } else {
        *cursor = end + sizeof(separator) - 1U;
    }
    while (end > start && (end[-1] == '\n' || end[-1] == '\r')) end--;
    size_t length = (size_t)(end - start);
    char *story = checked_calloc(length + 1U, 1);
    memcpy(story, start, length);
    return story;
}

static unsigned char *build_word_lexicon(
    Tokenizer *tokenizer,
    const char *stories
) {
    unsigned int *counts = checked_calloc(
        (size_t)tokenizer->vocab_size,
        sizeof(*counts)
    );
    int initialization_tokens[32];
    int initialization_count = 0;
    char initialization_text[] = "seed";
    encode(
        tokenizer,
        initialization_text,
        0,
        0,
        initialization_tokens,
        &initialization_count
    );
    (void)initialization_count;

    size_t word_capacity = tokenizer->max_token_length + 2U;
    char *word = checked_calloc(word_capacity, 1);
    const unsigned char *cursor = (const unsigned char *)stories;
    while (*cursor != '\0') {
        while (*cursor != '\0' && !isalpha(*cursor)) cursor++;
        const unsigned char *start = cursor;
        while (isalpha(*cursor) || *cursor == '\'' || *cursor == '-') cursor++;
        size_t length = (size_t)(cursor - start);
        if (length == 0 || length + 2U > word_capacity) continue;
        word[0] = ' ';
        memcpy(word + 1, start, length);
        word[length + 1U] = '\0';
        int token = str_lookup(
            word,
            tokenizer->sorted_vocab,
            tokenizer->vocab_size
        );
        if (token >= 3 && is_complete_word_piece(tokenizer, token)) {
            counts[token]++;
        }
    }
    free(word);
    unsigned char *lexicon = checked_calloc(
        (size_t)tokenizer->vocab_size,
        sizeof(*lexicon)
    );
    int recognized = 0;
    for (int token = 3; token < tokenizer->vocab_size; token++) {
        if (counts[token] >= 2U) {
            lexicon[token] = 1;
            recognized++;
        }
    }
    free(counts);
    fprintf(stderr, "corpus complete-word lexicon: %d tokens\n", recognized);
    return lexicon;
}

static void free_pair(CompanyPair *pair) {
    free(pair->candidate_tokens);
    free(pair->candidate_ar_logits);
    free(pair->positive_tokens);
    free(pair->negative_tokens);
    free(pair->positive_features);
    free(pair->negative_features);
    memset(pair, 0, sizeof(*pair));
}

static CompanyPair *make_dataset(
    Transformer *transformer,
    Tokenizer *tokenizer,
    char **story_cursor,
    const unsigned char *word_lexicon,
    int pair_count,
    int sequence_tokens,
    int min_suffix,
    int top_k,
    unsigned long long *rng,
    const char *split
) {
    Config *config = &transformer->config;
    int feature_dim = (config->n_layers + 1) * config->dim;
    CompanyPair *pairs = checked_calloc((size_t)pair_count, sizeof(*pairs));
    float *logits = checked_calloc(
        (size_t)sequence_tokens * config->vocab_size,
        sizeof(*logits)
    );

    int produced = 0;
    while (produced < pair_count) {
        char *story = next_story(story_cursor);
        if (story == NULL) fail("story file ended before dataset was complete");
        size_t capacity = strlen(story) + 3U;
        int *encoded = checked_calloc(capacity, sizeof(*encoded));
        int encoded_count = 0;
        encode(tokenizer, story, 1, 0, encoded, &encoded_count);
        free(story);
        if (encoded_count < 8) {
            free(encoded);
            continue;
        }

        CompanyPair *pair = &pairs[produced];
        pair->token_count = encoded_count < sequence_tokens ?
            encoded_count : sequence_tokens;
        pair->positive_tokens = checked_calloc(
            (size_t)pair->token_count,
            sizeof(*pair->positive_tokens)
        );
        pair->negative_tokens = checked_calloc(
            (size_t)pair->token_count,
            sizeof(*pair->negative_tokens)
        );
        pair->positive_features = checked_calloc(
            (size_t)pair->token_count * feature_dim,
            sizeof(*pair->positive_features)
        );
        pair->negative_features = checked_calloc(
            (size_t)pair->token_count * feature_dim,
            sizeof(*pair->negative_features)
        );
        memcpy(
            pair->positive_tokens,
            encoded,
            (size_t)pair->token_count * sizeof(*encoded)
        );
        memcpy(
            pair->negative_tokens,
            encoded,
            (size_t)pair->token_count * sizeof(*encoded)
        );
        free(encoded);

        capture_sequence(
            transformer,
            pair->positive_tokens,
            pair->token_count,
            pair->positive_features,
            logits
        );

        if (!choose_challenging_word_replacement(
                tokenizer,
                word_lexicon,
                pair->positive_tokens,
                pair->token_count,
                logits,
                top_k,
                min_suffix,
                rng,
                &pair->target,
                &pair->negative_token,
                &pair->ar_margin
            )) {
            free_pair(pair);
            continue;
        }
        pair->positive_token = pair->positive_tokens[pair->target];
        const float *target_logits = logits +
            (size_t)(pair->target - 1) * config->vocab_size;
        int *candidate_tokens = checked_calloc(
            (size_t)top_k,
            sizeof(*candidate_tokens)
        );
        int replacement_count = collect_word_replacements(
            tokenizer,
            word_lexicon,
            target_logits,
            pair->positive_token,
            top_k,
            candidate_tokens
        );
        pair->candidate_count = replacement_count + 1;
        pair->candidate_tokens = checked_calloc(
            (size_t)pair->candidate_count,
            sizeof(*pair->candidate_tokens)
        );
        pair->candidate_ar_logits = checked_calloc(
            (size_t)pair->candidate_count,
            sizeof(*pair->candidate_ar_logits)
        );
        pair->candidate_tokens[0] = pair->positive_token;
        pair->candidate_ar_logits[0] = target_logits[pair->positive_token];
        int retained_negative = 0;
        for (int index = 0; index < replacement_count; index++) {
            pair->candidate_tokens[index + 1] = candidate_tokens[index];
            pair->candidate_ar_logits[index + 1] =
                target_logits[candidate_tokens[index]];
            if (candidate_tokens[index] == pair->negative_token) {
                retained_negative = 1;
            }
        }
        free(candidate_tokens);
        if (!retained_negative) fail("selected negative missing from support");
        pair->negative_tokens[pair->target] = pair->negative_token;

        capture_sequence(
            transformer,
            pair->negative_tokens,
            pair->token_count,
            pair->negative_features,
            NULL
        );
        produced++;
        if (produced % 16 == 0 || produced == pair_count) {
            fprintf(
                stderr,
                "%s frozen activation pairs: %d/%d\n",
                split,
                produced,
                pair_count
            );
        }
    }
    free(logits);
    return pairs;
}

static HeadLayout head_layout(int feature_dim, int token_dim, int hidden_dim) {
    HeadLayout layout;
    size_t cursor = 0;
    layout.w_input = cursor;
    cursor += (size_t)hidden_dim * feature_dim;
    layout.w_recurrent = cursor;
    cursor += (size_t)hidden_dim * hidden_dim;
    layout.w_left = cursor;
    cursor += (size_t)hidden_dim * feature_dim;
    layout.w_token = cursor;
    cursor += (size_t)hidden_dim * token_dim;
    layout.recurrent_bias = cursor;
    cursor += (size_t)hidden_dim;
    layout.context_bias = cursor;
    cursor += (size_t)hidden_dim;
    layout.output = cursor;
    cursor += (size_t)hidden_dim;
    layout.output_bias = cursor++;
    layout.count = cursor;
    return layout;
}

static size_t active_parameter_count(
    const ReverseHead *head,
    ReadoutMode mode
) {
    size_t hidden = (size_t)head->hidden_dim;
    size_t feature = (size_t)head->feature_dim;
    size_t token = (size_t)head->token_dim;
    size_t count = hidden * feature; /* left-context matrix */
    count += hidden * token;         /* tested-token matrix */
    count += hidden;                 /* context bias */
    count += hidden;                 /* output vector */
    count += 1U;                     /* output bias */
    if (mode != READOUT_LEFT_ONLY) {
        size_t suffix = mode == READOUT_ALL_LAYERS_SUFFIX ? feature : token;
        count += hidden * suffix;    /* suffix-input matrix */
        count += hidden * hidden;    /* reverse recurrence */
        count += hidden;             /* recurrence bias */
    }
    return count;
}

static double symmetric_random(unsigned long long *rng) {
    return 2.0 * random_f32(rng) - 1.0;
}

static ReverseHead make_head(
    int feature_dim,
    int token_dim,
    int hidden_dim,
    unsigned long long *rng
) {
    ReverseHead head = {
        .feature_dim = feature_dim,
        .token_dim = token_dim,
        .hidden_dim = hidden_dim,
    };
    HeadLayout layout = head_layout(feature_dim, token_dim, hidden_dim);
    head.parameter_count = layout.count;
    head.parameters = checked_calloc(layout.count, sizeof(*head.parameters));

    double input_scale = sqrt(6.0 / (feature_dim + hidden_dim));
    double recurrent_scale = sqrt(3.0 / hidden_dim);
    double token_scale = sqrt(6.0 / (token_dim + hidden_dim));
    for (size_t index = layout.w_input; index < layout.w_recurrent; index++) {
        head.parameters[index] = input_scale * symmetric_random(rng);
    }
    for (size_t index = layout.w_recurrent; index < layout.w_left; index++) {
        head.parameters[index] = recurrent_scale * symmetric_random(rng);
    }
    for (size_t index = layout.w_left; index < layout.w_token; index++) {
        head.parameters[index] = input_scale * symmetric_random(rng);
    }
    for (size_t index = layout.w_token; index < layout.recurrent_bias; index++) {
        head.parameters[index] = token_scale * symmetric_random(rng);
    }
    for (int index = 0; index < hidden_dim; index++) {
        head.parameters[layout.output + (size_t)index] =
            symmetric_random(rng) / sqrt((double)hidden_dim);
    }
    return head;
}

static void free_head(ReverseHead *head) {
    free(head->parameters);
    memset(head, 0, sizeof(*head));
}

typedef struct {
    unsigned char magic[8];
    uint32_t version;
    uint32_t model_dim;
    uint32_t model_layers;
    uint32_t feature_dim;
    uint32_t token_dim;
    uint32_t hidden_dim;
    uint64_t parameter_count;
} SavedHeadHeader;

static void save_head(
    const ReverseHead *head,
    const Transformer *transformer,
    const char *path
) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) fail("could not create reverse head file");
    SavedHeadHeader header = {
        .magic = {'R', 'E', 'V', 'C', 'O', 'M', 'P', '1'},
        .version = 1,
        .model_dim = (uint32_t)transformer->config.dim,
        .model_layers = (uint32_t)transformer->config.n_layers,
        .feature_dim = (uint32_t)head->feature_dim,
        .token_dim = (uint32_t)head->token_dim,
        .hidden_dim = (uint32_t)head->hidden_dim,
        .parameter_count = (uint64_t)head->parameter_count,
    };
    if (fwrite(&header, sizeof(header), 1, file) != 1 ||
        fwrite(
            head->parameters,
            sizeof(*head->parameters),
            head->parameter_count,
            file
        ) != head->parameter_count) {
        fclose(file);
        fail("could not write reverse head file");
    }
    if (fclose(file) != 0) fail("could not close reverse head file");
}

static ScoreCache make_cache(int token_capacity, int hidden_dim) {
    ScoreCache cache = {
        .states = checked_calloc(
            (size_t)(token_capacity + 1) * hidden_dim,
            sizeof(*cache.states)
        ),
        .context = checked_calloc((size_t)hidden_dim, sizeof(*cache.context)),
        .capacity_tokens = token_capacity,
        .hidden_dim = hidden_dim,
    };
    return cache;
}

static void free_cache(ScoreCache *cache) {
    free(cache->states);
    free(cache->context);
    memset(cache, 0, sizeof(*cache));
}

static double score_company(
    const ReverseHead *head,
    const Transformer *transformer,
    const float *features,
    const int *tokens,
    int token_count,
    int target,
    ReadoutMode mode,
    ScoreCache *cache
) {
    if (token_count > cache->capacity_tokens || target <= 0 ||
        target >= token_count - 1) {
        fail("invalid score cache or target");
    }
    HeadLayout layout = head_layout(
        head->feature_dim,
        head->token_dim,
        head->hidden_dim
    );
    const double *parameters = head->parameters;
    int hidden_dim = head->hidden_dim;
    int feature_dim = head->feature_dim;
    int token_dim = head->token_dim;
    int suffix_feature_dim = mode == READOUT_ALL_LAYERS_SUFFIX ?
        feature_dim : mode == READOUT_EMBEDDING_SUFFIX ? token_dim : 0;
    memset(
        cache->states,
        0,
        (size_t)(token_count + 1) * hidden_dim * sizeof(*cache->states)
    );

    if (suffix_feature_dim > 0) {
        for (int position = token_count - 1; position > target; position--) {
            const float *feature = features + (size_t)position * feature_dim;
            const double *next = cache->states + (size_t)(position + 1) * hidden_dim;
            double *state = cache->states + (size_t)position * hidden_dim;
            for (int output = 0; output < hidden_dim; output++) {
                double value = parameters[layout.recurrent_bias + (size_t)output];
                const double *input_row = parameters + layout.w_input +
                    (size_t)output * feature_dim;
                const double *recurrent_row = parameters + layout.w_recurrent +
                    (size_t)output * hidden_dim;
                for (int input = 0; input < suffix_feature_dim; input++) {
                    value += input_row[input] * feature[input];
                }
                for (int input = 0; input < hidden_dim; input++) {
                    value += recurrent_row[input] * next[input];
                }
                state[output] = tanh(value);
            }
        }
    }

    const float *left = features + (size_t)(target - 1) * feature_dim;
    const float *embedding = transformer->weights.token_embedding_table +
        (size_t)tokens[target] * token_dim;
    const double *suffix = cache->states + (size_t)(target + 1) * hidden_dim;
    for (int output = 0; output < hidden_dim; output++) {
        double value = parameters[layout.context_bias + (size_t)output];
        const double *left_row = parameters + layout.w_left +
            (size_t)output * feature_dim;
        const double *token_row = parameters + layout.w_token +
            (size_t)output * token_dim;
        for (int input = 0; input < feature_dim; input++) {
            value += left_row[input] * left[input];
        }
        for (int input = 0; input < token_dim; input++) {
            value += token_row[input] * embedding[input];
        }
        if (suffix_feature_dim > 0) value += suffix[output];
        cache->context[output] = tanh(value);
    }

    double score = parameters[layout.output_bias];
    for (int index = 0; index < hidden_dim; index++) {
        score += parameters[layout.output + (size_t)index] *
            cache->context[index];
    }
    return score;
}

static void backward_company(
    const ReverseHead *head,
    const Transformer *transformer,
    const float *features,
    const int *tokens,
    int token_count,
    int target,
    ReadoutMode mode,
    const ScoreCache *cache,
    double score_gradient,
    double *gradient
) {
    HeadLayout layout = head_layout(
        head->feature_dim,
        head->token_dim,
        head->hidden_dim
    );
    const double *parameters = head->parameters;
    int hidden_dim = head->hidden_dim;
    int feature_dim = head->feature_dim;
    int token_dim = head->token_dim;
    int suffix_feature_dim = mode == READOUT_ALL_LAYERS_SUFFIX ?
        feature_dim : mode == READOUT_EMBEDDING_SUFFIX ? token_dim : 0;
    const float *left = features + (size_t)(target - 1) * feature_dim;
    const float *embedding = transformer->weights.token_embedding_table +
        (size_t)tokens[target] * token_dim;
    double *state_gradient = checked_calloc(
        (size_t)(token_count + 1) * hidden_dim,
        sizeof(*state_gradient)
    );

    gradient[layout.output_bias] += score_gradient;
    double *context_pre_gradient = checked_calloc(
        (size_t)hidden_dim,
        sizeof(*context_pre_gradient)
    );
    for (int output = 0; output < hidden_dim; output++) {
        gradient[layout.output + (size_t)output] +=
            score_gradient * cache->context[output];
        double context_gradient = score_gradient *
            parameters[layout.output + (size_t)output];
        double pre_gradient = context_gradient *
            (1.0 - cache->context[output] * cache->context[output]);
        context_pre_gradient[output] = pre_gradient;
        gradient[layout.context_bias + (size_t)output] += pre_gradient;
        double *left_row_gradient = gradient + layout.w_left +
            (size_t)output * feature_dim;
        double *token_row_gradient = gradient + layout.w_token +
            (size_t)output * token_dim;
        for (int input = 0; input < feature_dim; input++) {
            left_row_gradient[input] += pre_gradient * left[input];
        }
        for (int input = 0; input < token_dim; input++) {
            token_row_gradient[input] += pre_gradient * embedding[input];
        }
        if (suffix_feature_dim > 0) {
            state_gradient[(size_t)(target + 1) * hidden_dim + output] +=
                pre_gradient;
        }
    }
    free(context_pre_gradient);

    if (suffix_feature_dim > 0) {
        for (int position = target + 1; position < token_count; position++) {
            const float *feature = features + (size_t)position * feature_dim;
            const double *state = cache->states + (size_t)position * hidden_dim;
            const double *next = cache->states +
                (size_t)(position + 1) * hidden_dim;
            double *current_gradient = state_gradient +
                (size_t)position * hidden_dim;
            double *next_gradient = state_gradient +
                (size_t)(position + 1) * hidden_dim;
            for (int output = 0; output < hidden_dim; output++) {
                double pre_gradient = current_gradient[output] *
                    (1.0 - state[output] * state[output]);
                gradient[layout.recurrent_bias + (size_t)output] += pre_gradient;
                double *input_row_gradient = gradient + layout.w_input +
                    (size_t)output * feature_dim;
                double *recurrent_row_gradient = gradient + layout.w_recurrent +
                    (size_t)output * hidden_dim;
                const double *recurrent_row = parameters + layout.w_recurrent +
                    (size_t)output * hidden_dim;
                for (int input = 0; input < suffix_feature_dim; input++) {
                    input_row_gradient[input] += pre_gradient * feature[input];
                }
                for (int input = 0; input < hidden_dim; input++) {
                    recurrent_row_gradient[input] += pre_gradient * next[input];
                    next_gradient[input] += pre_gradient * recurrent_row[input];
                }
            }
        }
    }
    free(state_gradient);
}

static double logistic_pair_loss(double margin, double *margin_gradient) {
    if (margin >= 0.0) {
        double exponential = exp(-margin);
        if (margin_gradient != NULL) {
            *margin_gradient = -exponential / (1.0 + exponential);
        }
        return log1p(exponential);
    }
    double exponential = exp(margin);
    if (margin_gradient != NULL) {
        *margin_gradient = -1.0 / (1.0 + exponential);
    }
    return -margin + log1p(exponential);
}

static Metrics evaluate(
    const ReverseHead *head,
    const Transformer *transformer,
    const CompanyPair *pairs,
    int pair_count,
    ReadoutMode mode,
    int token_capacity
) {
    Metrics metrics = {0};
    ScoreCache positive_cache = make_cache(token_capacity, head->hidden_dim);
    ScoreCache negative_cache = make_cache(token_capacity, head->hidden_dim);
    for (int index = 0; index < pair_count; index++) {
        const CompanyPair *pair = &pairs[index];
        double positive = score_company(
            head,
            transformer,
            pair->positive_features,
            pair->positive_tokens,
            pair->token_count,
            pair->target,
            mode,
            &positive_cache
        );
        double negative = score_company(
            head,
            transformer,
            pair->negative_features,
            pair->negative_tokens,
            pair->token_count,
            pair->target,
            mode,
            &negative_cache
        );
        double margin = positive - negative;
        metrics.loss += logistic_pair_loss(margin, NULL);
        metrics.mean_margin += margin;
        metrics.accuracy += margin > 0.0;
        metrics.ar_accuracy += pair->ar_margin > 0.0;
    }
    metrics.loss /= pair_count;
    metrics.mean_margin /= pair_count;
    metrics.accuracy /= pair_count;
    metrics.ar_accuracy /= pair_count;
    free_cache(&negative_cache);
    free_cache(&positive_cache);
    return metrics;
}

static void shuffle_indices(int *indices, int count, unsigned long long *rng) {
    for (int index = count - 1; index > 0; index--) {
        int other = (int)(random_u32(rng) % (unsigned int)(index + 1));
        int temporary = indices[index];
        indices[index] = indices[other];
        indices[other] = temporary;
    }
}

static void train_head(
    ReverseHead *head,
    const Transformer *transformer,
    const CompanyPair *training,
    int training_count,
    const CompanyPair *validation,
    int validation_count,
    int token_capacity,
    ReadoutMode mode,
    const Options *options,
    unsigned long long *rng,
    const char *name
) {
    double *gradient = checked_calloc(
        head->parameter_count,
        sizeof(*gradient)
    );
    double *first_moment = checked_calloc(
        head->parameter_count,
        sizeof(*first_moment)
    );
    double *second_moment = checked_calloc(
        head->parameter_count,
        sizeof(*second_moment)
    );
    double *best = checked_calloc(head->parameter_count, sizeof(*best));
    int *indices = checked_calloc((size_t)training_count, sizeof(*indices));
    for (int index = 0; index < training_count; index++) indices[index] = index;
    ScoreCache positive_cache = make_cache(token_capacity, head->hidden_dim);
    ScoreCache negative_cache = make_cache(token_capacity, head->hidden_dim);
    double best_loss = DBL_MAX;
    unsigned long long update = 0;

    for (int epoch = 0; epoch < options->epochs; epoch++) {
        shuffle_indices(indices, training_count, rng);
        for (int start = 0; start < training_count; start += options->batch_size) {
            int end = start + options->batch_size;
            if (end > training_count) end = training_count;
            int batch_count = end - start;
            memset(gradient, 0, head->parameter_count * sizeof(*gradient));

            for (int offset = start; offset < end; offset++) {
                const CompanyPair *pair = &training[indices[offset]];
                double positive = score_company(
                    head,
                    transformer,
                    pair->positive_features,
                    pair->positive_tokens,
                    pair->token_count,
                    pair->target,
                    mode,
                    &positive_cache
                );
                double negative = score_company(
                    head,
                    transformer,
                    pair->negative_features,
                    pair->negative_tokens,
                    pair->token_count,
                    pair->target,
                    mode,
                    &negative_cache
                );
                double margin_gradient = 0.0;
                (void)logistic_pair_loss(positive - negative, &margin_gradient);
                backward_company(
                    head,
                    transformer,
                    pair->positive_features,
                    pair->positive_tokens,
                    pair->token_count,
                    pair->target,
                    mode,
                    &positive_cache,
                    margin_gradient,
                    gradient
                );
                backward_company(
                    head,
                    transformer,
                    pair->negative_features,
                    pair->negative_tokens,
                    pair->token_count,
                    pair->target,
                    mode,
                    &negative_cache,
                    -margin_gradient,
                    gradient
                );
            }

            double square_norm = 0.0;
            for (size_t parameter = 0; parameter < head->parameter_count; parameter++) {
                gradient[parameter] /= batch_count;
                square_norm += gradient[parameter] * gradient[parameter];
            }
            double gradient_scale = square_norm > 25.0 ?
                5.0 / sqrt(square_norm) : 1.0;
            update++;
            double first_correction = 1.0 - pow(0.9, (double)update);
            double second_correction = 1.0 - pow(0.999, (double)update);
            for (size_t parameter = 0; parameter < head->parameter_count; parameter++) {
                double value = gradient[parameter] * gradient_scale;
                first_moment[parameter] = 0.9 * first_moment[parameter] + 0.1 * value;
                second_moment[parameter] =
                    0.999 * second_moment[parameter] + 0.001 * value * value;
                double first_hat = first_moment[parameter] / first_correction;
                double second_hat = second_moment[parameter] / second_correction;
                head->parameters[parameter] -= options->learning_rate *
                    first_hat / (sqrt(second_hat) + 1e-8);
            }
        }

        Metrics train = evaluate(
            head,
            transformer,
            training,
            training_count,
            mode,
            token_capacity
        );
        Metrics valid = evaluate(
            head,
            transformer,
            validation,
            validation_count,
            mode,
            token_capacity
        );
        if (valid.loss < best_loss) {
            best_loss = valid.loss;
            memcpy(best, head->parameters, head->parameter_count * sizeof(*best));
        }
        fprintf(
            stderr,
            "%s epoch=%d train_loss=%.6f train_accuracy=%.4f "
            "validation_loss=%.6f validation_accuracy=%.4f "
            "validation_margin=%.6f\n",
            name,
            epoch + 1,
            train.loss,
            train.accuracy,
            valid.loss,
            valid.accuracy,
            valid.mean_margin
        );
    }
    memcpy(head->parameters, best, head->parameter_count * sizeof(*best));
    free_cache(&negative_cache);
    free_cache(&positive_cache);
    free(indices);
    free(best);
    free(second_moment);
    free(first_moment);
    free(gradient);
}

static void json_text(FILE *file, const char *text) {
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
        } else if (byte >= 0x20 && byte != 0x7f) {
            fputc(byte, file);
        } else {
            fprintf(file, "\\u%04x", byte);
        }
    }
    fputc('"', file);
}

static char *decode_tokens(
    Tokenizer *tokenizer,
    const int *tokens,
    int token_count
) {
    size_t capacity = (size_t)token_count *
        (tokenizer->max_token_length + 1U) + 1U;
    char *text = checked_calloc(capacity, 1);
    size_t length = 0;
    for (int index = 1; index < token_count; index++) {
        const char *piece = decode(tokenizer, tokens[index - 1], tokens[index]);
        size_t piece_length = strlen(piece);
        if (length + piece_length + 1U > capacity) {
            capacity = (length + piece_length + 1U) * 2U;
            char *resized = realloc(text, capacity);
            if (resized == NULL) fail("decoded text allocation failed");
            text = resized;
        }
        memcpy(text + length, piece, piece_length);
        length += piece_length;
        text[length] = '\0';
    }
    return text;
}

static double candidate_coordinate(
    const CandidateObservation *candidate,
    CandidateCoordinate coordinate
) {
    if (coordinate == CANDIDATE_AR) return candidate->ar_score;
    if (coordinate == CANDIDATE_REVERSE) return candidate->reverse_score;
    if (coordinate == CANDIDATE_EMBEDDING) {
        return candidate->embedding_score;
    }
    return candidate->left_score;
}

static int candidate_rank(
    const CandidateObservation *candidates,
    int candidate_count,
    int candidate_index,
    CandidateCoordinate coordinate
) {
    const CandidateObservation *candidate = &candidates[candidate_index];
    double score = candidate_coordinate(candidate, coordinate);
    int rank = 1;
    for (int index = 0; index < candidate_count; index++) {
        if (index == candidate_index) continue;
        double other = candidate_coordinate(&candidates[index], coordinate);
        if (other > score ||
            (other == score && candidates[index].token < candidate->token)) {
            rank++;
        }
    }
    return rank;
}

static void report_candidate_support(
    const ReverseHead *full,
    const ReverseHead *embedding,
    const ReverseHead *left,
    Transformer *transformer,
    Tokenizer *tokenizer,
    const CompanyPair *pair,
    int pair_index,
    int display,
    ScoreCache *full_cache,
    ScoreCache *embedding_cache,
    ScoreCache *left_cache,
    int *scratch_tokens,
    float *scratch_features,
    FILE *trace
) {
    int candidate_count = pair->candidate_count;
    CandidateObservation *candidates = checked_calloc(
        (size_t)candidate_count,
        sizeof(*candidates)
    );

    for (int index = 0; index < candidate_count; index++) {
        CandidateObservation *candidate = &candidates[index];
        candidate->support_index = index;
        candidate->token = pair->candidate_tokens[index];
        candidate->observed = candidate->token == pair->positive_token;
        candidate->selected_negative = candidate->token == pair->negative_token;
        candidate->ar_score = pair->candidate_ar_logits[index];

        const int *tokens = NULL;
        const float *features = NULL;
        if (candidate->observed) {
            tokens = pair->positive_tokens;
            features = pair->positive_features;
        } else if (candidate->selected_negative) {
            tokens = pair->negative_tokens;
            features = pair->negative_features;
        } else {
            memcpy(
                scratch_tokens,
                pair->positive_tokens,
                (size_t)pair->token_count * sizeof(*scratch_tokens)
            );
            scratch_tokens[pair->target] = candidate->token;
            capture_sequence(
                transformer,
                scratch_tokens,
                pair->token_count,
                scratch_features,
                NULL
            );
            tokens = scratch_tokens;
            features = scratch_features;
        }

        candidate->reverse_score = score_company(
            full,
            transformer,
            features,
            tokens,
            pair->token_count,
            pair->target,
            READOUT_ALL_LAYERS_SUFFIX,
            full_cache
        );
        candidate->embedding_score = score_company(
            embedding,
            transformer,
            features,
            tokens,
            pair->token_count,
            pair->target,
            READOUT_EMBEDDING_SUFFIX,
            embedding_cache
        );
        candidate->left_score = score_company(
            left,
            transformer,
            features,
            tokens,
            pair->token_count,
            pair->target,
            READOUT_LEFT_ONLY,
            left_cache
        );
        candidate->text = decode_tokens(
            tokenizer,
            tokens,
            pair->token_count
        );
    }

    for (int index = 0; index < candidate_count; index++) {
        candidates[index].ar_rank = candidate_rank(
            candidates,
            candidate_count,
            index,
            CANDIDATE_AR
        );
        candidates[index].reverse_rank = candidate_rank(
            candidates,
            candidate_count,
            index,
            CANDIDATE_REVERSE
        );
        candidates[index].embedding_rank = candidate_rank(
            candidates,
            candidate_count,
            index,
            CANDIDATE_EMBEDDING
        );
        candidates[index].left_rank = candidate_rank(
            candidates,
            candidate_count,
            index,
            CANDIDATE_LEFT
        );
    }

    const CandidateObservation *observed = &candidates[0];
    if (display) {
        printf(
            "  candidate_support=%d observed_ranks="
            "reverse:%d embedding:%d left:%d ar:%d\n",
            candidate_count,
            observed->reverse_rank,
            observed->embedding_rank,
            observed->left_rank,
            observed->ar_rank
        );
        for (int rank = 1; rank <= candidate_count; rank++) {
            for (int index = 0; index < candidate_count; index++) {
                const CandidateObservation *candidate = &candidates[index];
                if (candidate->reverse_rank != rank) continue;
                const char *piece = tokenizer->vocab[candidate->token];
                printf(
                    "    reverse_rank=%d piece=\"%s\" token=%d "
                    "reverse=%.9f delta=%.9f embedding=%.9f "
                    "left=%.9f ar=%.9f ranks[e:%d l:%d ar:%d]%s%s\n",
                    candidate->reverse_rank,
                    piece,
                    candidate->token,
                    candidate->reverse_score,
                    candidate->reverse_score - observed->reverse_score,
                    candidate->embedding_score,
                    candidate->left_score,
                    candidate->ar_score,
                    candidate->embedding_rank,
                    candidate->left_rank,
                    candidate->ar_rank,
                    candidate->observed ? " observed" : "",
                    candidate->selected_negative ? " selected-negative" : ""
                );
                printf("      text: %s\n", candidate->text);
            }
        }
    }

    if (trace != NULL) {
        for (int rank = 1; rank <= candidate_count; rank++) {
            for (int index = 0; index < candidate_count; index++) {
                const CandidateObservation *candidate = &candidates[index];
                if (candidate->reverse_rank != rank) continue;
                fprintf(
                    trace,
                    "{\"event\":\"company_candidate\",\"pair\":%d,"
                    "\"support_index\":%d,\"candidate_count\":%d,"
                    "\"position\":%d,\"token_count\":%d,"
                    "\"suffix_tokens\":%d,\"token\":%d,\"piece\":",
                    pair_index,
                    candidate->support_index,
                    candidate_count,
                    pair->target,
                    pair->token_count,
                    pair->token_count - pair->target - 1,
                    candidate->token
                );
                json_text(trace, tokenizer->vocab[candidate->token]);
                fprintf(
                    trace,
                    ",\"observed\":%s,\"selected_negative\":%s,"
                    "\"reverse_rank\":%d,\"reverse_score\":%.17g,"
                    "\"reverse_delta_from_observed\":%.17g,"
                    "\"embedding_rank\":%d,\"embedding_score\":%.17g,"
                    "\"embedding_delta_from_observed\":%.17g,"
                    "\"left_rank\":%d,\"left_score\":%.17g,"
                    "\"left_delta_from_observed\":%.17g,"
                    "\"ar_rank\":%d,\"ar_score\":%.17g,"
                    "\"ar_delta_from_observed\":%.17g,\"text\":",
                    candidate->observed ? "true" : "false",
                    candidate->selected_negative ? "true" : "false",
                    candidate->reverse_rank,
                    candidate->reverse_score,
                    candidate->reverse_score - observed->reverse_score,
                    candidate->embedding_rank,
                    candidate->embedding_score,
                    candidate->embedding_score - observed->embedding_score,
                    candidate->left_rank,
                    candidate->left_score,
                    candidate->left_score - observed->left_score,
                    candidate->ar_rank,
                    candidate->ar_score,
                    candidate->ar_score - observed->ar_score
                );
                json_text(trace, candidate->text);
                fputs("}\n", trace);
                fflush(trace);
            }
        }
    }

    for (int index = 0; index < candidate_count; index++) {
        free(candidates[index].text);
    }
    free(candidates);
}

static void report_pairs(
    const ReverseHead *full,
    const ReverseHead *embedding,
    const ReverseHead *left,
    Transformer *transformer,
    Tokenizer *tokenizer,
    const CompanyPair *pairs,
    int pair_count,
    int shown,
    int token_capacity,
    FILE *trace
) {
    ScoreCache full_positive = make_cache(token_capacity, full->hidden_dim);
    ScoreCache full_negative = make_cache(token_capacity, full->hidden_dim);
    ScoreCache embedding_positive = make_cache(
        token_capacity,
        embedding->hidden_dim
    );
    ScoreCache embedding_negative = make_cache(
        token_capacity,
        embedding->hidden_dim
    );
    ScoreCache left_positive = make_cache(token_capacity, left->hidden_dim);
    ScoreCache left_negative = make_cache(token_capacity, left->hidden_dim);
    int display_count = shown < pair_count ? shown : pair_count;
    int *scratch_tokens = checked_calloc(
        (size_t)token_capacity,
        sizeof(*scratch_tokens)
    );
    float *scratch_features = checked_calloc(
        (size_t)token_capacity * full->feature_dim,
        sizeof(*scratch_features)
    );

    for (int index = 0; index < pair_count; index++) {
        const CompanyPair *pair = &pairs[index];
        double full_positive_score = score_company(
            full,
            transformer,
            pair->positive_features,
            pair->positive_tokens,
            pair->token_count,
            pair->target,
            READOUT_ALL_LAYERS_SUFFIX,
            &full_positive
        );
        double full_negative_score = score_company(
            full,
            transformer,
            pair->negative_features,
            pair->negative_tokens,
            pair->token_count,
            pair->target,
            READOUT_ALL_LAYERS_SUFFIX,
            &full_negative
        );
        double embedding_positive_score = score_company(
            embedding,
            transformer,
            pair->positive_features,
            pair->positive_tokens,
            pair->token_count,
            pair->target,
            READOUT_EMBEDDING_SUFFIX,
            &embedding_positive
        );
        double embedding_negative_score = score_company(
            embedding,
            transformer,
            pair->negative_features,
            pair->negative_tokens,
            pair->token_count,
            pair->target,
            READOUT_EMBEDDING_SUFFIX,
            &embedding_negative
        );
        double left_positive_score = score_company(
            left,
            transformer,
            pair->positive_features,
            pair->positive_tokens,
            pair->token_count,
            pair->target,
            READOUT_LEFT_ONLY,
            &left_positive
        );
        double left_negative_score = score_company(
            left,
            transformer,
            pair->negative_features,
            pair->negative_tokens,
            pair->token_count,
            pair->target,
            READOUT_LEFT_ONLY,
            &left_negative
        );
        char *positive_text = decode_tokens(
            tokenizer,
            pair->positive_tokens,
            pair->token_count
        );
        char *negative_text = decode_tokens(
            tokenizer,
            pair->negative_tokens,
            pair->token_count
        );
        const char *positive_piece = tokenizer->vocab[pair->positive_token];
        const char *negative_piece = tokenizer->vocab[pair->negative_token];

        if (index < display_count) {
            printf(
                "pair=%d position=%d tokens=%d suffix=%d "
                "positive=\"%s\" negative=\"%s\" "
                "reverse_margin=%.6f embedding_margin=%.6f "
                "left_margin=%.6f ar_margin=%.6f\n",
                index,
                pair->target,
                pair->token_count,
                pair->token_count - pair->target - 1,
                positive_piece,
                negative_piece,
                full_positive_score - full_negative_score,
                embedding_positive_score - embedding_negative_score,
                left_positive_score - left_negative_score,
                pair->ar_margin
            );
            printf("  coherent: %s\n", positive_text);
            printf("  corrupt:  %s\n", negative_text);
        }
        if (trace != NULL) {
            fprintf(
                trace,
                "{\"event\":\"company_pair\",\"index\":%d,\"position\":%d,"
                "\"token_count\":%d,\"suffix_tokens\":%d,"
                "\"positive_token\":%d,\"negative_token\":%d,"
                "\"positive_piece\":",
                index,
                pair->target,
                pair->token_count,
                pair->token_count - pair->target - 1,
                pair->positive_token,
                pair->negative_token
            );
            json_text(trace, positive_piece);
            fputs(",\"negative_piece\":", trace);
            json_text(trace, negative_piece);
            fprintf(
                trace,
                ",\"reverse_positive\":%.17g,\"reverse_negative\":%.17g,"
                "\"reverse_margin\":%.17g,"
                "\"embedding_positive\":%.17g,"
                "\"embedding_negative\":%.17g,"
                "\"embedding_margin\":%.17g,\"left_positive\":%.17g,"
                "\"left_negative\":%.17g,\"left_margin\":%.17g,"
                "\"ar_margin\":%.17g,\"coherent_text\":",
                full_positive_score,
                full_negative_score,
                full_positive_score - full_negative_score,
                embedding_positive_score,
                embedding_negative_score,
                embedding_positive_score - embedding_negative_score,
                left_positive_score,
                left_negative_score,
                left_positive_score - left_negative_score,
                pair->ar_margin
            );
            json_text(trace, positive_text);
            fputs(",\"corrupt_text\":", trace);
            json_text(trace, negative_text);
            fputs("}\n", trace);
            fflush(trace);
        }
        report_candidate_support(
            full,
            embedding,
            left,
            transformer,
            tokenizer,
            pair,
            index,
            index < display_count,
            &full_positive,
            &embedding_positive,
            &left_positive,
            scratch_tokens,
            scratch_features,
            trace
        );
        free(negative_text);
        free(positive_text);
    }
    free(scratch_features);
    free(scratch_tokens);
    free_cache(&left_negative);
    free_cache(&left_positive);
    free_cache(&embedding_negative);
    free_cache(&embedding_positive);
    free_cache(&full_negative);
    free_cache(&full_positive);
}

int main(int argc, char **argv) {
    Options options = parse_options(argc, argv);
    const char *checkpoint_path = argv[1];
    const char *tokenizer_path = argv[2];
    const char *stories_path = argv[3];

    Transformer transformer;
    build_transformer(&transformer, (char *)checkpoint_path);
    Tokenizer tokenizer;
    build_tokenizer(
        &tokenizer,
        (char *)tokenizer_path,
        transformer.config.vocab_size
    );
    if (options.sequence_tokens > transformer.config.seq_len) {
        fail("requested token count exceeds model context length");
    }
    if (options.top_k >= transformer.config.vocab_size - 3) {
        fail("top-k exceeds usable vocabulary");
    }

    size_t story_bytes = 0;
    char *stories = read_file(stories_path, &story_bytes);
    (void)story_bytes;
    unsigned char *word_lexicon = build_word_lexicon(
        &tokenizer,
        stories
    );
    char *story_cursor = stories;
    unsigned long long data_rng = options.seed;
    fprintf(
        stderr,
        "frozen model dim=%d layers=%d vocab=%d context=%d\n",
        transformer.config.dim,
        transformer.config.n_layers,
        transformer.config.vocab_size,
        transformer.config.seq_len
    );
    CompanyPair *training = make_dataset(
        &transformer,
        &tokenizer,
        &story_cursor,
        word_lexicon,
        options.train_count,
        options.sequence_tokens,
        options.min_suffix,
        options.top_k,
        &data_rng,
        "training"
    );
    CompanyPair *validation = make_dataset(
        &transformer,
        &tokenizer,
        &story_cursor,
        word_lexicon,
        options.validation_count,
        options.sequence_tokens,
        options.min_suffix,
        options.top_k,
        &data_rng,
        "validation"
    );

    int feature_dim = (transformer.config.n_layers + 1) * transformer.config.dim;
    unsigned long long full_rng = options.seed ^ 0xa0761d6478bd642fULL;
    unsigned long long embedding_rng = options.seed ^ 0x8ebc6af09c88c6e3ULL;
    unsigned long long left_rng = options.seed ^ 0xe7037ed1a0b428dbULL;
    ReverseHead full = make_head(
        feature_dim,
        transformer.config.dim,
        options.head_dim,
        &full_rng
    );
    ReverseHead left = make_head(
        feature_dim,
        transformer.config.dim,
        options.head_dim,
        &left_rng
    );
    ReverseHead embedding = make_head(
        feature_dim,
        transformer.config.dim,
        options.head_dim,
        &embedding_rng
    );
    fprintf(
        stderr,
        "readout allocated_parameters=%zu active_parameters="
        "left:%zu embedding:%zu full:%zu training_pairs=%d "
        "validation_pairs=%d\n",
        full.parameter_count,
        active_parameter_count(&left, READOUT_LEFT_ONLY),
        active_parameter_count(&embedding, READOUT_EMBEDDING_SUFFIX),
        active_parameter_count(&full, READOUT_ALL_LAYERS_SUFFIX),
        options.train_count,
        options.validation_count
    );

    train_head(
        &left,
        &transformer,
        training,
        options.train_count,
        validation,
        options.validation_count,
        options.sequence_tokens,
        READOUT_LEFT_ONLY,
        &options,
        &left_rng,
        "left_control"
    );
    train_head(
        &embedding,
        &transformer,
        training,
        options.train_count,
        validation,
        options.validation_count,
        options.sequence_tokens,
        READOUT_EMBEDDING_SUFFIX,
        &options,
        &embedding_rng,
        "embedding_reverse_control"
    );
    train_head(
        &full,
        &transformer,
        training,
        options.train_count,
        validation,
        options.validation_count,
        options.sequence_tokens,
        READOUT_ALL_LAYERS_SUFFIX,
        &options,
        &full_rng,
        "reverse_company"
    );

    Metrics left_metrics = evaluate(
        &left,
        &transformer,
        validation,
        options.validation_count,
        READOUT_LEFT_ONLY,
        options.sequence_tokens
    );
    Metrics embedding_metrics = evaluate(
        &embedding,
        &transformer,
        validation,
        options.validation_count,
        READOUT_EMBEDDING_SUFFIX,
        options.sequence_tokens
    );
    Metrics full_metrics = evaluate(
        &full,
        &transformer,
        validation,
        options.validation_count,
        READOUT_ALL_LAYERS_SUFFIX,
        options.sequence_tokens
    );
    printf(
        "validation ar_accuracy=%.4f left_accuracy=%.4f "
        "embedding_accuracy=%.4f reverse_accuracy=%.4f "
        "left_margin=%.6f embedding_margin=%.6f reverse_margin=%.6f\n",
        full_metrics.ar_accuracy,
        left_metrics.accuracy,
        embedding_metrics.accuracy,
        full_metrics.accuracy,
        left_metrics.mean_margin,
        embedding_metrics.mean_margin,
        full_metrics.mean_margin
    );

    FILE *trace = NULL;
    if (options.trace_path != NULL) {
        trace = fopen(options.trace_path, "wb");
        if (trace == NULL) fail("could not create trace file");
    }
    report_pairs(
        &full,
        &embedding,
        &left,
        &transformer,
        &tokenizer,
        validation,
        options.validation_count,
        options.shown,
        options.sequence_tokens,
        trace
    );
    if (trace != NULL) fclose(trace);
    if (options.save_path != NULL) {
        save_head(&full, &transformer, options.save_path);
    }

    free_head(&left);
    free_head(&embedding);
    free_head(&full);
    for (int index = 0; index < options.validation_count; index++) {
        free_pair(&validation[index]);
    }
    for (int index = 0; index < options.train_count; index++) {
        free_pair(&training[index]);
    }
    free(validation);
    free(training);
    free(word_lexicon);
    free(stories);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return EXIT_SUCCESS;
}
