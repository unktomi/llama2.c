#define CANDIDATE_PROBE_LIBRARY
#include "candidate_probe.c"

#include <limits.h>
#include <stdint.h>

typedef struct {
    uint32_t pair;
    double score;
} RankedPair;

static void scale_usage(const char *program) {
    fprintf(
        stderr,
        "usage: %s CHECKPOINT TOKENIZER PROMPT OUTPUT_TSV [TOP_N]\n",
        program
    );
    exit(EXIT_FAILURE);
}

static void capture_scale_logits(
    Transformer *transformer,
    const float *hidden,
    float *logits
) {
    int dim = transformer->config.dim;
    float normalized[dim];
    rmsnorm(
        normalized,
        (float *)hidden,
        transformer->weights.rms_final_weight,
        dim
    );
    matmul(
        logits,
        normalized,
        transformer->weights.wcls,
        dim,
        transformer->config.vocab_size
    );
}

/* The numerical forward path is unchanged. The only addition is applying the
 * existing final RMS/classifier to the embedding and each completed layer so
 * those retained scales can be compared without combining them. */
static void forward_scales(
    Transformer *transformer,
    int token,
    int position,
    float *scale_logits
) {
    Config *config = &transformer->config;
    TransformerWeights *weights = &transformer->weights;
    RunState *state = &transformer->state;
    int dim = config->dim;
    int kv_dim = dim * config->n_kv_heads / config->n_heads;
    int kv_mul = config->n_heads / config->n_kv_heads;
    int head_size = dim / config->n_heads;

    memcpy(
        state->x,
        weights->token_embedding_table + (size_t)token * dim,
        (size_t)dim * sizeof(*state->x)
    );
    capture_scale_logits(transformer, state->x, scale_logits);

    for (int layer = 0; layer < config->n_layers; layer++) {
        rmsnorm(
            state->xb,
            state->x,
            weights->rms_att_weight + (size_t)layer * dim,
            dim
        );
        size_t layer_offset = (size_t)layer * config->seq_len * kv_dim;
        state->k = state->key_cache + layer_offset +
            (size_t)position * kv_dim;
        state->v = state->value_cache + layer_offset +
            (size_t)position * kv_dim;
        matmul(
            state->q,
            state->xb,
            weights->wq + (size_t)layer * dim * dim,
            dim,
            dim
        );
        matmul(
            state->k,
            state->xb,
            weights->wk + (size_t)layer * dim * kv_dim,
            dim,
            kv_dim
        );
        matmul(
            state->v,
            state->xb,
            weights->wv + (size_t)layer * dim * kv_dim,
            dim,
            kv_dim
        );

        for (int index = 0; index < dim; index += 2) {
            int head_dimension = index % head_size;
            float frequency = 1.0f /
                powf(10000.0f, head_dimension / (float)head_size);
            float angle = position * frequency;
            float real = cosf(angle);
            float imaginary = sinf(angle);
            int rotations = index < kv_dim ? 2 : 1;
            for (int vector = 0; vector < rotations; vector++) {
                float *values = vector == 0 ? state->q : state->k;
                float first = values[index];
                float second = values[index + 1];
                values[index] = first * real - second * imaginary;
                values[index + 1] = first * imaginary + second * real;
            }
        }

        for (int head = 0; head < config->n_heads; head++) {
            float *query = state->q + head * head_size;
            float *attention = state->att +
                (size_t)head * config->seq_len;
            for (int timestep = 0; timestep <= position; timestep++) {
                float *key = state->key_cache + layer_offset +
                    (size_t)timestep * kv_dim +
                    (head / kv_mul) * head_size;
                float score = 0.0f;
                for (int lane = 0; lane < head_size; lane++) {
                    score += query[lane] * key[lane];
                }
                attention[timestep] = score / sqrtf(head_size);
            }
            softmax(attention, position + 1);

            float *head_output = state->xb + head * head_size;
            memset(head_output, 0, (size_t)head_size * sizeof(*head_output));
            for (int timestep = 0; timestep <= position; timestep++) {
                float *value = state->value_cache + layer_offset +
                    (size_t)timestep * kv_dim +
                    (head / kv_mul) * head_size;
                float mass = attention[timestep];
                for (int lane = 0; lane < head_size; lane++) {
                    head_output[lane] += mass * value[lane];
                }
            }
        }

        matmul(
            state->xb2,
            state->xb,
            weights->wo + (size_t)layer * dim * dim,
            dim,
            dim
        );
        for (int lane = 0; lane < dim; lane++) {
            state->x[lane] += state->xb2[lane];
        }

        rmsnorm(
            state->xb,
            state->x,
            weights->rms_ffn_weight + (size_t)layer * dim,
            dim
        );
        matmul(
            state->hb,
            state->xb,
            weights->w1 + (size_t)layer * dim * config->hidden_dim,
            dim,
            config->hidden_dim
        );
        matmul(
            state->hb2,
            state->xb,
            weights->w3 + (size_t)layer * dim * config->hidden_dim,
            dim,
            config->hidden_dim
        );
        for (int lane = 0; lane < config->hidden_dim; lane++) {
            float value = state->hb[lane];
            value *= 1.0f / (1.0f + expf(-value));
            state->hb[lane] = value * state->hb2[lane];
        }
        matmul(
            state->xb,
            state->hb,
            weights->w2 + (size_t)layer * dim * config->hidden_dim,
            config->hidden_dim,
            dim
        );
        for (int lane = 0; lane < dim; lane++) {
            state->x[lane] += state->xb[lane];
        }
        capture_scale_logits(
            transformer,
            state->x,
            scale_logits +
                (size_t)(layer + 1) * config->vocab_size
        );
    }
}

static void logits_to_log_probabilities(
    const float *logits,
    double *log_probabilities,
    int count
) {
    double maximum = -DBL_MAX;
    for (int index = 0; index < count; index++) {
        if ((double)logits[index] > maximum) maximum = logits[index];
    }
    double partition = 0.0;
    for (int index = 0; index < count; index++) {
        partition += exp((double)logits[index] - maximum);
    }
    double log_partition = maximum + log(partition);
    for (int index = 0; index < count; index++) {
        log_probabilities[index] = logits[index] - log_partition;
    }
}

static void token_ranks(
    const double *log_probabilities,
    uint16_t *ranks,
    int count
) {
    for (int token = 0; token < count; token++) {
        int rank = 1;
        for (int candidate = 0; candidate < count; candidate++) {
            if (log_probabilities[candidate] > log_probabilities[token] ||
                (log_probabilities[candidate] == log_probabilities[token] &&
                 candidate < token)) {
                rank++;
            }
        }
        ranks[token] = (uint16_t)rank;
    }
}

static int compare_ranked_pairs(const void *left_pointer, const void *right_pointer) {
    const RankedPair *left = left_pointer;
    const RankedPair *right = right_pointer;
    if (left->score > right->score) return -1;
    if (left->score < right->score) return 1;
    if (left->pair < right->pair) return -1;
    if (left->pair > right->pair) return 1;
    return 0;
}

static void fprint_json_piece(FILE *file, const char *piece) {
    fputc('"', file);
    for (const unsigned char *cursor = (const unsigned char *)piece;
         *cursor != '\0'; cursor++) {
        unsigned char byte = *cursor;
        if (byte == '\\' || byte == '"') {
            fputc('\\', file);
            fputc(byte, file);
        } else if (byte == '\n') {
            fputs("\\n", file);
        } else if (byte == '\r') {
            fputs("\\r", file);
        } else if (byte == '\t') {
            fputs("\\t", file);
        } else if (byte >= 0x20 && byte < 0x7f) {
            fputc(byte, file);
        } else {
            fprintf(file, "\\u%04x", byte);
        }
    }
    fputc('"', file);
}

static long double rank_spearman(
    const uint32_t *left,
    const uint32_t *right,
    size_t count
) {
    long double square_sum = 0.0L;
    for (size_t index = 0; index < count; index++) {
        long double difference =
            (long double)left[index] - right[index];
        square_sum += difference * difference;
    }
    long double n = count;
    return 1.0L - 6.0L * square_sum / (n * (n * n - 1.0L));
}

static int top_overlap(
    const uint32_t *left,
    const uint32_t *right,
    size_t count,
    uint32_t top
) {
    int overlap = 0;
    for (size_t index = 0; index < count; index++) {
        if (left[index] <= top && right[index] <= top) overlap++;
    }
    return overlap;
}

static void print_pair(
    Tokenizer *tokenizer,
    int previous_token,
    int first,
    int second
) {
    fprint_json_piece(stdout, decode(tokenizer, previous_token, first));
    fputc('+', stdout);
    fprint_json_piece(stdout, decode(tokenizer, first, second));
}

int main(int argc, char **argv) {
    if (argc < 5 || argc > 6) scale_usage(argv[0]);
    long parsed_top = argc == 6 ? parse_long(argv[5], "top count") : 16;
    if (parsed_top <= 0 || parsed_top > INT_MAX) scale_usage(argv[0]);
    int top_count = (int)parsed_top;

    Transformer transformer;
    build_transformer(&transformer, argv[1]);
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, argv[2], transformer.config.vocab_size);
    int vocab = transformer.config.vocab_size;
    int scale_count = transformer.config.n_layers + 1;
    size_t pair_count = (size_t)vocab * vocab;

    int *prompt_tokens = malloc((strlen(argv[3]) + 3) * sizeof(*prompt_tokens));
    int prompt_count = 0;
    if (prompt_tokens == NULL) {
        fprintf(stderr, "could not allocate prompt tokens\n");
        exit(EXIT_FAILURE);
    }
    encode(&tokenizer, argv[3], 1, 0, prompt_tokens, &prompt_count);
    if (prompt_count < 2 || prompt_count + 2 > transformer.config.seq_len) {
        fprintf(stderr, "prefill must contain at least two model tokens\n");
        exit(EXIT_FAILURE);
    }

    float *scale_logits = malloc(
        (size_t)scale_count * vocab * sizeof(*scale_logits)
    );
    double *first_logp = malloc(
        (size_t)scale_count * vocab * sizeof(*first_logp)
    );
    double *second_logp = malloc(
        (size_t)scale_count * pair_count * sizeof(*second_logp)
    );
    uint16_t *first_ranks = malloc(
        (size_t)scale_count * vocab * sizeof(*first_ranks)
    );
    uint16_t *second_ranks = malloc(
        (size_t)scale_count * pair_count * sizeof(*second_ranks)
    );
    uint32_t *pair_ranks = malloc(
        (size_t)scale_count * pair_count * sizeof(*pair_ranks)
    );
    RankedPair *ordered = malloc(pair_count * sizeof(*ordered));
    if (scale_logits == NULL || first_logp == NULL || second_logp == NULL ||
        first_ranks == NULL || second_ranks == NULL || pair_ranks == NULL ||
        ordered == NULL) {
        fprintf(stderr, "could not allocate exhaustive scale tables\n");
        exit(EXIT_FAILURE);
    }

    for (int position = 0; position < prompt_count; position++) {
        forward_scales(
            &transformer,
            prompt_tokens[position],
            position,
            scale_logits
        );
    }
    for (int scale = 0; scale < scale_count; scale++) {
        logits_to_log_probabilities(
            scale_logits + (size_t)scale * vocab,
            first_logp + (size_t)scale * vocab,
            vocab
        );
        token_ranks(
            first_logp + (size_t)scale * vocab,
            first_ranks + (size_t)scale * vocab,
            vocab
        );
    }

    for (int first = 0; first < vocab; first++) {
        forward_scales(&transformer, first, prompt_count, scale_logits);
        for (int scale = 0; scale < scale_count; scale++) {
            double *distribution = second_logp +
                ((size_t)scale * vocab + first) * vocab;
            logits_to_log_probabilities(
                scale_logits + (size_t)scale * vocab,
                distribution,
                vocab
            );
            token_ranks(
                distribution,
                second_ranks + ((size_t)scale * vocab + first) * vocab,
                vocab
            );
        }
    }

    for (int scale = 0; scale < scale_count; scale++) {
        const double *first_distribution = first_logp + (size_t)scale * vocab;
        for (int first = 0; first < vocab; first++) {
            const double *second_distribution = second_logp +
                ((size_t)scale * vocab + first) * vocab;
            for (int second = 0; second < vocab; second++) {
                uint32_t pair = (uint32_t)((size_t)first * vocab + second);
                ordered[pair] = (RankedPair){
                    .pair = pair,
                    .score = first_distribution[first] +
                        second_distribution[second],
                };
            }
        }
        qsort(ordered, pair_count, sizeof(*ordered), compare_ranked_pairs);
        uint32_t *ranks = pair_ranks + (size_t)scale * pair_count;
        for (size_t index = 0; index < pair_count; index++) {
            ranks[ordered[index].pair] = (uint32_t)index + 1;
        }
    }

    FILE *table = fopen(argv[4], "w");
    if (table == NULL) {
        fprintf(stderr, "could not open exhaustive ranking table\n");
        exit(EXIT_FAILURE);
    }
    size_t output_buffer_size = 1 << 20;
    char *output_buffer = malloc(output_buffer_size);
    if (output_buffer != NULL) {
        setvbuf(table, output_buffer, _IOFBF, output_buffer_size);
    }
    fputs("token0\ttoken1\tpiece0\tpiece1", table);
    for (int scale = 0; scale < scale_count; scale++) {
        fprintf(
            table,
            "\ts%d_logp0\ts%d_rank0\ts%d_logp1\ts%d_rank1"
            "\ts%d_pair_score\ts%d_pair_rank",
            scale,
            scale,
            scale,
            scale,
            scale,
            scale
        );
    }
    fputc('\n', table);
    int previous_token = prompt_tokens[prompt_count - 1];
    for (int first = 0; first < vocab; first++) {
        for (int second = 0; second < vocab; second++) {
            uint32_t pair = (uint32_t)((size_t)first * vocab + second);
            fprintf(table, "%d\t%d\t", first, second);
            fprint_json_piece(table, decode(&tokenizer, previous_token, first));
            fputc('\t', table);
            fprint_json_piece(table, decode(&tokenizer, first, second));
            for (int scale = 0; scale < scale_count; scale++) {
                const double *first_distribution =
                    first_logp + (size_t)scale * vocab;
                const double *second_distribution = second_logp +
                    ((size_t)scale * vocab + first) * vocab;
                fprintf(
                    table,
                    "\t%.17g\t%u\t%.17g\t%u\t%.17g\t%u",
                    first_distribution[first],
                    first_ranks[(size_t)scale * vocab + first],
                    second_distribution[second],
                    second_ranks[((size_t)scale * vocab + first) * vocab +
                        second],
                    first_distribution[first] + second_distribution[second],
                    pair_ranks[(size_t)scale * pair_count + pair]
                );
            }
            fputc('\n', table);
        }
    }
    if (fclose(table) != 0) {
        fprintf(stderr, "could not finish exhaustive ranking table\n");
        exit(EXIT_FAILURE);
    }
    free(output_buffer);

    printf("prompt=");
    fprint_json_piece(stdout, argv[3]);
    printf(" prompt_tokens=%d ids=[", prompt_count);
    for (int index = 0; index < prompt_count; index++) {
        if (index != 0) fputc(',', stdout);
        printf("%d", prompt_tokens[index]);
    }
    printf("] completion_tokens=2 candidates=%zu scales=%d\n", pair_count, scale_count);

    for (int scale = 0; scale < scale_count; scale++) {
        for (size_t pair = 0; pair < pair_count; pair++) {
            int first = (int)(pair / vocab);
            int second = (int)(pair % vocab);
            ordered[pair] = (RankedPair){
                .pair = (uint32_t)pair,
                .score = first_logp[(size_t)scale * vocab + first] +
                    second_logp[((size_t)scale * vocab + first) * vocab +
                        second],
            };
        }
        qsort(ordered, pair_count, sizeof(*ordered), compare_ranked_pairs);
        printf("scale=%d kind=%s top=%d\n", scale,
            scale == 0 ? "embedding_probe" :
            (scale == transformer.config.n_layers ? "final_model" :
             "completed_layer_probe"),
            top_count);
        int shown = top_count < (int)pair_count ? top_count : (int)pair_count;
        for (int index = 0; index < shown; index++) {
            int first = (int)(ordered[index].pair / vocab);
            int second = (int)(ordered[index].pair % vocab);
            double first_score = first_logp[(size_t)scale * vocab + first];
            double second_score = second_logp[
                ((size_t)scale * vocab + first) * vocab + second
            ];
            printf(
                "  pair_rank=%d ids=[%d,%d] token_ranks=[%u,%u] "
                "token_logp=[%.9g,%.9g] pair_score=%.9g text=",
                index + 1,
                first,
                second,
                first_ranks[(size_t)scale * vocab + first],
                second_ranks[((size_t)scale * vocab + first) * vocab +
                    second],
                first_score,
                second_score,
                first_score + second_score
            );
            print_pair(&tokenizer, previous_token, first, second);
            fputc('\n', stdout);
        }
    }

    const uint32_t *final_ranks = pair_ranks +
        (size_t)(scale_count - 1) * pair_count;
    puts("rank_comparison_to_final_model:");
    for (int scale = 0; scale < scale_count; scale++) {
        const uint32_t *ranks = pair_ranks + (size_t)scale * pair_count;
        printf(
            "  scale=%d spearman=%.12Lf top10_overlap=%d "
            "top100_overlap=%d top1000_overlap=%d\n",
            scale,
            rank_spearman(ranks, final_ranks, pair_count),
            top_overlap(ranks, final_ranks, pair_count, 10),
            top_overlap(ranks, final_ranks, pair_count, 100),
            top_overlap(ranks, final_ranks, pair_count, 1000)
        );
    }
    printf("ranking_table=%s rows=%zu\n", argv[4], pair_count);

    free(ordered);
    free(pair_ranks);
    free(second_ranks);
    free(first_ranks);
    free(second_logp);
    free(first_logp);
    free(scale_logits);
    free(prompt_tokens);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
