/*
 * Ordered masked joint decoder for hidden-feedback llama2.c.
 *
 * The retained hidden tape is not assigned a scalar path energy.  A proposed
 * token tuple is placed in the company of the fixed prefill and the observer
 * returns the complete tuple of token-indexed observations:
 *
 *   observe : (Hidden^N, Token^N) -> Logits^N.
 *
 * The decoder is a tied-embedding, bidirectional masked language model.
 * Queries come from retained hidden states. Keys/values come from the fixed
 * prompt constructors and from every proposed completion constructor.
 * Completion position i is masked out of output row i, so the head cannot
 * copy the token it is rating. RoPE is applied to every query and key using
 * its absolute sequence position; unlike the rejected bag-of-embeddings
 * observer, this decoder can distinguish differently ordered company.
 *
 * Training denoises complete teacher tuples after deterministic deletion,
 * model-native wrong-token substitution, and repeated-company corruption;
 * the rated slot remains self-masked. Inference never folds the resulting row
 * observations to a path score. For a finite carrier K at each of N slots, a
 * leaf outcome is the complete covector family (R^K)^N. At depth i, backward
 * induction obtains each child's complete backed outcome and asks, inside
 * that one outcome covector, whether the branch token is the model's selected
 * filler. The local selection function returns the first attaining witness
 * (or the final fallback, as in finite searchable-set `find`) and passes that
 * complete outcome family upward unchanged. No scalar is compared across
 * candidate-conditioned contexts. Only the root terminalizes the tuple.
 */

#define TESTING
#include "run_hidden_feedback.c"

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>

typedef struct {
    float *hidden;             /* completion positions x dim */
    float *base_logits;        /* completion positions x vocab */
    int *targets;              /* completion positions, NULL for inference */
    int *prompt_tokens;
    int prompt_count;
} ObserverExample;

typedef struct {
    int dim;
    int head_count;
    int head_dim;
    size_t matrix_size;
    size_t parameter_count;
    double *parameters;        /* Wq, Wk, Wv, Wo, Wh */
} ObserverHead;

typedef struct {
    int positions;
    int sources;
    int vocab;
    double *source_input;
    double *query;
    double *key;
    double *value;
    double *attention;
    double *context;
    double *output;
    double *logits;
    double *grad_query;
    double *grad_key;
    double *grad_value;
    double *grad_attention;
    double *grad_context;
    double *grad_output;
} ObserverWorkspace;

typedef struct {
    int training_count;
    int validation_count;
    int prompt_count;
    int completion_count;
    int teacher_steps;
    int epochs;
    unsigned long long seed;
    double learning_rate;
    double l2;
} ObserverTrainOptions;

typedef struct {
    const char *prompt;
    int length;
    int top_k;
    int sample_ms;
    int exact;
    unsigned long long seed;
    const char *trace_path;
} ObserverInferOptions;

static void observer_fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

static void *observer_allocate(size_t count, size_t size) {
    if (count != 0 && size > SIZE_MAX / count) {
        observer_fail("joint observer allocation overflow");
    }
    void *memory = calloc(count, size);
    if (memory == NULL) observer_fail("joint observer allocation failed");
    return memory;
}

static int observer_parse_positive(const char *text, const char *name) {
    errno = 0;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value <= 0 || value > INT_MAX) {
        fprintf(stderr, "%s must be a positive integer\n", name);
        exit(EXIT_FAILURE);
    }
    return (int)value;
}

static unsigned long long observer_parse_seed(const char *text) {
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0) {
        observer_fail("seed must be a positive integer");
    }
    return value;
}

static double observer_parse_positive_double(const char *text, const char *name) {
    errno = 0;
    char *end = NULL;
    double value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' ||
        !isfinite(value) || value <= 0.0) {
        fprintf(stderr, "%s must be positive\n", name);
        exit(EXIT_FAILURE);
    }
    return value;
}

static uint64_t observer_random_u64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}

static double observer_random_signed(uint64_t *state) {
    return 2.0 * (double)(observer_random_u64(state) >> 11) /
        9007199254740992.0 - 1.0;
}

static uint64_t observer_mix_u64(uint64_t value) {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31;
    return value;
}

static double *observer_matrix(ObserverHead *head, int matrix) {
    return head->parameters + (size_t)matrix * head->matrix_size;
}

static const double *observer_const_matrix(
    const ObserverHead *head,
    int matrix
) {
    return head->parameters + (size_t)matrix * head->matrix_size;
}

static void observer_head_initialize(
    ObserverHead *head,
    int dim,
    int head_count,
    unsigned long long seed
) {
    if (dim <= 0 || head_count <= 0 || dim % head_count != 0) {
        observer_fail("invalid observer head dimensions");
    }
    *head = (ObserverHead){
        .dim = dim,
        .head_count = head_count,
        .head_dim = dim / head_count,
        .matrix_size = (size_t)dim * dim,
        .parameter_count = 5U * (size_t)dim * dim,
    };
    head->parameters = observer_allocate(head->parameter_count, sizeof(double));
    uint64_t random_state = seed != 0 ? seed : UINT64_C(1);
    for (int matrix = 0; matrix < 4; matrix++) {
        double *weight = observer_matrix(head, matrix);
        for (int row = 0; row < dim; row++) {
            for (int column = 0; column < dim; column++) {
                double identity = row == column ? 1.0 : 0.0;
                weight[(size_t)row * dim + column] = identity +
                    0.01 * observer_random_signed(&random_state) /
                    sqrt((double)dim);
            }
        }
    }
    double *hidden_output = observer_matrix(head, 4);
    for (size_t index = 0; index < head->matrix_size; index++) {
        hidden_output[index] = 0.01 * observer_random_signed(&random_state) /
            sqrt((double)dim);
    }
}

static void observer_head_free(ObserverHead *head) {
    free(head->parameters);
    memset(head, 0, sizeof(*head));
}

static double observer_initial_parameter(
    const ObserverHead *head,
    size_t parameter
) {
    int matrix = (int)(parameter / head->matrix_size);
    size_t coordinate = parameter % head->matrix_size;
    int row = (int)(coordinate / (size_t)head->dim);
    int column = (int)(coordinate % (size_t)head->dim);
    return matrix < 4 && row == column ? 1.0 : 0.0;
}

static void observer_head_save(const char *path, const ObserverHead *head) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        fprintf(stderr, "could not open observer output %s\n", path);
        exit(EXIT_FAILURE);
    }
    const unsigned char magic[8] = {'T','J','M','A','E','0','0','5'};
    int ok = fwrite(magic, sizeof(magic), 1, file) == 1 &&
        fwrite(&head->dim, sizeof(head->dim), 1, file) == 1 &&
        fwrite(&head->head_count, sizeof(head->head_count), 1, file) == 1 &&
        fwrite(
            head->parameters,
            sizeof(double),
            head->parameter_count,
            file
        ) == head->parameter_count;
    if (!ok || fclose(file) != 0) observer_fail("could not write observer head");
}

static void observer_head_load(const char *path, ObserverHead *head) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "could not open observer head %s\n", path);
        exit(EXIT_FAILURE);
    }
    unsigned char magic[8];
    int dim = 0;
    int heads = 0;
    if (fread(magic, sizeof(magic), 1, file) != 1 ||
        memcmp(magic, "TJMAE005", sizeof(magic)) != 0 ||
        fread(&dim, sizeof(dim), 1, file) != 1 ||
        fread(&heads, sizeof(heads), 1, file) != 1) {
        observer_fail("invalid observer head file");
    }
    observer_head_initialize(head, dim, heads, 1);
    if (fread(
            head->parameters,
            sizeof(double),
            head->parameter_count,
            file
        ) != head->parameter_count) {
        observer_fail("truncated observer head file");
    }
    if (fgetc(file) != EOF) observer_fail("observer head has trailing data");
    fclose(file);
}

static void observer_example_free(ObserverExample *example) {
    free(example->prompt_tokens);
    free(example->targets);
    free(example->base_logits);
    free(example->hidden);
    memset(example, 0, sizeof(*example));
}

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
} ObserverText;

static void observer_text_initialize(ObserverText *text, size_t capacity) {
    if (capacity < 64) capacity = 64;
    text->data = observer_allocate(capacity, sizeof(char));
    text->capacity = capacity;
    text->length = 0;
}

static void observer_text_append(ObserverText *text, const char *piece) {
    if (piece == NULL || piece[0] == '\0') return;
    size_t length = strlen(piece);
    if (length == 1) {
        unsigned char byte = (unsigned char)piece[0];
        if (!(isprint(byte) || isspace(byte))) return;
    }
    size_t required = text->length + length + 1;
    if (required > text->capacity) {
        size_t capacity = text->capacity;
        while (capacity < required) {
            if (capacity > SIZE_MAX / 2) observer_fail("teacher text too large");
            capacity *= 2;
        }
        char *grown = realloc(text->data, capacity);
        if (grown == NULL) observer_fail("could not grow teacher text");
        text->data = grown;
        text->capacity = capacity;
    }
    memcpy(text->data + text->length, piece, length);
    text->length += length;
    text->data[text->length] = '\0';
}

static char *observer_generate_teacher_story(
    Transformer *teacher,
    Tokenizer *tokenizer,
    Sampler *sampler,
    int steps
) {
    ObserverText text;
    observer_text_initialize(
        &text,
        (size_t)steps * (tokenizer->max_token_length + 1U) + 1U
    );
    int token = 1;
    for (int position = 0; position < steps; position++) {
        float *logits = forward(teacher, token, position);
        int next = sample(sampler, logits);
        if (next == 1 || next == 2) break;
        observer_text_append(&text, decode(tokenizer, token, next));
        token = next;
    }
    return text.data;
}

static void observer_build_example_from_tokens(
    ObserverExample *example,
    Transformer *student,
    const int *story_tokens,
    int content_start,
    int prompt_count,
    int completion_count
) {
    if (prompt_count + completion_count - 1 > student->config.seq_len) {
        observer_fail("student training span exceeds sequence length");
    }
    example->prompt_count = prompt_count;
    example->prompt_tokens = observer_allocate((size_t)prompt_count, sizeof(int));
    example->targets = observer_allocate((size_t)completion_count, sizeof(int));
    example->hidden = observer_allocate(
        (size_t)completion_count * student->config.dim,
        sizeof(float)
    );
    example->base_logits = observer_allocate(
        (size_t)completion_count * student->config.vocab_size,
        sizeof(float)
    );
    example->prompt_tokens[0] = 1;
    if (prompt_count > 1) {
        memcpy(
            example->prompt_tokens + 1,
            story_tokens + content_start,
            (size_t)(prompt_count - 1) * sizeof(int)
        );
    }
    memcpy(
        example->targets,
        story_tokens + content_start + prompt_count - 1,
        (size_t)completion_count * sizeof(int)
    );

    for (int position = 0; position < prompt_count; position++) {
        float *hidden = forward_token_hidden(
            student,
            example->prompt_tokens[position],
            position
        );
        if (position == prompt_count - 1) {
            memcpy(
                example->hidden,
                hidden,
                (size_t)student->config.dim * sizeof(float)
            );
        }
    }
    for (int index = 1; index < completion_count; index++) {
        float *hidden = forward_feedback_hidden(
            student,
            prompt_count + index - 1
        );
        memcpy(
            example->hidden + (size_t)index * student->config.dim,
            hidden,
            (size_t)student->config.dim * sizeof(float)
        );
    }
    for (int position = 0; position < completion_count; position++) {
        matmul(
            example->base_logits +
                (size_t)position * student->config.vocab_size,
            example->hidden + (size_t)position * student->config.dim,
            student->weights.wcls,
            student->config.dim,
            student->config.vocab_size
        );
    }
}

static ObserverExample *observer_generate_dataset(
    int count,
    Transformer *teacher,
    Tokenizer *teacher_tokenizer,
    Sampler *teacher_sampler,
    Transformer *student,
    Tokenizer *student_tokenizer,
    int prompt_count,
    int completion_count,
    int teacher_steps,
    const char *name
) {
    ObserverExample *examples = observer_allocate(
        (size_t)count,
        sizeof(*examples)
    );
    int built = 0;
    int stories = 0;
    int content_span = prompt_count - 1 + completion_count;
    while (built < count && stories < count * 30) {
        stories++;
        char *story = observer_generate_teacher_story(
            teacher,
            teacher_tokenizer,
            teacher_sampler,
            teacher_steps
        );
        size_t story_bytes = strlen(story);
        int *story_tokens = observer_allocate(story_bytes + 3U, sizeof(int));
        int story_token_count = 0;
        encode(
            student_tokenizer,
            story,
            1,
            0,
            story_tokens,
            &story_token_count
        );
        free(story);
        int window_count = story_token_count - content_span;
        if (window_count <= 0) {
            free(story_tokens);
            continue;
        }
        int first_window = (int)(
            random_u32(&teacher_sampler->rng_state) % (unsigned int)window_count
        );
        for (int window = 0; window < window_count && built < count; window++) {
            int content_start = 1 + (first_window + window) % window_count;
            observer_build_example_from_tokens(
                &examples[built],
                student,
                story_tokens,
                content_start,
                prompt_count,
                completion_count
            );
            built++;
            if (built == count || built % 128 == 0) {
                fprintf(
                    stderr,
                    "%s corpus: %d/%d teacher_stories=%d\r",
                    name,
                    built,
                    count,
                    stories
                );
                fflush(stderr);
            }
        }
        free(story_tokens);
    }
    fputc('\n', stderr);
    if (built != count) {
        fprintf(
            stderr,
            "built only %d/%d %s examples from %d teacher stories\n",
            built,
            count,
            name,
            stories
        );
        exit(EXIT_FAILURE);
    }
    return examples;
}

static void observer_build_prompt_example(
    ObserverExample *example,
    Transformer *student,
    Tokenizer *tokenizer,
    const char *prompt,
    int positions
) {
    size_t prompt_bytes = strlen(prompt);
    example->prompt_tokens = observer_allocate(prompt_bytes + 3U, sizeof(int));
    encode(
        tokenizer,
        (char *)prompt,
        1,
        0,
        example->prompt_tokens,
        &example->prompt_count
    );
    if (example->prompt_count <= 0) observer_fail("prompt encoded to no tokens");
    if (example->prompt_count + positions - 1 > student->config.seq_len) {
        observer_fail("prompt plus completion exceeds sequence length");
    }
    example->hidden = observer_allocate(
        (size_t)positions * student->config.dim,
        sizeof(float)
    );
    example->base_logits = observer_allocate(
        (size_t)positions * student->config.vocab_size,
        sizeof(float)
    );
    for (int position = 0; position < example->prompt_count; position++) {
        float *hidden = forward_token_hidden(
            student,
            example->prompt_tokens[position],
            position
        );
        if (position == example->prompt_count - 1) {
            memcpy(
                example->hidden,
                hidden,
                (size_t)student->config.dim * sizeof(float)
            );
        }
    }
    for (int index = 1; index < positions; index++) {
        float *hidden = forward_feedback_hidden(
            student,
            example->prompt_count + index - 1
        );
        memcpy(
            example->hidden + (size_t)index * student->config.dim,
            hidden,
            (size_t)student->config.dim * sizeof(float)
        );
    }
    for (int position = 0; position < positions; position++) {
        matmul(
            example->base_logits +
                (size_t)position * student->config.vocab_size,
            example->hidden + (size_t)position * student->config.dim,
            student->weights.wcls,
            student->config.dim,
            student->config.vocab_size
        );
    }
}

static void observer_workspace_initialize(
    ObserverWorkspace *workspace,
    const ObserverHead *head,
    int positions,
    int sources,
    int vocab,
    int with_gradients
) {
    *workspace = (ObserverWorkspace){
        .positions = positions,
        .sources = sources,
        .vocab = vocab,
    };
    size_t query_count = (size_t)positions * head->dim;
    size_t source_count = (size_t)sources * head->dim;
    size_t attention_count =
        (size_t)head->head_count * positions * sources;
    workspace->source_input = observer_allocate(source_count, sizeof(double));
    workspace->query = observer_allocate(query_count, sizeof(double));
    workspace->key = observer_allocate(source_count, sizeof(double));
    workspace->value = observer_allocate(source_count, sizeof(double));
    workspace->attention = observer_allocate(attention_count, sizeof(double));
    workspace->context = observer_allocate(query_count, sizeof(double));
    workspace->output = observer_allocate(query_count, sizeof(double));
    workspace->logits = observer_allocate(
        (size_t)positions * vocab,
        sizeof(double)
    );
    if (with_gradients) {
        workspace->grad_query = observer_allocate(query_count, sizeof(double));
        workspace->grad_key = observer_allocate(source_count, sizeof(double));
        workspace->grad_value = observer_allocate(source_count, sizeof(double));
        workspace->grad_attention = observer_allocate(
            attention_count,
            sizeof(double)
        );
        workspace->grad_context = observer_allocate(query_count, sizeof(double));
        workspace->grad_output = observer_allocate(query_count, sizeof(double));
    }
}

static void observer_workspace_free(ObserverWorkspace *workspace) {
    free(workspace->grad_output);
    free(workspace->grad_context);
    free(workspace->grad_attention);
    free(workspace->grad_value);
    free(workspace->grad_key);
    free(workspace->grad_query);
    free(workspace->logits);
    free(workspace->output);
    free(workspace->context);
    free(workspace->attention);
    free(workspace->value);
    free(workspace->key);
    free(workspace->query);
    free(workspace->source_input);
    memset(workspace, 0, sizeof(*workspace));
}

static void observer_linear(
    const double *weight,
    const double *input,
    int rows,
    int dim,
    double *output
) {
    for (int item = 0; item < rows; item++) {
        for (int out = 0; out < dim; out++) {
            double value = 0.0;
            for (int in = 0; in < dim; in++) {
                value += weight[(size_t)out * dim + in] *
                    input[(size_t)item * dim + in];
            }
            output[(size_t)item * dim + out] = value;
        }
    }
}

/*
 * Give the masked company its sequence order. The old observer fed an
 * unordered multiset of token embeddings to attention, making every
 * permutation of a completion observationally identical. This is the same
 * per-head rotary map used by llama2.c, applied to the decoder's Q/K vectors.
 * direction=-1 applies the transpose map needed by back-propagation.
 */
static void observer_apply_rope(
    double *vectors,
    int rows,
    int dim,
    int head_count,
    int first_position,
    int direction
) {
    int head_dim = dim / head_count;
    for (int row = 0; row < rows; row++) {
        int position = first_position + row;
        for (int head = 0; head < head_count; head++) {
            double *vector = vectors + (size_t)row * dim + head * head_dim;
            for (int lane = 0; lane + 1 < head_dim; lane += 2) {
                double frequency = 1.0 /
                    pow(10000.0, lane / (double)head_dim);
                double angle = direction * position * frequency;
                double cosine = cos(angle);
                double sine = sin(angle);
                double first = vector[lane];
                double second = vector[lane + 1];
                vector[lane] = first * cosine - second * sine;
                vector[lane + 1] = first * sine + second * cosine;
            }
        }
    }
}

static int observer_source_visible(
    const int *candidate_tokens,
    int prompt_count,
    int source,
    int self_source
) {
    if (source == self_source) return 0;
    if (source < prompt_count) return 1;
    return candidate_tokens[source - prompt_count] >= 0;
}

static void observer_forward(
    const ObserverExample *example,
    const int *candidate_tokens,
    const Transformer *student,
    const ObserverHead *head,
    ObserverWorkspace *workspace
) {
    int positions = workspace->positions;
    int sources = workspace->sources;
    int dim = head->dim;
    int prompt_count = example->prompt_count;
    int head_dim = head->head_dim;
    memset(
        workspace->source_input,
        0,
        (size_t)sources * dim * sizeof(double)
    );
    memset(
        workspace->context,
        0,
        (size_t)positions * dim * sizeof(double)
    );
    for (int source = 0; source < prompt_count; source++) {
        const float *embedding = student->weights.wcls +
            (size_t)example->prompt_tokens[source] * dim;
        for (int lane = 0; lane < dim; lane++) {
            workspace->source_input[(size_t)source * dim + lane] =
                embedding[lane];
        }
    }
    for (int position = 0; position < positions; position++) {
        int source = prompt_count + position;
        if (candidate_tokens[position] < 0) continue;
        const float *embedding = student->weights.wcls +
            (size_t)candidate_tokens[position] * dim;
        for (int lane = 0; lane < dim; lane++) {
            workspace->source_input[(size_t)source * dim + lane] =
                embedding[lane];
        }
    }
    double *query_input = observer_allocate(
        (size_t)positions * dim,
        sizeof(double)
    );
    for (int position = 0; position < positions; position++) {
        for (int lane = 0; lane < dim; lane++) {
            query_input[(size_t)position * dim + lane] =
                example->hidden[(size_t)position * dim + lane];
        }
    }
    observer_linear(
        observer_const_matrix(head, 0),
        query_input,
        positions,
        dim,
        workspace->query
    );
    free(query_input);
    observer_apply_rope(
        workspace->query,
        positions,
        dim,
        head->head_count,
        prompt_count,
        1
    );
    observer_linear(
        observer_const_matrix(head, 1),
        workspace->source_input,
        sources,
        dim,
        workspace->key
    );
    observer_apply_rope(
        workspace->key,
        sources,
        dim,
        head->head_count,
        0,
        1
    );
    observer_linear(
        observer_const_matrix(head, 2),
        workspace->source_input,
        sources,
        dim,
        workspace->value
    );

    double scale = 1.0 / sqrt((double)head_dim);
    for (int attention_head = 0;
         attention_head < head->head_count;
         attention_head++) {
        int lane_start = attention_head * head_dim;
        for (int output_position = 0;
             output_position < positions;
             output_position++) {
            double *row = workspace->attention +
                ((size_t)attention_head * positions + output_position) * sources;
            int masked_source = prompt_count + output_position;
            double maximum = -DBL_MAX;
            for (int source = 0; source < sources; source++) {
                if (!observer_source_visible(
                        candidate_tokens,
                        prompt_count,
                        source,
                        masked_source)) {
                    row[source] = 0.0;
                    continue;
                }
                double score = 0.0;
                for (int lane = 0; lane < head_dim; lane++) {
                    int coordinate = lane_start + lane;
                    score += workspace->query[
                        (size_t)output_position * dim + coordinate
                    ] * workspace->key[(size_t)source * dim + coordinate];
                }
                score *= scale;
                row[source] = score;
                if (score > maximum) maximum = score;
            }
            double partition = 0.0;
            for (int source = 0; source < sources; source++) {
                if (!observer_source_visible(
                        candidate_tokens,
                        prompt_count,
                        source,
                        masked_source)) continue;
                row[source] = exp(row[source] - maximum);
                partition += row[source];
            }
            for (int source = 0; source < sources; source++) {
                if (!observer_source_visible(
                        candidate_tokens,
                        prompt_count,
                        source,
                        masked_source)) {
                    row[source] = 0.0;
                    continue;
                }
                row[source] /= partition;
                for (int lane = 0; lane < head_dim; lane++) {
                    int coordinate = lane_start + lane;
                    workspace->context[
                        (size_t)output_position * dim + coordinate
                    ] += row[source] * workspace->value[
                        (size_t)source * dim + coordinate
                    ];
                }
            }
        }
    }

    const double *output_weight = observer_const_matrix(head, 3);
    const double *hidden_delta = observer_const_matrix(head, 4);
    for (int position = 0; position < positions; position++) {
        for (int out = 0; out < dim; out++) {
            /* Preserve the frozen model's predictive covector. Matrix 4 is a
             * zero-centred residual adapter, not a replacement initialized
             * near zero for the entire hidden path. */
            double value = example->hidden[(size_t)position * dim + out];
            for (int in = 0; in < dim; in++) {
                value += output_weight[(size_t)out * dim + in] *
                    workspace->context[(size_t)position * dim + in];
                value += hidden_delta[(size_t)out * dim + in] *
                    example->hidden[(size_t)position * dim + in];
            }
            workspace->output[(size_t)position * dim + out] = value;
        }
    }
    for (int position = 0; position < positions; position++) {
        for (int token = 0; token < workspace->vocab; token++) {
            const float *embedding = student->weights.wcls +
                (size_t)token * dim;
            double value = 0.0;
            for (int lane = 0; lane < dim; lane++) {
                value += embedding[lane] * workspace->output[
                    (size_t)position * dim + lane
                ];
            }
            workspace->logits[(size_t)position * workspace->vocab + token] =
                value;
        }
    }
}

enum {
    OBSERVER_NATIVE_CORRUPTION_WIDTH = 4,
    OBSERVER_CORRUPTION_MODES = 5,
};

/* Pick a plausible wrong constructor from the frozen student's native
 * covector at one position.  Keeping this deterministic makes every training
 * epoch reproducible while the rank argument exposes more than one local
 * alternative over successive corruption phases. */
static int observer_native_wrong_token(
    const ObserverExample *example,
    int position,
    int vocab,
    unsigned int requested_rank
) {
    int top[OBSERVER_NATIVE_CORRUPTION_WIDTH];
    for (int rank = 0; rank < OBSERVER_NATIVE_CORRUPTION_WIDTH; rank++) {
        top[rank] = -1;
    }
    const float *logits = example->base_logits + (size_t)position * vocab;
    int target = example->targets[position];
    for (int token = 0; token < vocab; token++) {
        if (token == target) continue;
        int insert = OBSERVER_NATIVE_CORRUPTION_WIDTH;
        for (int rank = 0; rank < OBSERVER_NATIVE_CORRUPTION_WIDTH; rank++) {
            if (top[rank] < 0 || logits[token] > logits[top[rank]]) {
                insert = rank;
                break;
            }
        }
        if (insert == OBSERVER_NATIVE_CORRUPTION_WIDTH) continue;
        for (int rank = OBSERVER_NATIVE_CORRUPTION_WIDTH - 1;
             rank > insert;
             rank--) {
            top[rank] = top[rank - 1];
        }
        top[insert] = token;
    }
    int available = 0;
    while (available < OBSERVER_NATIVE_CORRUPTION_WIDTH && top[available] >= 0) {
        available++;
    }
    if (available == 0) observer_fail("native corruption has no alternative");
    return top[requested_rank % (unsigned int)available];
}

static double observer_dataset_loss(
    const ObserverExample *examples,
    int example_count,
    const Transformer *student,
    ObserverHead *head,
    int positions,
    int corruption_phase,
    double l2,
    double *gradient,
    double *token_accuracy,
    double *sequence_accuracy
) {
    int dim = head->dim;
    int vocab = student->config.vocab_size;
    int head_dim = head->head_dim;
    if (gradient != NULL) {
        memset(gradient, 0, head->parameter_count * sizeof(double));
    }
    double *gradient_q = gradient;
    double *gradient_k = gradient == NULL ? NULL : gradient + head->matrix_size;
    double *gradient_v = gradient == NULL ? NULL : gradient_k + head->matrix_size;
    double *gradient_o = gradient == NULL ? NULL : gradient_v + head->matrix_size;
    double *gradient_h = gradient == NULL ? NULL : gradient_o + head->matrix_size;
    double loss = 0.0;
    unsigned long long correct_tokens = 0;
    unsigned long long correct_sequences = 0;

    for (int example_index = 0;
         example_index < example_count;
         example_index++) {
        const ObserverExample *example = &examples[example_index];
        int *candidate_tokens = observer_allocate(
            (size_t)positions,
            sizeof(int)
        );
        int corruption_mode =
            (example_index + corruption_phase) % OBSERVER_CORRUPTION_MODES;
        uint64_t example_key =
            ((uint64_t)(unsigned int)(example_index + 1) << 32) ^
            ((uint64_t)(unsigned int)(corruption_phase + 1) << 48);
        int repeated_position = (int)(
            observer_mix_u64(example_key ^ UINT64_C(0x7265706561746564)) %
            (uint64_t)positions
        );
        int repeated_token = observer_native_wrong_token(
            example,
            repeated_position,
            vocab,
            (unsigned int)(observer_mix_u64(
                example_key ^ UINT64_C(0x636f6d70616e7900)
            ) % OBSERVER_NATIVE_CORRUPTION_WIDTH)
        );
        for (int position = 0; position < positions; position++) {
            uint64_t key = example_key ^
                (uint64_t)(unsigned int)(position + 1);
            int draw = (int)(observer_mix_u64(key) % UINT64_C(100));
            int wrong = observer_native_wrong_token(
                example,
                position,
                vocab,
                (unsigned int)(observer_mix_u64(
                    key ^ UINT64_C(0x77726f6e67000000)
                ) % OBSERVER_NATIVE_CORRUPTION_WIDTH)
            );
            if (corruption_mode == 0) {
                candidate_tokens[position] = example->targets[position];
            } else if (corruption_mode == 1) {
                candidate_tokens[position] = draw < 50 ?
                    -1 : example->targets[position];
            } else if (corruption_mode == 2) {
                candidate_tokens[position] = draw < 50 ?
                    wrong : example->targets[position];
            } else if (corruption_mode == 3) {
                /* Directly train against the repeated-token fixed points seen
                 * in the v4/v5 exact traces.  The target remains the original
                 * teacher tuple and observer_forward still masks the source at
                 * the row being rated. */
                candidate_tokens[position] = repeated_token;
            } else {
                candidate_tokens[position] = draw < 25 ? -1 :
                    (draw < 75 ? wrong : example->targets[position]);
            }
        }
        int sources = example->prompt_count + positions;
        ObserverWorkspace workspace;
        observer_workspace_initialize(
            &workspace,
            head,
            positions,
            sources,
            vocab,
            gradient != NULL
        );
        observer_forward(example, candidate_tokens, student, head, &workspace);
        if (gradient != NULL) {
            memset(
                workspace.grad_query,
                0,
                (size_t)positions * dim * sizeof(double)
            );
            memset(
                workspace.grad_key,
                0,
                (size_t)sources * dim * sizeof(double)
            );
            memset(
                workspace.grad_value,
                0,
                (size_t)sources * dim * sizeof(double)
            );
            memset(
                workspace.grad_attention,
                0,
                (size_t)head->head_count * positions * sources *
                    sizeof(double)
            );
            memset(
                workspace.grad_context,
                0,
                (size_t)positions * dim * sizeof(double)
            );
            memset(
                workspace.grad_output,
                0,
                (size_t)positions * dim * sizeof(double)
            );
        }
        int sequence_correct = 1;
        for (int position = 0; position < positions; position++) {
            double maximum = workspace.logits[(size_t)position * vocab];
            int selected = 0;
            for (int token = 1; token < vocab; token++) {
                double value = workspace.logits[(size_t)position * vocab + token];
                if (value > maximum) {
                    maximum = value;
                    selected = token;
                }
            }
            int target = example->targets[position];
            if (selected == target) {
                correct_tokens++;
            } else {
                sequence_correct = 0;
            }
            double partition = 0.0;
            for (int token = 0; token < vocab; token++) {
                partition += exp(
                    workspace.logits[(size_t)position * vocab + token] - maximum
                );
            }
            double log_partition = maximum + log(partition);
            loss += log_partition -
                workspace.logits[(size_t)position * vocab + target];
            if (gradient != NULL) {
                for (int token = 0; token < vocab; token++) {
                    double probability = exp(
                        workspace.logits[(size_t)position * vocab + token] -
                        log_partition
                    );
                    double error = probability - (token == target ? 1.0 : 0.0);
                    const float *embedding = student->weights.wcls +
                        (size_t)token * dim;
                    for (int lane = 0; lane < dim; lane++) {
                        workspace.grad_output[(size_t)position * dim + lane] +=
                            error * embedding[lane];
                    }
                }
            }
        }
        if (sequence_correct) correct_sequences++;

        if (gradient != NULL) {
            const double *output_weight = observer_const_matrix(head, 3);
            for (int position = 0; position < positions; position++) {
                for (int out = 0; out < dim; out++) {
                    double output_gradient = workspace.grad_output[
                        (size_t)position * dim + out
                    ];
                    for (int in = 0; in < dim; in++) {
                        gradient_o[(size_t)out * dim + in] +=
                            output_gradient * workspace.context[
                                (size_t)position * dim + in
                            ];
                        gradient_h[(size_t)out * dim + in] +=
                            output_gradient * example->hidden[
                                (size_t)position * dim + in
                            ];
                        workspace.grad_context[(size_t)position * dim + in] +=
                            output_weight[(size_t)out * dim + in] *
                            output_gradient;
                    }
                }
            }

            double attention_scale = 1.0 / sqrt((double)head_dim);
            for (int attention_head = 0;
                 attention_head < head->head_count;
                 attention_head++) {
                int lane_start = attention_head * head_dim;
                for (int output_position = 0;
                     output_position < positions;
                     output_position++) {
                    int masked_source = example->prompt_count + output_position;
                    double *attention_row = workspace.attention +
                        ((size_t)attention_head * positions + output_position) *
                            sources;
                    double *gradient_row = workspace.grad_attention +
                        ((size_t)attention_head * positions + output_position) *
                            sources;
                    for (int source = 0; source < sources; source++) {
                        if (!observer_source_visible(
                                candidate_tokens,
                                example->prompt_count,
                                source,
                                masked_source)) {
                            gradient_row[source] = 0.0;
                            continue;
                        }
                        double attention_gradient = 0.0;
                        for (int lane = 0; lane < head_dim; lane++) {
                            int coordinate = lane_start + lane;
                            attention_gradient += workspace.grad_context[
                                (size_t)output_position * dim + coordinate
                            ] * workspace.value[(size_t)source * dim + coordinate];
                            workspace.grad_value[
                                (size_t)source * dim + coordinate
                            ] += attention_row[source] * workspace.grad_context[
                                (size_t)output_position * dim + coordinate
                            ];
                        }
                        gradient_row[source] = attention_gradient;
                    }
                    double softmax_dot = 0.0;
                    for (int source = 0; source < sources; source++) {
                        if (!observer_source_visible(
                                candidate_tokens,
                                example->prompt_count,
                                source,
                                masked_source)) continue;
                        softmax_dot += gradient_row[source] * attention_row[source];
                    }
                    for (int source = 0; source < sources; source++) {
                        if (!observer_source_visible(
                                candidate_tokens,
                                example->prompt_count,
                                source,
                                masked_source)) continue;
                        double score_gradient = attention_row[source] *
                            (gradient_row[source] - softmax_dot);
                        for (int lane = 0; lane < head_dim; lane++) {
                            int coordinate = lane_start + lane;
                            workspace.grad_query[
                                (size_t)output_position * dim + coordinate
                            ] += score_gradient * workspace.key[
                                (size_t)source * dim + coordinate
                            ] * attention_scale;
                            workspace.grad_key[
                                (size_t)source * dim + coordinate
                            ] += score_gradient * workspace.query[
                                (size_t)output_position * dim + coordinate
                            ] * attention_scale;
                        }
                    }
                }
            }

            /* Q/K were rotary-positioned before attention. Undo that fixed
             * orthogonal map before differentiating their learned matrices. */
            observer_apply_rope(
                workspace.grad_query,
                positions,
                dim,
                head->head_count,
                example->prompt_count,
                -1
            );
            observer_apply_rope(
                workspace.grad_key,
                sources,
                dim,
                head->head_count,
                0,
                -1
            );

            for (int out = 0; out < dim; out++) {
                for (int in = 0; in < dim; in++) {
                    for (int position = 0; position < positions; position++) {
                        gradient_q[(size_t)out * dim + in] +=
                            workspace.grad_query[(size_t)position * dim + out] *
                            example->hidden[(size_t)position * dim + in];
                    }
                    for (int source = 0; source < sources; source++) {
                        double source_input = workspace.source_input[
                            (size_t)source * dim + in
                        ];
                        gradient_k[(size_t)out * dim + in] +=
                            workspace.grad_key[(size_t)source * dim + out] *
                            source_input;
                        gradient_v[(size_t)out * dim + in] +=
                            workspace.grad_value[(size_t)source * dim + out] *
                            source_input;
                    }
                }
            }
        }
        observer_workspace_free(&workspace);
        free(candidate_tokens);
    }

    double observation_count = (double)example_count * positions;
    loss /= observation_count;
    if (gradient != NULL) {
        for (size_t parameter = 0;
             parameter < head->parameter_count;
             parameter++) {
            gradient[parameter] /= observation_count;
        }
    }
    for (size_t parameter = 0;
         parameter < head->parameter_count;
         parameter++) {
        double displacement = head->parameters[parameter] -
            observer_initial_parameter(head, parameter);
        loss += 0.5 * l2 * displacement * displacement;
        if (gradient != NULL) gradient[parameter] += l2 * displacement;
    }
    if (token_accuracy != NULL) {
        *token_accuracy = (double)correct_tokens / observation_count;
    }
    if (sequence_accuracy != NULL) {
        *sequence_accuracy =
            (double)correct_sequences / (double)example_count;
    }
    return loss;
}

static void observer_train(
    const ObserverExample *training,
    int training_count,
    const ObserverExample *validation,
    int validation_count,
    const Transformer *student,
    ObserverHead *head,
    int positions,
    int epochs,
    double learning_rate,
    double l2
) {
    size_t count = head->parameter_count;
    double *gradient = observer_allocate(count, sizeof(double));
    double *first_moment = observer_allocate(count, sizeof(double));
    double *second_moment = observer_allocate(count, sizeof(double));
    double *best = observer_allocate(count, sizeof(double));
    memcpy(best, head->parameters, count * sizeof(double));
    double best_validation = DBL_MAX;
    double beta1_power = 1.0;
    double beta2_power = 1.0;
    for (int epoch = 1; epoch <= epochs; epoch++) {
        double train_token = 0.0;
        double train_sequence = 0.0;
        double train_loss = observer_dataset_loss(
            training,
            training_count,
            student,
            head,
            positions,
            epoch,
            l2,
            gradient,
            &train_token,
            &train_sequence
        );
        double squared_norm = 0.0;
        for (size_t parameter = 0; parameter < count; parameter++) {
            squared_norm += gradient[parameter] * gradient[parameter];
        }
        double gradient_scale = squared_norm > 25.0 ?
            5.0 / sqrt(squared_norm) : 1.0;
        beta1_power *= 0.9;
        beta2_power *= 0.999;
        for (size_t parameter = 0; parameter < count; parameter++) {
            double value = gradient[parameter] * gradient_scale;
            first_moment[parameter] =
                0.9 * first_moment[parameter] + 0.1 * value;
            second_moment[parameter] =
                0.999 * second_moment[parameter] + 0.001 * value * value;
            double first = first_moment[parameter] / (1.0 - beta1_power);
            double second = second_moment[parameter] / (1.0 - beta2_power);
            head->parameters[parameter] -= learning_rate * first /
                (sqrt(second) + 1e-8);
        }
        double validation_token = 0.0;
        double validation_sequence = 0.0;
        double validation_loss = observer_dataset_loss(
            validation,
            validation_count,
            student,
            head,
            positions,
            0,
            l2,
            NULL,
            &validation_token,
            &validation_sequence
        );
        if (validation_loss < best_validation) {
            best_validation = validation_loss;
            memcpy(best, head->parameters, count * sizeof(double));
        }
        fprintf(
            stderr,
            "epoch=%d train_loss=%.6f train_token=%.4f "
            "train_sequence=%.4f validation_loss=%.6f "
            "validation_token=%.4f validation_sequence=%.4f "
            "gradient_norm=%.6f\n",
            epoch,
            train_loss,
            train_token,
            train_sequence,
            validation_loss,
            validation_token,
            validation_sequence,
            sqrt(squared_norm)
        );
    }
    memcpy(head->parameters, best, count * sizeof(double));
    free(best);
    free(second_moment);
    free(first_moment);
    free(gradient);
}

static void observer_select_top_k(
    const float *logits,
    int vocab,
    int top_k,
    int *tokens
) {
    for (int rank = 0; rank < top_k; rank++) tokens[rank] = -1;
    for (int token = 0; token < vocab; token++) {
        int insert = top_k;
        for (int rank = 0; rank < top_k; rank++) {
            if (tokens[rank] < 0 || logits[token] > logits[tokens[rank]]) {
                insert = rank;
                break;
            }
        }
        if (insert == top_k) continue;
        for (int rank = top_k - 1; rank > insert; rank--) {
            tokens[rank] = tokens[rank - 1];
        }
        tokens[insert] = token;
    }
}

static void observer_select_top_k_double(
    const double *logits,
    int vocab,
    int top_k,
    int *tokens
) {
    for (int rank = 0; rank < top_k; rank++) tokens[rank] = -1;
    for (int token = 0; token < vocab; token++) {
        int insert = top_k;
        for (int rank = 0; rank < top_k; rank++) {
            if (tokens[rank] < 0 || logits[token] > logits[tokens[rank]]) {
                insert = rank;
                break;
            }
        }
        if (insert == top_k) continue;
        for (int rank = top_k - 1; rank > insert; rank--) {
            tokens[rank] = tokens[rank - 1];
        }
        tokens[insert] = token;
    }
}

static void observer_json_string(FILE *file, const char *text) {
    fputc('"', file);
    const unsigned char *cursor = (const unsigned char *)text;
    while (*cursor != '\0') {
        unsigned char byte = *cursor++;
        if (byte == '"' || byte == '\\') {
            fputc('\\', file);
            fputc(byte, file);
        } else if (byte == '\n') {
            fputs("\\n", file);
        } else if (byte == '\r') {
            fputs("\\r", file);
        } else if (byte == '\t') {
            fputs("\\t", file);
        } else if (byte < 0x20) {
            fprintf(file, "\\u%04x", byte);
        } else {
            fputc(byte, file);
        }
    }
    fputc('"', file);
}

static void observer_trace_flush(FILE *trace) {
    if (trace != NULL && fflush(trace) != 0) observer_fail("trace flush failed");
}

typedef struct {
    int depth;
    int parent;
    int parent_rank;
    int winner_outcome;
    unsigned long long visits;
} ObserverTrieNode;

typedef struct {
    int positions;
    int top_k;
    int node_count;
    int node_capacity;
    ObserverTrieNode *nodes;
    int *children;             /* node_capacity x top_k, -1 means absent */
    int outcome_count;
    int outcome_capacity;
    int *outcome_tokens;       /* outcome_capacity x positions */
    double *outcome_covectors; /* outcome_capacity x positions x top_k */
} ObserverTrie;

static void observer_trie_initialize(
    ObserverTrie *trie,
    int positions,
    int top_k
) {
    *trie = (ObserverTrie){
        .positions = positions,
        .top_k = top_k,
        .node_capacity = 1024,
        .outcome_capacity = 256,
    };
    trie->nodes = observer_allocate(
        (size_t)trie->node_capacity,
        sizeof(*trie->nodes)
    );
    trie->children = observer_allocate(
        (size_t)trie->node_capacity * top_k,
        sizeof(int)
    );
    for (size_t index = 0;
         index < (size_t)trie->node_capacity * top_k;
         index++) {
        trie->children[index] = -1;
    }
    trie->outcome_tokens = observer_allocate(
        (size_t)trie->outcome_capacity * positions,
        sizeof(int)
    );
    trie->outcome_covectors = observer_allocate(
        (size_t)trie->outcome_capacity * positions * top_k,
        sizeof(double)
    );
    trie->node_count = 1;
    trie->nodes[0] = (ObserverTrieNode){
        .depth = 0,
        .parent = -1,
        .parent_rank = -1,
        .winner_outcome = -1,
    };
}

static void observer_trie_grow_nodes(ObserverTrie *trie) {
    int old_capacity = trie->node_capacity;
    if (old_capacity > INT_MAX / 2) observer_fail("too many trie nodes");
    trie->node_capacity *= 2;
    ObserverTrieNode *nodes = realloc(
        trie->nodes,
        (size_t)trie->node_capacity * sizeof(*nodes)
    );
    int *children = realloc(
        trie->children,
        (size_t)trie->node_capacity * trie->top_k * sizeof(int)
    );
    if (nodes == NULL || children == NULL) observer_fail("could not grow trie");
    trie->nodes = nodes;
    trie->children = children;
    size_t old_count = (size_t)old_capacity * trie->top_k;
    size_t new_count = (size_t)trie->node_capacity * trie->top_k;
    for (size_t index = old_count; index < new_count; index++) {
        trie->children[index] = -1;
    }
}

static void observer_trie_grow_outcomes(ObserverTrie *trie) {
    int old_capacity = trie->outcome_capacity;
    if (old_capacity > INT_MAX / 2) observer_fail("too many outcomes");
    trie->outcome_capacity *= 2;
    int *tokens = realloc(
        trie->outcome_tokens,
        (size_t)trie->outcome_capacity * trie->positions * sizeof(int)
    );
    double *covectors = realloc(
        trie->outcome_covectors,
        (size_t)trie->outcome_capacity * trie->positions * trie->top_k *
            sizeof(double)
    );
    if (tokens == NULL || covectors == NULL) {
        observer_fail("could not grow outcomes");
    }
    trie->outcome_tokens = tokens;
    trie->outcome_covectors = covectors;
}

static int observer_trie_insert_path(
    ObserverTrie *trie,
    const int *ranks,
    int *path_nodes
) {
    int node = 0;
    path_nodes[0] = 0;
    trie->nodes[node].visits++;
    for (int position = 0; position < trie->positions; position++) {
        int rank = ranks[position];
        int *child_slot = trie->children +
            (size_t)node * trie->top_k + rank;
        if (*child_slot < 0) {
            if (trie->node_count == trie->node_capacity) {
                int parent = node;
                observer_trie_grow_nodes(trie);
                node = parent;
                child_slot = trie->children +
                    (size_t)node * trie->top_k + rank;
            }
            int child = trie->node_count++;
            trie->nodes[child] = (ObserverTrieNode){
                .depth = position + 1,
                .parent = node,
                .parent_rank = rank,
                .winner_outcome = -1,
            };
            *child_slot = child;
        }
        node = *child_slot;
        trie->nodes[node].visits++;
        path_nodes[position + 1] = node;
    }
    return node;
}

static int observer_trie_store_outcome(
    ObserverTrie *trie,
    const int *tokens,
    const double *covectors
) {
    if (trie->outcome_count == trie->outcome_capacity) {
        observer_trie_grow_outcomes(trie);
    }
    int outcome = trie->outcome_count++;
    memcpy(
        trie->outcome_tokens + (size_t)outcome * trie->positions,
        tokens,
        (size_t)trie->positions * sizeof(int)
    );
    memcpy(
        trie->outcome_covectors +
            (size_t)outcome * trie->positions * trie->top_k,
        covectors,
        (size_t)trie->positions * trie->top_k * sizeof(double)
    );
    return outcome;
}

static int observer_trie_recompute_node(
    ObserverTrie *trie,
    int node,
    FILE *trace
) {
    int depth = trie->nodes[node].depth;
    if (depth >= trie->positions) return trie->nodes[node].winner_outcome;
    int winner = -1;
    int winning_rank = -1;
    int local_winner_rank = -1;
    int fallback = -1;
    int fallback_rank = -1;
    int fallback_local_winner = -1;
    for (int rank = 0; rank < trie->top_k; rank++) {
        int child = trie->children[(size_t)node * trie->top_k + rank];
        if (child < 0) continue;
        int outcome = trie->nodes[child].winner_outcome;
        if (outcome < 0) continue;
        const double *covector = trie->outcome_covectors +
            ((size_t)outcome * trie->positions + depth) * trie->top_k;
        int selected = 0;
        for (int alternative = 1;
             alternative < trie->top_k;
             alternative++) {
            if (covector[alternative] > covector[selected]) {
                selected = alternative;
            }
        }
        fallback = outcome;
        fallback_rank = rank;
        fallback_local_winner = selected;
        if (selected == rank) {
            winner = outcome;
            winning_rank = rank;
            local_winner_rank = selected;
            break;
        }
    }
    if (winner < 0) {
        winner = fallback;
        winning_rank = fallback_rank;
        local_winner_rank = fallback_local_winner;
    }
    int changed = winner != trie->nodes[node].winner_outcome;
    trie->nodes[node].winner_outcome = winner;
    if (changed && trace != NULL && winner >= 0) {
        fprintf(
            trace,
            "{\"event\":\"select_backup\",\"node\":%d,\"depth\":%d,"
            "\"winning_rank\":%d,\"outcome\":%d," 
            "\"local_covector_winner_rank\":%d,"
            "\"attains\":%s,"
            "\"propagated\":\"complete_covector_family\"}\n",
            node,
            depth,
            winning_rank + 1,
            winner,
            local_winner_rank + 1,
            local_winner_rank == winning_rank ? "true" : "false"
        );
        observer_trace_flush(trace);
    }
    return winner;
}

static void observer_trie_backup_path(
    ObserverTrie *trie,
    const int *path_nodes,
    int outcome,
    FILE *trace
) {
    int leaf = path_nodes[trie->positions];
    trie->nodes[leaf].winner_outcome = outcome;
    for (int depth = trie->positions - 1; depth >= 0; depth--) {
        observer_trie_recompute_node(trie, path_nodes[depth], trace);
    }
}

static void observer_trie_free(ObserverTrie *trie) {
    free(trie->outcome_covectors);
    free(trie->outcome_tokens);
    free(trie->children);
    free(trie->nodes);
    memset(trie, 0, sizeof(*trie));
}

static void observer_trace_outcome(
    FILE *trace,
    int outcome,
    const ObserverExample *example,
    Tokenizer *tokenizer,
    const int *tokens,
    const double *covectors,
    int positions,
    int top_k
) {
    if (trace == NULL) return;
    fprintf(trace, "{\"event\":\"outcome\",\"outcome\":%d,\"tokens\":[", outcome);
    for (int position = 0; position < positions; position++) {
        if (position != 0) fputc(',', trace);
        fprintf(trace, "%d", tokens[position]);
    }
    fputs("],\"covectors\":[", trace);
    for (int position = 0; position < positions; position++) {
        if (position != 0) fputc(',', trace);
        fputc('[', trace);
        for (int rank = 0; rank < top_k; rank++) {
            if (rank != 0) fputc(',', trace);
            fprintf(
                trace,
                "%.17g",
                covectors[(size_t)position * top_k + rank]
            );
        }
        fputc(']', trace);
    }
    fputs("],\"text\":", trace);
    fputc('"', trace);
    int previous = example->prompt_tokens[example->prompt_count - 1];
    for (int position = 0; position < positions; position++) {
        const unsigned char *piece = (const unsigned char *)decode(
            tokenizer,
            previous,
            tokens[position]
        );
        while (*piece != '\0') {
            unsigned char byte = *piece++;
            if (byte == '"' || byte == '\\') {
                fputc('\\', trace);
                fputc(byte, trace);
            } else if (byte == '\n') {
                fputs("\\n", trace);
            } else if (byte == '\r') {
                fputs("\\r", trace);
            } else if (byte == '\t') {
                fputs("\\t", trace);
            } else if (byte >= 0x20) {
                fputc(byte, trace);
            }
        }
        previous = tokens[position];
    }
    fputs("\"}\n", trace);
    observer_trace_flush(trace);
}

static int observer_evaluate_path(
    ObserverTrie *trie,
    const int *ranks,
    const int *carrier,
    const ObserverExample *example,
    const Transformer *student,
    const ObserverHead *head,
    Tokenizer *tokenizer,
    ObserverWorkspace *workspace,
    FILE *trace,
    int *path_nodes,
    int *tokens,
    double *covectors
) {
    int leaf = observer_trie_insert_path(trie, ranks, path_nodes);
    if (trie->nodes[leaf].winner_outcome >= 0) return 0;
    for (int position = 0; position < trie->positions; position++) {
        tokens[position] = carrier[
            (size_t)position * trie->top_k + ranks[position]
        ];
    }
    observer_forward(example, tokens, student, head, workspace);
    for (int position = 0; position < trie->positions; position++) {
        for (int rank = 0; rank < trie->top_k; rank++) {
            int token = carrier[(size_t)position * trie->top_k + rank];
            covectors[(size_t)position * trie->top_k + rank] =
                workspace->logits[
                    (size_t)position * workspace->vocab + token
                ];
        }
    }
    int outcome = observer_trie_store_outcome(trie, tokens, covectors);
    observer_trace_outcome(
        trace,
        outcome,
        example,
        tokenizer,
        tokens,
        covectors,
        trie->positions,
        trie->top_k
    );
    observer_trie_backup_path(trie, path_nodes, outcome, trace);
    return 1;
}

static unsigned long long observer_exact_path_count(
    int top_k,
    int positions
) {
    unsigned long long count = 1;
    for (int position = 0; position < positions; position++) {
        if (count > ULLONG_MAX / (unsigned long long)top_k) {
            observer_fail("exact carrier is too large to enumerate");
        }
        count *= (unsigned long long)top_k;
    }
    return count;
}

static void observer_index_ranks(
    unsigned long long index,
    int top_k,
    int positions,
    int *ranks
) {
    for (int position = positions - 1; position >= 0; position--) {
        ranks[position] = (int)(index % (unsigned long long)top_k);
        index /= (unsigned long long)top_k;
    }
}

static int *observer_search(
    const ObserverExample *example,
    const Transformer *student,
    const ObserverHead *head,
    Tokenizer *tokenizer,
    int positions,
    int top_k,
    int exact,
    int sample_ms,
    unsigned long long seed,
    FILE *trace,
    unsigned long long *attempted_out,
    int *unique_out,
    int *node_count_out
) {
    int vocab = student->config.vocab_size;
    if (top_k > vocab) top_k = vocab;
    int *carrier = observer_allocate(
        (size_t)positions * top_k,
        sizeof(int)
    );
    /* The local selections are the model-native covectors retained by the
     * hidden-feedback term. The learned leave-one-out decoder observes a
     * completed tuple; it must not replace those selections with an unrelated
     * all-slots-masked proposal whose rows collapse to the same generic words. */
    for (int position = 0; position < positions; position++) {
        observer_select_top_k(
            example->base_logits + (size_t)position * vocab,
            vocab,
            top_k,
            carrier + (size_t)position * top_k
        );
    }
    if (trace != NULL) {
        fprintf(
            trace,
            "{\"event\":\"run\","
            "\"reward_type\":\"finite_covector_family\","
            "\"backup\":\"selection_product_model_attainment\","
            "\"proposal\":\"model_native_hidden_feedback_covectors\","
            "\"positions\":%d,\"top_k\":%d,\"exact\":%s," 
            "\"sample_ms\":%d,\"seed\":%llu," 
            "\"root_terminalizations\":1}\n",
            positions,
            top_k,
            exact ? "true" : "false",
            sample_ms,
            seed
        );
        observer_trace_flush(trace);
        for (int position = 0; position < positions; position++) {
            for (int rank = 0; rank < top_k; rank++) {
                int token = carrier[(size_t)position * top_k + rank];
                fprintf(
                    trace,
                    "{\"event\":\"carrier_candidate\",\"position\":%d," 
                    "\"rank\":%d,\"token\":%d,\"piece\":",
                    position,
                    rank + 1,
                    token
                );
                observer_json_string(trace, tokenizer->vocab[token]);
                fprintf(
                    trace,
                    ",\"native_proposal_logit\":%.17g}\n",
                    example->base_logits[(size_t)position * vocab + token]
                );
                observer_trace_flush(trace);
            }
        }
    }

    ObserverTrie trie;
    observer_trie_initialize(&trie, positions, top_k);
    ObserverWorkspace workspace;
    observer_workspace_initialize(
        &workspace,
        head,
        positions,
        example->prompt_count + positions,
        vocab,
        0
    );
    int *ranks = observer_allocate((size_t)positions, sizeof(int));
    int *path_nodes = observer_allocate((size_t)positions + 1U, sizeof(int));
    int *tokens = observer_allocate((size_t)positions, sizeof(int));
    double *covectors = observer_allocate(
        (size_t)positions * top_k,
        sizeof(double)
    );
    unsigned long long attempted = 0;
    long start = time_in_ms();
    if (exact) {
        unsigned long long total = observer_exact_path_count(top_k, positions);
        for (unsigned long long index = 0; index < total; index++) {
            observer_index_ranks(index, top_k, positions, ranks);
            observer_evaluate_path(
                &trie,
                ranks,
                carrier,
                example,
                student,
                head,
                tokenizer,
                &workspace,
                trace,
                path_nodes,
                tokens,
                covectors
            );
            attempted++;
        }
    } else {
        uint64_t random_state = seed != 0 ? seed : UINT64_C(1);
        unsigned long long systematic = 0;
        do {
            if (systematic < (unsigned long long)top_k) {
                for (int position = 0; position < positions; position++) {
                    ranks[position] = (int)(
                        (systematic + (unsigned long long)position) %
                        (unsigned long long)top_k
                    );
                }
                systematic++;
            } else {
                for (int position = 0; position < positions; position++) {
                    ranks[position] = (int)(
                        observer_random_u64(&random_state) %
                        (uint64_t)top_k
                    );
                }
            }
            observer_evaluate_path(
                &trie,
                ranks,
                carrier,
                example,
                student,
                head,
                tokenizer,
                &workspace,
                trace,
                path_nodes,
                tokens,
                covectors
            );
            attempted++;
        } while (time_in_ms() - start < sample_ms);
    }
    int winner = trie.nodes[0].winner_outcome;
    if (winner < 0) observer_fail("search produced no complete outcome");
    int *selected = observer_allocate((size_t)positions, sizeof(int));
    memcpy(
        selected,
        trie.outcome_tokens + (size_t)winner * positions,
        (size_t)positions * sizeof(int)
    );
    if (trace != NULL) {
        fprintf(
            trace,
            "{\"event\":\"root_terminalized\",\"outcome\":%d," 
            "\"attempted\":%llu,\"unique\":%d,\"nodes\":%d," 
            "\"tokens\":[",
            winner,
            attempted,
            trie.outcome_count,
            trie.node_count
        );
        for (int position = 0; position < positions; position++) {
            if (position != 0) fputc(',', trace);
            fprintf(trace, "%d", selected[position]);
        }
        fputs("]}\n", trace);
        observer_trace_flush(trace);
    }
    *attempted_out = attempted;
    *unique_out = trie.outcome_count;
    *node_count_out = trie.node_count;
    free(covectors);
    free(tokens);
    free(path_nodes);
    free(ranks);
    observer_workspace_free(&workspace);
    observer_trie_free(&trie);
    free(carrier);
    return selected;
}

static void observer_print_token_span(
    Tokenizer *tokenizer,
    int previous,
    const int *tokens,
    int count
) {
    for (int index = 0; index < count; index++) {
        safe_printf(decode(tokenizer, previous, tokens[index]));
        previous = tokens[index];
    }
}

static void observer_usage(const char *program) {
    fprintf(
        stderr,
        "usage:\n"
        "  %s train TEACHER.bin TEACHER_TOKENIZER.bin STUDENT.bin "
        "STUDENT_TOKENIZER.bin OUTPUT.bin [options]\n"
        "  %s infer STUDENT.bin STUDENT_TOKENIZER.bin HEAD.bin [options]\n\n"
        "train options:\n"
        "  --train-examples N       default 64\n"
        "  --validation-examples N  default 16\n"
        "  --prompt-tokens N        default 16\n"
        "  --completion-tokens N    default 16\n"
        "  --teacher-steps N        default 96\n"
        "  --epochs N               default 30\n"
        "  --seed N                 default 26015\n"
        "  --learning-rate X        default 0.001\n"
        "  --l2 X                   default 0.000001\n\n"
        "infer options:\n"
        "  --prompt TEXT            default 'Lily was'\n"
        "  --length N               default 32\n"
        "  --top-k N                default 4\n"
        "  --sample-ms N            default 1000\n"
        "  --seed N                 default 42\n"
        "  --trace PATH             default observer-candidates.jsonl\n"
        "  --exact                  enumerate the whole finite carrier\n",
        program,
        program
    );
    exit(EXIT_FAILURE);
}

static void observer_parse_train_options(
    int argc,
    char **argv,
    int start,
    ObserverTrainOptions *options
) {
    *options = (ObserverTrainOptions){
        .training_count = 64,
        .validation_count = 16,
        .prompt_count = 16,
        .completion_count = 16,
        .teacher_steps = 96,
        .epochs = 30,
        .seed = 26015,
        .learning_rate = 0.001,
        .l2 = 1e-6,
    };
    for (int index = start; index < argc; index += 2) {
        if (index + 1 >= argc) observer_usage(argv[0]);
        const char *flag = argv[index];
        const char *value = argv[index + 1];
        if (strcmp(flag, "--train-examples") == 0) {
            options->training_count = observer_parse_positive(value, "train examples");
        } else if (strcmp(flag, "--validation-examples") == 0) {
            options->validation_count = observer_parse_positive(
                value,
                "validation examples"
            );
        } else if (strcmp(flag, "--prompt-tokens") == 0) {
            options->prompt_count = observer_parse_positive(value, "prompt tokens");
        } else if (strcmp(flag, "--completion-tokens") == 0) {
            options->completion_count = observer_parse_positive(
                value,
                "completion tokens"
            );
        } else if (strcmp(flag, "--teacher-steps") == 0) {
            options->teacher_steps = observer_parse_positive(value, "teacher steps");
        } else if (strcmp(flag, "--epochs") == 0) {
            options->epochs = observer_parse_positive(value, "epochs");
        } else if (strcmp(flag, "--seed") == 0) {
            options->seed = observer_parse_seed(value);
        } else if (strcmp(flag, "--learning-rate") == 0) {
            options->learning_rate = observer_parse_positive_double(
                value,
                "learning rate"
            );
        } else if (strcmp(flag, "--l2") == 0) {
            options->l2 = observer_parse_positive_double(value, "l2");
        } else {
            observer_usage(argv[0]);
        }
    }
}

static void observer_parse_infer_options(
    int argc,
    char **argv,
    int start,
    ObserverInferOptions *options
) {
    *options = (ObserverInferOptions){
        .prompt = "Lily was",
        .length = 32,
        .top_k = 4,
        .sample_ms = 1000,
        .exact = 0,
        .seed = 42,
        .trace_path = "observer-candidates.jsonl",
    };
    for (int index = start; index < argc;) {
        const char *flag = argv[index];
        if (strcmp(flag, "--exact") == 0) {
            options->exact = 1;
            index++;
            continue;
        }
        if (index + 1 >= argc) observer_usage(argv[0]);
        const char *value = argv[index + 1];
        if (strcmp(flag, "--prompt") == 0) {
            options->prompt = value;
        } else if (strcmp(flag, "--length") == 0) {
            options->length = observer_parse_positive(value, "length");
        } else if (strcmp(flag, "--top-k") == 0) {
            options->top_k = observer_parse_positive(value, "top-k");
        } else if (strcmp(flag, "--sample-ms") == 0) {
            options->sample_ms = observer_parse_positive(value, "sample-ms");
        } else if (strcmp(flag, "--seed") == 0) {
            options->seed = observer_parse_seed(value);
        } else if (strcmp(flag, "--trace") == 0) {
            options->trace_path = value;
        } else {
            observer_usage(argv[0]);
        }
        index += 2;
    }
}

static int observer_train_main(int argc, char **argv) {
    if (argc < 7) observer_usage(argv[0]);
    const char *teacher_checkpoint = argv[2];
    const char *teacher_tokenizer_path = argv[3];
    const char *student_checkpoint = argv[4];
    const char *student_tokenizer_path = argv[5];
    const char *output_path = argv[6];
    ObserverTrainOptions options;
    observer_parse_train_options(argc, argv, 7, &options);
    Transformer teacher;
    Transformer student;
    build_transformer(&teacher, (char *)teacher_checkpoint);
    build_transformer(&student, (char *)student_checkpoint);
    Tokenizer teacher_tokenizer;
    Tokenizer student_tokenizer;
    build_tokenizer(
        &teacher_tokenizer,
        (char *)teacher_tokenizer_path,
        teacher.config.vocab_size
    );
    build_tokenizer(
        &student_tokenizer,
        (char *)student_tokenizer_path,
        student.config.vocab_size
    );
    if (options.teacher_steps > teacher.config.seq_len) {
        observer_fail("teacher steps exceed teacher sequence length");
    }
    if (options.prompt_count + options.completion_count - 1 >
        student.config.seq_len) {
        observer_fail("training span exceeds student sequence length");
    }
    Sampler teacher_sampler;
    build_sampler(
        &teacher_sampler,
        teacher.config.vocab_size,
        0.8f,
        0.9f,
        options.seed
    );
    fprintf(
        stderr,
        "mode=train_joint_observer teacher_vocab=%d student_vocab=%d "
        "student_dim=%d train=%d validation=%d prompt=%d completion=%d "
        "epochs=%d seed=%llu\n",
        teacher.config.vocab_size,
        student.config.vocab_size,
        student.config.dim,
        options.training_count,
        options.validation_count,
        options.prompt_count,
        options.completion_count,
        options.epochs,
        options.seed
    );
    ObserverExample *training = observer_generate_dataset(
        options.training_count,
        &teacher,
        &teacher_tokenizer,
        &teacher_sampler,
        &student,
        &student_tokenizer,
        options.prompt_count,
        options.completion_count,
        options.teacher_steps,
        "training"
    );
    ObserverExample *validation = observer_generate_dataset(
        options.validation_count,
        &teacher,
        &teacher_tokenizer,
        &teacher_sampler,
        &student,
        &student_tokenizer,
        options.prompt_count,
        options.completion_count,
        options.teacher_steps,
        "validation"
    );
    ObserverHead head;
    observer_head_initialize(
        &head,
        student.config.dim,
        student.config.n_heads,
        options.seed ^ UINT64_C(0x9e3779b97f4a7c15)
    );
    double initial_token = 0.0;
    double initial_sequence = 0.0;
    double initial_loss = observer_dataset_loss(
        validation,
        options.validation_count,
        &student,
        &head,
        options.completion_count,
        0,
        options.l2,
        NULL,
        &initial_token,
        &initial_sequence
    );
    fprintf(
        stderr,
        "initial validation_loss=%.6f validation_token=%.4f "
        "validation_sequence=%.4f\n",
        initial_loss,
        initial_token,
        initial_sequence
    );
    observer_train(
        training,
        options.training_count,
        validation,
        options.validation_count,
        &student,
        &head,
        options.completion_count,
        options.epochs,
        options.learning_rate,
        options.l2
    );
    observer_head_save(output_path, &head);
    observer_head_free(&head);
    for (int index = 0; index < options.validation_count; index++) {
        observer_example_free(&validation[index]);
    }
    for (int index = 0; index < options.training_count; index++) {
        observer_example_free(&training[index]);
    }
    free(validation);
    free(training);
    free_sampler(&teacher_sampler);
    free_tokenizer(&student_tokenizer);
    free_tokenizer(&teacher_tokenizer);
    free_transformer(&student);
    free_transformer(&teacher);
    return 0;
}

static int observer_infer_main(int argc, char **argv) {
    if (argc < 5) observer_usage(argv[0]);
    const char *checkpoint = argv[2];
    const char *tokenizer_path = argv[3];
    const char *head_path = argv[4];
    ObserverInferOptions options;
    observer_parse_infer_options(argc, argv, 5, &options);
    Transformer student;
    build_transformer(&student, (char *)checkpoint);
    Tokenizer tokenizer;
    build_tokenizer(
        &tokenizer,
        (char *)tokenizer_path,
        student.config.vocab_size
    );
    ObserverHead head;
    observer_head_load(head_path, &head);
    if (head.dim != student.config.dim) {
        observer_fail("observer head dimension does not match checkpoint");
    }
    ObserverExample example = {0};
    long recurrence_start = time_in_ms();
    observer_build_prompt_example(
        &example,
        &student,
        &tokenizer,
        options.prompt,
        options.length
    );
    long recurrence_end = time_in_ms();
    FILE *trace = fopen(options.trace_path, "w");
    if (trace == NULL) {
        fprintf(stderr, "could not open trace %s\n", options.trace_path);
        exit(EXIT_FAILURE);
    }
    unsigned long long attempted = 0;
    int unique = 0;
    int nodes = 0;
    long search_start = time_in_ms();
    int *selected = observer_search(
        &example,
        &student,
        &head,
        &tokenizer,
        options.length,
        options.top_k,
        options.exact,
        options.sample_ms,
        options.seed,
        trace,
        &attempted,
        &unique,
        &nodes
    );
    long search_end = time_in_ms();
    if (fclose(trace) != 0) observer_fail("could not close trace");
    observer_print_token_span(
        &tokenizer,
        example.prompt_tokens[0],
        example.prompt_tokens + 1,
        example.prompt_count - 1
    );
    observer_print_token_span(
        &tokenizer,
        example.prompt_tokens[example.prompt_count - 1],
        selected,
        options.length
    );
    printf("\n");
    fflush(stdout);
    fprintf(
        stderr,
        "mode=ordered_masked_joint_decoder positions=%d top_k=%d exact=%d "
        "attempted=%llu unique=%d nodes=%d recurrence_ms=%ld search_ms=%ld "
        "trace=%s\n",
        options.length,
        options.top_k,
        options.exact,
        attempted,
        unique,
        nodes,
        recurrence_end - recurrence_start,
        search_end - search_start,
        options.trace_path
    );
    free(selected);
    observer_example_free(&example);
    observer_head_free(&head);
    free_tokenizer(&tokenizer);
    free_transformer(&student);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) observer_usage(argv[0]);
    if (strcmp(argv[1], "train") == 0) return observer_train_main(argc, argv);
    if (strcmp(argv[1], "infer") == 0) return observer_infer_main(argc, argv);
    observer_usage(argv[0]);
    return EXIT_FAILURE;
}
