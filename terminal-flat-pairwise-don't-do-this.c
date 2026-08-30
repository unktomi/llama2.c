/*
 * QUARANTINED: DO NOT USE THIS FLAT PAIRWISE CHAIN AS THE JOINT PROJECTION.
 *
 * Although `select_suffix` recursively forced continuations, the observer it
 * propagated was only a scalar sum of adjacent pair energies.  It therefore
 * collapsed each suffix to a first-order Markov value rather than retaining
 * a selection/observer at every recursively composed span.  In real 260K
 * runs it learned repeated-token attractors (for example long runs of "her"
 * and "the") while lowering its training loss.  That is precisely the flat
 * failure this experiment was meant to avoid.
 *
 * Historical description follows.
 *
 * Joint terminal projection for the hidden-feedback llama2.c experiment.
 *
 * The prompt is processed normally.  After the final prompt token, the final
 * hidden state is fed into the next causal position without first selecting a
 * token.  The complete retained hidden sequence is then exposed to ONE joint
 * observer of a token tuple.
 *
 * The observer is a globally-normalized, hidden-conditioned chain energy:
 *
 *   P_H(t[0..n)) =
 *       u_scale * sum_i centered_logit(H[i], t[i])
 *     + p_scale * sum_i <Q z(H[i], t[i-1]), K z(H[i], t[i])> / sqrt(rank)
 *
 * where t[-1] is the final prompt token and
 *
 *   z(H,t) = normalize([ E[t], H .* E[t] ]).
 *
 * This is not the quarantined row-wise head and not a corpus bigram table:
 * changing a token changes the hidden-conditioned continuation score.  The
 * teacher supplies only an observed complete tuple.  Training uses the
 * log-semiring forward/backward contraction over the complete finite carrier;
 * it never trains independent token classifiers.  Inference uses the same
 * term in the max-plus semiring and stores its witnesses.  `select_suffix` is
 * Escardo's product specialized by this factorization: each local candidate
 * is rated by forcing its memoized continuation, and only the root is run.
 *
 * The finite carrier is explicit (`--top-k`) and every target is inserted into
 * its training carrier.  The matrices are shared over positions, so training
 * completion length does not become an inference horizon bound.
 */

#define TESTING
#include "run_hidden_feedback.c"

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>

typedef struct {
    float *hidden;             /* positions x dim */
    float *logits;             /* positions x vocab */
    int *targets;              /* positions */
    int *prompt_tokens;
    int prompt_count;
} JointExample;

typedef struct {
    int dim;
    int feature_dim;
    int rank;
    double unary_scale;
    double pair_scale;
    double *query_weight;      /* rank x feature_dim */
    double *key_weight;        /* rank x feature_dim */
} JointHead;

typedef struct {
    int positions;
    int top_k;
    int dim;
    int feature_dim;
    int rank;
    int vocab_size;
    int preceding_token;
    int *tokens;               /* positions x top_k */
    double *unary;             /* centered model logits */
    double *query_feature;     /* positions x top_k x feature_dim */
    double *key_feature;       /* positions x top_k x feature_dim */
    double *query;             /* positions x top_k x rank */
    double *key;               /* positions x top_k x rank */
    int *target_rank;          /* positions, training/evaluation only */
    unsigned char *target_natural;
} JointLattice;

typedef struct {
    unsigned char state;       /* 0=unforced, 1=forcing, 2=forced */
    int choice;
    double value;
} SelectionCell;

typedef struct {
    const JointHead *head;
    const JointLattice *lattice;
    Tokenizer *tokenizer;
    FILE *trace;
    SelectionCell *cells;      /* positions x top_k; root uses cell [0,0] */
} SelectionRun;

typedef struct {
    int training_count;
    int validation_count;
    int prompt_count;
    int completion_count;
    int teacher_steps;
    int epochs;
    int top_k;
    int rank;
    unsigned long long seed;
    double learning_rate;
    double l2;
} TrainOptions;

typedef struct {
    const char *prompt;
    int length;
    int top_k;
    const char *trace_path;
} InferOptions;

static void joint_fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

static void *joint_allocate(size_t count, size_t size) {
    if (count != 0 && size > SIZE_MAX / count) {
        joint_fail("joint projection allocation overflow");
    }
    void *memory = calloc(count, size);
    if (memory == NULL) joint_fail("joint projection allocation failed");
    return memory;
}

static int parse_positive(const char *text, const char *name) {
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

static unsigned long long parse_seed(const char *text) {
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0) {
        joint_fail("seed must be a positive integer");
    }
    return value;
}

static double parse_positive_double(const char *text, const char *name) {
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

static uint64_t joint_random_u64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}

static double joint_random_signed(uint64_t *state) {
    return 2.0 * (double)(joint_random_u64(state) >> 11) /
        9007199254740992.0 - 1.0;
}

static void joint_head_initialize(
    JointHead *head,
    int dim,
    int rank,
    unsigned long long seed
) {
    if (dim <= 0 || rank <= 0) joint_fail("invalid joint head dimensions");
    head->dim = dim;
    head->feature_dim = 2 * dim;
    head->rank = rank;
    head->unary_scale = 0.25;
    head->pair_scale = 1.0;
    size_t matrix_count = (size_t)rank * head->feature_dim;
    head->query_weight = joint_allocate(matrix_count, sizeof(double));
    head->key_weight = joint_allocate(matrix_count, sizeof(double));
    uint64_t random_state = seed != 0 ? seed : UINT64_C(1);
    double scale = 1.0 / sqrt((double)head->feature_dim);
    for (size_t index = 0; index < matrix_count; index++) {
        head->query_weight[index] = scale * joint_random_signed(&random_state);
        head->key_weight[index] = scale * joint_random_signed(&random_state);
    }
}

static void joint_head_free(JointHead *head) {
    free(head->key_weight);
    free(head->query_weight);
    memset(head, 0, sizeof(*head));
}

static size_t joint_parameter_count(const JointHead *head) {
    return 2U + 2U * (size_t)head->rank * head->feature_dim;
}

static void joint_parameters_read(const JointHead *head, double *parameters) {
    size_t matrix_count = (size_t)head->rank * head->feature_dim;
    parameters[0] = head->unary_scale;
    parameters[1] = head->pair_scale;
    memcpy(parameters + 2, head->query_weight, matrix_count * sizeof(double));
    memcpy(
        parameters + 2 + matrix_count,
        head->key_weight,
        matrix_count * sizeof(double)
    );
}

static void joint_parameters_write(JointHead *head, const double *parameters) {
    size_t matrix_count = (size_t)head->rank * head->feature_dim;
    head->unary_scale = parameters[0];
    head->pair_scale = parameters[1];
    memcpy(head->query_weight, parameters + 2, matrix_count * sizeof(double));
    memcpy(
        head->key_weight,
        parameters + 2 + matrix_count,
        matrix_count * sizeof(double)
    );
}

static void joint_head_save(const char *path, const JointHead *head) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        fprintf(stderr, "could not open joint head output %s\n", path);
        exit(EXIT_FAILURE);
    }
    const unsigned char magic[8] = {'T','J','P','R','0','0','0','1'};
    size_t matrix_count = (size_t)head->rank * head->feature_dim;
    int ok = fwrite(magic, sizeof(magic), 1, file) == 1 &&
        fwrite(&head->dim, sizeof(head->dim), 1, file) == 1 &&
        fwrite(&head->rank, sizeof(head->rank), 1, file) == 1 &&
        fwrite(&head->unary_scale, sizeof(head->unary_scale), 1, file) == 1 &&
        fwrite(&head->pair_scale, sizeof(head->pair_scale), 1, file) == 1 &&
        fwrite(head->query_weight, sizeof(double), matrix_count, file) ==
            matrix_count &&
        fwrite(head->key_weight, sizeof(double), matrix_count, file) ==
            matrix_count;
    if (!ok || fclose(file) != 0) joint_fail("could not write joint head");
}

static void joint_head_load(const char *path, JointHead *head) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "could not open joint head %s\n", path);
        exit(EXIT_FAILURE);
    }
    unsigned char magic[8];
    int dim = 0;
    int rank = 0;
    if (fread(magic, sizeof(magic), 1, file) != 1 ||
        memcmp(magic, "TJPR0001", sizeof(magic)) != 0 ||
        fread(&dim, sizeof(dim), 1, file) != 1 ||
        fread(&rank, sizeof(rank), 1, file) != 1) {
        joint_fail("invalid joint head file");
    }
    joint_head_initialize(head, dim, rank, 1);
    size_t matrix_count = (size_t)rank * head->feature_dim;
    int ok = fread(&head->unary_scale, sizeof(double), 1, file) == 1 &&
        fread(&head->pair_scale, sizeof(double), 1, file) == 1 &&
        fread(head->query_weight, sizeof(double), matrix_count, file) ==
            matrix_count &&
        fread(head->key_weight, sizeof(double), matrix_count, file) ==
            matrix_count;
    if (!ok) joint_fail("truncated joint head file");
    int trailing = fgetc(file);
    if (trailing != EOF) joint_fail("joint head has trailing data");
    fclose(file);
}

static void joint_example_free(JointExample *example) {
    free(example->prompt_tokens);
    free(example->targets);
    free(example->logits);
    free(example->hidden);
    memset(example, 0, sizeof(*example));
}

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
} JointText;

static void joint_text_initialize(JointText *text, size_t capacity) {
    if (capacity < 64) capacity = 64;
    text->data = joint_allocate(capacity, sizeof(char));
    text->capacity = capacity;
    text->length = 0;
}

static void joint_text_append(JointText *text, const char *piece) {
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
            if (capacity > SIZE_MAX / 2) joint_fail("teacher text too large");
            capacity *= 2;
        }
        char *grown = realloc(text->data, capacity);
        if (grown == NULL) joint_fail("could not grow teacher text");
        text->data = grown;
        text->capacity = capacity;
    }
    memcpy(text->data + text->length, piece, length);
    text->length += length;
    text->data[text->length] = '\0';
}

static char *joint_generate_teacher_story(
    Transformer *teacher,
    Tokenizer *tokenizer,
    Sampler *sampler,
    int steps
) {
    JointText text;
    joint_text_initialize(
        &text,
        (size_t)steps * (tokenizer->max_token_length + 1U) + 1U
    );
    int token = 1;
    for (int position = 0; position < steps; position++) {
        float *logits = forward(teacher, token, position);
        int next = sample(sampler, logits);
        if (next == 1 || next == 2) break;
        joint_text_append(&text, decode(tokenizer, token, next));
        token = next;
    }
    return text.data;
}

static int joint_build_example(
    JointExample *example,
    Transformer *teacher,
    Tokenizer *teacher_tokenizer,
    Sampler *teacher_sampler,
    Transformer *student,
    Tokenizer *student_tokenizer,
    int prompt_count,
    int completion_count,
    int teacher_steps
) {
    char *story = joint_generate_teacher_story(
        teacher,
        teacher_tokenizer,
        teacher_sampler,
        teacher_steps
    );
    size_t story_bytes = strlen(story);
    int *story_tokens = joint_allocate(story_bytes + 3U, sizeof(int));
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
    int content_span = (prompt_count - 1) + completion_count;
    if (story_token_count - 1 < content_span) {
        free(story_tokens);
        return 0;
    }
    if (prompt_count + completion_count - 1 > student->config.seq_len) {
        joint_fail("student prompt and completion exceed sequence length");
    }

    example->prompt_count = prompt_count;
    example->prompt_tokens = joint_allocate((size_t)prompt_count, sizeof(int));
    example->targets = joint_allocate((size_t)completion_count, sizeof(int));
    example->hidden = joint_allocate(
        (size_t)completion_count * student->config.dim,
        sizeof(float)
    );
    example->logits = joint_allocate(
        (size_t)completion_count * student->config.vocab_size,
        sizeof(float)
    );
    int maximum_start = story_token_count - content_span;
    int content_start = 1;
    if (maximum_start > 1) {
        content_start += (int)(random_u32(&teacher_sampler->rng_state) %
            (unsigned int)maximum_start);
    }
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
    free(story_tokens);

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
        int position = prompt_count + index - 1;
        float *hidden = forward_feedback_hidden(student, position);
        memcpy(
            example->hidden + (size_t)index * student->config.dim,
            hidden,
            (size_t)student->config.dim * sizeof(float)
        );
    }
    for (int index = 0; index < completion_count; index++) {
        matmul(
            example->logits + (size_t)index * student->config.vocab_size,
            example->hidden + (size_t)index * student->config.dim,
            student->weights.wcls,
            student->config.dim,
            student->config.vocab_size
        );
    }
    return 1;
}

static JointExample *joint_generate_dataset(
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
    JointExample *examples = joint_allocate((size_t)count, sizeof(*examples));
    int built = 0;
    int attempts = 0;
    while (built < count && attempts < count * 30) {
        attempts++;
        if (!joint_build_example(
                &examples[built],
                teacher,
                teacher_tokenizer,
                teacher_sampler,
                student,
                student_tokenizer,
                prompt_count,
                completion_count,
                teacher_steps)) {
            continue;
        }
        built++;
        if (built == count || built % 8 == 0) {
            fprintf(stderr, "%s corpus: %d/%d\r", name, built, count);
            fflush(stderr);
        }
    }
    fputc('\n', stderr);
    if (built != count) {
        fprintf(stderr, "built only %d/%d %s examples\n", built, count, name);
        exit(EXIT_FAILURE);
    }
    return examples;
}

static void joint_select_top_k(
    const float *logits,
    int vocab_size,
    int top_k,
    int *tokens
) {
    for (int rank = 0; rank < top_k; rank++) tokens[rank] = -1;
    for (int token = 0; token < vocab_size; token++) {
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

static int joint_find_token(const int *tokens, int count, int token) {
    for (int index = 0; index < count; index++) {
        if (tokens[index] == token) return index;
    }
    return -1;
}

static void joint_make_feature(
    const float *hidden,
    const float *embedding,
    int dim,
    double *feature
) {
    double norm = 0.0;
    for (int lane = 0; lane < dim; lane++) {
        feature[lane] = embedding[lane];
        feature[dim + lane] = (double)hidden[lane] * embedding[lane];
        norm += feature[lane] * feature[lane] +
            feature[dim + lane] * feature[dim + lane];
    }
    norm = sqrt(norm / (2.0 * dim) + 1e-12);
    for (int lane = 0; lane < 2 * dim; lane++) feature[lane] /= norm;
}

static void joint_project(
    const double *weight,
    const double *feature,
    int rank,
    int feature_dim,
    double *output
) {
    for (int out = 0; out < rank; out++) {
        double value = 0.0;
        for (int in = 0; in < feature_dim; in++) {
            value += weight[(size_t)out * feature_dim + in] * feature[in];
        }
        output[out] = value;
    }
}

static void joint_lattice_initialize(
    JointLattice *lattice,
    const JointExample *example,
    const Transformer *student,
    const JointHead *head,
    int positions,
    int top_k,
    int include_targets
) {
    if (top_k > student->config.vocab_size) {
        top_k = student->config.vocab_size;
    }
    *lattice = (JointLattice){
        .positions = positions,
        .top_k = top_k,
        .dim = student->config.dim,
        .feature_dim = head->feature_dim,
        .rank = head->rank,
        .vocab_size = student->config.vocab_size,
        .preceding_token = example->prompt_tokens[example->prompt_count - 1],
    };
    size_t candidate_count = (size_t)positions * top_k;
    lattice->tokens = joint_allocate(candidate_count, sizeof(int));
    lattice->unary = joint_allocate(candidate_count, sizeof(double));
    lattice->query_feature = joint_allocate(
        candidate_count * head->feature_dim,
        sizeof(double)
    );
    lattice->key_feature = joint_allocate(
        candidate_count * head->feature_dim,
        sizeof(double)
    );
    lattice->query = joint_allocate(
        candidate_count * head->rank,
        sizeof(double)
    );
    lattice->key = joint_allocate(
        candidate_count * head->rank,
        sizeof(double)
    );
    lattice->target_rank = joint_allocate((size_t)positions, sizeof(int));
    lattice->target_natural = joint_allocate(
        (size_t)positions,
        sizeof(unsigned char)
    );
    for (int position = 0; position < positions; position++) {
        const float *row = example->logits +
            (size_t)position * student->config.vocab_size;
        int *tokens = lattice->tokens + (size_t)position * top_k;
        joint_select_top_k(row, student->config.vocab_size, top_k, tokens);
        int target_rank = -1;
        if (include_targets) {
            int target = example->targets[position];
            target_rank = joint_find_token(tokens, top_k, target);
            lattice->target_natural[position] = target_rank >= 0;
            if (target_rank < 0) {
                tokens[top_k - 1] = target;
                target_rank = top_k - 1;
            }
        }
        lattice->target_rank[position] = target_rank;
        double maximum = row[0];
        for (int token = 1; token < student->config.vocab_size; token++) {
            if (row[token] > maximum) maximum = row[token];
        }
        const float *hidden = example->hidden +
            (size_t)position * student->config.dim;
        for (int rank = 0; rank < top_k; rank++) {
            int token = tokens[rank];
            size_t candidate = (size_t)position * top_k + rank;
            lattice->unary[candidate] = (double)row[token] - maximum;
            double *key_feature = lattice->key_feature +
                candidate * head->feature_dim;
            joint_make_feature(
                hidden,
                student->weights.wcls + (size_t)token * student->config.dim,
                student->config.dim,
                key_feature
            );
            joint_project(
                head->key_weight,
                key_feature,
                head->rank,
                head->feature_dim,
                lattice->key + candidate * head->rank
            );
        }

        int previous_count = position == 0 ? 1 : top_k;
        for (int previous_rank = 0;
             previous_rank < previous_count;
             previous_rank++) {
            int previous_token = position == 0 ? lattice->preceding_token :
                lattice->tokens[(size_t)(position - 1) * top_k + previous_rank];
            size_t query_index = (size_t)position * top_k + previous_rank;
            double *query_feature = lattice->query_feature +
                query_index * head->feature_dim;
            joint_make_feature(
                hidden,
                student->weights.wcls +
                    (size_t)previous_token * student->config.dim,
                student->config.dim,
                query_feature
            );
            joint_project(
                head->query_weight,
                query_feature,
                head->rank,
                head->feature_dim,
                lattice->query + query_index * head->rank
            );
        }
    }
}

static void joint_lattice_refresh_projections(
    JointLattice *lattice,
    const JointHead *head
) {
    int positions = lattice->positions;
    int top_k = lattice->top_k;
    for (int position = 0; position < positions; position++) {
        for (int rank = 0; rank < top_k; rank++) {
            size_t candidate = (size_t)position * top_k + rank;
            joint_project(
                head->key_weight,
                lattice->key_feature + candidate * head->feature_dim,
                head->rank,
                head->feature_dim,
                lattice->key + candidate * head->rank
            );
        }
        int previous_count = position == 0 ? 1 : top_k;
        for (int rank = 0; rank < previous_count; rank++) {
            size_t candidate = (size_t)position * top_k + rank;
            joint_project(
                head->query_weight,
                lattice->query_feature + candidate * head->feature_dim,
                head->rank,
                head->feature_dim,
                lattice->query + candidate * head->rank
            );
        }
    }
}

static void joint_lattice_free(JointLattice *lattice) {
    free(lattice->target_natural);
    free(lattice->target_rank);
    free(lattice->key);
    free(lattice->query);
    free(lattice->key_feature);
    free(lattice->query_feature);
    free(lattice->unary);
    free(lattice->tokens);
    memset(lattice, 0, sizeof(*lattice));
}

static double joint_raw_pair(
    const JointLattice *lattice,
    int position,
    int previous_rank,
    int token_rank
) {
    const double *query = lattice->query +
        ((size_t)position * lattice->top_k + previous_rank) * lattice->rank;
    const double *key = lattice->key +
        ((size_t)position * lattice->top_k + token_rank) * lattice->rank;
    double value = 0.0;
    for (int lane = 0; lane < lattice->rank; lane++) {
        value += query[lane] * key[lane];
    }
    return value / sqrt((double)lattice->rank);
}

static double joint_local_energy(
    const JointHead *head,
    const JointLattice *lattice,
    int position,
    int previous_rank,
    int token_rank
) {
    return head->unary_scale *
        lattice->unary[(size_t)position * lattice->top_k + token_rank] +
        head->pair_scale * joint_raw_pair(
            lattice,
            position,
            previous_rank,
            token_rank
        );
}

static double joint_logsumexp(const double *values, int count) {
    double maximum = values[0];
    for (int index = 1; index < count; index++) {
        if (values[index] > maximum) maximum = values[index];
    }
    double sum = 0.0;
    for (int index = 0; index < count; index++) {
        sum += exp(values[index] - maximum);
    }
    return maximum + log(sum);
}

static void joint_json_string(FILE *file, const char *text) {
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

static void joint_trace_flush(FILE *trace) {
    if (trace != NULL && fflush(trace) != 0) joint_fail("trace flush failed");
}

static void joint_trace_metadata(
    FILE *trace,
    const JointHead *head,
    const JointLattice *lattice
) {
    if (trace == NULL) return;
    fprintf(
        trace,
        "{\"event\":\"run\",\"observer\":\"joint_hidden_conditioned_energy\"," 
        "\"semiring\":\"max_plus\",\"root_terminalizations\":1," 
        "\"positions\":%d,\"top_k\":%d,\"rank\":%d," 
        "\"unary_scale\":%.17g,\"pair_scale\":%.17g}\n",
        lattice->positions,
        lattice->top_k,
        head->rank,
        head->unary_scale,
        head->pair_scale
    );
    joint_trace_flush(trace);
}

static void joint_trace_candidate_rows(
    FILE *trace,
    const JointLattice *lattice,
    Tokenizer *tokenizer
) {
    if (trace == NULL) return;
    for (int position = 0; position < lattice->positions; position++) {
        for (int rank = 0; rank < lattice->top_k; rank++) {
            int token = lattice->tokens[(size_t)position * lattice->top_k + rank];
            fprintf(
                trace,
                "{\"event\":\"carrier_candidate\",\"position\":%d," 
                "\"rank\":%d,\"token\":%d,\"piece\":",
                position,
                rank + 1,
                token
            );
            joint_json_string(trace, tokenizer->vocab[token]);
            fprintf(
                trace,
                ",\"centered_logit\":%.17g}\n",
                lattice->unary[(size_t)position * lattice->top_k + rank]
            );
            joint_trace_flush(trace);
        }
    }
}

static SelectionCell *joint_selection_cell(
    SelectionRun *run,
    int position,
    int previous_rank
) {
    return run->cells +
        (size_t)position * run->lattice->top_k + previous_rank;
}

/*
 * Memoized dependent product of local selections.
 *
 * For the state (position, previous token), every current filler x is placed
 * in the composed continuation b(x).  Only after b(x) has returned its backed
 * value is x compared with its siblings.  The chosen witness is stored in the
 * cell and the root caller reconstructs the selected sequence from those
 * witnesses.  No token is committed before its continuation is rated.
 */
static double joint_select_suffix(
    SelectionRun *run,
    int position,
    int previous_rank
) {
    SelectionCell *cell = joint_selection_cell(run, position, previous_rank);
    if (cell->state == 2) return cell->value;
    if (cell->state == 1) joint_fail("cycle in finite selection product");
    cell->state = 1;
    cell->choice = -1;
    cell->value = -DBL_MAX;

    const JointLattice *lattice = run->lattice;
    for (int token_rank = 0; token_rank < lattice->top_k; token_rank++) {
        double local = joint_local_energy(
            run->head,
            lattice,
            position,
            previous_rank,
            token_rank
        );
        double continuation = position + 1 < lattice->positions ?
            joint_select_suffix(run, position + 1, token_rank) : 0.0;
        double backed = local + continuation;
        if (run->trace != NULL) {
            int previous_token = position == 0 ? lattice->preceding_token :
                lattice->tokens[
                    (size_t)(position - 1) * lattice->top_k + previous_rank
                ];
            int token = lattice->tokens[
                (size_t)position * lattice->top_k + token_rank
            ];
            fprintf(
                run->trace,
                "{\"event\":\"candidate_rated\",\"position\":%d," 
                "\"previous_rank\":%d,\"previous_token\":%d," 
                "\"candidate_rank\":%d,\"token\":%d,\"piece\":",
                position,
                position == 0 ? 0 : previous_rank + 1,
                previous_token,
                token_rank + 1,
                token
            );
            joint_json_string(run->trace, run->tokenizer->vocab[token]);
            fprintf(
                run->trace,
                ",\"unary\":%.17g,\"pair\":%.17g," 
                "\"local_energy\":%.17g,\"continuation\":%.17g," 
                "\"backed_value\":%.17g}\n",
                run->head->unary_scale * lattice->unary[
                    (size_t)position * lattice->top_k + token_rank
                ],
                run->head->pair_scale * joint_raw_pair(
                    lattice,
                    position,
                    previous_rank,
                    token_rank
                ),
                local,
                continuation,
                backed
            );
            joint_trace_flush(run->trace);
        }
        if (cell->choice < 0 || backed > cell->value) {
            cell->choice = token_rank;
            cell->value = backed;
        }
    }
    cell->state = 2;
    if (run->trace != NULL) {
        int token = lattice->tokens[
            (size_t)position * lattice->top_k + cell->choice
        ];
        fprintf(
            run->trace,
            "{\"event\":\"selection_return\",\"position\":%d," 
            "\"previous_rank\":%d,\"selected_rank\":%d," 
            "\"selected_token\":%d,\"selected_piece\":",
            position,
            position == 0 ? 0 : previous_rank + 1,
            cell->choice + 1,
            token
        );
        joint_json_string(run->trace, run->tokenizer->vocab[token]);
        fprintf(run->trace, ",\"backed_value\":%.17g}\n", cell->value);
        joint_trace_flush(run->trace);
    }
    return cell->value;
}

static void joint_trace_selected(
    FILE *trace,
    Tokenizer *tokenizer,
    const JointLattice *lattice,
    const int *selected,
    double value
) {
    if (trace == NULL) return;
    fprintf(
        trace,
        "{\"event\":\"root_terminalized\",\"score\":%.17g," 
        "\"tokens\":[",
        value
    );
    for (int position = 0; position < lattice->positions; position++) {
        if (position != 0) fputc(',', trace);
        fprintf(trace, "%d", selected[position]);
    }
    fputs("],\"text\":", trace);
    fputc('"', trace);
    int previous = lattice->preceding_token;
    for (int position = 0; position < lattice->positions; position++) {
        const unsigned char *piece = (const unsigned char *)decode(
            tokenizer,
            previous,
            selected[position]
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
        previous = selected[position];
    }
    fputs("\"}\n", trace);
    joint_trace_flush(trace);
}

static double joint_select(
    const JointHead *head,
    const JointLattice *lattice,
    Tokenizer *tokenizer,
    FILE *trace,
    int *selected
) {
    SelectionRun run = {
        .head = head,
        .lattice = lattice,
        .tokenizer = tokenizer,
        .trace = trace,
        .cells = joint_allocate(
            (size_t)lattice->positions * lattice->top_k,
            sizeof(SelectionCell)
        ),
    };
    joint_trace_metadata(trace, head, lattice);
    joint_trace_candidate_rows(trace, lattice, tokenizer);
    double value = joint_select_suffix(&run, 0, 0);
    int previous_rank = 0;
    for (int position = 0; position < lattice->positions; position++) {
        SelectionCell *cell = joint_selection_cell(
            &run,
            position,
            previous_rank
        );
        int rank = cell->choice;
        selected[position] = lattice->tokens[
            (size_t)position * lattice->top_k + rank
        ];
        previous_rank = rank;
    }
    joint_trace_selected(trace, tokenizer, lattice, selected, value);
    free(run.cells);
    return value;
}

static void joint_print_token_span(
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

static void joint_print_example(
    const JointExample *example,
    const Transformer *student,
    Tokenizer *tokenizer,
    const JointHead *head,
    int positions,
    int top_k,
    const char *label
) {
    JointLattice lattice;
    joint_lattice_initialize(
        &lattice,
        example,
        student,
        head,
        positions,
        top_k,
        1
    );
    int *selected = joint_allocate((size_t)positions, sizeof(int));
    int *independent = joint_allocate((size_t)positions, sizeof(int));
    for (int position = 0; position < positions; position++) {
        independent[position] = lattice.tokens[(size_t)position * top_k];
    }
    double score = joint_select(head, &lattice, tokenizer, NULL, selected);
    printf("\n%s\nprompt: ", label);
    joint_print_token_span(
        tokenizer,
        example->prompt_tokens[0],
        example->prompt_tokens + 1,
        example->prompt_count - 1
    );
    int previous = example->prompt_tokens[example->prompt_count - 1];
    printf("\nteacher tuple: ");
    joint_print_token_span(tokenizer, previous, example->targets, positions);
    printf("\nrowwise delayed: ");
    joint_print_token_span(tokenizer, previous, independent, positions);
    printf("\njoint selected: ");
    joint_print_token_span(tokenizer, previous, selected, positions);
    printf("\njoint score: %.9f\n", score);
    fflush(stdout);
    free(independent);
    free(selected);
    joint_lattice_free(&lattice);
}

static double joint_dataset_loss(
    const JointExample *examples,
    int example_count,
    const Transformer *student,
    Tokenizer *tokenizer,
    JointHead *head,
    int positions,
    int top_k,
    double l2,
    double *gradient,
    double *token_accuracy,
    double *sequence_accuracy,
    double *carrier_coverage
) {
    size_t parameter_count = joint_parameter_count(head);
    size_t matrix_count = (size_t)head->rank * head->feature_dim;
    if (gradient != NULL) {
        memset(gradient, 0, parameter_count * sizeof(double));
    }
    double *query_gradient = gradient == NULL ? NULL : gradient + 2;
    double *key_gradient = gradient == NULL ? NULL :
        gradient + 2 + matrix_count;
    double loss = 0.0;
    unsigned long long correct_tokens = 0;
    unsigned long long correct_sequences = 0;
    unsigned long long covered_tokens = 0;

    for (int example_index = 0;
         example_index < example_count;
         example_index++) {
        const JointExample *example = &examples[example_index];
        JointLattice lattice;
        joint_lattice_initialize(
            &lattice,
            example,
            student,
            head,
            positions,
            top_k,
            1
        );
        int k = lattice.top_k;
        double *alpha = joint_allocate((size_t)positions * k, sizeof(double));
        double *beta = joint_allocate((size_t)positions * k, sizeof(double));
        double *scratch = joint_allocate((size_t)k, sizeof(double));

        for (int token_rank = 0; token_rank < k; token_rank++) {
            alpha[token_rank] = joint_local_energy(
                head,
                &lattice,
                0,
                0,
                token_rank
            );
        }
        for (int position = 1; position < positions; position++) {
            for (int token_rank = 0; token_rank < k; token_rank++) {
                for (int previous_rank = 0;
                     previous_rank < k;
                     previous_rank++) {
                    scratch[previous_rank] = alpha[
                        (size_t)(position - 1) * k + previous_rank
                    ] + joint_local_energy(
                        head,
                        &lattice,
                        position,
                        previous_rank,
                        token_rank
                    );
                }
                alpha[(size_t)position * k + token_rank] =
                    joint_logsumexp(scratch, k);
            }
        }
        double log_partition = joint_logsumexp(
            alpha + (size_t)(positions - 1) * k,
            k
        );
        for (int token_rank = 0; token_rank < k; token_rank++) {
            beta[(size_t)(positions - 1) * k + token_rank] = 0.0;
        }
        for (int position = positions - 2; position >= 0; position--) {
            for (int previous_rank = 0;
                 previous_rank < k;
                 previous_rank++) {
                for (int token_rank = 0; token_rank < k; token_rank++) {
                    scratch[token_rank] = joint_local_energy(
                        head,
                        &lattice,
                        position + 1,
                        previous_rank,
                        token_rank
                    ) + beta[(size_t)(position + 1) * k + token_rank];
                }
                beta[(size_t)position * k + previous_rank] =
                    joint_logsumexp(scratch, k);
            }
        }

        double target_score = 0.0;
        int previous_rank = 0;
        for (int position = 0; position < positions; position++) {
            int target_rank = lattice.target_rank[position];
            target_score += joint_local_energy(
                head,
                &lattice,
                position,
                previous_rank,
                target_rank
            );
            previous_rank = target_rank;
            covered_tokens += lattice.target_natural[position];
        }
        loss += log_partition - target_score;

        int *selected = joint_allocate((size_t)positions, sizeof(int));
        joint_select(head, &lattice, tokenizer, NULL, selected);
        int sequence_correct = 1;
        for (int position = 0; position < positions; position++) {
            if (selected[position] == example->targets[position]) {
                correct_tokens++;
            } else {
                sequence_correct = 0;
            }
        }
        if (sequence_correct) correct_sequences++;
        free(selected);

        if (gradient != NULL) {
            for (int position = 0; position < positions; position++) {
                int target_rank = lattice.target_rank[position];
                for (int token_rank = 0; token_rank < k; token_rank++) {
                    double probability = exp(
                        alpha[(size_t)position * k + token_rank] +
                        beta[(size_t)position * k + token_rank] -
                        log_partition
                    );
                    double coefficient = probability -
                        (token_rank == target_rank ? 1.0 : 0.0);
                    gradient[0] += coefficient * lattice.unary[
                        (size_t)position * k + token_rank
                    ];
                }
            }

            double inverse_rank_scale = 1.0 / sqrt((double)head->rank);
            for (int position = 0; position < positions; position++) {
                int previous_count = position == 0 ? 1 : k;
                int teacher_previous = position == 0 ? 0 :
                    lattice.target_rank[position - 1];
                int teacher_current = lattice.target_rank[position];
                for (int edge_previous = 0;
                     edge_previous < previous_count;
                     edge_previous++) {
                    for (int token_rank = 0;
                         token_rank < k;
                         token_rank++) {
                        double probability;
                        if (position == 0) {
                            probability = exp(
                                alpha[token_rank] + beta[token_rank] -
                                log_partition
                            );
                        } else {
                            probability = exp(
                                alpha[(size_t)(position - 1) * k +
                                    edge_previous] +
                                joint_local_energy(
                                    head,
                                    &lattice,
                                    position,
                                    edge_previous,
                                    token_rank
                                ) +
                                beta[(size_t)position * k + token_rank] -
                                log_partition
                            );
                        }
                        double coefficient = probability -
                            (edge_previous == teacher_previous &&
                             token_rank == teacher_current ? 1.0 : 0.0);
                        double raw_pair = joint_raw_pair(
                            &lattice,
                            position,
                            edge_previous,
                            token_rank
                        );
                        gradient[1] += coefficient * raw_pair;
                        double projected_coefficient = coefficient *
                            head->pair_scale * inverse_rank_scale;
                        size_t query_index =
                            (size_t)position * k + edge_previous;
                        size_t key_index =
                            (size_t)position * k + token_rank;
                        const double *query = lattice.query +
                            query_index * head->rank;
                        const double *key = lattice.key +
                            key_index * head->rank;
                        const double *query_feature = lattice.query_feature +
                            query_index * head->feature_dim;
                        const double *key_feature = lattice.key_feature +
                            key_index * head->feature_dim;
                        for (int projection = 0;
                             projection < head->rank;
                             projection++) {
                            double query_back = projected_coefficient *
                                key[projection];
                            double key_back = projected_coefficient *
                                query[projection];
                            for (int feature = 0;
                                 feature < head->feature_dim;
                                 feature++) {
                                query_gradient[
                                    (size_t)projection * head->feature_dim +
                                    feature
                                ] += query_back * query_feature[feature];
                                key_gradient[
                                    (size_t)projection * head->feature_dim +
                                    feature
                                ] += key_back * key_feature[feature];
                            }
                        }
                    }
                }
            }
        }
        free(scratch);
        free(beta);
        free(alpha);
        joint_lattice_free(&lattice);
    }

    double normalizer = (double)example_count;
    loss /= normalizer;
    if (gradient != NULL) {
        for (size_t parameter = 0; parameter < parameter_count; parameter++) {
            gradient[parameter] /= normalizer;
        }
    }
    const double *query_weight = head->query_weight;
    const double *key_weight = head->key_weight;
    for (size_t parameter = 0; parameter < matrix_count; parameter++) {
        loss += 0.5 * l2 *
            (query_weight[parameter] * query_weight[parameter] +
             key_weight[parameter] * key_weight[parameter]);
        if (gradient != NULL) {
            query_gradient[parameter] += l2 * query_weight[parameter];
            key_gradient[parameter] += l2 * key_weight[parameter];
        }
    }
    loss += 0.5 * l2 *
        (head->unary_scale * head->unary_scale +
         head->pair_scale * head->pair_scale);
    if (gradient != NULL) {
        gradient[0] += l2 * head->unary_scale;
        gradient[1] += l2 * head->pair_scale;
    }
    double token_count = (double)example_count * positions;
    if (token_accuracy != NULL) {
        *token_accuracy = (double)correct_tokens / token_count;
    }
    if (sequence_accuracy != NULL) {
        *sequence_accuracy =
            (double)correct_sequences / (double)example_count;
    }
    if (carrier_coverage != NULL) {
        *carrier_coverage = (double)covered_tokens / token_count;
    }
    return loss;
}

static void joint_train(
    const JointExample *training,
    int training_count,
    const JointExample *validation,
    int validation_count,
    const Transformer *student,
    Tokenizer *tokenizer,
    int positions,
    int top_k,
    int epochs,
    double learning_rate,
    double l2,
    JointHead *head
) {
    size_t parameter_count = joint_parameter_count(head);
    double *parameters = joint_allocate(parameter_count, sizeof(double));
    double *gradient = joint_allocate(parameter_count, sizeof(double));
    double *first_moment = joint_allocate(parameter_count, sizeof(double));
    double *second_moment = joint_allocate(parameter_count, sizeof(double));
    double *best = joint_allocate(parameter_count, sizeof(double));
    joint_parameters_read(head, parameters);
    memcpy(best, parameters, parameter_count * sizeof(double));
    double best_validation = DBL_MAX;
    double beta1_power = 1.0;
    double beta2_power = 1.0;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        double train_token = 0.0;
        double train_sequence = 0.0;
        double train_coverage = 0.0;
        double training_loss = joint_dataset_loss(
            training,
            training_count,
            student,
            tokenizer,
            head,
            positions,
            top_k,
            l2,
            gradient,
            &train_token,
            &train_sequence,
            &train_coverage
        );
        double squared_norm = 0.0;
        for (size_t parameter = 0; parameter < parameter_count; parameter++) {
            squared_norm += gradient[parameter] * gradient[parameter];
        }
        double gradient_scale = squared_norm > 25.0 ?
            5.0 / sqrt(squared_norm) : 1.0;
        beta1_power *= 0.9;
        beta2_power *= 0.999;
        for (size_t parameter = 0; parameter < parameter_count; parameter++) {
            double value = gradient[parameter] * gradient_scale;
            first_moment[parameter] =
                0.9 * first_moment[parameter] + 0.1 * value;
            second_moment[parameter] =
                0.999 * second_moment[parameter] + 0.001 * value * value;
            double first = first_moment[parameter] / (1.0 - beta1_power);
            double second = second_moment[parameter] / (1.0 - beta2_power);
            parameters[parameter] -= learning_rate * first /
                (sqrt(second) + 1e-8);
        }
        joint_parameters_write(head, parameters);

        double validation_token = 0.0;
        double validation_sequence = 0.0;
        double validation_coverage = 0.0;
        double validation_loss = joint_dataset_loss(
            validation,
            validation_count,
            student,
            tokenizer,
            head,
            positions,
            top_k,
            l2,
            NULL,
            &validation_token,
            &validation_sequence,
            &validation_coverage
        );
        if (validation_loss < best_validation) {
            best_validation = validation_loss;
            joint_parameters_read(head, best);
        }
        fprintf(
            stderr,
            "epoch=%d train_nll=%.6f train_token=%.4f train_sequence=%.4f "
            "validation_nll=%.6f validation_token=%.4f "
            "validation_sequence=%.4f carrier_coverage=%.4f "
            "u_scale=%.6f p_scale=%.6f gradient_norm=%.6f\n",
            epoch,
            training_loss,
            train_token,
            train_sequence,
            validation_loss,
            validation_token,
            validation_sequence,
            validation_coverage,
            head->unary_scale,
            head->pair_scale,
            sqrt(squared_norm)
        );
    }
    joint_parameters_write(head, best);
    free(best);
    free(second_moment);
    free(first_moment);
    free(gradient);
    free(parameters);
}

static void joint_build_prompt_example(
    JointExample *example,
    Transformer *student,
    Tokenizer *tokenizer,
    const char *prompt,
    int positions
) {
    size_t prompt_bytes = strlen(prompt);
    int *prompt_tokens = joint_allocate(prompt_bytes + 3U, sizeof(int));
    int prompt_count = 0;
    encode(tokenizer, (char *)prompt, 1, 0, prompt_tokens, &prompt_count);
    if (prompt_count <= 0) joint_fail("prompt encoded to no tokens");
    if (prompt_count + positions - 1 > student->config.seq_len) {
        joint_fail("prompt plus joint completion exceeds sequence length");
    }
    example->prompt_tokens = prompt_tokens;
    example->prompt_count = prompt_count;
    example->hidden = joint_allocate(
        (size_t)positions * student->config.dim,
        sizeof(float)
    );
    example->logits = joint_allocate(
        (size_t)positions * student->config.vocab_size,
        sizeof(float)
    );
    for (int position = 0; position < prompt_count; position++) {
        float *hidden = forward_token_hidden(
            student,
            prompt_tokens[position],
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
    for (int index = 1; index < positions; index++) {
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
    for (int position = 0; position < positions; position++) {
        matmul(
            example->logits +
                (size_t)position * student->config.vocab_size,
            example->hidden + (size_t)position * student->config.dim,
            student->weights.wcls,
            student->config.dim,
            student->config.vocab_size
        );
    }
}

static void joint_usage(const char *program) {
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
        "  --teacher-steps N        default 64\n"
        "  --epochs N               default 20\n"
        "  --top-k N                default 8\n"
        "  --rank N                 default 8\n"
        "  --seed N                 default 26015\n"
        "  --learning-rate X        default 0.003\n"
        "  --l2 X                   default 0.00001\n\n"
        "infer options:\n"
        "  --prompt TEXT            default 'Lily was'\n"
        "  --length N               default 32\n"
        "  --top-k N                default 8\n"
        "  --trace PATH             default joint-candidates.jsonl\n",
        program,
        program
    );
    exit(EXIT_FAILURE);
}

static void joint_parse_train_options(
    int argc,
    char **argv,
    int start,
    TrainOptions *options
) {
    *options = (TrainOptions){
        .training_count = 64,
        .validation_count = 16,
        .prompt_count = 16,
        .completion_count = 16,
        .teacher_steps = 64,
        .epochs = 20,
        .top_k = 8,
        .rank = 8,
        .seed = 26015,
        .learning_rate = 0.003,
        .l2 = 1e-5,
    };
    for (int index = start; index < argc; index += 2) {
        if (index + 1 >= argc) joint_usage(argv[0]);
        const char *flag = argv[index];
        const char *value = argv[index + 1];
        if (strcmp(flag, "--train-examples") == 0) {
            options->training_count = parse_positive(value, "train examples");
        } else if (strcmp(flag, "--validation-examples") == 0) {
            options->validation_count = parse_positive(
                value,
                "validation examples"
            );
        } else if (strcmp(flag, "--prompt-tokens") == 0) {
            options->prompt_count = parse_positive(value, "prompt tokens");
        } else if (strcmp(flag, "--completion-tokens") == 0) {
            options->completion_count = parse_positive(
                value,
                "completion tokens"
            );
        } else if (strcmp(flag, "--teacher-steps") == 0) {
            options->teacher_steps = parse_positive(value, "teacher steps");
        } else if (strcmp(flag, "--epochs") == 0) {
            options->epochs = parse_positive(value, "epochs");
        } else if (strcmp(flag, "--top-k") == 0) {
            options->top_k = parse_positive(value, "top-k");
        } else if (strcmp(flag, "--rank") == 0) {
            options->rank = parse_positive(value, "rank");
        } else if (strcmp(flag, "--seed") == 0) {
            options->seed = parse_seed(value);
        } else if (strcmp(flag, "--learning-rate") == 0) {
            options->learning_rate = parse_positive_double(
                value,
                "learning rate"
            );
        } else if (strcmp(flag, "--l2") == 0) {
            options->l2 = parse_positive_double(value, "l2");
        } else {
            joint_usage(argv[0]);
        }
    }
}

static void joint_parse_infer_options(
    int argc,
    char **argv,
    int start,
    InferOptions *options
) {
    *options = (InferOptions){
        .prompt = "Lily was",
        .length = 32,
        .top_k = 8,
        .trace_path = "joint-candidates.jsonl",
    };
    for (int index = start; index < argc; index += 2) {
        if (index + 1 >= argc) joint_usage(argv[0]);
        const char *flag = argv[index];
        const char *value = argv[index + 1];
        if (strcmp(flag, "--prompt") == 0) {
            options->prompt = value;
        } else if (strcmp(flag, "--length") == 0) {
            options->length = parse_positive(value, "length");
        } else if (strcmp(flag, "--top-k") == 0) {
            options->top_k = parse_positive(value, "top-k");
        } else if (strcmp(flag, "--trace") == 0) {
            options->trace_path = value;
        } else {
            joint_usage(argv[0]);
        }
    }
}

static int joint_train_main(int argc, char **argv) {
    if (argc < 7) joint_usage(argv[0]);
    const char *teacher_checkpoint = argv[2];
    const char *teacher_tokenizer_path = argv[3];
    const char *student_checkpoint = argv[4];
    const char *student_tokenizer_path = argv[5];
    const char *output_path = argv[6];
    TrainOptions options;
    joint_parse_train_options(argc, argv, 7, &options);

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
        joint_fail("teacher steps exceed teacher sequence length");
    }
    if (options.prompt_count + options.completion_count - 1 >
        student.config.seq_len) {
        joint_fail("training span exceeds student sequence length");
    }
    if (options.top_k > student.config.vocab_size) {
        options.top_k = student.config.vocab_size;
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
        "mode=train teacher_vocab=%d student_vocab=%d student_dim=%d "
        "train=%d validation=%d prompt=%d completion=%d top_k=%d rank=%d "
        "epochs=%d seed=%llu\n",
        teacher.config.vocab_size,
        student.config.vocab_size,
        student.config.dim,
        options.training_count,
        options.validation_count,
        options.prompt_count,
        options.completion_count,
        options.top_k,
        options.rank,
        options.epochs,
        options.seed
    );
    JointExample *training = joint_generate_dataset(
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
    JointExample *validation = joint_generate_dataset(
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
    JointHead head;
    joint_head_initialize(
        &head,
        student.config.dim,
        options.rank,
        options.seed ^ UINT64_C(0x9e3779b97f4a7c15)
    );
    double initial_token = 0.0;
    double initial_sequence = 0.0;
    double coverage = 0.0;
    double initial_loss = joint_dataset_loss(
        validation,
        options.validation_count,
        &student,
        &student_tokenizer,
        &head,
        options.completion_count,
        options.top_k,
        options.l2,
        NULL,
        &initial_token,
        &initial_sequence,
        &coverage
    );
    fprintf(
        stderr,
        "initial validation_nll=%.6f validation_token=%.4f "
        "validation_sequence=%.4f carrier_coverage=%.4f\n",
        initial_loss,
        initial_token,
        initial_sequence,
        coverage
    );
    joint_train(
        training,
        options.training_count,
        validation,
        options.validation_count,
        &student,
        &student_tokenizer,
        options.completion_count,
        options.top_k,
        options.epochs,
        options.learning_rate,
        options.l2,
        &head
    );
    joint_head_save(output_path, &head);
    int shown = options.validation_count < 6 ? options.validation_count : 6;
    for (int index = 0; index < shown; index++) {
        char label[64];
        snprintf(label, sizeof(label), "heldout=%d", index);
        joint_print_example(
            &validation[index],
            &student,
            &student_tokenizer,
            &head,
            options.completion_count,
            options.top_k,
            label
        );
    }

    joint_head_free(&head);
    for (int index = 0; index < options.validation_count; index++) {
        joint_example_free(&validation[index]);
    }
    for (int index = 0; index < options.training_count; index++) {
        joint_example_free(&training[index]);
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

static int joint_infer_main(int argc, char **argv) {
    if (argc < 5) joint_usage(argv[0]);
    const char *checkpoint = argv[2];
    const char *tokenizer_path = argv[3];
    const char *head_path = argv[4];
    InferOptions options;
    joint_parse_infer_options(argc, argv, 5, &options);

    Transformer student;
    build_transformer(&student, (char *)checkpoint);
    Tokenizer tokenizer;
    build_tokenizer(
        &tokenizer,
        (char *)tokenizer_path,
        student.config.vocab_size
    );
    JointHead head;
    joint_head_load(head_path, &head);
    if (head.dim != student.config.dim) {
        joint_fail("joint head dimension does not match checkpoint");
    }
    if (options.top_k > student.config.vocab_size) {
        options.top_k = student.config.vocab_size;
    }
    JointExample example = {0};
    long recurrence_start = time_in_ms();
    joint_build_prompt_example(
        &example,
        &student,
        &tokenizer,
        options.prompt,
        options.length
    );
    long recurrence_end = time_in_ms();
    JointLattice lattice;
    joint_lattice_initialize(
        &lattice,
        &example,
        &student,
        &head,
        options.length,
        options.top_k,
        0
    );
    FILE *trace = fopen(options.trace_path, "w");
    if (trace == NULL) {
        fprintf(stderr, "could not open trace %s\n", options.trace_path);
        exit(EXIT_FAILURE);
    }
    int *selected = joint_allocate((size_t)options.length, sizeof(int));
    long selection_start = time_in_ms();
    double value = joint_select(
        &head,
        &lattice,
        &tokenizer,
        trace,
        selected
    );
    long selection_end = time_in_ms();
    if (fclose(trace) != 0) joint_fail("could not close trace");

    int previous = example.prompt_tokens[0];
    joint_print_token_span(
        &tokenizer,
        previous,
        example.prompt_tokens + 1,
        example.prompt_count - 1
    );
    previous = example.prompt_tokens[example.prompt_count - 1];
    joint_print_token_span(&tokenizer, previous, selected, options.length);
    printf("\n");
    fflush(stdout);
    fprintf(
        stderr,
        "mode=joint_terminal_projection positions=%d top_k=%d rank=%d "
        "score=%.9f recurrence_ms=%ld selection_ms=%ld trace=%s\n",
        options.length,
        options.top_k,
        head.rank,
        value,
        recurrence_end - recurrence_start,
        selection_end - selection_start,
        options.trace_path
    );

    free(selected);
    joint_lattice_free(&lattice);
    joint_example_free(&example);
    joint_head_free(&head);
    free_tokenizer(&tokenizer);
    free_transformer(&student);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) joint_usage(argv[0]);
    if (strcmp(argv[1], "train") == 0) return joint_train_main(argc, argv);
    if (strcmp(argv[1], "infer") == 0) return joint_infer_main(argc, argv);
    joint_usage(argv[0]);
    return EXIT_FAILURE;
}
