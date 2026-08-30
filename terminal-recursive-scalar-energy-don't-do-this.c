/*
 * QUARANTINED INCOMPLETE DESIGN: DO NOT FINISH OR USE THIS SCALAR TREE CRF.
 *
 * This version noticed that adjacent pair scores must compose recursively but
 * made the more basic mistake of projecting every leaf and span to semiring
 * scalars before composition.  Its latent-state vector is an invented energy
 * model, not the retained token-indexed observation tuple produced by a joint
 * model callback.  It therefore repeats the same early-observation error one
 * level higher.  The source is retained as provenance only.
 *
 * Historical design follows.
 *
 * Recursive joint terminal projection for hidden-feedback llama2.c.
 *
 * A retained hidden tape H[0..n) is decoded by a balanced tree of selection
 * products.  A leaf does not collapse to a token.  It retains an R-indexed
 * observer over every token constructor:
 *
 *   leaf(i,s) = Select_t emission(H[i], t, s).
 *
 * An internal span does not collapse its children.  For every parent observer
 * coordinate p it retains the product of the complete left/right observer
 * vectors:
 *
 *   span(p) = Select_(l,r)
 *       left(l) + right(r) + rule(p,l,r) + context(H_span,p).
 *
 * Only the root applies its terminal observer.  In the log-sum-exp semiring
 * the same tree is an exact inside/outside contraction over all V^N token
 * tuples and supplies the globally-normalized training loss.  In max-plus it
 * is exact Bellman/Escardo backward induction with witness backpointers.  No
 * top-k carrier and no corpus bigram table are involved.
 *
 * The matrices are shared at every span, so training length does not impose an
 * inference horizon.  The balanced bracketing makes the recursive scales
 * explicit: characters/subwords remain fillers at leaves; phrases are fillers
 * for their parent spans; the complete phrase tree is observed only at root.
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
    int *targets;              /* positions, NULL for inference */
    int *prompt_tokens;
    int prompt_count;
} RecursiveExample;

typedef struct {
    int dim;
    int feature_dim;
    int state_count;
    size_t parameter_count;
    double *parameters;
} RecursiveHead;

typedef struct {
    int start;
    int end;
    int left;
    int right;
} RecursiveNode;

typedef struct {
    int positions;
    int node_count;
    int root;
    int *leaf_for_position;
    RecursiveNode *nodes;      /* postorder: children precede parents */
} RecursiveTree;

typedef struct {
    int training_count;
    int validation_count;
    int prompt_count;
    int completion_count;
    int teacher_steps;
    int epochs;
    int state_count;
    unsigned long long seed;
    double learning_rate;
    double l2;
} RecursiveTrainOptions;

typedef struct {
    const char *prompt;
    int length;
    const char *trace_path;
} RecursiveInferOptions;

static void recursive_fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

static void *recursive_allocate(size_t count, size_t size) {
    if (count != 0 && size > SIZE_MAX / count) {
        recursive_fail("recursive projection allocation overflow");
    }
    void *memory = calloc(count, size);
    if (memory == NULL) recursive_fail("recursive projection allocation failed");
    return memory;
}

static int recursive_parse_positive(const char *text, const char *name) {
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

static unsigned long long recursive_parse_seed(const char *text) {
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0) {
        recursive_fail("seed must be a positive integer");
    }
    return value;
}

static double recursive_parse_positive_double(
    const char *text,
    const char *name
) {
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

static uint64_t recursive_random_u64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}

static double recursive_random_signed(uint64_t *state) {
    return 2.0 * (double)(recursive_random_u64(state) >> 11) /
        9007199254740992.0 - 1.0;
}

static size_t recursive_leaf_count(const RecursiveHead *head) {
    return (size_t)head->state_count * head->feature_dim;
}

static size_t recursive_rule_count(const RecursiveHead *head) {
    return (size_t)head->state_count * head->state_count * head->state_count;
}

static size_t recursive_span_count(const RecursiveHead *head) {
    return (size_t)head->state_count * head->dim;
}

static double *recursive_unary(RecursiveHead *head) {
    return head->parameters;
}

static const double *recursive_const_unary(const RecursiveHead *head) {
    return head->parameters;
}

static double *recursive_leaf_weight(RecursiveHead *head) {
    return head->parameters + 1;
}

static const double *recursive_const_leaf_weight(const RecursiveHead *head) {
    return head->parameters + 1;
}

static double *recursive_rule(RecursiveHead *head) {
    return recursive_leaf_weight(head) + recursive_leaf_count(head);
}

static const double *recursive_const_rule(const RecursiveHead *head) {
    return recursive_const_leaf_weight(head) + recursive_leaf_count(head);
}

static double *recursive_span_weight(RecursiveHead *head) {
    return recursive_rule(head) + recursive_rule_count(head);
}

static const double *recursive_const_span_weight(const RecursiveHead *head) {
    return recursive_const_rule(head) + recursive_rule_count(head);
}

static double *recursive_root_weight(RecursiveHead *head) {
    return recursive_span_weight(head) + recursive_span_count(head);
}

static const double *recursive_const_root_weight(const RecursiveHead *head) {
    return recursive_const_span_weight(head) + recursive_span_count(head);
}

static void recursive_head_initialize(
    RecursiveHead *head,
    int dim,
    int state_count,
    unsigned long long seed
) {
    if (dim <= 0 || state_count <= 0) {
        recursive_fail("invalid recursive head dimensions");
    }
    *head = (RecursiveHead){
        .dim = dim,
        .feature_dim = 2 * dim,
        .state_count = state_count,
    };
    head->parameter_count = 1U +
        (size_t)state_count * head->feature_dim +
        (size_t)state_count * state_count * state_count +
        (size_t)state_count * dim +
        (size_t)state_count;
    head->parameters = recursive_allocate(
        head->parameter_count,
        sizeof(double)
    );
    *recursive_unary(head) = 0.25;
    uint64_t random_state = seed != 0 ? seed : UINT64_C(1);
    double leaf_scale = 0.15 / sqrt((double)head->feature_dim);
    double rule_scale = 0.05 / sqrt((double)state_count);
    double span_scale = 0.05 / sqrt((double)dim);
    double *leaf = recursive_leaf_weight(head);
    for (size_t index = 0; index < recursive_leaf_count(head); index++) {
        leaf[index] = leaf_scale * recursive_random_signed(&random_state);
    }
    double *rule = recursive_rule(head);
    for (size_t index = 0; index < recursive_rule_count(head); index++) {
        rule[index] = rule_scale * recursive_random_signed(&random_state);
    }
    double *span = recursive_span_weight(head);
    for (size_t index = 0; index < recursive_span_count(head); index++) {
        span[index] = span_scale * recursive_random_signed(&random_state);
    }
}

static void recursive_head_free(RecursiveHead *head) {
    free(head->parameters);
    memset(head, 0, sizeof(*head));
}

static void recursive_head_save(const char *path, const RecursiveHead *head) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        fprintf(stderr, "could not open recursive head output %s\n", path);
        exit(EXIT_FAILURE);
    }
    const unsigned char magic[8] = {'T','R','P','R','0','0','0','1'};
    int ok = fwrite(magic, sizeof(magic), 1, file) == 1 &&
        fwrite(&head->dim, sizeof(head->dim), 1, file) == 1 &&
        fwrite(&head->state_count, sizeof(head->state_count), 1, file) == 1 &&
        fwrite(
            head->parameters,
            sizeof(double),
            head->parameter_count,
            file
        ) == head->parameter_count;
    if (!ok || fclose(file) != 0) recursive_fail("could not write recursive head");
}

static void recursive_head_load(const char *path, RecursiveHead *head) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "could not open recursive head %s\n", path);
        exit(EXIT_FAILURE);
    }
    unsigned char magic[8];
    int dim = 0;
    int states = 0;
    if (fread(magic, sizeof(magic), 1, file) != 1 ||
        memcmp(magic, "TRPR0001", sizeof(magic)) != 0 ||
        fread(&dim, sizeof(dim), 1, file) != 1 ||
        fread(&states, sizeof(states), 1, file) != 1) {
        recursive_fail("invalid recursive head file");
    }
    recursive_head_initialize(head, dim, states, 1);
    if (fread(
            head->parameters,
            sizeof(double),
            head->parameter_count,
            file
        ) != head->parameter_count) {
        recursive_fail("truncated recursive head file");
    }
    if (fgetc(file) != EOF) recursive_fail("recursive head has trailing data");
    fclose(file);
}

static int recursive_tree_build_span(
    RecursiveTree *tree,
    int start,
    int end,
    int *next
) {
    if (end - start == 1) {
        int index = (*next)++;
        tree->nodes[index] = (RecursiveNode){
            .start = start,
            .end = end,
            .left = -1,
            .right = -1,
        };
        tree->leaf_for_position[start] = index;
        return index;
    }
    int middle = start + (end - start) / 2;
    int left = recursive_tree_build_span(tree, start, middle, next);
    int right = recursive_tree_build_span(tree, middle, end, next);
    int index = (*next)++;
    tree->nodes[index] = (RecursiveNode){
        .start = start,
        .end = end,
        .left = left,
        .right = right,
    };
    return index;
}

static void recursive_tree_initialize(RecursiveTree *tree, int positions) {
    if (positions <= 0) recursive_fail("recursive tree needs positions");
    *tree = (RecursiveTree){
        .positions = positions,
        .node_count = 2 * positions - 1,
    };
    tree->nodes = recursive_allocate(
        (size_t)tree->node_count,
        sizeof(*tree->nodes)
    );
    tree->leaf_for_position = recursive_allocate(
        (size_t)positions,
        sizeof(int)
    );
    int next = 0;
    tree->root = recursive_tree_build_span(tree, 0, positions, &next);
    if (next != tree->node_count || tree->root != tree->node_count - 1) {
        recursive_fail("recursive tree construction invariant failed");
    }
}

static void recursive_tree_free(RecursiveTree *tree) {
    free(tree->leaf_for_position);
    free(tree->nodes);
    memset(tree, 0, sizeof(*tree));
}

static void recursive_example_free(RecursiveExample *example) {
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
} RecursiveText;

static void recursive_text_initialize(RecursiveText *text, size_t capacity) {
    if (capacity < 64) capacity = 64;
    text->data = recursive_allocate(capacity, sizeof(char));
    text->capacity = capacity;
    text->length = 0;
}

static void recursive_text_append(RecursiveText *text, const char *piece) {
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
            if (capacity > SIZE_MAX / 2) recursive_fail("teacher text too large");
            capacity *= 2;
        }
        char *grown = realloc(text->data, capacity);
        if (grown == NULL) recursive_fail("could not grow teacher text");
        text->data = grown;
        text->capacity = capacity;
    }
    memcpy(text->data + text->length, piece, length);
    text->length += length;
    text->data[text->length] = '\0';
}

static char *recursive_generate_teacher_story(
    Transformer *teacher,
    Tokenizer *tokenizer,
    Sampler *sampler,
    int steps
) {
    RecursiveText text;
    recursive_text_initialize(
        &text,
        (size_t)steps * (tokenizer->max_token_length + 1U) + 1U
    );
    int token = 1;
    for (int position = 0; position < steps; position++) {
        float *logits = forward(teacher, token, position);
        int next = sample(sampler, logits);
        if (next == 1 || next == 2) break;
        recursive_text_append(&text, decode(tokenizer, token, next));
        token = next;
    }
    return text.data;
}

static int recursive_build_example(
    RecursiveExample *example,
    Transformer *teacher,
    Tokenizer *teacher_tokenizer,
    Sampler *teacher_sampler,
    Transformer *student,
    Tokenizer *student_tokenizer,
    int prompt_count,
    int completion_count,
    int teacher_steps
) {
    char *story = recursive_generate_teacher_story(
        teacher,
        teacher_tokenizer,
        teacher_sampler,
        teacher_steps
    );
    size_t story_bytes = strlen(story);
    int *story_tokens = recursive_allocate(story_bytes + 3U, sizeof(int));
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
    int content_span = prompt_count - 1 + completion_count;
    if (story_token_count - 1 < content_span) {
        free(story_tokens);
        return 0;
    }
    if (prompt_count + completion_count - 1 > student->config.seq_len) {
        recursive_fail("student training span exceeds sequence length");
    }
    int maximum_start = story_token_count - content_span;
    int content_start = 1;
    if (maximum_start > 1) {
        content_start += (int)(random_u32(&teacher_sampler->rng_state) %
            (unsigned int)maximum_start);
    }
    example->prompt_count = prompt_count;
    example->prompt_tokens = recursive_allocate((size_t)prompt_count, sizeof(int));
    example->targets = recursive_allocate((size_t)completion_count, sizeof(int));
    example->hidden = recursive_allocate(
        (size_t)completion_count * student->config.dim,
        sizeof(float)
    );
    example->logits = recursive_allocate(
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
            example->logits +
                (size_t)position * student->config.vocab_size,
            example->hidden + (size_t)position * student->config.dim,
            student->weights.wcls,
            student->config.dim,
            student->config.vocab_size
        );
    }
    return 1;
}

static RecursiveExample *recursive_generate_dataset(
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
    RecursiveExample *examples = recursive_allocate(
        (size_t)count,
        sizeof(*examples)
    );
    int built = 0;
    int attempts = 0;
    while (built < count && attempts < count * 30) {
        attempts++;
        if (!recursive_build_example(
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

static void recursive_build_prompt_example(
    RecursiveExample *example,
    Transformer *student,
    Tokenizer *tokenizer,
    const char *prompt,
    int positions
) {
    size_t prompt_bytes = strlen(prompt);
    example->prompt_tokens = recursive_allocate(prompt_bytes + 3U, sizeof(int));
    encode(
        tokenizer,
        (char *)prompt,
        1,
        0,
        example->prompt_tokens,
        &example->prompt_count
    );
    if (example->prompt_count <= 0) recursive_fail("prompt encoded to no tokens");
    if (example->prompt_count + positions - 1 > student->config.seq_len) {
        recursive_fail("prompt plus completion exceeds sequence length");
    }
    example->hidden = recursive_allocate(
        (size_t)positions * student->config.dim,
        sizeof(float)
    );
    example->logits = recursive_allocate(
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
            example->logits +
                (size_t)position * student->config.vocab_size,
            example->hidden + (size_t)position * student->config.dim,
            student->weights.wcls,
            student->config.dim,
            student->config.vocab_size
        );
    }
}

static void recursive_make_feature(
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

static double recursive_logadd(double left, double right) {
    if (left == -INFINITY) return right;
    if (right == -INFINITY) return left;
    double maximum = left > right ? left : right;
    return maximum + log(exp(left - maximum) + exp(right - maximum));
}

static double recursive_logsumexp(const double *values, int count) {
    double result = -INFINITY;
    for (int index = 0; index < count; index++) {
        result = recursive_logadd(result, values[index]);
    }
    return result;
}

static double recursive_row_maximum(
    const RecursiveExample *example,
    int position,
    int vocab_size
) {
    const float *row = example->logits + (size_t)position * vocab_size;
    float maximum = row[0];
    for (int candidate = 1; candidate < vocab_size; candidate++) {
        if (row[candidate] > maximum) maximum = row[candidate];
    }
    return maximum;
}

static double recursive_leaf_energy(
    const RecursiveHead *head,
    const RecursiveExample *example,
    const Transformer *student,
    int position,
    int token,
    int state,
    double centered_logit,
    double *feature_scratch
) {
    recursive_make_feature(
        example->hidden + (size_t)position * head->dim,
        student->weights.wcls + (size_t)token * head->dim,
        head->dim,
        feature_scratch
    );
    const double *weight = recursive_const_leaf_weight(head) +
        (size_t)state * head->feature_dim;
    double value = *recursive_const_unary(head) * centered_logit;
    double scale = 1.0 / sqrt((double)head->feature_dim);
    for (int feature = 0; feature < head->feature_dim; feature++) {
        value += scale * weight[feature] * feature_scratch[feature];
    }
    return value;
}

static void recursive_span_mean(
    const RecursiveNode *node,
    const RecursiveExample *example,
    int dim,
    double *mean
) {
    memset(mean, 0, (size_t)dim * sizeof(double));
    for (int position = node->start; position < node->end; position++) {
        const float *hidden = example->hidden + (size_t)position * dim;
        for (int lane = 0; lane < dim; lane++) mean[lane] += hidden[lane];
    }
    double inverse = 1.0 / (double)(node->end - node->start);
    for (int lane = 0; lane < dim; lane++) mean[lane] *= inverse;
}

static double recursive_span_bias(
    const RecursiveHead *head,
    const double *span_mean,
    int state
) {
    const double *weight = recursive_const_span_weight(head) +
        (size_t)state * head->dim;
    double value = 0.0;
    for (int lane = 0; lane < head->dim; lane++) {
        value += weight[lane] * span_mean[lane];
    }
    return value / sqrt((double)head->dim);
}

static size_t recursive_rule_index(
    const RecursiveHead *head,
    int parent,
    int left,
    int right
) {
    return ((size_t)parent * head->state_count + left) *
        head->state_count + right;
}

static double *recursive_span_means(
    const RecursiveTree *tree,
    const RecursiveExample *example,
    int dim
) {
    double *means = recursive_allocate(
        (size_t)tree->node_count * dim,
        sizeof(double)
    );
    for (int node = 0; node < tree->node_count; node++) {
        recursive_span_mean(
            &tree->nodes[node],
            example,
            dim,
            means + (size_t)node * dim
        );
    }
    return means;
}

static void recursive_inside_pair(
    const RecursiveHead *head,
    const RecursiveTree *tree,
    const RecursiveExample *example,
    const Transformer *student,
    const double *span_means,
    double *inside_all,
    double *inside_teacher
) {
    int states = head->state_count;
    int vocab = student->config.vocab_size;
    double *feature = recursive_allocate(
        (size_t)head->feature_dim,
        sizeof(double)
    );
    for (int node_index = 0;
         node_index < tree->node_count;
         node_index++) {
        const RecursiveNode *node = &tree->nodes[node_index];
        double *all = inside_all + (size_t)node_index * states;
        double *teacher = inside_teacher + (size_t)node_index * states;
        if (node->left < 0) {
            int position = node->start;
            const float *row = example->logits + (size_t)position * vocab;
            double maximum = recursive_row_maximum(example, position, vocab);
            for (int state = 0; state < states; state++) all[state] = -INFINITY;
            for (int token = 0; token < vocab; token++) {
                double centered = (double)row[token] - maximum;
                recursive_make_feature(
                    example->hidden + (size_t)position * head->dim,
                    student->weights.wcls + (size_t)token * head->dim,
                    head->dim,
                    feature
                );
                for (int state = 0; state < states; state++) {
                    const double *weight = recursive_const_leaf_weight(head) +
                        (size_t)state * head->feature_dim;
                    double energy = *recursive_const_unary(head) * centered;
                    double scale = 1.0 / sqrt((double)head->feature_dim);
                    for (int coordinate = 0;
                         coordinate < head->feature_dim;
                         coordinate++) {
                        energy += scale * weight[coordinate] * feature[coordinate];
                    }
                    all[state] = recursive_logadd(all[state], energy);
                }
            }
            int target = example->targets[position];
            double target_centered = (double)row[target] - maximum;
            for (int state = 0; state < states; state++) {
                teacher[state] = recursive_leaf_energy(
                    head,
                    example,
                    student,
                    position,
                    target,
                    state,
                    target_centered,
                    feature
                );
            }
        } else {
            const double *left_all = inside_all +
                (size_t)node->left * states;
            const double *right_all = inside_all +
                (size_t)node->right * states;
            const double *left_teacher = inside_teacher +
                (size_t)node->left * states;
            const double *right_teacher = inside_teacher +
                (size_t)node->right * states;
            const double *mean = span_means + (size_t)node_index * head->dim;
            const double *rule = recursive_const_rule(head);
            for (int parent = 0; parent < states; parent++) {
                double all_value = -INFINITY;
                double teacher_value = -INFINITY;
                for (int left = 0; left < states; left++) {
                    for (int right = 0; right < states; right++) {
                        double relation = rule[
                            recursive_rule_index(head, parent, left, right)
                        ];
                        all_value = recursive_logadd(
                            all_value,
                            left_all[left] + right_all[right] + relation
                        );
                        teacher_value = recursive_logadd(
                            teacher_value,
                            left_teacher[left] + right_teacher[right] + relation
                        );
                    }
                }
                double bias = recursive_span_bias(head, mean, parent);
                all[parent] = all_value + bias;
                teacher[parent] = teacher_value + bias;
            }
        }
    }
    free(feature);
}

static double recursive_root_partition(
    const RecursiveHead *head,
    const RecursiveTree *tree,
    const double *inside
) {
    const double *root_inside = inside +
        (size_t)tree->root * head->state_count;
    const double *root = recursive_const_root_weight(head);
    double value = -INFINITY;
    for (int state = 0; state < head->state_count; state++) {
        value = recursive_logadd(value, root_inside[state] + root[state]);
    }
    return value;
}

static void recursive_outside(
    const RecursiveHead *head,
    const RecursiveTree *tree,
    const double *inside,
    const double *span_means,
    double *outside
) {
    int states = head->state_count;
    size_t count = (size_t)tree->node_count * states;
    for (size_t index = 0; index < count; index++) outside[index] = -INFINITY;
    const double *root = recursive_const_root_weight(head);
    for (int state = 0; state < states; state++) {
        outside[(size_t)tree->root * states + state] = root[state];
    }
    const double *rule = recursive_const_rule(head);
    for (int node_index = tree->root; node_index >= 0; node_index--) {
        const RecursiveNode *node = &tree->nodes[node_index];
        if (node->left < 0) continue;
        const double *parent_outside = outside + (size_t)node_index * states;
        const double *left_inside = inside + (size_t)node->left * states;
        const double *right_inside = inside + (size_t)node->right * states;
        double *left_outside = outside + (size_t)node->left * states;
        double *right_outside = outside + (size_t)node->right * states;
        const double *mean = span_means + (size_t)node_index * head->dim;
        for (int left = 0; left < states; left++) {
            double value = -INFINITY;
            for (int parent = 0; parent < states; parent++) {
                double bias = recursive_span_bias(head, mean, parent);
                for (int right = 0; right < states; right++) {
                    value = recursive_logadd(
                        value,
                        parent_outside[parent] + bias +
                        rule[recursive_rule_index(head, parent, left, right)] +
                        right_inside[right]
                    );
                }
            }
            left_outside[left] = value;
        }
        for (int right = 0; right < states; right++) {
            double value = -INFINITY;
            for (int parent = 0; parent < states; parent++) {
                double bias = recursive_span_bias(head, mean, parent);
                for (int left = 0; left < states; left++) {
                    value = recursive_logadd(
                        value,
                        parent_outside[parent] + bias +
                        rule[recursive_rule_index(head, parent, left, right)] +
                        left_inside[left]
                    );
                }
            }
            right_outside[right] = value;
        }
    }
}

static void recursive_json_string(FILE *file, const char *text) {
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

static void recursive_trace_flush(FILE *trace) {
    if (trace != NULL && fflush(trace) != 0) recursive_fail("trace flush failed");
}

typedef struct {
    double score;
    int root_state;
    double *inside;
    int *leaf_token;           /* node x state */
    int *left_state;           /* node x parent state */
    int *right_state;
} RecursiveSelection;

static RecursiveSelection recursive_select(
    const RecursiveHead *head,
    const RecursiveTree *tree,
    const RecursiveExample *example,
    const Transformer *student,
    Tokenizer *tokenizer,
    const double *span_means,
    FILE *trace
) {
    int states = head->state_count;
    int vocab = student->config.vocab_size;
    size_t cell_count = (size_t)tree->node_count * states;
    RecursiveSelection selection = {
        .score = -DBL_MAX,
        .root_state = -1,
        .inside = recursive_allocate(cell_count, sizeof(double)),
        .leaf_token = recursive_allocate(cell_count, sizeof(int)),
        .left_state = recursive_allocate(cell_count, sizeof(int)),
        .right_state = recursive_allocate(cell_count, sizeof(int)),
    };
    double *feature = recursive_allocate(
        (size_t)head->feature_dim,
        sizeof(double)
    );
    if (trace != NULL) {
        fprintf(
            trace,
            "{\"event\":\"run\",\"observer\":\"recursive_span_selection\"," 
            "\"training_semiring\":\"logsumexp\"," 
            "\"inference_semiring\":\"max_plus\"," 
            "\"positions\":%d,\"vocab\":%d,\"states\":%d," 
            "\"nodes\":%d,\"root_terminalizations\":1}\n",
            tree->positions,
            vocab,
            states,
            tree->node_count
        );
        recursive_trace_flush(trace);
    }
    const double *rule = recursive_const_rule(head);
    for (int node_index = 0;
         node_index < tree->node_count;
         node_index++) {
        const RecursiveNode *node = &tree->nodes[node_index];
        double *inside = selection.inside + (size_t)node_index * states;
        if (node->left < 0) {
            int position = node->start;
            const float *row = example->logits + (size_t)position * vocab;
            double maximum = recursive_row_maximum(example, position, vocab);
            for (int state = 0; state < states; state++) {
                inside[state] = -DBL_MAX;
                selection.leaf_token[(size_t)node_index * states + state] = -1;
            }
            for (int token = 0; token < vocab; token++) {
                double centered = (double)row[token] - maximum;
                recursive_make_feature(
                    example->hidden + (size_t)position * head->dim,
                    student->weights.wcls + (size_t)token * head->dim,
                    head->dim,
                    feature
                );
                if (trace != NULL) {
                    fprintf(
                        trace,
                        "{\"event\":\"leaf_candidate\",\"position\":%d," 
                        "\"token\":%d,\"piece\":",
                        position,
                        token
                    );
                    recursive_json_string(trace, tokenizer->vocab[token]);
                    fprintf(trace, ",\"centered_logit\":%.17g,\"scores\":[", centered);
                }
                for (int state = 0; state < states; state++) {
                    const double *weight = recursive_const_leaf_weight(head) +
                        (size_t)state * head->feature_dim;
                    double energy = *recursive_const_unary(head) * centered;
                    double scale = 1.0 / sqrt((double)head->feature_dim);
                    for (int coordinate = 0;
                         coordinate < head->feature_dim;
                         coordinate++) {
                        energy += scale * weight[coordinate] * feature[coordinate];
                    }
                    if (trace != NULL) {
                        if (state != 0) fputc(',', trace);
                        fprintf(trace, "%.17g", energy);
                    }
                    if (selection.leaf_token[
                            (size_t)node_index * states + state
                        ] < 0 || energy > inside[state]) {
                        inside[state] = energy;
                        selection.leaf_token[
                            (size_t)node_index * states + state
                        ] = token;
                    }
                }
                if (trace != NULL) {
                    fputs("]}\n", trace);
                    recursive_trace_flush(trace);
                }
            }
        } else {
            const double *left_inside = selection.inside +
                (size_t)node->left * states;
            const double *right_inside = selection.inside +
                (size_t)node->right * states;
            const double *mean = span_means + (size_t)node_index * head->dim;
            for (int parent = 0; parent < states; parent++) {
                inside[parent] = -DBL_MAX;
                int best_left = -1;
                int best_right = -1;
                for (int left = 0; left < states; left++) {
                    for (int right = 0; right < states; right++) {
                        double value = left_inside[left] + right_inside[right] +
                            rule[recursive_rule_index(head, parent, left, right)];
                        if (best_left < 0 || value > inside[parent]) {
                            inside[parent] = value;
                            best_left = left;
                            best_right = right;
                        }
                    }
                }
                inside[parent] += recursive_span_bias(head, mean, parent);
                selection.left_state[(size_t)node_index * states + parent] =
                    best_left;
                selection.right_state[(size_t)node_index * states + parent] =
                    best_right;
                if (trace != NULL) {
                    fprintf(
                        trace,
                        "{\"event\":\"span_selection\",\"node\":%d," 
                        "\"start\":%d,\"end\":%d,\"parent_state\":%d," 
                        "\"left_state\":%d,\"right_state\":%d," 
                        "\"backed_value\":%.17g}\n",
                        node_index,
                        node->start,
                        node->end,
                        parent,
                        best_left,
                        best_right,
                        inside[parent]
                    );
                    recursive_trace_flush(trace);
                }
            }
        }
    }
    const double *root_inside = selection.inside +
        (size_t)tree->root * states;
    const double *root_weight = recursive_const_root_weight(head);
    for (int state = 0; state < states; state++) {
        double value = root_inside[state] + root_weight[state];
        if (selection.root_state < 0 || value > selection.score) {
            selection.root_state = state;
            selection.score = value;
        }
    }
    free(feature);
    return selection;
}

static void recursive_backtrack(
    const RecursiveHead *head,
    const RecursiveTree *tree,
    const RecursiveSelection *selection,
    int node_index,
    int state,
    int *tokens
) {
    const RecursiveNode *node = &tree->nodes[node_index];
    size_t cell = (size_t)node_index * head->state_count + state;
    if (node->left < 0) {
        tokens[node->start] = selection->leaf_token[cell];
        return;
    }
    recursive_backtrack(
        head,
        tree,
        selection,
        node->left,
        selection->left_state[cell],
        tokens
    );
    recursive_backtrack(
        head,
        tree,
        selection,
        node->right,
        selection->right_state[cell],
        tokens
    );
}

static void recursive_selection_free(RecursiveSelection *selection) {
    free(selection->right_state);
    free(selection->left_state);
    free(selection->leaf_token);
    free(selection->inside);
    memset(selection, 0, sizeof(*selection));
}

static double recursive_dataset_loss(
    const RecursiveExample *examples,
    int example_count,
    const Transformer *student,
    Tokenizer *tokenizer,
    RecursiveHead *head,
    const RecursiveTree *tree,
    double l2,
    double *gradient,
    double *token_accuracy,
    double *sequence_accuracy
) {
    int states = head->state_count;
    int vocab = student->config.vocab_size;
    size_t cell_count = (size_t)tree->node_count * states;
    if (gradient != NULL) {
        memset(gradient, 0, head->parameter_count * sizeof(double));
    }
    double *gradient_unary = gradient;
    double *gradient_leaf = gradient == NULL ? NULL : gradient + 1;
    double *gradient_rule = gradient == NULL ? NULL :
        gradient_leaf + recursive_leaf_count(head);
    double *gradient_span = gradient == NULL ? NULL :
        gradient_rule + recursive_rule_count(head);
    double *gradient_root = gradient == NULL ? NULL :
        gradient_span + recursive_span_count(head);
    double loss = 0.0;
    unsigned long long correct_tokens = 0;
    unsigned long long correct_sequences = 0;

    for (int example_index = 0;
         example_index < example_count;
         example_index++) {
        const RecursiveExample *example = &examples[example_index];
        double *means = recursive_span_means(tree, example, head->dim);
        double *inside_all = recursive_allocate(cell_count, sizeof(double));
        double *inside_teacher = recursive_allocate(cell_count, sizeof(double));
        double *outside_all = recursive_allocate(cell_count, sizeof(double));
        double *outside_teacher = recursive_allocate(cell_count, sizeof(double));
        recursive_inside_pair(
            head,
            tree,
            example,
            student,
            means,
            inside_all,
            inside_teacher
        );
        double partition_all = recursive_root_partition(head, tree, inside_all);
        double partition_teacher = recursive_root_partition(
            head,
            tree,
            inside_teacher
        );
        loss += partition_all - partition_teacher;
        recursive_outside(
            head,
            tree,
            inside_all,
            means,
            outside_all
        );
        recursive_outside(
            head,
            tree,
            inside_teacher,
            means,
            outside_teacher
        );

        RecursiveSelection selection = recursive_select(
            head,
            tree,
            example,
            student,
            tokenizer,
            means,
            NULL
        );
        int *selected = recursive_allocate(
            (size_t)tree->positions,
            sizeof(int)
        );
        recursive_backtrack(
            head,
            tree,
            &selection,
            tree->root,
            selection.root_state,
            selected
        );
        int sequence_correct = 1;
        for (int position = 0; position < tree->positions; position++) {
            if (selected[position] == example->targets[position]) {
                correct_tokens++;
            } else {
                sequence_correct = 0;
            }
        }
        if (sequence_correct) correct_sequences++;
        free(selected);
        recursive_selection_free(&selection);

        if (gradient != NULL) {
            const double *root_all = inside_all +
                (size_t)tree->root * states;
            const double *root_teacher = inside_teacher +
                (size_t)tree->root * states;
            const double *root_weight = recursive_const_root_weight(head);
            for (int state = 0; state < states; state++) {
                double all_probability = exp(
                    root_all[state] + root_weight[state] - partition_all
                );
                double teacher_probability = exp(
                    root_teacher[state] + root_weight[state] -
                    partition_teacher
                );
                gradient_root[state] += all_probability - teacher_probability;
            }

            const double *rule = recursive_const_rule(head);
            for (int node_index = 0;
                 node_index < tree->node_count;
                 node_index++) {
                const RecursiveNode *node = &tree->nodes[node_index];
                if (node->left < 0) continue;
                const double *mean = means + (size_t)node_index * head->dim;
                const double *left_all = inside_all +
                    (size_t)node->left * states;
                const double *right_all = inside_all +
                    (size_t)node->right * states;
                const double *left_teacher = inside_teacher +
                    (size_t)node->left * states;
                const double *right_teacher = inside_teacher +
                    (size_t)node->right * states;
                const double *node_outside_all = outside_all +
                    (size_t)node_index * states;
                const double *node_outside_teacher = outside_teacher +
                    (size_t)node_index * states;
                for (int parent = 0; parent < states; parent++) {
                    double state_all = exp(
                        node_outside_all[parent] +
                        inside_all[(size_t)node_index * states + parent] -
                        partition_all
                    );
                    double state_teacher = exp(
                        node_outside_teacher[parent] +
                        inside_teacher[(size_t)node_index * states + parent] -
                        partition_teacher
                    );
                    double state_difference = state_all - state_teacher;
                    for (int lane = 0; lane < head->dim; lane++) {
                        gradient_span[(size_t)parent * head->dim + lane] +=
                            state_difference * mean[lane] /
                            sqrt((double)head->dim);
                    }
                    double bias = recursive_span_bias(head, mean, parent);
                    for (int left = 0; left < states; left++) {
                        for (int right = 0; right < states; right++) {
                            size_t rule_index = recursive_rule_index(
                                head,
                                parent,
                                left,
                                right
                            );
                            double all_probability = exp(
                                node_outside_all[parent] + bias +
                                rule[rule_index] + left_all[left] +
                                right_all[right] - partition_all
                            );
                            double teacher_probability = exp(
                                node_outside_teacher[parent] + bias +
                                rule[rule_index] + left_teacher[left] +
                                right_teacher[right] - partition_teacher
                            );
                            gradient_rule[rule_index] +=
                                all_probability - teacher_probability;
                        }
                    }
                }
            }

            double *feature = recursive_allocate(
                (size_t)head->feature_dim,
                sizeof(double)
            );
            double feature_scale = 1.0 / sqrt((double)head->feature_dim);
            for (int position = 0; position < tree->positions; position++) {
                int leaf_node = tree->leaf_for_position[position];
                const double *leaf_outside_all = outside_all +
                    (size_t)leaf_node * states;
                const double *leaf_outside_teacher = outside_teacher +
                    (size_t)leaf_node * states;
                const float *row = example->logits + (size_t)position * vocab;
                double maximum = recursive_row_maximum(example, position, vocab);
                int target = example->targets[position];
                for (int token = 0; token < vocab; token++) {
                    double centered = (double)row[token] - maximum;
                    recursive_make_feature(
                        example->hidden + (size_t)position * head->dim,
                        student->weights.wcls + (size_t)token * head->dim,
                        head->dim,
                        feature
                    );
                    for (int state = 0; state < states; state++) {
                        double energy = recursive_leaf_energy(
                            head,
                            example,
                            student,
                            position,
                            token,
                            state,
                            centered,
                            feature
                        );
                        double all_probability = exp(
                            leaf_outside_all[state] + energy - partition_all
                        );
                        double teacher_probability = 0.0;
                        if (token == target) {
                            teacher_probability = exp(
                                leaf_outside_teacher[state] + energy -
                                partition_teacher
                            );
                        }
                        double difference =
                            all_probability - teacher_probability;
                        *gradient_unary += difference * centered;
                        double *leaf_gradient = gradient_leaf +
                            (size_t)state * head->feature_dim;
                        for (int coordinate = 0;
                             coordinate < head->feature_dim;
                             coordinate++) {
                            leaf_gradient[coordinate] +=
                                difference * feature_scale * feature[coordinate];
                        }
                    }
                }
            }
            free(feature);
        }
        free(outside_teacher);
        free(outside_all);
        free(inside_teacher);
        free(inside_all);
        free(means);
    }

    double inverse_examples = 1.0 / (double)example_count;
    loss *= inverse_examples;
    if (gradient != NULL) {
        for (size_t parameter = 0;
             parameter < head->parameter_count;
             parameter++) {
            gradient[parameter] *= inverse_examples;
        }
    }
    for (size_t parameter = 1;
         parameter < head->parameter_count;
         parameter++) {
        double value = head->parameters[parameter];
        loss += 0.5 * l2 * value * value;
        if (gradient != NULL) gradient[parameter] += l2 * value;
    }
    loss += 0.5 * l2 * head->parameters[0] * head->parameters[0];
    if (gradient != NULL) gradient[0] += l2 * head->parameters[0];
    double token_count = (double)example_count * tree->positions;
    if (token_accuracy != NULL) {
        *token_accuracy = (double)correct_tokens / token_count;
    }
    if (sequence_accuracy != NULL) {
        *sequence_accuracy =
            (double)correct_sequences / (double)example_count;
    }
    return loss;
}

static void recursive_train(
    const RecursiveExample *training,
    int training_count,
    const RecursiveExample *validation,
    int validation_count,
    const Transformer *student,
    Tokenizer *tokenizer,
    RecursiveHead *head,
    const RecursiveTree *tree,
    int epochs,
    double learning_rate,
    double l2
) {
    size_t count = head->parameter_count;
    double *gradient = recursive_allocate(count, sizeof(double));
    double *first_moment = recursive_allocate(count, sizeof(double));
    double *second_moment = recursive_allocate(count, sizeof(double));
    double *best = recursive_allocate(count, sizeof(double));
    memcpy(best, head->parameters, count * sizeof(double));
    double best_validation = DBL_MAX;
    double beta1_power = 1.0;
    double beta2_power = 1.0;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        double train_token = 0.0;
        double train_sequence = 0.0;
        double training_loss = recursive_dataset_loss(
            training,
            training_count,
            student,
            tokenizer,
            head,
            tree,
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
        double validation_loss = recursive_dataset_loss(
            validation,
            validation_count,
            student,
            tokenizer,
            head,
            tree,
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
            "epoch=%d train_nll=%.6f train_token=%.4f train_sequence=%.4f "
            "validation_nll=%.6f validation_token=%.4f "
            "validation_sequence=%.4f unary_scale=%.6f "
            "gradient_norm=%.6f\n",
            epoch,
            training_loss,
            train_token,
            train_sequence,
            validation_loss,
            validation_token,
            validation_sequence,
            head->parameters[0],
            sqrt(squared_norm)
        );
    }
    memcpy(head->parameters, best, count * sizeof(double));
    free(best);
    free(second_moment);
    free(first_moment);
    free(gradient);
}
