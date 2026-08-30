/*
 * QUARANTINED: DO NOT USE THIS AS A JOINT PROJECTION.
 *
 * This experiment terminalized the retained hidden sequence too early.  Its
 * position and attention heads were trained with independent row-wise cross
 * entropy, while its "joint" variant merely appended a corpus bigram table
 * and Viterbi.  Neither makes a complete candidate tuple the argument of one
 * observer, so neither implements the requested selection-product semantics.
 *
 * Historical description follows.
 *
 * Train the smallest many-to-many terminal projection for the
 * hidden-feedback experiment.
 *
 * A 15M TinyStories model generates an ordinary text corpus.  The text is
 * retokenized with the 260K model's tokenizer; no logits, hidden states, or
 * token identities cross between the two models.  For each student example,
 * the 260K transformer is frozen and produces an N-position hidden-feedback
 * trajectory without projecting an intermediate token.
 *
 * If B[j,t] = <h_j,E_t> is the frozen output-head observation, the terminal
 * head is
 *
 *     L[i,t] = B[i,t] + sum_j Delta[i,j] B[j,t].
 *
 * This is exactly U=(I+Delta)H followed by L=U E^T.  Delta is the only learned
 * parameter.  Delta=0 is the delayed independent projection baseline.
 */

#define TESTING
#include "run_hidden_feedback.c"

#include <errno.h>
#include <float.h>
#include <limits.h>

typedef struct {
    float *base_logits;       /* completion_count x vocab_size */
    float *hidden_sequence;   /* completion_count x student dimension */
    int *targets;             /* completion_count */
    int *prompt_tokens;       /* prompt_count */
    int prompt_count;
} TerminalExample;

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
} TextBuffer;

static void *terminal_allocate(size_t count, size_t size) {
    if (count != 0 && size > SIZE_MAX / count) {
        fprintf(stderr, "terminal mixer allocation overflow\n");
        exit(EXIT_FAILURE);
    }
    void *allocation = calloc(count, size);
    if (allocation == NULL) {
        fprintf(stderr, "terminal mixer allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return allocation;
}

static void text_buffer_initialize(TextBuffer *buffer, size_t capacity) {
    if (capacity < 64) capacity = 64;
    buffer->data = terminal_allocate(capacity, sizeof(*buffer->data));
    buffer->length = 0;
    buffer->capacity = capacity;
}

static void text_buffer_append_piece(TextBuffer *buffer, const char *piece) {
    if (piece == NULL || piece[0] == '\0') return;
    size_t piece_length = strlen(piece);
    if (piece_length == 1) {
        unsigned char byte = (unsigned char)piece[0];
        if (!(isprint(byte) || isspace(byte))) return;
    }
    size_t required = buffer->length + piece_length + 1;
    if (required > buffer->capacity) {
        size_t capacity = buffer->capacity;
        while (capacity < required) {
            if (capacity > SIZE_MAX / 2) {
                fprintf(stderr, "teacher text is too large\n");
                exit(EXIT_FAILURE);
            }
            capacity *= 2;
        }
        char *grown = realloc(buffer->data, capacity);
        if (grown == NULL) {
            fprintf(stderr, "could not grow teacher text\n");
            exit(EXIT_FAILURE);
        }
        buffer->data = grown;
        buffer->capacity = capacity;
    }
    memcpy(buffer->data + buffer->length, piece, piece_length);
    buffer->length += piece_length;
    buffer->data[buffer->length] = '\0';
}

static char *generate_teacher_story(
    Transformer *teacher,
    Tokenizer *tokenizer,
    Sampler *sampler,
    int steps
) {
    TextBuffer text;
    text_buffer_initialize(
        &text,
        (size_t)steps * (tokenizer->max_token_length + 1U) + 1U
    );
    int token = 1;
    for (int position = 0; position < steps; position++) {
        float *logits = forward(teacher, token, position);
        int next = sample(sampler, logits);
        if (next == 1) break;
        text_buffer_append_piece(&text, decode(tokenizer, token, next));
        token = next;
    }
    return text.data;
}

static void free_terminal_example(TerminalExample *example) {
    free(example->base_logits);
    free(example->hidden_sequence);
    free(example->targets);
    free(example->prompt_tokens);
    memset(example, 0, sizeof(*example));
}

static int build_terminal_example(
    TerminalExample *example,
    Transformer *teacher,
    Tokenizer *teacher_tokenizer,
    Sampler *teacher_sampler,
    Transformer *student,
    Tokenizer *student_tokenizer,
    int prompt_count,
    int completion_count,
    int teacher_steps
) {
    char *story = generate_teacher_story(
        teacher,
        teacher_tokenizer,
        teacher_sampler,
        teacher_steps
    );
    size_t story_bytes = strlen(story);
    int *story_tokens = terminal_allocate(story_bytes + 3U, sizeof(*story_tokens));
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

    if (story_token_count < prompt_count + completion_count) {
        free(story_tokens);
        return 0;
    }
    if (prompt_count + completion_count - 1 > student->config.seq_len) {
        fprintf(stderr, "student prompt and completion exceed its sequence length\n");
        exit(EXIT_FAILURE);
    }

    example->prompt_count = prompt_count;
    example->prompt_tokens = terminal_allocate(
        (size_t)prompt_count,
        sizeof(*example->prompt_tokens)
    );
    example->targets = terminal_allocate(
        (size_t)completion_count,
        sizeof(*example->targets)
    );
    example->base_logits = terminal_allocate(
        (size_t)completion_count * student->config.vocab_size,
        sizeof(*example->base_logits)
    );
    memcpy(
        example->prompt_tokens,
        story_tokens,
        (size_t)prompt_count * sizeof(*story_tokens)
    );
    memcpy(
        example->targets,
        story_tokens + prompt_count,
        (size_t)completion_count * sizeof(*story_tokens)
    );
    free(story_tokens);

    size_t hidden_count = (size_t)completion_count * student->config.dim;
    example->hidden_sequence = terminal_allocate(
        hidden_count,
        sizeof(*example->hidden_sequence)
    );
    for (int position = 0; position < prompt_count; position++) {
        float *hidden = forward_token_hidden(
            student,
            example->prompt_tokens[position],
            position
        );
        if (position == prompt_count - 1) {
            memcpy(
                example->hidden_sequence,
                hidden,
                (size_t)student->config.dim *
                    sizeof(*example->hidden_sequence)
            );
        }
    }
    for (int index = 1; index < completion_count; index++) {
        int position = prompt_count + index - 1;
        float *hidden = forward_feedback_hidden(student, position);
        memcpy(
            example->hidden_sequence + (size_t)index * student->config.dim,
            hidden,
            (size_t)student->config.dim *
                sizeof(*example->hidden_sequence)
        );
    }

    /* Projection is terminal: observe the complete retained sequence now. */
    for (int index = 0; index < completion_count; index++) {
        matmul(
            example->base_logits +
                (size_t)index * student->config.vocab_size,
            example->hidden_sequence +
                (size_t)index * student->config.dim,
            student->weights.wcls,
            student->config.dim,
            student->config.vocab_size
        );
    }
    return 1;
}

static TerminalExample *generate_terminal_dataset(
    int example_count,
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
    TerminalExample *examples = terminal_allocate(
        (size_t)example_count,
        sizeof(*examples)
    );
    int built = 0;
    int attempts = 0;
    int maximum_attempts = example_count * 20;
    while (built < example_count && attempts < maximum_attempts) {
        attempts++;
        if (!build_terminal_example(
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
        if (built == example_count || built % 16 == 0) {
            fprintf(
                stderr,
                "%s corpus: %d/%d examples\r",
                name,
                built,
                example_count
            );
            fflush(stderr);
        }
    }
    fputc('\n', stderr);
    if (built != example_count) {
        fprintf(
            stderr,
            "could only build %d of %d %s examples\n",
            built,
            example_count,
            name
        );
        exit(EXIT_FAILURE);
    }
    return examples;
}

static double terminal_mixer_loss(
    const TerminalExample *examples,
    int example_count,
    int completion_count,
    int vocab_size,
    const double *delta,
    double l2,
    double *gradient,
    double *token_accuracy,
    double *sequence_accuracy
) {
    size_t parameter_count = (size_t)completion_count * completion_count;
    if (gradient != NULL) {
        memset(gradient, 0, parameter_count * sizeof(*gradient));
    }
    double loss = 0.0;
    unsigned long long correct_tokens = 0;
    unsigned long long correct_sequences = 0;
    double *mixed = terminal_allocate((size_t)vocab_size, sizeof(*mixed));

    for (int example_index = 0;
         example_index < example_count;
         example_index++) {
        const TerminalExample *example = &examples[example_index];
        int sequence_correct = 1;
        for (int output = 0; output < completion_count; output++) {
            double maximum = -DBL_MAX;
            int selected = -1;
            for (int token = 0; token < vocab_size; token++) {
                double value = example->base_logits[
                    (size_t)output * vocab_size + token
                ];
                for (int source = 0; source < completion_count; source++) {
                    value += delta[(size_t)output * completion_count + source] *
                        example->base_logits[
                            (size_t)source * vocab_size + token
                        ];
                }
                mixed[token] = value;
                if (selected < 0 || value > maximum) {
                    maximum = value;
                    selected = token;
                }
            }
            int target = example->targets[output];
            if (selected == target) {
                correct_tokens++;
            } else {
                sequence_correct = 0;
            }
            double partition = 0.0;
            for (int token = 0; token < vocab_size; token++) {
                partition += exp(mixed[token] - maximum);
            }
            double log_partition = maximum + log(partition);
            loss += log_partition - mixed[target];

            if (gradient != NULL) {
                for (int token = 0; token < vocab_size; token++) {
                    double error = exp(mixed[token] - log_partition) -
                        (token == target ? 1.0 : 0.0);
                    for (int source = 0;
                         source < completion_count;
                         source++) {
                        gradient[
                            (size_t)output * completion_count + source
                        ] += error * example->base_logits[
                            (size_t)source * vocab_size + token
                        ];
                    }
                }
            }
        }
        if (sequence_correct) correct_sequences++;
    }

    double observation_count = (double)example_count * completion_count;
    loss /= observation_count;
    if (gradient != NULL) {
        for (size_t parameter = 0; parameter < parameter_count; parameter++) {
            gradient[parameter] = gradient[parameter] / observation_count +
                l2 * delta[parameter];
        }
    }
    for (size_t parameter = 0; parameter < parameter_count; parameter++) {
        loss += 0.5 * l2 * delta[parameter] * delta[parameter];
    }
    if (token_accuracy != NULL) {
        *token_accuracy = (double)correct_tokens / observation_count;
    }
    if (sequence_accuracy != NULL) {
        *sequence_accuracy =
            (double)correct_sequences / (double)example_count;
    }
    free(mixed);
    return loss;
}

static void train_terminal_mixer(
    const TerminalExample *training,
    int training_count,
    const TerminalExample *validation,
    int validation_count,
    int completion_count,
    int vocab_size,
    int epochs,
    double learning_rate,
    double l2,
    double *delta
) {
    size_t parameter_count = (size_t)completion_count * completion_count;
    double *gradient = terminal_allocate(parameter_count, sizeof(*gradient));
    double *first_moment = terminal_allocate(parameter_count, sizeof(*first_moment));
    double *second_moment = terminal_allocate(parameter_count, sizeof(*second_moment));
    double *best_delta = terminal_allocate(parameter_count, sizeof(*best_delta));
    double best_validation = DBL_MAX;
    double beta1_power = 1.0;
    double beta2_power = 1.0;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        double training_accuracy = 0.0;
        double training_sequence_accuracy = 0.0;
        double training_loss = terminal_mixer_loss(
            training,
            training_count,
            completion_count,
            vocab_size,
            delta,
            l2,
            gradient,
            &training_accuracy,
            &training_sequence_accuracy
        );
        beta1_power *= 0.9;
        beta2_power *= 0.999;
        for (size_t parameter = 0; parameter < parameter_count; parameter++) {
            first_moment[parameter] =
                0.9 * first_moment[parameter] + 0.1 * gradient[parameter];
            second_moment[parameter] =
                0.999 * second_moment[parameter] +
                0.001 * gradient[parameter] * gradient[parameter];
            double corrected_first = first_moment[parameter] /
                (1.0 - beta1_power);
            double corrected_second = second_moment[parameter] /
                (1.0 - beta2_power);
            delta[parameter] -= learning_rate * corrected_first /
                (sqrt(corrected_second) + 1e-8);
        }

        double validation_accuracy = 0.0;
        double validation_sequence_accuracy = 0.0;
        double validation_loss = terminal_mixer_loss(
            validation,
            validation_count,
            completion_count,
            vocab_size,
            delta,
            l2,
            NULL,
            &validation_accuracy,
            &validation_sequence_accuracy
        );
        if (validation_loss < best_validation) {
            best_validation = validation_loss;
            memcpy(
                best_delta,
                delta,
                parameter_count * sizeof(*best_delta)
            );
        }
        if (epoch == 1 || epoch == epochs || epoch % 10 == 0) {
            fprintf(
                stderr,
                "epoch=%d train_loss=%.6f train_token_accuracy=%.4f "
                "train_sequence_accuracy=%.4f validation_loss=%.6f "
                "validation_token_accuracy=%.4f "
                "validation_sequence_accuracy=%.4f\n",
                epoch,
                training_loss,
                training_accuracy,
                training_sequence_accuracy,
                validation_loss,
                validation_accuracy,
                validation_sequence_accuracy
            );
        }
    }
    memcpy(delta, best_delta, parameter_count * sizeof(*delta));
    free(best_delta);
    free(second_moment);
    free(first_moment);
    free(gradient);
}

static void select_terminal_tokens(
    const TerminalExample *example,
    int completion_count,
    int vocab_size,
    const double *delta,
    int *tokens
) {
    for (int output = 0; output < completion_count; output++) {
        double maximum = -DBL_MAX;
        int selected = -1;
        for (int token = 0; token < vocab_size; token++) {
            double value = example->base_logits[
                (size_t)output * vocab_size + token
            ];
            for (int source = 0; source < completion_count; source++) {
                value += delta[(size_t)output * completion_count + source] *
                    example->base_logits[
                        (size_t)source * vocab_size + token
                    ];
            }
            if (selected < 0 || value > maximum) {
                maximum = value;
                selected = token;
            }
        }
        tokens[output] = selected;
    }
}

static void print_token_span(
    Tokenizer *tokenizer,
    int previous,
    const int *tokens,
    int count
) {
    for (int index = 0; index < count; index++) {
        int token = tokens[index];
        if (token == 1) break;
        safe_printf(decode(tokenizer, previous, token));
        previous = token;
    }
}

static void print_terminal_examples(
    const TerminalExample *examples,
    int example_count,
    int shown,
    int completion_count,
    int vocab_size,
    Tokenizer *tokenizer,
    const double *delta
) {
    if (shown > example_count) shown = example_count;
    double *zero = terminal_allocate(
        (size_t)completion_count * completion_count,
        sizeof(*zero)
    );
    int *baseline = terminal_allocate(
        (size_t)completion_count,
        sizeof(*baseline)
    );
    int *mixed = terminal_allocate(
        (size_t)completion_count,
        sizeof(*mixed)
    );
    for (int index = 0; index < shown; index++) {
        const TerminalExample *example = &examples[index];
        select_terminal_tokens(
            example,
            completion_count,
            vocab_size,
            zero,
            baseline
        );
        select_terminal_tokens(
            example,
            completion_count,
            vocab_size,
            delta,
            mixed
        );
        int previous = example->prompt_tokens[0];
        printf("\nheldout=%d\nprompt: ", index);
        print_token_span(
            tokenizer,
            previous,
            example->prompt_tokens + 1,
            example->prompt_count - 1
        );
        previous = example->prompt_tokens[example->prompt_count - 1];
        printf("\n15M-text target: ");
        print_token_span(
            tokenizer,
            previous,
            example->targets,
            completion_count
        );
        printf("\ndelayed independent: ");
        print_token_span(tokenizer, previous, baseline, completion_count);
        printf("\ntrained many-to-many: ");
        print_token_span(tokenizer, previous, mixed, completion_count);
        printf("\n");
    }
    free(mixed);
    free(baseline);
    free(zero);
}

static void save_terminal_mixer(
    const char *path,
    int completion_count,
    const double *delta
) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        fprintf(stderr, "could not open terminal mixer output %s\n", path);
        exit(EXIT_FAILURE);
    }
    const unsigned char magic[8] = {'T','M','I','X','E','R','0','1'};
    if (fwrite(magic, sizeof(magic), 1, file) != 1 ||
        fwrite(&completion_count, sizeof(completion_count), 1, file) != 1 ||
        fwrite(
            delta,
            sizeof(*delta),
            (size_t)completion_count * completion_count,
            file
        ) != (size_t)completion_count * completion_count) {
        fprintf(stderr, "could not write terminal mixer output %s\n", path);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    fclose(file);
}

/* ------------------------------------------------------------------------- */
/* One bidirectional terminal self-attention block.                           */

typedef struct {
    int position_count;
    int dim;
    int head_count;
    int head_dim;
    double *parameters; /* Wq, Wk, Wv, Wo; each dim x dim, output-major. */
} TerminalAttention;

typedef struct {
    double *query;
    double *key;
    double *value;
    double *attention;
    double *context;
    double *output;
    double *grad_query;
    double *grad_key;
    double *grad_value;
    double *grad_attention;
    double *grad_context;
    double *grad_output;
    double *logits;
    double *grad_hidden;
} TerminalAttentionWorkspace;

static size_t terminal_attention_matrix_size(const TerminalAttention *head) {
    return (size_t)head->dim * head->dim;
}

static size_t terminal_attention_parameter_count(
    const TerminalAttention *head
) {
    return 4U * terminal_attention_matrix_size(head);
}

static double *terminal_attention_matrix(
    TerminalAttention *head,
    int matrix
) {
    return head->parameters +
        (size_t)matrix * terminal_attention_matrix_size(head);
}

static const double *terminal_attention_const_matrix(
    const TerminalAttention *head,
    int matrix
) {
    return head->parameters +
        (size_t)matrix * terminal_attention_matrix_size(head);
}

static void terminal_attention_initialize(
    TerminalAttention *head,
    int position_count,
    int dim,
    int head_count
) {
    if (position_count <= 0 || dim <= 0 || head_count <= 0 ||
        dim % head_count != 0) {
        fprintf(stderr, "invalid terminal attention dimensions\n");
        exit(EXIT_FAILURE);
    }
    *head = (TerminalAttention){
        .position_count = position_count,
        .dim = dim,
        .head_count = head_count,
        .head_dim = dim / head_count,
    };
    head->parameters = terminal_allocate(
        terminal_attention_parameter_count(head),
        sizeof(*head->parameters)
    );
    /* Identity Q/K/V and zero output projection make U=H exactly at epoch 0. */
    for (int matrix = 0; matrix < 3; matrix++) {
        double *weight = terminal_attention_matrix(head, matrix);
        for (int lane = 0; lane < dim; lane++) {
            weight[(size_t)lane * dim + lane] = 1.0;
        }
    }
}

static void terminal_attention_free(TerminalAttention *head) {
    free(head->parameters);
    memset(head, 0, sizeof(*head));
}

static void terminal_attention_workspace_initialize(
    TerminalAttentionWorkspace *workspace,
    const TerminalAttention *head,
    int vocab_size,
    int with_gradients
) {
    size_t hidden_count = (size_t)head->position_count * head->dim;
    size_t attention_count =
        (size_t)head->head_count * head->position_count *
        head->position_count;
    memset(workspace, 0, sizeof(*workspace));
    workspace->query = terminal_allocate(hidden_count, sizeof(*workspace->query));
    workspace->key = terminal_allocate(hidden_count, sizeof(*workspace->key));
    workspace->value = terminal_allocate(hidden_count, sizeof(*workspace->value));
    workspace->attention = terminal_allocate(
        attention_count,
        sizeof(*workspace->attention)
    );
    workspace->context = terminal_allocate(hidden_count, sizeof(*workspace->context));
    workspace->output = terminal_allocate(hidden_count, sizeof(*workspace->output));
    workspace->logits = terminal_allocate((size_t)vocab_size, sizeof(*workspace->logits));
    workspace->grad_hidden = terminal_allocate(
        (size_t)head->dim,
        sizeof(*workspace->grad_hidden)
    );
    if (with_gradients) {
        workspace->grad_query = terminal_allocate(
            hidden_count,
            sizeof(*workspace->grad_query)
        );
        workspace->grad_key = terminal_allocate(hidden_count, sizeof(*workspace->grad_key));
        workspace->grad_value = terminal_allocate(
            hidden_count,
            sizeof(*workspace->grad_value)
        );
        workspace->grad_attention = terminal_allocate(
            attention_count,
            sizeof(*workspace->grad_attention)
        );
        workspace->grad_context = terminal_allocate(
            hidden_count,
            sizeof(*workspace->grad_context)
        );
        workspace->grad_output = terminal_allocate(
            hidden_count,
            sizeof(*workspace->grad_output)
        );
    }
}

static void terminal_attention_workspace_free(
    TerminalAttentionWorkspace *workspace
) {
    free(workspace->grad_hidden);
    free(workspace->logits);
    free(workspace->grad_output);
    free(workspace->grad_context);
    free(workspace->grad_attention);
    free(workspace->grad_value);
    free(workspace->grad_key);
    free(workspace->grad_query);
    free(workspace->output);
    free(workspace->context);
    free(workspace->attention);
    free(workspace->value);
    free(workspace->key);
    free(workspace->query);
    memset(workspace, 0, sizeof(*workspace));
}

static void terminal_dense_sequence(
    double *output,
    const float *input,
    const double *weight,
    int position_count,
    int dim
) {
    for (int position = 0; position < position_count; position++) {
        for (int out = 0; out < dim; out++) {
            double value = 0.0;
            for (int in = 0; in < dim; in++) {
                value += weight[(size_t)out * dim + in] *
                    input[(size_t)position * dim + in];
            }
            output[(size_t)position * dim + out] = value;
        }
    }
}

static void terminal_attention_forward(
    const TerminalExample *example,
    const TerminalAttention *head,
    TerminalAttentionWorkspace *workspace
) {
    int positions = head->position_count;
    int dim = head->dim;
    int head_dim = head->head_dim;
    terminal_dense_sequence(
        workspace->query,
        example->hidden_sequence,
        terminal_attention_const_matrix(head, 0),
        positions,
        dim
    );
    terminal_dense_sequence(
        workspace->key,
        example->hidden_sequence,
        terminal_attention_const_matrix(head, 1),
        positions,
        dim
    );
    terminal_dense_sequence(
        workspace->value,
        example->hidden_sequence,
        terminal_attention_const_matrix(head, 2),
        positions,
        dim
    );
    memset(
        workspace->context,
        0,
        (size_t)positions * dim * sizeof(*workspace->context)
    );
    double scale = 1.0 / sqrt((double)head_dim);
    for (int attention_head = 0;
         attention_head < head->head_count;
         attention_head++) {
        int lane_start = attention_head * head_dim;
        for (int output_position = 0;
             output_position < positions;
             output_position++) {
            double maximum = -DBL_MAX;
            double *row = workspace->attention +
                ((size_t)attention_head * positions + output_position) *
                    positions;
            for (int source_position = 0;
                 source_position < positions;
                 source_position++) {
                double score = 0.0;
                for (int lane = 0; lane < head_dim; lane++) {
                    int coordinate = lane_start + lane;
                    score += workspace->query[
                        (size_t)output_position * dim + coordinate
                    ] * workspace->key[
                        (size_t)source_position * dim + coordinate
                    ];
                }
                score *= scale;
                row[source_position] = score;
                if (score > maximum) maximum = score;
            }
            double partition = 0.0;
            for (int source_position = 0;
                 source_position < positions;
                 source_position++) {
                row[source_position] = exp(row[source_position] - maximum);
                partition += row[source_position];
            }
            for (int source_position = 0;
                 source_position < positions;
                 source_position++) {
                row[source_position] /= partition;
                for (int lane = 0; lane < head_dim; lane++) {
                    int coordinate = lane_start + lane;
                    workspace->context[
                        (size_t)output_position * dim + coordinate
                    ] += row[source_position] * workspace->value[
                        (size_t)source_position * dim + coordinate
                    ];
                }
            }
        }
    }

    const double *output_weight = terminal_attention_const_matrix(head, 3);
    for (int position = 0; position < positions; position++) {
        for (int out = 0; out < dim; out++) {
            double value = 0.0;
            for (int in = 0; in < dim; in++) {
                value += output_weight[(size_t)out * dim + in] *
                    workspace->context[(size_t)position * dim + in];
            }
            workspace->output[(size_t)position * dim + out] = value;
        }
    }
}

static double terminal_attention_initial_parameter(
    const TerminalAttention *head,
    size_t parameter
) {
    size_t matrix_size = terminal_attention_matrix_size(head);
    int matrix = (int)(parameter / matrix_size);
    size_t coordinate = parameter % matrix_size;
    int row = (int)(coordinate / (size_t)head->dim);
    int column = (int)(coordinate % (size_t)head->dim);
    return matrix < 3 && row == column ? 1.0 : 0.0;
}

static double terminal_attention_loss(
    const TerminalExample *examples,
    int example_count,
    const Transformer *student,
    const TerminalAttention *head,
    double l2,
    double *parameter_gradient,
    double *token_accuracy,
    double *sequence_accuracy
) {
    int positions = head->position_count;
    int dim = head->dim;
    int vocab_size = student->config.vocab_size;
    int head_dim = head->head_dim;
    size_t hidden_count = (size_t)positions * dim;
    size_t attention_count =
        (size_t)head->head_count * positions * positions;
    size_t parameter_count = terminal_attention_parameter_count(head);
    if (parameter_gradient != NULL) {
        memset(
            parameter_gradient,
            0,
            parameter_count * sizeof(*parameter_gradient)
        );
    }
    TerminalAttentionWorkspace workspace;
    terminal_attention_workspace_initialize(
        &workspace,
        head,
        vocab_size,
        parameter_gradient != NULL
    );
    double loss = 0.0;
    unsigned long long correct_tokens = 0;
    unsigned long long correct_sequences = 0;
    double scale = 1.0 / sqrt((double)head_dim);

    for (int example_index = 0;
         example_index < example_count;
         example_index++) {
        const TerminalExample *example = &examples[example_index];
        terminal_attention_forward(example, head, &workspace);
        if (parameter_gradient != NULL) {
            memset(workspace.grad_query, 0, hidden_count * sizeof(*workspace.grad_query));
            memset(workspace.grad_key, 0, hidden_count * sizeof(*workspace.grad_key));
            memset(workspace.grad_value, 0, hidden_count * sizeof(*workspace.grad_value));
            memset(
                workspace.grad_attention,
                0,
                attention_count * sizeof(*workspace.grad_attention)
            );
            memset(workspace.grad_context, 0, hidden_count * sizeof(*workspace.grad_context));
            memset(workspace.grad_output, 0, hidden_count * sizeof(*workspace.grad_output));
        }
        int sequence_correct = 1;
        for (int position = 0; position < positions; position++) {
            double maximum = -DBL_MAX;
            int selected = -1;
            for (int token = 0; token < vocab_size; token++) {
                double value = 0.0;
                const float *embedding = student->weights.wcls +
                    (size_t)token * dim;
                for (int lane = 0; lane < dim; lane++) {
                    double hidden = example->hidden_sequence[
                        (size_t)position * dim + lane
                    ] + workspace.output[(size_t)position * dim + lane];
                    value += (double)embedding[lane] * hidden;
                }
                workspace.logits[token] = value;
                if (selected < 0 || value > maximum) {
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
            for (int token = 0; token < vocab_size; token++) {
                partition += exp(workspace.logits[token] - maximum);
            }
            double log_partition = maximum + log(partition);
            loss += log_partition - workspace.logits[target];

            if (parameter_gradient != NULL) {
                memset(
                    workspace.grad_hidden,
                    0,
                    (size_t)dim * sizeof(*workspace.grad_hidden)
                );
                for (int token = 0; token < vocab_size; token++) {
                    double error = exp(
                        workspace.logits[token] - log_partition
                    ) - (token == target ? 1.0 : 0.0);
                    const float *embedding = student->weights.wcls +
                        (size_t)token * dim;
                    for (int lane = 0; lane < dim; lane++) {
                        workspace.grad_hidden[lane] +=
                            error * (double)embedding[lane];
                    }
                }
                for (int lane = 0; lane < dim; lane++) {
                    workspace.grad_output[(size_t)position * dim + lane] =
                        workspace.grad_hidden[lane];
                }
            }
        }
        if (sequence_correct) correct_sequences++;

        if (parameter_gradient == NULL) continue;
        double *grad_wq = parameter_gradient;
        double *grad_wk = grad_wq + terminal_attention_matrix_size(head);
        double *grad_wv = grad_wk + terminal_attention_matrix_size(head);
        double *grad_wo = grad_wv + terminal_attention_matrix_size(head);
        const double *wo = terminal_attention_const_matrix(head, 3);

        for (int position = 0; position < positions; position++) {
            for (int out = 0; out < dim; out++) {
                double grad = workspace.grad_output[
                    (size_t)position * dim + out
                ];
                for (int in = 0; in < dim; in++) {
                    grad_wo[(size_t)out * dim + in] += grad *
                        workspace.context[(size_t)position * dim + in];
                    workspace.grad_context[
                        (size_t)position * dim + in
                    ] += wo[(size_t)out * dim + in] * grad;
                }
            }
        }

        for (int attention_head = 0;
             attention_head < head->head_count;
             attention_head++) {
            int lane_start = attention_head * head_dim;
            for (int output_position = 0;
                 output_position < positions;
                 output_position++) {
                double *attention_row = workspace.attention +
                    ((size_t)attention_head * positions + output_position) *
                        positions;
                double *grad_attention_row = workspace.grad_attention +
                    ((size_t)attention_head * positions + output_position) *
                        positions;
                for (int source_position = 0;
                     source_position < positions;
                     source_position++) {
                    double grad = 0.0;
                    for (int lane = 0; lane < head_dim; lane++) {
                        int coordinate = lane_start + lane;
                        grad += workspace.grad_context[
                            (size_t)output_position * dim + coordinate
                        ] * workspace.value[
                            (size_t)source_position * dim + coordinate
                        ];
                        workspace.grad_value[
                            (size_t)source_position * dim + coordinate
                        ] += attention_row[source_position] *
                            workspace.grad_context[
                                (size_t)output_position * dim + coordinate
                            ];
                    }
                    grad_attention_row[source_position] = grad;
                }
                double softmax_dot = 0.0;
                for (int source_position = 0;
                     source_position < positions;
                     source_position++) {
                    softmax_dot += grad_attention_row[source_position] *
                        attention_row[source_position];
                }
                for (int source_position = 0;
                     source_position < positions;
                     source_position++) {
                    double grad_score = attention_row[source_position] *
                        (grad_attention_row[source_position] - softmax_dot);
                    for (int lane = 0; lane < head_dim; lane++) {
                        int coordinate = lane_start + lane;
                        workspace.grad_query[
                            (size_t)output_position * dim + coordinate
                        ] += grad_score * workspace.key[
                            (size_t)source_position * dim + coordinate
                        ] * scale;
                        workspace.grad_key[
                            (size_t)source_position * dim + coordinate
                        ] += grad_score * workspace.query[
                            (size_t)output_position * dim + coordinate
                        ] * scale;
                    }
                }
            }
        }

        for (int out = 0; out < dim; out++) {
            for (int in = 0; in < dim; in++) {
                double q_gradient = 0.0;
                double k_gradient = 0.0;
                double v_gradient = 0.0;
                for (int position = 0; position < positions; position++) {
                    double hidden = example->hidden_sequence[
                        (size_t)position * dim + in
                    ];
                    q_gradient += workspace.grad_query[
                        (size_t)position * dim + out
                    ] * hidden;
                    k_gradient += workspace.grad_key[
                        (size_t)position * dim + out
                    ] * hidden;
                    v_gradient += workspace.grad_value[
                        (size_t)position * dim + out
                    ] * hidden;
                }
                grad_wq[(size_t)out * dim + in] += q_gradient;
                grad_wk[(size_t)out * dim + in] += k_gradient;
                grad_wv[(size_t)out * dim + in] += v_gradient;
            }
        }
    }

    double observation_count = (double)example_count * positions;
    loss /= observation_count;
    for (size_t parameter = 0; parameter < parameter_count; parameter++) {
        double displacement = head->parameters[parameter] -
            terminal_attention_initial_parameter(head, parameter);
        loss += 0.5 * l2 * displacement * displacement;
        if (parameter_gradient != NULL) {
            parameter_gradient[parameter] =
                parameter_gradient[parameter] / observation_count +
                l2 * displacement;
        }
    }
    if (token_accuracy != NULL) {
        *token_accuracy = (double)correct_tokens / observation_count;
    }
    if (sequence_accuracy != NULL) {
        *sequence_accuracy =
            (double)correct_sequences / (double)example_count;
    }
    terminal_attention_workspace_free(&workspace);
    return loss;
}

static void train_terminal_attention(
    const TerminalExample *training,
    int training_count,
    const TerminalExample *validation,
    int validation_count,
    const Transformer *student,
    int epochs,
    double learning_rate,
    double l2,
    TerminalAttention *head
) {
    size_t parameter_count = terminal_attention_parameter_count(head);
    double *gradient = terminal_allocate(parameter_count, sizeof(*gradient));
    double *first_moment = terminal_allocate(parameter_count, sizeof(*first_moment));
    double *second_moment = terminal_allocate(parameter_count, sizeof(*second_moment));
    double *best_parameters = terminal_allocate(
        parameter_count,
        sizeof(*best_parameters)
    );
    double best_validation = DBL_MAX;
    double beta1_power = 1.0;
    double beta2_power = 1.0;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        double training_accuracy = 0.0;
        double training_sequence_accuracy = 0.0;
        double training_loss = terminal_attention_loss(
            training,
            training_count,
            student,
            head,
            l2,
            gradient,
            &training_accuracy,
            &training_sequence_accuracy
        );
        beta1_power *= 0.9;
        beta2_power *= 0.999;
        for (size_t parameter = 0; parameter < parameter_count; parameter++) {
            first_moment[parameter] =
                0.9 * first_moment[parameter] + 0.1 * gradient[parameter];
            second_moment[parameter] =
                0.999 * second_moment[parameter] +
                0.001 * gradient[parameter] * gradient[parameter];
            double corrected_first = first_moment[parameter] /
                (1.0 - beta1_power);
            double corrected_second = second_moment[parameter] /
                (1.0 - beta2_power);
            head->parameters[parameter] -= learning_rate * corrected_first /
                (sqrt(corrected_second) + 1e-8);
        }

        double validation_accuracy = 0.0;
        double validation_sequence_accuracy = 0.0;
        double validation_loss = terminal_attention_loss(
            validation,
            validation_count,
            student,
            head,
            l2,
            NULL,
            &validation_accuracy,
            &validation_sequence_accuracy
        );
        if (validation_loss < best_validation) {
            best_validation = validation_loss;
            memcpy(
                best_parameters,
                head->parameters,
                parameter_count * sizeof(*best_parameters)
            );
        }
        if (epoch == 1 || epoch == epochs || epoch % 10 == 0) {
            fprintf(
                stderr,
                "attention_epoch=%d train_loss=%.6f "
                "train_token_accuracy=%.4f train_sequence_accuracy=%.4f "
                "validation_loss=%.6f validation_token_accuracy=%.4f "
                "validation_sequence_accuracy=%.4f\n",
                epoch,
                training_loss,
                training_accuracy,
                training_sequence_accuracy,
                validation_loss,
                validation_accuracy,
                validation_sequence_accuracy
            );
        }
    }
    memcpy(
        head->parameters,
        best_parameters,
        parameter_count * sizeof(*head->parameters)
    );
    free(best_parameters);
    free(second_moment);
    free(first_moment);
    free(gradient);
}

static void select_terminal_attention_tokens(
    const TerminalExample *example,
    const Transformer *student,
    const TerminalAttention *head,
    int *tokens
) {
    TerminalAttentionWorkspace workspace;
    terminal_attention_workspace_initialize(
        &workspace,
        head,
        student->config.vocab_size,
        0
    );
    terminal_attention_forward(example, head, &workspace);
    for (int position = 0; position < head->position_count; position++) {
        int selected = -1;
        double maximum = -DBL_MAX;
        for (int token = 0; token < student->config.vocab_size; token++) {
            const float *embedding = student->weights.wcls +
                (size_t)token * head->dim;
            double logit = 0.0;
            for (int lane = 0; lane < head->dim; lane++) {
                logit += (double)embedding[lane] *
                    (example->hidden_sequence[
                        (size_t)position * head->dim + lane
                    ] + workspace.output[
                        (size_t)position * head->dim + lane
                    ]);
            }
            if (selected < 0 || logit > maximum) {
                selected = token;
                maximum = logit;
            }
        }
        tokens[position] = selected;
    }
    terminal_attention_workspace_free(&workspace);
}

static void print_terminal_attention_examples(
    const TerminalExample *examples,
    int example_count,
    int shown,
    const Transformer *student,
    Tokenizer *tokenizer,
    const TerminalAttention *head
) {
    if (shown > example_count) shown = example_count;
    int positions = head->position_count;
    double *zero = terminal_allocate(
        (size_t)positions * positions,
        sizeof(*zero)
    );
    int *baseline = terminal_allocate((size_t)positions, sizeof(*baseline));
    int *attention = terminal_allocate((size_t)positions, sizeof(*attention));
    for (int index = 0; index < shown; index++) {
        const TerminalExample *example = &examples[index];
        select_terminal_tokens(
            example,
            positions,
            student->config.vocab_size,
            zero,
            baseline
        );
        select_terminal_attention_tokens(
            example,
            student,
            head,
            attention
        );
        int previous = example->prompt_tokens[0];
        printf("\nheldout=%d\nprompt: ", index);
        print_token_span(
            tokenizer,
            previous,
            example->prompt_tokens + 1,
            example->prompt_count - 1
        );
        previous = example->prompt_tokens[example->prompt_count - 1];
        printf("\n15M-text target: ");
        print_token_span(tokenizer, previous, example->targets, positions);
        printf("\ndelayed independent: ");
        print_token_span(tokenizer, previous, baseline, positions);
        printf("\ntrained terminal attention: ");
        print_token_span(tokenizer, previous, attention, positions);
        printf("\n");
    }
    free(attention);
    free(baseline);
    free(zero);
}

static void save_terminal_attention(
    const char *path,
    const TerminalAttention *head
) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        fprintf(stderr, "could not open terminal attention output %s\n", path);
        exit(EXIT_FAILURE);
    }
    const unsigned char magic[8] = {'T','A','T','T','N','0','0','1'};
    size_t parameter_count = terminal_attention_parameter_count(head);
    if (fwrite(magic, sizeof(magic), 1, file) != 1 ||
        fwrite(&head->position_count, sizeof(head->position_count), 1, file) != 1 ||
        fwrite(&head->dim, sizeof(head->dim), 1, file) != 1 ||
        fwrite(&head->head_count, sizeof(head->head_count), 1, file) != 1 ||
        fwrite(
            head->parameters,
            sizeof(*head->parameters),
            parameter_count,
            file
        ) != parameter_count) {
        fprintf(stderr, "could not write terminal attention output %s\n", path);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    fclose(file);
}

/* ------------------------------------------------------------------------- */
/* Terminal linear-chain company model and exact Viterbi projection.          */

typedef struct {
    int vocab_size;
    double smoothing;
    double *transition_log_probability; /* previous x next */
} TerminalCompany;

static void terminal_company_build(
    TerminalCompany *company,
    const TerminalExample *examples,
    int example_count,
    int completion_count,
    int vocab_size,
    double smoothing
) {
    if (vocab_size <= 0 || !(smoothing > 0.0)) {
        fprintf(stderr, "invalid terminal company dimensions\n");
        exit(EXIT_FAILURE);
    }
    *company = (TerminalCompany){
        .vocab_size = vocab_size,
        .smoothing = smoothing,
    };
    size_t transition_count = (size_t)vocab_size * vocab_size;
    double *counts = terminal_allocate(transition_count, sizeof(*counts));
    double *row_counts = terminal_allocate((size_t)vocab_size, sizeof(*row_counts));
    company->transition_log_probability = terminal_allocate(
        transition_count,
        sizeof(*company->transition_log_probability)
    );

    for (int example_index = 0;
         example_index < example_count;
         example_index++) {
        const TerminalExample *example = &examples[example_index];
        int previous = example->prompt_tokens[0];
        for (int index = 1; index < example->prompt_count; index++) {
            int token = example->prompt_tokens[index];
            counts[(size_t)previous * vocab_size + token] += 1.0;
            row_counts[previous] += 1.0;
            previous = token;
        }
        for (int index = 0; index < completion_count; index++) {
            int token = example->targets[index];
            counts[(size_t)previous * vocab_size + token] += 1.0;
            row_counts[previous] += 1.0;
            previous = token;
        }
    }

    for (int previous = 0; previous < vocab_size; previous++) {
        double denominator = row_counts[previous] +
            smoothing * vocab_size;
        for (int token = 0; token < vocab_size; token++) {
            double numerator = counts[
                (size_t)previous * vocab_size + token
            ] + smoothing;
            company->transition_log_probability[
                (size_t)previous * vocab_size + token
            ] = log(numerator / denominator);
        }
    }
    free(row_counts);
    free(counts);
}

static void terminal_company_free(TerminalCompany *company) {
    free(company->transition_log_probability);
    memset(company, 0, sizeof(*company));
}

static void terminal_attention_logits(
    const TerminalExample *example,
    const Transformer *student,
    const TerminalAttention *head,
    double *logits
) {
    TerminalAttentionWorkspace workspace;
    terminal_attention_workspace_initialize(
        &workspace,
        head,
        student->config.vocab_size,
        0
    );
    terminal_attention_forward(example, head, &workspace);
    for (int position = 0; position < head->position_count; position++) {
        for (int token = 0; token < student->config.vocab_size; token++) {
            const float *embedding = student->weights.wcls +
                (size_t)token * head->dim;
            double value = 0.0;
            for (int lane = 0; lane < head->dim; lane++) {
                value += (double)embedding[lane] *
                    (example->hidden_sequence[
                        (size_t)position * head->dim + lane
                    ] + workspace.output[
                        (size_t)position * head->dim + lane
                    ]);
            }
            logits[(size_t)position * student->config.vocab_size + token] =
                value;
        }
    }
    terminal_attention_workspace_free(&workspace);
}

static void terminal_company_viterbi(
    const TerminalCompany *company,
    const double *unary_logits,
    int position_count,
    int preceding_token,
    int *selected_tokens
) {
    int vocab_size = company->vocab_size;
    double *previous = terminal_allocate((size_t)vocab_size, sizeof(*previous));
    double *current = terminal_allocate((size_t)vocab_size, sizeof(*current));
    int *backpointers = terminal_allocate(
        (size_t)position_count * vocab_size,
        sizeof(*backpointers)
    );

    for (int token = 0; token < vocab_size; token++) {
        previous[token] = unary_logits[token] +
            company->transition_log_probability[
                (size_t)preceding_token * vocab_size + token
            ];
        backpointers[token] = preceding_token;
    }
    for (int position = 1; position < position_count; position++) {
        for (int token = 0; token < vocab_size; token++) {
            int best_previous = -1;
            double best = -DBL_MAX;
            for (int candidate = 0; candidate < vocab_size; candidate++) {
                double score = previous[candidate] +
                    company->transition_log_probability[
                        (size_t)candidate * vocab_size + token
                    ];
                if (best_previous < 0 || score > best) {
                    best_previous = candidate;
                    best = score;
                }
            }
            current[token] = unary_logits[(size_t)position * vocab_size + token] +
                best;
            backpointers[(size_t)position * vocab_size + token] = best_previous;
        }
        double *swap = previous;
        previous = current;
        current = swap;
    }
    int selected = 0;
    for (int token = 1; token < vocab_size; token++) {
        if (previous[token] > previous[selected]) selected = token;
    }
    selected_tokens[position_count - 1] = selected;
    for (int position = position_count - 1; position > 0; position--) {
        selected = backpointers[(size_t)position * vocab_size + selected];
        selected_tokens[position - 1] = selected;
    }
    free(backpointers);
    free(current);
    free(previous);
}

static void terminal_company_accuracy(
    const TerminalExample *examples,
    int example_count,
    const Transformer *student,
    const TerminalAttention *head,
    const TerminalCompany *company,
    double *token_accuracy,
    double *sequence_accuracy
) {
    int positions = head->position_count;
    int vocab_size = student->config.vocab_size;
    double *logits = terminal_allocate(
        (size_t)positions * vocab_size,
        sizeof(*logits)
    );
    int *tokens = terminal_allocate((size_t)positions, sizeof(*tokens));
    unsigned long long correct_tokens = 0;
    unsigned long long correct_sequences = 0;
    for (int example_index = 0;
         example_index < example_count;
         example_index++) {
        const TerminalExample *example = &examples[example_index];
        terminal_attention_logits(example, student, head, logits);
        terminal_company_viterbi(
            company,
            logits,
            positions,
            example->prompt_tokens[example->prompt_count - 1],
            tokens
        );
        int sequence_correct = 1;
        for (int position = 0; position < positions; position++) {
            if (tokens[position] == example->targets[position]) {
                correct_tokens++;
            } else {
                sequence_correct = 0;
            }
        }
        if (sequence_correct) correct_sequences++;
    }
    *token_accuracy = (double)correct_tokens /
        ((double)example_count * positions);
    *sequence_accuracy = (double)correct_sequences / example_count;
    free(tokens);
    free(logits);
}

static void print_terminal_company_examples(
    const TerminalExample *examples,
    int example_count,
    int shown,
    const Transformer *student,
    Tokenizer *tokenizer,
    const TerminalAttention *head,
    const TerminalCompany *company
) {
    if (shown > example_count) shown = example_count;
    int positions = head->position_count;
    int vocab_size = student->config.vocab_size;
    double *zero = terminal_allocate(
        (size_t)positions * positions,
        sizeof(*zero)
    );
    double *logits = terminal_allocate(
        (size_t)positions * vocab_size,
        sizeof(*logits)
    );
    int *baseline = terminal_allocate((size_t)positions, sizeof(*baseline));
    int *attention = terminal_allocate((size_t)positions, sizeof(*attention));
    int *joint = terminal_allocate((size_t)positions, sizeof(*joint));
    for (int index = 0; index < shown; index++) {
        const TerminalExample *example = &examples[index];
        select_terminal_tokens(
            example,
            positions,
            vocab_size,
            zero,
            baseline
        );
        terminal_attention_logits(example, student, head, logits);
        for (int position = 0; position < positions; position++) {
            int selected = 0;
            for (int token = 1; token < vocab_size; token++) {
                if (logits[(size_t)position * vocab_size + token] >
                    logits[(size_t)position * vocab_size + selected]) {
                    selected = token;
                }
            }
            attention[position] = selected;
        }
        terminal_company_viterbi(
            company,
            logits,
            positions,
            example->prompt_tokens[example->prompt_count - 1],
            joint
        );

        int previous = example->prompt_tokens[0];
        printf("\nheldout=%d\nprompt: ", index);
        print_token_span(
            tokenizer,
            previous,
            example->prompt_tokens + 1,
            example->prompt_count - 1
        );
        previous = example->prompt_tokens[example->prompt_count - 1];
        printf("\n15M-text target: ");
        print_token_span(tokenizer, previous, example->targets, positions);
        printf("\ndelayed independent: ");
        print_token_span(tokenizer, previous, baseline, positions);
        printf("\nterminal attention rowwise: ");
        print_token_span(tokenizer, previous, attention, positions);
        printf("\nterminal attention + company Viterbi: ");
        print_token_span(tokenizer, previous, joint, positions);
        printf("\n");
    }
    free(joint);
    free(attention);
    free(baseline);
    free(logits);
    free(zero);
}

static void save_terminal_company(
    const char *path,
    const TerminalAttention *head,
    const TerminalCompany *company
) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        fprintf(stderr, "could not open terminal company output %s\n", path);
        exit(EXIT_FAILURE);
    }
    const unsigned char magic[8] = {'T','C','R','F','0','0','0','1'};
    size_t attention_parameters = terminal_attention_parameter_count(head);
    size_t transition_count =
        (size_t)company->vocab_size * company->vocab_size;
    if (fwrite(magic, sizeof(magic), 1, file) != 1 ||
        fwrite(&head->position_count, sizeof(head->position_count), 1, file) != 1 ||
        fwrite(&head->dim, sizeof(head->dim), 1, file) != 1 ||
        fwrite(&head->head_count, sizeof(head->head_count), 1, file) != 1 ||
        fwrite(&company->vocab_size, sizeof(company->vocab_size), 1, file) != 1 ||
        fwrite(&company->smoothing, sizeof(company->smoothing), 1, file) != 1 ||
        fwrite(
            head->parameters,
            sizeof(*head->parameters),
            attention_parameters,
            file
        ) != attention_parameters ||
        fwrite(
            company->transition_log_probability,
            sizeof(*company->transition_log_probability),
            transition_count,
            file
        ) != transition_count) {
        fprintf(stderr, "could not write terminal company output %s\n", path);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    fclose(file);
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
        fprintf(stderr, "seed must be a positive integer\n");
        exit(EXIT_FAILURE);
    }
    return value;
}

static void terminal_usage(const char *program) {
    fprintf(
        stderr,
        "usage: %s TEACHER_CHECKPOINT TEACHER_TOKENIZER "
        "STUDENT_CHECKPOINT STUDENT_TOKENIZER [TRAIN_EXAMPLES] "
        "[VALIDATION_EXAMPLES] [COMPLETION_TOKENS] [PROMPT_TOKENS] "
        "[TEACHER_STEPS] [EPOCHS] [SEED] [OUTPUT] [HEAD]\n"
        "HEAD is position, attention, or crf; default position.\n",
        program
    );
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
    if (argc < 5 || argc > 14) terminal_usage(argv[0]);
    const char *teacher_checkpoint = argv[1];
    const char *teacher_tokenizer_path = argv[2];
    const char *student_checkpoint = argv[3];
    const char *student_tokenizer_path = argv[4];
    int training_count = argc > 5 ? parse_positive(argv[5], "train examples") : 192;
    int validation_count = argc > 6 ? parse_positive(argv[6], "validation examples") : 48;
    int completion_count = argc > 7 ? parse_positive(argv[7], "completion tokens") : 24;
    int prompt_count = argc > 8 ? parse_positive(argv[8], "prompt tokens") : 24;
    int teacher_steps = argc > 9 ? parse_positive(argv[9], "teacher steps") : 64;
    int epochs = argc > 10 ? parse_positive(argv[10], "epochs") : 80;
    unsigned long long seed = argc > 11 ? parse_seed(argv[11]) : 26015ULL;
    const char *output_path = argc > 12 ? argv[12] : "terminal_mixer.bin";
    const char *head_kind = argc > 13 ? argv[13] : "position";
    if (strcmp(head_kind, "position") != 0 &&
        strcmp(head_kind, "attention") != 0 &&
        strcmp(head_kind, "crf") != 0) {
        terminal_usage(argv[0]);
    }

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
    if (teacher_steps > teacher.config.seq_len) {
        fprintf(stderr, "teacher steps exceed its sequence length\n");
        exit(EXIT_FAILURE);
    }
    if (prompt_count + completion_count - 1 > student.config.seq_len) {
        fprintf(stderr, "student prompt and completion exceed its sequence length\n");
        exit(EXIT_FAILURE);
    }

    Sampler teacher_sampler;
    build_sampler(
        &teacher_sampler,
        teacher.config.vocab_size,
        0.8f,
        0.9f,
        seed
    );
    fprintf(
        stderr,
        "teacher_vocab=%d student_vocab=%d student_dim=%d "
        "prompt_tokens=%d completion_tokens=%d temperature=0.8 top_p=0.9 "
        "seed=%llu head=%s\n",
        teacher.config.vocab_size,
        student.config.vocab_size,
        student.config.dim,
        prompt_count,
        completion_count,
        seed,
        head_kind
    );
    TerminalExample *training = generate_terminal_dataset(
        training_count,
        &teacher,
        &teacher_tokenizer,
        &teacher_sampler,
        &student,
        &student_tokenizer,
        prompt_count,
        completion_count,
        teacher_steps,
        "training"
    );
    TerminalExample *validation = generate_terminal_dataset(
        validation_count,
        &teacher,
        &teacher_tokenizer,
        &teacher_sampler,
        &student,
        &student_tokenizer,
        prompt_count,
        completion_count,
        teacher_steps,
        "validation"
    );

    if (strcmp(head_kind, "position") == 0) {
        size_t parameter_count = (size_t)completion_count * completion_count;
        double *delta = terminal_allocate(parameter_count, sizeof(*delta));
        double baseline_token_accuracy = 0.0;
        double baseline_sequence_accuracy = 0.0;
        double baseline_loss = terminal_mixer_loss(
            validation,
            validation_count,
            completion_count,
            student.config.vocab_size,
            delta,
            0.0,
            NULL,
            &baseline_token_accuracy,
            &baseline_sequence_accuracy
        );
        fprintf(
            stderr,
            "baseline validation_loss=%.6f validation_token_accuracy=%.4f "
            "validation_sequence_accuracy=%.4f\n",
            baseline_loss,
            baseline_token_accuracy,
            baseline_sequence_accuracy
        );

        train_terminal_mixer(
            training,
            training_count,
            validation,
            validation_count,
            completion_count,
            student.config.vocab_size,
            epochs,
            0.01,
            1e-4,
            delta
        );
        double trained_token_accuracy = 0.0;
        double trained_sequence_accuracy = 0.0;
        double trained_loss = terminal_mixer_loss(
            validation,
            validation_count,
            completion_count,
            student.config.vocab_size,
            delta,
            0.0,
            NULL,
            &trained_token_accuracy,
            &trained_sequence_accuracy
        );
        fprintf(
            stderr,
            "trained validation_loss=%.6f validation_token_accuracy=%.4f "
            "validation_sequence_accuracy=%.4f\n",
            trained_loss,
            trained_token_accuracy,
            trained_sequence_accuracy
        );
        save_terminal_mixer(output_path, completion_count, delta);
        print_terminal_examples(
            validation,
            validation_count,
            8,
            completion_count,
            student.config.vocab_size,
            &student_tokenizer,
            delta
        );
        free(delta);
    } else {
        TerminalAttention attention;
        terminal_attention_initialize(
            &attention,
            completion_count,
            student.config.dim,
            student.config.n_heads
        );
        double baseline_token_accuracy = 0.0;
        double baseline_sequence_accuracy = 0.0;
        double baseline_loss = terminal_attention_loss(
            validation,
            validation_count,
            &student,
            &attention,
            0.0,
            NULL,
            &baseline_token_accuracy,
            &baseline_sequence_accuracy
        );
        fprintf(
            stderr,
            "attention baseline validation_loss=%.6f "
            "validation_token_accuracy=%.4f "
            "validation_sequence_accuracy=%.4f parameters=%zu\n",
            baseline_loss,
            baseline_token_accuracy,
            baseline_sequence_accuracy,
            terminal_attention_parameter_count(&attention)
        );
        train_terminal_attention(
            training,
            training_count,
            validation,
            validation_count,
            &student,
            epochs,
            0.001,
            1e-6,
            &attention
        );
        double trained_token_accuracy = 0.0;
        double trained_sequence_accuracy = 0.0;
        double trained_loss = terminal_attention_loss(
            validation,
            validation_count,
            &student,
            &attention,
            0.0,
            NULL,
            &trained_token_accuracy,
            &trained_sequence_accuracy
        );
        fprintf(
            stderr,
            "trained attention validation_loss=%.6f "
            "validation_token_accuracy=%.4f "
            "validation_sequence_accuracy=%.4f\n",
            trained_loss,
            trained_token_accuracy,
            trained_sequence_accuracy
        );
        if (strcmp(head_kind, "attention") == 0) {
            save_terminal_attention(output_path, &attention);
            print_terminal_attention_examples(
                validation,
                validation_count,
                8,
                &student,
                &student_tokenizer,
                &attention
            );
        } else {
            TerminalCompany company;
            terminal_company_build(
                &company,
                training,
                training_count,
                completion_count,
                student.config.vocab_size,
                0.5
            );
            double joint_token_accuracy = 0.0;
            double joint_sequence_accuracy = 0.0;
            terminal_company_accuracy(
                validation,
                validation_count,
                &student,
                &attention,
                &company,
                &joint_token_accuracy,
                &joint_sequence_accuracy
            );
            fprintf(
                stderr,
                "terminal company smoothing=%.3f validation_token_accuracy=%.4f "
                "validation_sequence_accuracy=%.4f\n",
                company.smoothing,
                joint_token_accuracy,
                joint_sequence_accuracy
            );
            save_terminal_company(output_path, &attention, &company);
            print_terminal_company_examples(
                validation,
                validation_count,
                8,
                &student,
                &student_tokenizer,
                &attention,
                &company
            );
            terminal_company_free(&company);
        }
        terminal_attention_free(&attention);
    }

    for (int index = 0; index < validation_count; index++) {
        free_terminal_example(&validation[index]);
    }
    for (int index = 0; index < training_count; index++) {
        free_terminal_example(&training[index]);
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
