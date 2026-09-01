/*
 * Three-action grammatical cubes with task-relative token observations.
 *
 * The eight supplied terms are indexed by the A, B, and C action bits:
 *
 *   x, Ax, Bx, ABx, Cx, ACx, BCx, ABCx.
 *
 * A and B must form the same aligned token-constructor square in both C
 * fibers.  C must be one uniform suffix constructor: every C corner must have
 * its corresponding non-C corner as an exact token prefix and all four token
 * suffixes must be identical.
 *
 * At every typed transformer boundary, the complete local A/B action jet for
 * each C fiber is written to an append-only float32 sidecar.  C changes the
 * frontier type by changing the number of positions, so the program does not
 * invent a local subtraction between those differently typed fibers.  At the
 * terminal boundary both fibers have the same fixed token-observation type;
 * there the program applies a genuine three-bit fast Moebius transform and
 * retains carrier, all first-order, all pairwise, and the third-order terms.
 *
 * The terminal observation is a vector of learned classifier contrasts
 *
 *   logit(token_i) - logit(reference_token).
 *
 * No probability, norm, sum across candidates, or scalar completion reward
 * is constructed.
 */

#define CPS_GRAMMAR_ACTIONS_NO_MAIN
#include "cps_grammar_actions.c"

#include <sys/types.h>

enum {
    CUBE_X = 0,
    CUBE_A = 1,
    CUBE_B = 2,
    CUBE_AB = 3,
    CUBE_C = 4,
    CUBE_AC = 5,
    CUBE_BC = 6,
    CUBE_ABC = 7,
    CUBE_CORNER_COUNT = 8,
    SQUARE_CORNER_COUNT = 4
};

typedef struct {
    const char *trace_path;
    const char *jet_path;
    const char *observer_path;
    int reference_token;
    bool retain_local_jets;
} CubeOptions;

typedef struct {
    Transformer *transformer;
    int reference_token;
    int *tokens;
    int count;
} TokenContrastObserver;

typedef struct {
    FILE *trace;
    FILE *jet;
    const char *fiber;
    int boundary_index;
    double maximum_mobius_inverse_defect;
    double maximum_chain_output_defect;
} LocalJetWriter;

static CubeOptions parse_cube_options(int argc, char **argv) {
    if (argc < 12) {
        fprintf(
            stderr,
            "usage: %s CHECKPOINT TOKENIZER X AX BX ABX CX ACX BCX ABCX "
            "OBSERVER_TSV [--reference-token ID] "
            "[--jet-bin PATH | --terminal-only] [--trace PATH]\n",
            argv[0]
        );
        exit(EXIT_FAILURE);
    }
    CubeOptions options = {
        .observer_path = argv[11],
        .reference_token = 1,
        .retain_local_jets = true
    };
    for (int index = 12; index < argc;) {
        if (strcmp(argv[index], "--reference-token") == 0 &&
            index + 1 < argc) {
            errno = 0;
            char *end = NULL;
            long value = strtol(argv[index + 1], &end, 10);
            if (errno != 0 || end == argv[index + 1] || *end != '\0' ||
                value < 0 || value > INT_MAX) {
                fail("invalid cube observer reference token");
            }
            options.reference_token = (int)value;
            index += 2;
        } else if (strcmp(argv[index], "--jet-bin") == 0 &&
                   index + 1 < argc) {
            options.jet_path = argv[index + 1];
            index += 2;
        } else if (strcmp(argv[index], "--trace") == 0 &&
                   index + 1 < argc) {
            options.trace_path = argv[index + 1];
            index += 2;
        } else if (strcmp(argv[index], "--terminal-only") == 0) {
            options.retain_local_jets = false;
            index++;
        } else {
            fail("unrecognized cps_grammar_cube option");
        }
    }
    if (options.retain_local_jets && options.jet_path == NULL) {
        fail("three-action cube requires --jet-bin PATH");
    }
    if (!options.retain_local_jets && options.jet_path != NULL) {
        fail("--terminal-only cannot be combined with --jet-bin");
    }
    if (options.trace_path == NULL) {
        fail("three-action cube requires --trace PATH");
    }
    return options;
}

static TokenContrastObserver read_token_observer(
    Transformer *transformer,
    const char *path,
    int reference_token
) {
    if (reference_token < 0 ||
        reference_token >= transformer->config.vocab_size) {
        fail("observer reference token is outside the vocabulary");
    }
    FILE *file = fopen(path, "rb");
    if (file == NULL) fail("could not open observer token TSV");
    int capacity = 128;
    int count = 0;
    int *tokens = checked_calloc((size_t)capacity, sizeof(*tokens));
    char line[4096];
    while (fgets(line, sizeof(line), file) != NULL) {
        char *cursor = line;
        while (*cursor == ' ' || *cursor == '\t') cursor++;
        if (*cursor == '\0' || *cursor == '\n' || *cursor == '#') continue;
        errno = 0;
        char *end = NULL;
        long value = strtol(cursor, &end, 10);
        if (errno != 0 || end == cursor || value < 0 ||
            value >= transformer->config.vocab_size) {
            fail("invalid observer token TSV row");
        }
        if ((int)value == reference_token) {
            fail("reference token must not occur among contrast tokens");
        }
        for (int index = 0; index < count; index++) {
            if (tokens[index] == (int)value) {
                fail("observer token TSV contains a duplicate ID");
            }
        }
        if (count == capacity) {
            capacity *= 2;
            int *grown = realloc(tokens, (size_t)capacity * sizeof(*tokens));
            if (grown == NULL) fail("observer token allocation failed");
            tokens = grown;
        }
        tokens[count++] = (int)value;
    }
    if (ferror(file) || fclose(file) != 0) {
        fail("could not read observer token TSV");
    }
    if (count == 0) fail("observer token TSV is empty");
    return (TokenContrastObserver){
        .transformer = transformer,
        .reference_token = reference_token,
        .tokens = tokens,
        .count = count
    };
}

static void free_token_observer(TokenContrastObserver *observer) {
    free(observer->tokens);
    memset(observer, 0, sizeof(*observer));
}

static float classifier_coordinate(
    const Transformer *transformer,
    const float *hidden,
    int token
) {
    int dim = transformer->config.dim;
    const float *weights = transformer->weights.wcls + (size_t)token * dim;
    float value = 0.0f;
    for (int lane = 0; lane < dim; lane++) {
        value += weights[lane] * hidden[lane];
    }
    return value;
}

static void observe_token_contrasts(
    const TokenContrastObserver *observer,
    const float *hidden,
    float *result
) {
    float reference = classifier_coordinate(
        observer->transformer,
        hidden,
        observer->reference_token
    );
    for (int index = 0; index < observer->count; index++) {
        result[index] = classifier_coordinate(
            observer->transformer,
            hidden,
            observer->tokens[index]
        ) - reference;
    }
}

static bool factorized_four_corner_square(
    const EncodedContext contexts[SQUARE_CORNER_COUNT]
) {
    int positions = contexts[CUBE_X].count;
    for (int role = 1; role < SQUARE_CORNER_COUNT; role++) {
        if (contexts[role].count != positions) return false;
    }
    bool saw_a = false;
    bool saw_b = false;
    for (int position = 0; position < positions; position++) {
        int x = contexts[CUBE_X].tokens[position];
        int a = contexts[CUBE_A].tokens[position];
        int b = contexts[CUBE_B].tokens[position];
        int ab = contexts[CUBE_AB].tokens[position];
        if (x == a && x == b && x == ab) continue;
        if (x == b && a == ab && x != a) {
            saw_a = true;
        } else if (x == a && b == ab && x != b) {
            saw_b = true;
        } else {
            return false;
        }
    }
    return saw_a && saw_b;
}

static int validate_uniform_extension(
    const EncodedContext cube[CUBE_CORNER_COUNT]
) {
    int suffix_count = cube[CUBE_C].count - cube[CUBE_X].count;
    if (suffix_count <= 0) fail("C action must add a nonempty token suffix");
    const int base_roles[SQUARE_CORNER_COUNT] = {
        CUBE_X, CUBE_A, CUBE_B, CUBE_AB
    };
    const int extended_roles[SQUARE_CORNER_COUNT] = {
        CUBE_C, CUBE_AC, CUBE_BC, CUBE_ABC
    };
    for (int index = 0; index < SQUARE_CORNER_COUNT; index++) {
        const EncodedContext *base = &cube[base_roles[index]];
        const EncodedContext *extended = &cube[extended_roles[index]];
        if (extended->count != base->count + suffix_count) {
            fail("C action changes token length differently between corners");
        }
        if (memcmp(
                extended->tokens,
                base->tokens,
                (size_t)base->count * sizeof(*base->tokens)
            ) != 0) {
            fail("C action does not preserve a corner as an exact token prefix");
        }
        if (memcmp(
                extended->tokens + base->count,
                cube[CUBE_C].tokens + cube[CUBE_X].count,
                (size_t)suffix_count * sizeof(*extended->tokens)
            ) != 0) {
            fail("C action is not one uniform token suffix constructor");
        }
    }
    return suffix_count;
}

static void fast_mobius(float *values, int bits, int width) {
    int corners = 1 << bits;
    for (int bit = 0; bit < bits; bit++) {
        int flag = 1 << bit;
        for (int mask = 0; mask < corners; mask++) {
            if ((mask & flag) == 0) continue;
            float *target = values + (size_t)mask * width;
            const float *source = values + (size_t)(mask ^ flag) * width;
            for (int index = 0; index < width; index++) {
                target[index] -= source[index];
            }
        }
    }
}

static void inverse_fast_mobius(float *values, int bits, int width) {
    int corners = 1 << bits;
    for (int bit = 0; bit < bits; bit++) {
        int flag = 1 << bit;
        for (int mask = 0; mask < corners; mask++) {
            if ((mask & flag) == 0) continue;
            float *target = values + (size_t)mask * width;
            const float *source = values + (size_t)(mask ^ flag) * width;
            for (int index = 0; index < width; index++) {
                target[index] += source[index];
            }
        }
    }
}

static double maximum_difference(
    const float *left,
    const float *right,
    size_t count
) {
    double maximum = 0.0;
    for (size_t index = 0; index < count; index++) {
        double difference = fabs((double)left[index] - right[index]);
        if (difference > maximum) maximum = difference;
    }
    return maximum;
}

static void write_binary_reference(
    FILE *trace,
    const char *fiber,
    int boundary_index,
    int layer,
    const char *phase,
    const char *boundary,
    int local_width,
    off_t byte_offset,
    size_t value_count
) {
    if (trace == NULL) return;
    fprintf(
        trace,
        "{\"kind\":\"grammatical_cube_local_jet\",\"fiber\":"
    );
    fprint_json_string(trace, fiber);
    fprintf(
        trace,
        ",\"boundary_index\":%d,\"layer\":%d,\"phase\":",
        boundary_index,
        layer
    );
    fprint_json_string(trace, phase);
    fputs(",\"boundary\":", trace);
    fprint_json_string(trace, boundary);
    fprintf(
        trace,
        ",\"local_width\":%d,\"mobius_subsets\":[\"carrier\","
        "\"A\",\"B\",\"AB\"],\"binary_byte_offset\":%lld,"
        "\"binary_float32_count\":%zu,\"binary_shape\":[4,%d]}\n",
        local_width,
        (long long)byte_offset,
        value_count,
        local_width
    );
    fflush(trace);
}

static void retain_local_square_jet(
    LocalJetWriter *writer,
    int layer,
    const char *phase,
    const char *boundary,
    const float *const states[SQUARE_CORNER_COUNT],
    int local_width
) {
    if (writer->jet == NULL) {
        writer->boundary_index++;
        return;
    }
    size_t value_count = (size_t)SQUARE_CORNER_COUNT * local_width;
    float *raw = checked_calloc(value_count, sizeof(*raw));
    float *mobius = checked_calloc(value_count, sizeof(*mobius));
    float *reconstructed = checked_calloc(value_count, sizeof(*reconstructed));
    for (int corner = 0; corner < SQUARE_CORNER_COUNT; corner++) {
        memcpy(
            raw + (size_t)corner * local_width,
            states[corner],
            (size_t)local_width * sizeof(*raw)
        );
    }
    memcpy(mobius, raw, value_count * sizeof(*mobius));
    fast_mobius(mobius, 2, local_width);
    memcpy(reconstructed, mobius, value_count * sizeof(*reconstructed));
    inverse_fast_mobius(reconstructed, 2, local_width);
    double defect = maximum_difference(raw, reconstructed, value_count);
    if (defect > writer->maximum_mobius_inverse_defect) {
        writer->maximum_mobius_inverse_defect = defect;
    }
    off_t offset = ftello(writer->jet);
    if (offset < 0) fail("could not locate cube jet binary offset");
    if (fwrite(mobius, sizeof(*mobius), value_count, writer->jet) != value_count) {
        fail("could not write cube jet binary");
    }
    if (fflush(writer->jet) != 0) fail("could not flush cube jet binary");
    write_binary_reference(
        writer->trace,
        writer->fiber,
        writer->boundary_index,
        layer,
        phase,
        boundary,
        local_width,
        offset,
        value_count
    );
    writer->boundary_index++;
    free(reconstructed);
    free(mobius);
    free(raw);
}

static double retain_square_stage_chain(
    LocalJetWriter *writer,
    FrontierMap *maps,
    int map_count,
    int layer,
    const char *phase,
    const float *const inputs[SQUARE_CORNER_COUNT],
    const float *const expected_outputs[SQUARE_CORNER_COUNT]
) {
    const float *current[SQUARE_CORNER_COUNT];
    for (int corner = 0; corner < SQUARE_CORNER_COUNT; corner++) {
        current[corner] = inputs[corner];
    }
    bool current_owned = false;
    for (int stage = 0; stage < map_count; stage++) {
        float *next[SQUARE_CORNER_COUNT];
        const float *next_const[SQUARE_CORNER_COUNT];
        for (int corner = 0; corner < SQUARE_CORNER_COUNT; corner++) {
            next[corner] = checked_calloc(
                (size_t)maps[stage].output_width,
                sizeof(*next[corner])
            );
            maps[stage].apply(
                maps[stage].environment,
                current[corner],
                next[corner]
            );
            next_const[corner] = next[corner];
        }
        retain_local_square_jet(
            writer,
            layer,
            phase,
            maps[stage].name,
            next_const,
            maps[stage].output_width
        );
        if (current_owned) {
            for (int corner = 0; corner < SQUARE_CORNER_COUNT; corner++) {
                free((void *)current[corner]);
            }
        }
        for (int corner = 0; corner < SQUARE_CORNER_COUNT; corner++) {
            current[corner] = next[corner];
        }
        current_owned = true;
    }
    double maximum = 0.0;
    for (int corner = 0; corner < SQUARE_CORNER_COUNT; corner++) {
        double defect = difference_l2(
            current[corner],
            expected_outputs[corner],
            maps[map_count - 1].output_width
        );
        if (defect > maximum) maximum = defect;
    }
    if (current_owned) {
        for (int corner = 0; corner < SQUARE_CORNER_COUNT; corner++) {
            free((void *)current[corner]);
        }
    }
    return maximum;
}

static void retain_square_jets(
    LocalJetWriter *writer,
    GrammarTerm *term,
    ContextFrontiers captures[SQUARE_CORNER_COUNT]
) {
    for (int layer = 0; layer < term->layers; layer++) {
        const float *layer_inputs[SQUARE_CORNER_COUNT];
        const float *post_attention[SQUARE_CORNER_COUNT];
        const float *layer_outputs[SQUARE_CORNER_COUNT];
        for (int corner = 0; corner < SQUARE_CORNER_COUNT; corner++) {
            layer_inputs[corner] = layer_frontier(&captures[corner], layer);
            post_attention[corner] = post_attention_frontier(
                &captures[corner],
                layer
            );
            layer_outputs[corner] = layer_frontier(
                &captures[corner],
                layer + 1
            );
        }
        if (layer == 0) {
            retain_local_square_jet(
                writer,
                layer,
                "layer",
                "token_embedding_frontier",
                layer_inputs,
                term->frontier_width
            );
        }
        FrontierMap attention_stages[7];
        int attention_count = fill_attention_stage_maps(
            &term->runtimes[layer],
            attention_stages
        );
        double attention_defect = retain_square_stage_chain(
            writer,
            attention_stages,
            attention_count,
            layer,
            "attention",
            layer_inputs,
            post_attention
        );
        if (attention_defect > writer->maximum_chain_output_defect) {
            writer->maximum_chain_output_defect = attention_defect;
        }
        FrontierMap ffn_stages[6];
        int ffn_count = fill_ffn_stage_maps(
            &term->runtimes[layer],
            ffn_stages
        );
        double ffn_defect = retain_square_stage_chain(
            writer,
            ffn_stages,
            ffn_count,
            layer,
            "ffn",
            post_attention,
            layer_outputs
        );
        if (ffn_defect > writer->maximum_chain_output_defect) {
            writer->maximum_chain_output_defect = ffn_defect;
        }
    }
    const float *final_inputs[SQUARE_CORNER_COUNT];
    float *final_outputs[SQUARE_CORNER_COUNT];
    const float *final_const[SQUARE_CORNER_COUNT];
    for (int corner = 0; corner < SQUARE_CORNER_COUNT; corner++) {
        final_inputs[corner] = layer_frontier(&captures[corner], term->layers);
        final_outputs[corner] = checked_calloc(
            (size_t)term->frontier_width,
            sizeof(*final_outputs[corner])
        );
        term->final_rms_map.apply(
            term->final_rms_map.environment,
            final_inputs[corner],
            final_outputs[corner]
        );
        final_const[corner] = final_outputs[corner];
    }
    retain_local_square_jet(
        writer,
        term->layers,
        "root",
        "final_rms",
        final_const,
        term->frontier_width
    );
    for (int corner = 0; corner < SQUARE_CORNER_COUNT; corner++) {
        free(final_outputs[corner]);
    }
}

static void write_cube_context(
    FILE *trace,
    Tokenizer *tokenizer,
    const EncodedContext *context,
    int corner
) {
    static const char *names[CUBE_CORNER_COUNT] = {
        "x", "ax", "bx", "abx", "cx", "acx", "bcx", "abcx"
    };
    fprintf(trace, "{\"kind\":\"grammatical_cube_context\",\"corner\":");
    fprint_json_string(trace, names[corner]);
    fputs(",\"text\":", trace);
    fprint_json_string(trace, context->text);
    fputs(",\"tokens\":[", trace);
    for (int position = 0; position < context->count; position++) {
        if (position != 0) fputc(',', trace);
        int previous = position == 0 ? 0 : context->tokens[position - 1];
        int token = context->tokens[position];
        fprintf(trace, "{\"position\":%d,\"id\":%d,\"piece\":", position, token);
        fprint_json_string(trace, decode(tokenizer, previous, token));
        fputc('}', trace);
    }
    fputs("]}\n", trace);
    fflush(trace);
}

static double reference_logit_contrast_defect(
    Transformer *transformer,
    const TokenContrastObserver *observer,
    const EncodedContext *context,
    const float *expected,
    double *relative_defect
) {
    Config *config = &transformer->config;
    int dim = config->dim;
    int kv_dim = dim * config->n_kv_heads / config->n_heads;
    size_t cache_count =
        (size_t)config->n_layers * config->seq_len * kv_dim;
    memset(transformer->state.key_cache, 0, cache_count * sizeof(float));
    memset(transformer->state.value_cache, 0, cache_count * sizeof(float));
    float *logits = NULL;
    for (int position = 0; position < context->count; position++) {
        logits = forward(transformer, context->tokens[position], position);
    }
    float reference = logits[observer->reference_token];
    double square = 0.0;
    double reference_square = 0.0;
    for (int index = 0; index < observer->count; index++) {
        double actual = (double)logits[observer->tokens[index]] - reference;
        double difference = actual - expected[index];
        square += difference * difference;
        reference_square += actual * actual;
    }
    double defect = sqrt(square);
    double reference_norm = sqrt(reference_square);
    *relative_defect = reference_norm == 0.0 ? 0.0 :
        defect / reference_norm;
    return defect;
}

static void write_terminal_cube(
    FILE *trace,
    const TokenContrastObserver *observer,
    GrammarTerm *base_term,
    GrammarTerm *extended_term,
    ContextFrontiers base_captures[SQUARE_CORNER_COUNT],
    ContextFrontiers extended_captures[SQUARE_CORNER_COUNT],
    float **raw_out,
    double *inverse_defect
) {
    int width = observer->count;
    size_t value_count = (size_t)CUBE_CORNER_COUNT * width;
    float *raw = checked_calloc(value_count, sizeof(*raw));
    float *mobius = checked_calloc(value_count, sizeof(*mobius));
    float *reconstructed = checked_calloc(value_count, sizeof(*reconstructed));
    for (int cube_corner = 0; cube_corner < CUBE_CORNER_COUNT; cube_corner++) {
        bool extended = cube_corner >= CUBE_C;
        int square_corner = cube_corner & CUBE_AB;
        GrammarTerm *term = extended ? extended_term : base_term;
        ContextFrontiers *capture = extended ?
            &extended_captures[square_corner] : &base_captures[square_corner];
        float *normalized = checked_calloc(
            (size_t)term->frontier_width,
            sizeof(*normalized)
        );
        term->final_rms_map.apply(
            term->final_rms_map.environment,
            layer_frontier(capture, term->layers),
            normalized
        );
        const float *last = normalized +
            (size_t)(term->positions - 1) * term->dim;
        observe_token_contrasts(
            observer,
            last,
            raw + (size_t)cube_corner * width
        );
        free(normalized);
    }
    memcpy(mobius, raw, value_count * sizeof(*mobius));
    fast_mobius(mobius, 3, width);
    memcpy(reconstructed, mobius, value_count * sizeof(*reconstructed));
    inverse_fast_mobius(reconstructed, 3, width);
    *inverse_defect = maximum_difference(raw, reconstructed, value_count);
    fprintf(
        trace,
        "{\"kind\":\"grammatical_cube_terminal\","
        "\"observer_width\":%d,\"mobius_subsets\":["
        "\"carrier\",\"A\",\"B\",\"AB\",\"C\",\"AC\","
        "\"BC\",\"ABC\"],\"coefficients\":{",
        width
    );
    static const char *names[CUBE_CORNER_COUNT] = {
        "carrier", "A", "B", "AB", "C", "AC", "BC", "ABC"
    };
    for (int mask = 0; mask < CUBE_CORNER_COUNT; mask++) {
        if (mask != 0) fputc(',', trace);
        fprint_json_string(trace, names[mask]);
        fputc(':', trace);
        write_float_vector(trace, mobius + (size_t)mask * width, width);
    }
    fputs("}}\n", trace);
    fflush(trace);
    free(reconstructed);
    free(mobius);
    *raw_out = raw;
}

int main(int argc, char **argv) {
    CubeOptions options = parse_cube_options(argc, argv);
    Transformer transformer;
    build_transformer(&transformer, argv[1]);
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, argv[2], transformer.config.vocab_size);
    TokenContrastObserver observer = read_token_observer(
        &transformer,
        options.observer_path,
        options.reference_token
    );

    EncodedContext cube[CUBE_CORNER_COUNT];
    for (int corner = 0; corner < CUBE_CORNER_COUNT; corner++) {
        cube[corner] = encode_context(&tokenizer, argv[3 + corner]);
        if (cube[corner].count > transformer.config.seq_len) {
            fail("grammatical cube term exceeds model sequence length");
        }
    }
    if (!factorized_four_corner_square(cube) ||
        !factorized_four_corner_square(cube + CUBE_C)) {
        fail("A/B actions do not form aligned constructor squares in both C fibers");
    }
    int suffix_tokens = validate_uniform_extension(cube);
    int base_positions = cube[CUBE_X].count;
    int extended_positions = cube[CUBE_C].count;

    GrammarTerm base_term = {0};
    GrammarTerm extended_term = {0};
    build_grammar_term(&base_term, &transformer, base_positions);
    build_grammar_term(&extended_term, &transformer, extended_positions);
    ContextFrontiers base_captures[SQUARE_CORNER_COUNT];
    ContextFrontiers extended_captures[SQUARE_CORNER_COUNT];
    for (int corner = 0; corner < SQUARE_CORNER_COUNT; corner++) {
        base_captures[corner] = allocate_frontiers(
            base_term.layers,
            base_term.frontier_width
        );
        extended_captures[corner] = allocate_frontiers(
            extended_term.layers,
            extended_term.frontier_width
        );
        capture_context_frontiers(
            &transformer,
            &cube[corner],
            base_term.runtimes,
            &base_captures[corner]
        );
        capture_context_frontiers(
            &transformer,
            &cube[corner + CUBE_C],
            extended_term.runtimes,
            &extended_captures[corner]
        );
    }

    FILE *trace = fopen(options.trace_path, "wb");
    if (trace == NULL) fail("could not create grammatical cube trace");
    FILE *jet = NULL;
    if (options.retain_local_jets) {
        jet = fopen(options.jet_path, "wb");
        if (jet == NULL) fail("could not create grammatical cube jet binary");
    }
    fprintf(
        trace,
        "{\"kind\":\"grammatical_cube_meta\",\"schema_version\":1,"
        "\"semantics\":\"carrier_conditioned_action_jet_with_terminal_token_contrasts\","
        "\"base_positions\":%d,\"extended_positions\":%d,"
        "\"extension_token_count\":%d,\"layers\":%d,\"dim\":%d,"
        "\"observer\":\"logit_token_minus_logit_reference\","
        "\"observer_reference_token\":%d,\"observer_tokens\":[",
        base_positions,
        extended_positions,
        suffix_tokens,
        base_term.layers,
        base_term.dim,
        observer.reference_token
    );
    for (int index = 0; index < observer.count; index++) {
        if (index != 0) fputc(',', trace);
        fprintf(trace, "%d", observer.tokens[index]);
    }
    fprintf(trace, "] ,\"local_jets_retained\":%s,",
        options.retain_local_jets ? "true" : "false");
    fputs("\"terminal_probabilities_used\":false,"
          "\"scalar_completion_reward_used\":false,"
          "\"local_c_difference_defined\":false,"
          "\"local_c_reason\":\"C changes the position-indexed frontier type\","
          "\"local_jet_binary_dtype\":\"native_float32\","
          "\"norms_are_diagnostics_not_scores\":true}\n", trace);
    for (int corner = 0; corner < CUBE_CORNER_COUNT; corner++) {
        write_cube_context(trace, &tokenizer, &cube[corner], corner);
    }

    LocalJetWriter base_writer = {
        .trace = trace,
        .jet = jet,
        .fiber = "without_C"
    };
    LocalJetWriter extended_writer = {
        .trace = trace,
        .jet = jet,
        .fiber = "with_C"
    };
    retain_square_jets(&base_writer, &base_term, base_captures);
    retain_square_jets(&extended_writer, &extended_term, extended_captures);
    if (base_writer.boundary_index != extended_writer.boundary_index) {
        fail("C fibers produced different typed-boundary counts");
    }

    float *terminal_raw = NULL;
    double terminal_inverse_defect = 0.0;
    write_terminal_cube(
        trace,
        &observer,
        &base_term,
        &extended_term,
        base_captures,
        extended_captures,
        &terminal_raw,
        &terminal_inverse_defect
    );

    double maximum_hidden_relative_defect = 0.0;
    double maximum_logit_contrast_defect = 0.0;
    double maximum_logit_contrast_relative_defect = 0.0;
    for (int cube_corner = 0; cube_corner < CUBE_CORNER_COUNT; cube_corner++) {
        bool extended = cube_corner >= CUBE_C;
        int square_corner = cube_corner & CUBE_AB;
        GrammarTerm *term = extended ? &extended_term : &base_term;
        ContextFrontiers *capture = extended ?
            &extended_captures[square_corner] : &base_captures[square_corner];
        float *normalized = checked_calloc(
            (size_t)term->frontier_width,
            sizeof(*normalized)
        );
        term->final_rms_map.apply(
            term->final_rms_map.environment,
            layer_frontier(capture, term->layers),
            normalized
        );
        double relative = 0.0;
        (void)check_reference_hidden_frontier(
            &transformer,
            &cube[cube_corner],
            normalized,
            &relative
        );
        if (relative > maximum_hidden_relative_defect) {
            maximum_hidden_relative_defect = relative;
        }
        double logit_relative = 0.0;
        double logit_defect = reference_logit_contrast_defect(
            &transformer,
            &observer,
            &cube[cube_corner],
            terminal_raw + (size_t)cube_corner * observer.count,
            &logit_relative
        );
        if (logit_defect > maximum_logit_contrast_defect) {
            maximum_logit_contrast_defect = logit_defect;
        }
        if (logit_relative > maximum_logit_contrast_relative_defect) {
            maximum_logit_contrast_relative_defect = logit_relative;
        }
        free(normalized);
    }
    free(terminal_raw);

    fprintf(
        trace,
        "{\"kind\":\"grammatical_cube_check\",\"typed_boundaries\":%d,"
        "\"maximum_typed_chain_output_l2_defect\":%.17g,"
        "\"maximum_local_mobius_inverse_absolute_defect\":%.17g,"
        "\"terminal_mobius_inverse_absolute_defect\":%.17g,"
        "\"maximum_stock_hidden_relative_defect\":%.17g,"
        "\"maximum_stock_logit_contrast_l2_defect\":%.17g,"
        "\"maximum_stock_logit_contrast_relative_defect\":%.17g}\n",
        base_writer.boundary_index,
        fmax(
            base_writer.maximum_chain_output_defect,
            extended_writer.maximum_chain_output_defect
        ),
        fmax(
            base_writer.maximum_mobius_inverse_defect,
            extended_writer.maximum_mobius_inverse_defect
        ),
        terminal_inverse_defect,
        maximum_hidden_relative_defect,
        maximum_logit_contrast_defect,
        maximum_logit_contrast_relative_defect
    );
    fflush(trace);
    printf(
        "grammar_cube base_positions=%d extended_positions=%d boundaries=%d "
        "observer_width=%d chain_defect=%.8g hidden_relative=%.8g "
        "logit_contrast_defect=%.8g logit_contrast_relative=%.8g\n",
        base_positions,
        extended_positions,
        base_writer.boundary_index,
        observer.count,
        fmax(
            base_writer.maximum_chain_output_defect,
            extended_writer.maximum_chain_output_defect
        ),
        maximum_hidden_relative_defect,
        maximum_logit_contrast_defect,
        maximum_logit_contrast_relative_defect
    );

    if (jet != NULL && fclose(jet) != 0) {
        fail("could not close grammatical cube jet binary");
    }
    if (fclose(trace) != 0) {
        fail("could not close grammatical cube trace");
    }
    for (int corner = 0; corner < SQUARE_CORNER_COUNT; corner++) {
        free_frontiers(&extended_captures[corner]);
        free_frontiers(&base_captures[corner]);
    }
    free_grammar_term(&extended_term);
    free_grammar_term(&base_term);
    for (int corner = 0; corner < CUBE_CORNER_COUNT; corner++) {
        free_context(&cube[corner]);
    }
    free_token_observer(&observer);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return EXIT_SUCCESS;
}
