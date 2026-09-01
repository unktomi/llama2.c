/*
 * Grammatical constructor actions observed through the exact CPS transformer.
 *
 * The caller supplies five terms
 *
 *     x, a(x), b(x), b(a(x)), a(b(x)).
 *
 * At every typed transformer boundary this program retains the literal mixed
 * difference and commutator
 *
 *     ab - a - b + x,                 ab - ba,
 *
 * together with their images under the mechanically composed suffix
 * continuation.  It also compares the actual joint point ab with the affine
 * torsor completion a + b - x.  No logits, classifier, scalar reward, parse
 * tree, or per-layer accumulation occurs here.  Norms are diagnostics for the
 * retained vectors, not scores.
 */

#define CPS_FIXED_POINTS_NO_MAIN
#include "cps_fixed_points.c"

typedef enum {
    GRAMMAR_ROOT_ALL,
    GRAMMAR_ROOT_LAST
} GrammarRootScope;

typedef enum {
    ACTION_X,
    ACTION_AX,
    ACTION_BX,
    ACTION_ABX,
    ACTION_BAX,
    ACTION_CONTEXT_COUNT
} ActionRole;

typedef struct {
    const char *trace_path;
    GrammarRootScope root_scope;
} GrammarOptions;

typedef struct {
    Transformer *transformer;
    int layers;
    int positions;
    int dim;
    int frontier_width;
    LayerRuntime *runtimes;
    FrontierMap *attention_maps;
    FrontierMap *ffn_maps;
    FrontierMap *layer_maps;
    FinalRmsRuntime final_rms_runtime;
    FrontierMap final_rms_map;
    Continuation root_identity;
    Continuation *layer_suffixes;
    Continuation *post_attention_suffixes;
    PullbackEnvironment final_rms_pullback;
    PullbackEnvironment *layer_pullbacks;
    PullbackEnvironment *ffn_pullbacks;
} GrammarTerm;

typedef struct {
    int boundary_index;
    int layer;
    const char *phase;
    const char *boundary;
    int local_width;
    int root_width;
    double local_mixed_l2;
    double local_mixed_relative;
    double local_commutator_l2;
    double local_commutator_relative;
    double pullback_mixed_l2;
    double pullback_mixed_relative;
    double pullback_commutator_l2;
    double pullback_commutator_relative;
    double torsor_visible_l2;
    double torsor_visible_relative;
    float *local_mixed;
    float *local_commutator;
    float *pullback_mixed;
    float *pullback_commutator;
    float *torsor_visible;
} ActionMeasurement;

static const char *action_role_name(ActionRole role) {
    switch (role) {
        case ACTION_X: return "x";
        case ACTION_AX: return "ax";
        case ACTION_BX: return "bx";
        case ACTION_ABX: return "abx";
        case ACTION_BAX: return "bax";
        case ACTION_CONTEXT_COUNT: break;
    }
    fail("invalid grammatical action role");
    return "invalid";
}

static const char *grammar_root_scope_name(GrammarRootScope scope) {
    switch (scope) {
        case GRAMMAR_ROOT_ALL: return "all";
        case GRAMMAR_ROOT_LAST: return "last";
    }
    fail("invalid grammatical root scope");
    return "invalid";
}

static GrammarOptions parse_grammar_options(int argc, char **argv) {
    if (argc < 8) {
        fprintf(
            stderr,
            "usage: %s CHECKPOINT TOKENIZER X AX BX ABX BAX "
            "[--root all|last] [--trace PATH]\n",
            argv[0]
        );
        exit(EXIT_FAILURE);
    }
    GrammarOptions options = {.root_scope = GRAMMAR_ROOT_LAST};
    for (int index = 8; index < argc;) {
        if (strcmp(argv[index], "--root") == 0 && index + 1 < argc) {
            if (strcmp(argv[index + 1], "all") == 0) {
                options.root_scope = GRAMMAR_ROOT_ALL;
            } else if (strcmp(argv[index + 1], "last") == 0) {
                options.root_scope = GRAMMAR_ROOT_LAST;
            } else {
                fail("grammatical root scope must be all or last");
            }
            index += 2;
        } else if (strcmp(argv[index], "--trace") == 0 &&
                   index + 1 < argc) {
            options.trace_path = argv[index + 1];
            index += 2;
        } else {
            fail("unrecognized cps_grammar_actions option");
        }
    }
    return options;
}

static GrammarTerm build_grammar_term(
    Transformer *transformer,
    int positions
) {
    GrammarTerm term = {
        .transformer = transformer,
        .layers = transformer->config.n_layers,
        .positions = positions,
        .dim = transformer->config.dim,
        .frontier_width = positions * transformer->config.dim
    };
    term.runtimes = checked_calloc(
        (size_t)term.layers,
        sizeof(*term.runtimes)
    );
    term.attention_maps = checked_calloc(
        (size_t)term.layers,
        sizeof(*term.attention_maps)
    );
    term.ffn_maps = checked_calloc(
        (size_t)term.layers,
        sizeof(*term.ffn_maps)
    );
    term.layer_maps = checked_calloc(
        (size_t)term.layers,
        sizeof(*term.layer_maps)
    );
    for (int layer = 0; layer < term.layers; layer++) {
        term.runtimes[layer] = (LayerRuntime){
            .transformer = transformer,
            .layer = layer,
            .positions = positions,
            .workspace = allocate_workspace(&transformer->config, positions)
        };
        term.attention_maps[layer] = (FrontierMap){
            .name = "attention_residual",
            .input_width = term.frontier_width,
            .output_width = term.frontier_width,
            .apply = attention_map_apply,
            .environment = &term.runtimes[layer]
        };
        term.ffn_maps[layer] = (FrontierMap){
            .name = "swiglu_residual",
            .input_width = term.frontier_width,
            .output_width = term.frontier_width,
            .apply = ffn_map_apply,
            .environment = &term.runtimes[layer]
        };
        term.layer_maps[layer] = (FrontierMap){
            .name = "whole_layer",
            .input_width = term.frontier_width,
            .output_width = term.frontier_width,
            .apply = layer_map_apply,
            .environment = &term.runtimes[layer]
        };
    }
    term.final_rms_runtime = (FinalRmsRuntime){
        .transformer = transformer,
        .positions = positions
    };
    term.final_rms_map = (FrontierMap){
        .name = "final_rms",
        .input_width = term.frontier_width,
        .output_width = term.frontier_width,
        .apply = final_rms_map_apply,
        .environment = &term.final_rms_runtime
    };
    term.root_identity = (Continuation){
        .input_width = term.frontier_width,
        .result_width = term.frontier_width,
        .apply = identity_continuation_apply,
        .environment = &term.frontier_width
    };
    term.layer_suffixes = checked_calloc(
        (size_t)term.layers + 1,
        sizeof(*term.layer_suffixes)
    );
    term.post_attention_suffixes = checked_calloc(
        (size_t)term.layers,
        sizeof(*term.post_attention_suffixes)
    );
    term.layer_pullbacks = checked_calloc(
        (size_t)term.layers,
        sizeof(*term.layer_pullbacks)
    );
    term.ffn_pullbacks = checked_calloc(
        (size_t)term.layers,
        sizeof(*term.ffn_pullbacks)
    );
    term.layer_suffixes[term.layers] = make_pullback(
        &term.final_rms_pullback,
        term.final_rms_map,
        term.root_identity
    );
    for (int layer = term.layers - 1; layer >= 0; layer--) {
        term.post_attention_suffixes[layer] = make_pullback(
            &term.ffn_pullbacks[layer],
            term.ffn_maps[layer],
            term.layer_suffixes[layer + 1]
        );
        term.layer_suffixes[layer] = make_pullback(
            &term.layer_pullbacks[layer],
            term.layer_maps[layer],
            term.layer_suffixes[layer + 1]
        );
    }
    return term;
}

static void free_grammar_term(GrammarTerm *term) {
    free_pullback(&term->final_rms_pullback);
    for (int layer = 0; layer < term->layers; layer++) {
        free_pullback(&term->layer_pullbacks[layer]);
        free_pullback(&term->ffn_pullbacks[layer]);
        free_workspace(&term->runtimes[layer].workspace);
    }
    free(term->ffn_pullbacks);
    free(term->layer_pullbacks);
    free(term->post_attention_suffixes);
    free(term->layer_suffixes);
    free(term->layer_maps);
    free(term->ffn_maps);
    free(term->attention_maps);
    free(term->runtimes);
    memset(term, 0, sizeof(*term));
}

static int fill_attention_stage_maps(
    LayerRuntime *runtime,
    FrontierMap maps[7]
) {
    maps[0] = (FrontierMap){
        .name = "attention_rms_pair",
        .input_width = frontier_width_for(runtime),
        .output_width = attention_norm_state_width(runtime),
        .apply = attention_norm_stage_apply,
        .environment = runtime
    };
    maps[1] = (FrontierMap){
        .name = "qkv_projection",
        .input_width = attention_norm_state_width(runtime),
        .output_width = qkv_state_width(runtime),
        .apply = qkv_stage_apply,
        .environment = runtime
    };
    maps[2] = (FrontierMap){
        .name = "qk_contraction",
        .input_width = qkv_state_width(runtime),
        .output_width = score_state_width(runtime),
        .apply = qk_stage_apply,
        .environment = runtime
    };
    maps[3] = (FrontierMap){
        .name = "softmax",
        .input_width = score_state_width(runtime),
        .output_width = score_state_width(runtime),
        .apply = softmax_stage_apply,
        .environment = runtime
    };
    maps[4] = (FrontierMap){
        .name = "attention_value_contraction",
        .input_width = score_state_width(runtime),
        .output_width = residual_pair_width(runtime),
        .apply = value_stage_apply,
        .environment = runtime
    };
    maps[5] = (FrontierMap){
        .name = "attention_output_projection",
        .input_width = residual_pair_width(runtime),
        .output_width = residual_pair_width(runtime),
        .apply = wo_stage_apply,
        .environment = runtime
    };
    maps[6] = (FrontierMap){
        .name = "attention_residual_addition",
        .input_width = residual_pair_width(runtime),
        .output_width = frontier_width_for(runtime),
        .apply = attention_residual_stage_apply,
        .environment = runtime
    };
    return 7;
}

static int fill_ffn_stage_maps(
    LayerRuntime *runtime,
    FrontierMap maps[6]
) {
    maps[0] = (FrontierMap){
        .name = "ffn_rms_pair",
        .input_width = frontier_width_for(runtime),
        .output_width = ffn_norm_state_width(runtime),
        .apply = ffn_norm_stage_apply,
        .environment = runtime
    };
    maps[1] = (FrontierMap){
        .name = "w1_w3_projection",
        .input_width = ffn_norm_state_width(runtime),
        .output_width = ffn_branch_state_width(runtime),
        .apply = ffn_projection_stage_apply,
        .environment = runtime
    };
    maps[2] = (FrontierMap){
        .name = "silu",
        .input_width = ffn_branch_state_width(runtime),
        .output_width = ffn_branch_state_width(runtime),
        .apply = silu_stage_apply,
        .environment = runtime
    };
    maps[3] = (FrontierMap){
        .name = "swiglu_product",
        .input_width = ffn_branch_state_width(runtime),
        .output_width = ffn_product_state_width(runtime),
        .apply = swiglu_product_stage_apply,
        .environment = runtime
    };
    maps[4] = (FrontierMap){
        .name = "ffn_output_projection",
        .input_width = ffn_product_state_width(runtime),
        .output_width = residual_pair_width(runtime),
        .apply = ffn_output_stage_apply,
        .environment = runtime
    };
    maps[5] = (FrontierMap){
        .name = "ffn_residual_addition",
        .input_width = residual_pair_width(runtime),
        .output_width = frontier_width_for(runtime),
        .apply = ffn_residual_stage_apply,
        .environment = runtime
    };
    return 6;
}

static int action_root_width(
    const GrammarTerm *term,
    GrammarRootScope scope
) {
    return scope == GRAMMAR_ROOT_ALL ? term->frontier_width : term->dim;
}

static void apply_root_observation(
    const GrammarTerm *term,
    GrammarRootScope scope,
    Continuation suffix,
    const float *input,
    float *whole_root,
    float *observation
) {
    if (suffix.result_width != term->frontier_width) {
        fail("grammatical suffix does not reach the complete root frontier");
    }
    suffix.apply(suffix.environment, input, whole_root);
    if (scope == GRAMMAR_ROOT_ALL) {
        memcpy(
            observation,
            whole_root,
            (size_t)term->frontier_width * sizeof(*observation)
        );
    } else {
        memcpy(
            observation,
            whole_root + (size_t)(term->positions - 1) * term->dim,
            (size_t)term->dim * sizeof(*observation)
        );
    }
}

static void affine_mixed(
    const float *x,
    const float *a,
    const float *b,
    const float *ab,
    float *mixed,
    int width
) {
    for (int index = 0; index < width; index++) {
        mixed[index] = ab[index] - a[index] - b[index] + x[index];
    }
}

static void subtract_vectors(
    const float *left,
    const float *right,
    float *result,
    int width
) {
    for (int index = 0; index < width; index++) {
        result[index] = left[index] - right[index];
    }
}

static void torsor_completion(
    const float *x,
    const float *a,
    const float *b,
    float *result,
    int width
) {
    for (int index = 0; index < width; index++) {
        result[index] = a[index] + b[index] - x[index];
    }
}

static ActionMeasurement measure_action_boundary(
    const GrammarTerm *term,
    GrammarRootScope scope,
    int boundary_index,
    int layer,
    const char *phase,
    const char *boundary,
    Continuation suffix,
    const float *const states[ACTION_CONTEXT_COUNT],
    int local_width
) {
    int root_width = action_root_width(term, scope);
    ActionMeasurement measurement = {
        .boundary_index = boundary_index,
        .layer = layer,
        .phase = phase,
        .boundary = boundary,
        .local_width = local_width,
        .root_width = root_width,
        .local_mixed = checked_calloc((size_t)local_width, sizeof(float)),
        .local_commutator = checked_calloc((size_t)local_width, sizeof(float)),
        .pullback_mixed = checked_calloc((size_t)root_width, sizeof(float)),
        .pullback_commutator = checked_calloc((size_t)root_width, sizeof(float)),
        .torsor_visible = checked_calloc((size_t)root_width, sizeof(float))
    };
    float *independent = checked_calloc((size_t)local_width, sizeof(float));
    affine_mixed(
        states[ACTION_X],
        states[ACTION_AX],
        states[ACTION_BX],
        states[ACTION_ABX],
        measurement.local_mixed,
        local_width
    );
    subtract_vectors(
        states[ACTION_ABX],
        states[ACTION_BAX],
        measurement.local_commutator,
        local_width
    );
    torsor_completion(
        states[ACTION_X],
        states[ACTION_AX],
        states[ACTION_BX],
        independent,
        local_width
    );

    float *whole_root = checked_calloc(
        (size_t)term->frontier_width,
        sizeof(float)
    );
    float *roots[ACTION_CONTEXT_COUNT];
    for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
        roots[role] = checked_calloc((size_t)root_width, sizeof(float));
        apply_root_observation(
            term,
            scope,
            suffix,
            states[role],
            whole_root,
            roots[role]
        );
    }
    float *independent_root = checked_calloc(
        (size_t)root_width,
        sizeof(float)
    );
    apply_root_observation(
        term,
        scope,
        suffix,
        independent,
        whole_root,
        independent_root
    );
    affine_mixed(
        roots[ACTION_X],
        roots[ACTION_AX],
        roots[ACTION_BX],
        roots[ACTION_ABX],
        measurement.pullback_mixed,
        root_width
    );
    subtract_vectors(
        roots[ACTION_ABX],
        roots[ACTION_BAX],
        measurement.pullback_commutator,
        root_width
    );
    subtract_vectors(
        roots[ACTION_ABX],
        independent_root,
        measurement.torsor_visible,
        root_width
    );

    double local_scale = vector_l2(states[ACTION_ABX], local_width);
    double root_scale = vector_l2(roots[ACTION_ABX], root_width);
    measurement.local_mixed_l2 = vector_l2(
        measurement.local_mixed,
        local_width
    );
    measurement.local_commutator_l2 = vector_l2(
        measurement.local_commutator,
        local_width
    );
    measurement.pullback_mixed_l2 = vector_l2(
        measurement.pullback_mixed,
        root_width
    );
    measurement.pullback_commutator_l2 = vector_l2(
        measurement.pullback_commutator,
        root_width
    );
    measurement.torsor_visible_l2 = vector_l2(
        measurement.torsor_visible,
        root_width
    );
    if (local_scale != 0.0) {
        measurement.local_mixed_relative =
            measurement.local_mixed_l2 / local_scale;
        measurement.local_commutator_relative =
            measurement.local_commutator_l2 / local_scale;
    }
    if (root_scale != 0.0) {
        measurement.pullback_mixed_relative =
            measurement.pullback_mixed_l2 / root_scale;
        measurement.pullback_commutator_relative =
            measurement.pullback_commutator_l2 / root_scale;
        measurement.torsor_visible_relative =
            measurement.torsor_visible_l2 / root_scale;
    }

    free(independent_root);
    for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
        free(roots[role]);
    }
    free(whole_root);
    free(independent);
    return measurement;
}

static void free_action_measurement(ActionMeasurement *measurement) {
    free(measurement->torsor_visible);
    free(measurement->pullback_commutator);
    free(measurement->pullback_mixed);
    free(measurement->local_commutator);
    free(measurement->local_mixed);
    memset(measurement, 0, sizeof(*measurement));
}

static void write_float_vector(FILE *trace, const float *values, int width) {
    fputc('[', trace);
    for (int index = 0; index < width; index++) {
        if (index != 0) fputc(',', trace);
        fprintf(trace, "%.9g", values[index]);
    }
    fputc(']', trace);
}

static void write_action_measurement(
    FILE *trace,
    const ActionMeasurement *measurement
) {
    if (trace == NULL) return;
    fprintf(
        trace,
        "{\"kind\":\"grammatical_action_boundary\","
        "\"boundary_index\":%d,\"layer\":%d,\"phase\":",
        measurement->boundary_index,
        measurement->layer
    );
    fprint_json_string(trace, measurement->phase);
    fputs(",\"boundary\":", trace);
    fprint_json_string(trace, measurement->boundary);
    fprintf(
        trace,
        ",\"local_width\":%d,\"root_width\":%d,"
        "\"local_mixed_l2\":%.17g,"
        "\"local_mixed_relative\":%.17g,"
        "\"local_commutator_l2\":%.17g,"
        "\"local_commutator_relative\":%.17g,"
        "\"pullback_mixed_l2\":%.17g,"
        "\"pullback_mixed_relative\":%.17g,"
        "\"pullback_commutator_l2\":%.17g,"
        "\"pullback_commutator_relative\":%.17g,"
        "\"torsor_visible_l2\":%.17g,"
        "\"torsor_visible_relative\":%.17g,"
        "\"local_mixed\":",
        measurement->local_width,
        measurement->root_width,
        measurement->local_mixed_l2,
        measurement->local_mixed_relative,
        measurement->local_commutator_l2,
        measurement->local_commutator_relative,
        measurement->pullback_mixed_l2,
        measurement->pullback_mixed_relative,
        measurement->pullback_commutator_l2,
        measurement->pullback_commutator_relative,
        measurement->torsor_visible_l2,
        measurement->torsor_visible_relative
    );
    write_float_vector(
        trace,
        measurement->local_mixed,
        measurement->local_width
    );
    fputs(",\"local_commutator\":", trace);
    write_float_vector(
        trace,
        measurement->local_commutator,
        measurement->local_width
    );
    fputs(",\"pullback_mixed\":", trace);
    write_float_vector(
        trace,
        measurement->pullback_mixed,
        measurement->root_width
    );
    fputs(",\"pullback_commutator\":", trace);
    write_float_vector(
        trace,
        measurement->pullback_commutator,
        measurement->root_width
    );
    fputs(",\"torsor_visible\":", trace);
    write_float_vector(
        trace,
        measurement->torsor_visible,
        measurement->root_width
    );
    fputs("}\n", trace);
    fflush(trace);
}

static void report_action_boundary(
    FILE *trace,
    const GrammarTerm *term,
    GrammarRootScope scope,
    int *boundary_index,
    int layer,
    const char *phase,
    const char *boundary,
    Continuation suffix,
    const float *const states[ACTION_CONTEXT_COUNT],
    int local_width
) {
    ActionMeasurement measurement = measure_action_boundary(
        term,
        scope,
        *boundary_index,
        layer,
        phase,
        boundary,
        suffix,
        states,
        local_width
    );
    printf(
        "boundary=%d layer=%d %-9s %-28s "
        "local_mixed=%.8g local_commutator=%.8g "
        "torsor_visible=%.8g pullback_mixed=%.8g\n",
        *boundary_index,
        layer,
        phase,
        boundary,
        measurement.local_mixed_l2,
        measurement.local_commutator_l2,
        measurement.torsor_visible_l2,
        measurement.pullback_mixed_l2
    );
    fflush(stdout);
    write_action_measurement(trace, &measurement);
    free_action_measurement(&measurement);
    (*boundary_index)++;
}

static void write_action_context(
    FILE *trace,
    Tokenizer *tokenizer,
    const EncodedContext *context,
    ActionRole role
) {
    if (trace == NULL) return;
    fprintf(trace, "{\"kind\":\"grammatical_action_context\",\"role\":");
    fprint_json_string(trace, action_role_name(role));
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

static bool token_sequences_equal(
    const EncodedContext *left,
    const EncodedContext *right
) {
    return left->count == right->count &&
        memcmp(
            left->tokens,
            right->tokens,
            (size_t)left->count * sizeof(*left->tokens)
        ) == 0;
}

static bool factorized_constructor_square(
    const EncodedContext contexts[ACTION_CONTEXT_COUNT]
) {
    bool saw_a = false;
    bool saw_b = false;
    for (int position = 0; position < contexts[ACTION_X].count; position++) {
        int x = contexts[ACTION_X].tokens[position];
        int a = contexts[ACTION_AX].tokens[position];
        int b = contexts[ACTION_BX].tokens[position];
        int ab = contexts[ACTION_ABX].tokens[position];
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

static double compare_stage_output(
    const float *const actual[ACTION_CONTEXT_COUNT],
    const float *const expected[ACTION_CONTEXT_COUNT],
    int width
) {
    double maximum = 0.0;
    for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
        double defect = difference_l2(actual[role], expected[role], width);
        if (defect > maximum) maximum = defect;
    }
    return maximum;
}

static double run_action_stage_chain(
    FILE *trace,
    const GrammarTerm *term,
    GrammarRootScope scope,
    int *boundary_index,
    int layer,
    const char *phase,
    FrontierMap *maps,
    int map_count,
    Continuation ending_suffix,
    const float *const inputs[ACTION_CONTEXT_COUNT],
    const float *const expected_outputs[ACTION_CONTEXT_COUNT]
) {
    PullbackEnvironment *pullbacks = checked_calloc(
        (size_t)map_count,
        sizeof(*pullbacks)
    );
    Continuation *suffixes = checked_calloc(
        (size_t)map_count + 1,
        sizeof(*suffixes)
    );
    suffixes[map_count] = ending_suffix;
    for (int stage = map_count - 1; stage >= 0; stage--) {
        suffixes[stage] = make_pullback(
            &pullbacks[stage],
            maps[stage],
            suffixes[stage + 1]
        );
    }

    const float *current[ACTION_CONTEXT_COUNT];
    for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
        current[role] = inputs[role];
    }
    bool current_owned = false;
    for (int stage = 0; stage < map_count; stage++) {
        float *next[ACTION_CONTEXT_COUNT];
        for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
            next[role] = checked_calloc(
                (size_t)maps[stage].output_width,
                sizeof(float)
            );
            maps[stage].apply(
                maps[stage].environment,
                current[role],
                next[role]
            );
        }
        const float *next_const[ACTION_CONTEXT_COUNT];
        for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
            next_const[role] = next[role];
        }
        report_action_boundary(
            trace,
            term,
            scope,
            boundary_index,
            layer,
            phase,
            maps[stage].name,
            suffixes[stage + 1],
            next_const,
            maps[stage].output_width
        );
        if (current_owned) {
            for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
                free((void *)current[role]);
            }
        }
        for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
            current[role] = next[role];
        }
        current_owned = true;
    }
    double output_defect = compare_stage_output(
        current,
        expected_outputs,
        maps[map_count - 1].output_width
    );
    if (current_owned) {
        for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
            free((void *)current[role]);
        }
    }
    for (int stage = 0; stage < map_count; stage++) {
        free_pullback(&pullbacks[stage]);
    }
    free(suffixes);
    free(pullbacks);
    return output_defect;
}

int main(int argc, char **argv) {
    GrammarOptions options = parse_grammar_options(argc, argv);
    Transformer transformer;
    build_transformer(&transformer, argv[1]);
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, argv[2], transformer.config.vocab_size);

    EncodedContext contexts[ACTION_CONTEXT_COUNT];
    for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
        contexts[role] = encode_context(&tokenizer, argv[3 + role]);
    }
    int positions = contexts[ACTION_X].count;
    for (int role = 1; role < ACTION_CONTEXT_COUNT; role++) {
        if (contexts[role].count != positions) {
            fail(
                "grammatical action terms must tokenize to equal lengths; "
                "choose aligned constructors rather than padding"
            );
        }
    }
    if (positions > transformer.config.seq_len) {
        fail("grammatical action term exceeds model sequence length");
    }
    bool factorized_square = factorized_constructor_square(contexts);
    bool constructors_commute = token_sequences_equal(
        &contexts[ACTION_ABX],
        &contexts[ACTION_BAX]
    );
    GrammarTerm term = build_grammar_term(&transformer, positions);

    FILE *trace = NULL;
    if (options.trace_path != NULL) {
        trace = fopen(options.trace_path, "wb");
        if (trace == NULL) fail("could not create grammatical action trace");
        fprintf(
            trace,
            "{\"kind\":\"grammatical_action_meta\",\"schema_version\":1,"
            "\"semantics\":\"exact_constructor_pullbacks\","
            "\"mixed_operator\":\"(U_a-I)(U_b-I)k\","
            "\"commutator_operator\":\"(U_aU_b-U_bU_a)k\","
            "\"constructor_order\":{\"abx\":\"b_after_a_on_x\","
            "\"bax\":\"a_after_b_on_x\"},"
            "\"root_observer\":\"post_final_rms_hidden\","
            "\"root_scope\":\"%s\",\"positions\":%d,\"dim\":%d,"
            "\"layers\":%d,\"factorized_token_square\":%s,"
            "\"constructors_commute_on_x\":%s,"
            "\"norms_are_diagnostics_not_scores\":true}\n",
            grammar_root_scope_name(options.root_scope),
            positions,
            term.dim,
            term.layers,
            factorized_square ? "true" : "false",
            constructors_commute ? "true" : "false"
        );
        fflush(trace);
        for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
            write_action_context(
                trace,
                &tokenizer,
                &contexts[role],
                (ActionRole)role
            );
        }
    }
    printf(
        "grammar_actions positions=%d layers=%d dim=%d root=%s "
        "factorized_square=%s constructors_commute=%s\n",
        positions,
        term.layers,
        term.dim,
        grammar_root_scope_name(options.root_scope),
        factorized_square ? "true" : "false",
        constructors_commute ? "true" : "false"
    );

    ContextFrontiers captures[ACTION_CONTEXT_COUNT];
    for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
        captures[role] = allocate_frontiers(term.layers, term.frontier_width);
        capture_context_frontiers(
            &transformer,
            &contexts[role],
            term.runtimes,
            &captures[role]
        );
    }

    int boundary_index = 0;
    double maximum_stage_output_defect = 0.0;
    for (int layer = 0; layer < term.layers; layer++) {
        const float *layer_inputs[ACTION_CONTEXT_COUNT];
        const float *post_attention[ACTION_CONTEXT_COUNT];
        const float *layer_outputs[ACTION_CONTEXT_COUNT];
        for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
            layer_inputs[role] = layer_frontier(&captures[role], layer);
            post_attention[role] = post_attention_frontier(
                &captures[role],
                layer
            );
            layer_outputs[role] = layer_frontier(
                &captures[role],
                layer + 1
            );
        }
        if (layer == 0) {
            report_action_boundary(
                trace,
                &term,
                options.root_scope,
                &boundary_index,
                layer,
                "layer",
                "token_embedding_frontier",
                term.layer_suffixes[layer],
                layer_inputs,
                term.frontier_width
            );
        }

        FrontierMap attention_stages[7];
        int attention_count = fill_attention_stage_maps(
            &term.runtimes[layer],
            attention_stages
        );
        double attention_defect = run_action_stage_chain(
            trace,
            &term,
            options.root_scope,
            &boundary_index,
            layer,
            "attention",
            attention_stages,
            attention_count,
            term.post_attention_suffixes[layer],
            layer_inputs,
            post_attention
        );
        if (attention_defect > maximum_stage_output_defect) {
            maximum_stage_output_defect = attention_defect;
        }

        FrontierMap ffn_stages[6];
        int ffn_count = fill_ffn_stage_maps(
            &term.runtimes[layer],
            ffn_stages
        );
        double ffn_defect = run_action_stage_chain(
            trace,
            &term,
            options.root_scope,
            &boundary_index,
            layer,
            "ffn",
            ffn_stages,
            ffn_count,
            term.layer_suffixes[layer + 1],
            post_attention,
            layer_outputs
        );
        if (ffn_defect > maximum_stage_output_defect) {
            maximum_stage_output_defect = ffn_defect;
        }
    }

    const float *final_inputs[ACTION_CONTEXT_COUNT];
    float *final_outputs[ACTION_CONTEXT_COUNT];
    const float *final_output_const[ACTION_CONTEXT_COUNT];
    for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
        final_inputs[role] = layer_frontier(
            &captures[role],
            term.layers
        );
        final_outputs[role] = checked_calloc(
            (size_t)term.frontier_width,
            sizeof(float)
        );
        term.final_rms_map.apply(
            term.final_rms_map.environment,
            final_inputs[role],
            final_outputs[role]
        );
        final_output_const[role] = final_outputs[role];
    }
    report_action_boundary(
        trace,
        &term,
        options.root_scope,
        &boundary_index,
        term.layers,
        "root",
        "final_rms",
        term.root_identity,
        final_output_const,
        term.frontier_width
    );

    printf(
        "boundaries=%d maximum_typed_stage_output_l2_defect=%.8g\n",
        boundary_index,
        maximum_stage_output_defect
    );
    if (trace != NULL) {
        fprintf(
            trace,
            "{\"kind\":\"grammatical_action_check\","
            "\"boundaries\":%d,"
            "\"maximum_typed_stage_output_l2_defect\":%.17g}\n",
            boundary_index,
            maximum_stage_output_defect
        );
        if (fclose(trace) != 0) {
            fail("could not close grammatical action trace");
        }
    }

    for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
        free(final_outputs[role]);
        free_frontiers(&captures[role]);
        free_context(&contexts[role]);
    }
    free_grammar_term(&term);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return EXIT_SUCCESS;
}
