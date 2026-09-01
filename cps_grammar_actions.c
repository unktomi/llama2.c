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
 * torsor completion a + b - x, retains the vector change in that comparison
 * across every adjacent boundary, and reconstructs each QK change from its
 * two separately retained bilinear cross terms before removing their sum
 * causally.  No logits, classifier, scalar reward, parse tree, or per-layer
 * scalar accumulation occurs here.  Norms are diagnostics for the retained
 * vectors, not scores.
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
    int pullback_depth;
    bool behavior_only;
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

typedef struct {
    int root_width;
    bool has_previous;
    int previous_boundary_index;
    int previous_layer;
    const char *previous_phase;
    const char *previous_boundary;
    int last_from_boundary_index;
    int last_to_boundary_index;
    float *first_torsor_visible;
    float *previous_torsor_visible;
    float *last_delta_tau;
    double *telescoping_sum;
    int transition_count;
} ActionTraceState;

typedef struct {
    int layer;
    int qkv_boundary_index;
    int qk_boundary_index;
    int score_width;
    int root_width;
    double copied_prefix_l2;
    double measured_score_defect_l2;
    double directed_a_query_b_key_l2;
    double directed_b_query_a_key_l2;
    double analytic_cross_terms_l2;
    double score_reconstruction_l2_defect;
    double score_reconstruction_relative_defect;
    double score_reconstruction_max_abs_defect;
    double exact_delta_tau_l2;
    double transition_delta_tau_l2;
    double transition_identity_l2_defect;
    double transition_identity_relative_defect;
    double cross_removed_root_l2;
    double cross_removed_root_relative;
    float *measured_score_defect;
    float *directed_a_query_b_key;
    float *directed_b_query_a_key;
    float *analytic_cross_terms;
    float *score_reconstruction_defect;
    float *exact_delta_tau;
    float *transition_delta_tau;
    float *cross_removed_root;
} QkCausalMeasurement;

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

static int parse_positive_integer(const char *text, const char *name) {
    errno = 0;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value <= 0 ||
        value > INT_MAX) {
        fprintf(stderr, "invalid %s: %s\n", name, text);
        exit(EXIT_FAILURE);
    }
    return (int)value;
}

static GrammarOptions parse_grammar_options(int argc, char **argv) {
    if (argc < 8) {
        fprintf(
            stderr,
            "usage: %s CHECKPOINT TOKENIZER X AX BX ABX BAX "
            "[--root all|last] [--pullback-depth N] [--behavior-only] "
            "[--trace PATH]\n",
            argv[0]
        );
        exit(EXIT_FAILURE);
    }
    GrammarOptions options = {
        .root_scope = GRAMMAR_ROOT_LAST,
        .pullback_depth = 1
    };
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
        } else if (strcmp(argv[index], "--pullback-depth") == 0 &&
                   index + 1 < argc) {
            options.pullback_depth = parse_positive_integer(
                argv[index + 1],
                "pullback depth"
            );
            index += 2;
        } else if (strcmp(argv[index], "--trace") == 0 &&
                   index + 1 < argc) {
            options.trace_path = argv[index + 1];
            index += 2;
        } else if (strcmp(argv[index], "--behavior-only") == 0) {
            options.behavior_only = true;
            index++;
        } else {
            fail("unrecognized cps_grammar_actions option");
        }
    }
    return options;
}

/*
 * Construct directly in caller-owned storage.  The final RMS map, identity
 * continuation, and terminal pullback point into this GrammarTerm, so a
 * self-referential value returned by copy would have dangling environments.
 */
static void build_grammar_term(
    GrammarTerm *term,
    Transformer *transformer,
    int positions
) {
    *term = (GrammarTerm){
        .transformer = transformer,
        .layers = transformer->config.n_layers,
        .positions = positions,
        .dim = transformer->config.dim,
        .frontier_width = positions * transformer->config.dim
    };
    term->runtimes = checked_calloc(
        (size_t)term->layers,
        sizeof(*term->runtimes)
    );
    term->attention_maps = checked_calloc(
        (size_t)term->layers,
        sizeof(*term->attention_maps)
    );
    term->ffn_maps = checked_calloc(
        (size_t)term->layers,
        sizeof(*term->ffn_maps)
    );
    term->layer_maps = checked_calloc(
        (size_t)term->layers,
        sizeof(*term->layer_maps)
    );
    for (int layer = 0; layer < term->layers; layer++) {
        term->runtimes[layer] = (LayerRuntime){
            .transformer = transformer,
            .layer = layer,
            .positions = positions,
            .workspace = allocate_workspace(&transformer->config, positions)
        };
        term->attention_maps[layer] = (FrontierMap){
            .name = "attention_residual",
            .input_width = term->frontier_width,
            .output_width = term->frontier_width,
            .apply = attention_map_apply,
            .environment = &term->runtimes[layer]
        };
        term->ffn_maps[layer] = (FrontierMap){
            .name = "swiglu_residual",
            .input_width = term->frontier_width,
            .output_width = term->frontier_width,
            .apply = ffn_map_apply,
            .environment = &term->runtimes[layer]
        };
        term->layer_maps[layer] = (FrontierMap){
            .name = "whole_layer",
            .input_width = term->frontier_width,
            .output_width = term->frontier_width,
            .apply = layer_map_apply,
            .environment = &term->runtimes[layer]
        };
    }
    term->final_rms_runtime = (FinalRmsRuntime){
        .transformer = transformer,
        .positions = positions
    };
    term->final_rms_map = (FrontierMap){
        .name = "final_rms",
        .input_width = term->frontier_width,
        .output_width = term->frontier_width,
        .apply = final_rms_map_apply,
        .environment = &term->final_rms_runtime
    };
    term->root_identity = (Continuation){
        .input_width = term->frontier_width,
        .result_width = term->frontier_width,
        .apply = identity_continuation_apply,
        .environment = &term->frontier_width
    };
    term->layer_suffixes = checked_calloc(
        (size_t)term->layers + 1,
        sizeof(*term->layer_suffixes)
    );
    term->post_attention_suffixes = checked_calloc(
        (size_t)term->layers,
        sizeof(*term->post_attention_suffixes)
    );
    term->layer_pullbacks = checked_calloc(
        (size_t)term->layers,
        sizeof(*term->layer_pullbacks)
    );
    term->ffn_pullbacks = checked_calloc(
        (size_t)term->layers,
        sizeof(*term->ffn_pullbacks)
    );
    term->layer_suffixes[term->layers] = make_pullback(
        &term->final_rms_pullback,
        term->final_rms_map,
        term->root_identity
    );
    for (int layer = term->layers - 1; layer >= 0; layer--) {
        term->post_attention_suffixes[layer] = make_pullback(
            &term->ffn_pullbacks[layer],
            term->ffn_maps[layer],
            term->layer_suffixes[layer + 1]
        );
        term->layer_suffixes[layer] = make_pullback(
            &term->layer_pullbacks[layer],
            term->layer_maps[layer],
            term->layer_suffixes[layer + 1]
        );
    }

    if (term->final_rms_map.environment != &term->final_rms_runtime ||
        term->root_identity.environment != &term->frontier_width ||
        term->layer_suffixes[term->layers].environment !=
            &term->final_rms_pullback ||
        term->final_rms_pullback.map.environment !=
            &term->final_rms_runtime ||
        term->final_rms_pullback.next.environment !=
            &term->frontier_width) {
        fail("grammatical term contains an unstable self-reference");
    }
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

static double vector_max_abs(const float *values, int width);

static void evaluate_mixed_continuation(
    const GrammarTerm *term,
    GrammarRootScope scope,
    Continuation continuation,
    const float *const states[ACTION_CONTEXT_COUNT],
    float *mixed
) {
    int root_width = action_root_width(term, scope);
    float *whole_root = checked_calloc(
        (size_t)term->frontier_width,
        sizeof(*whole_root)
    );
    float *roots[4];
    for (int role = ACTION_X; role <= ACTION_ABX; role++) {
        roots[role] = checked_calloc((size_t)root_width, sizeof(float));
        apply_root_observation(
            term,
            scope,
            continuation,
            states[role],
            whole_root,
            roots[role]
        );
    }
    affine_mixed(
        roots[ACTION_X],
        roots[ACTION_AX],
        roots[ACTION_BX],
        roots[ACTION_ABX],
        mixed,
        root_width
    );
    for (int role = ACTION_X; role <= ACTION_ABX; role++) {
        free(roots[role]);
    }
    free(whole_root);
}

/*
 * Retain the grammatical continuation orbit k, U_F k, ..., U_F^p k.
 * Each generation evaluates D_ab k after another real application of F.
 * `composed` independently evaluates the first pullback D_ab (k . F) by
 * constructing make_pullback(F,k); this checks the recursive orbit at depth 1.
 */
static double report_block_pullback(
    FILE *trace,
    const GrammarTerm *term,
    GrammarRootScope scope,
    int layer,
    const char *block,
    FrontierMap map,
    Continuation suffix,
    const float *const inputs[ACTION_CONTEXT_COUNT],
    const float *const mapped[ACTION_CONTEXT_COUNT],
    int pullback_depth
) {
    if (map.input_width != map.output_width ||
        map.output_width != suffix.input_width) {
        fail("grammatical block pullback is not endomorphic");
    }
    int root_width = action_root_width(term, scope);
    if ((size_t)pullback_depth + 1U > SIZE_MAX / (size_t)root_width) {
        fail("grammatical pullback generation size overflow");
    }
    size_t generation_values =
        ((size_t)pullback_depth + 1U) * (size_t)root_width;
    float *generations = checked_calloc(
        generation_values,
        sizeof(*generations)
    );
    float *composed = checked_calloc((size_t)root_width, sizeof(float));
    float *composition_defect = checked_calloc(
        (size_t)root_width,
        sizeof(float)
    );
    const float *current[ACTION_CONTEXT_COUNT];
    for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
        current[role] = inputs[role];
    }
    bool current_owned = false;
    for (int generation = 0; generation <= pullback_depth; generation++) {
        evaluate_mixed_continuation(
            term,
            scope,
            suffix,
            current,
            generations + (size_t)generation * root_width
        );
        if (generation == pullback_depth) break;
        const float *next[ACTION_CONTEXT_COUNT];
        bool next_owned = generation != 0;
        if (generation == 0) {
            for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
                next[role] = mapped[role];
            }
        } else {
            for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
                float *mapped_state = checked_calloc(
                    (size_t)map.output_width,
                    sizeof(float)
                );
                map.apply(map.environment, current[role], mapped_state);
                next[role] = mapped_state;
            }
        }
        if (current_owned) {
            for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
                free((void *)current[role]);
            }
        }
        for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
            current[role] = next[role];
        }
        current_owned = next_owned;
    }
    if (current_owned) {
        for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
            free((void *)current[role]);
        }
    }
    const float *unpulled = generations;
    const float *pulled = generations + root_width;
    PullbackEnvironment pullback_environment = {0};
    Continuation pullback = make_pullback(
        &pullback_environment,
        map,
        suffix
    );
    evaluate_mixed_continuation(
        term,
        scope,
        pullback,
        inputs,
        composed
    );
    subtract_vectors(
        composed,
        pulled,
        composition_defect,
        root_width
    );
    double defect_l2 = vector_l2(composition_defect, root_width);
    double pulled_l2 = vector_l2(pulled, root_width);
    double relative_defect = pulled_l2 == 0.0 ? defect_l2 :
        defect_l2 / pulled_l2;
    double maximum_absolute_defect = vector_max_abs(
        composition_defect,
        root_width
    );
    printf(
        "  block_pullback layer=%d block=%s depth=%d unpulled=%.8g "
        "pulled=%.8g final=%.8g composition=%.8g relative=%.8g\n",
        layer,
        block,
        pullback_depth,
        vector_l2(unpulled, root_width),
        pulled_l2,
        vector_l2(
            generations + (size_t)pullback_depth * root_width,
            root_width
        ),
        defect_l2,
        relative_defect
    );
    fflush(stdout);
    if (trace != NULL) {
        fprintf(
            trace,
            "{\"kind\":\"grammatical_block_pullback\","
            "\"layer\":%d,\"block\":",
            layer
        );
        fprint_json_string(trace, block);
        fputs(",\"map\":", trace);
        fprint_json_string(trace, map.name);
        fprintf(
            trace,
            ",\"operator\":\"U_F(k)=k_after_F\","
            "\"action\":\"(U_a-I)(U_b-I)\","
            "\"root_width\":%d,\"recorded_pullback_depth\":%d,"
            "\"unpulled_mixed_l2\":%.17g,"
            "\"pulled_mixed_l2\":%.17g,"
            "\"composition_l2_defect\":%.17g,"
            "\"composition_relative_defect\":%.17g,"
            "\"composition_maximum_absolute_defect\":%.17g,"
            "\"mixed_generation_l2\":[",
            root_width,
            pullback_depth,
            vector_l2(unpulled, root_width),
            pulled_l2,
            defect_l2,
            relative_defect,
            maximum_absolute_defect
        );
        for (int generation = 0; generation <= pullback_depth; generation++) {
            if (generation != 0) fputc(',', trace);
            fprintf(
                trace,
                "%.17g",
                vector_l2(
                    generations + (size_t)generation * root_width,
                    root_width
                )
            );
        }
        fputs("],\"mixed_generations\":[", trace);
        for (int generation = 0; generation <= pullback_depth; generation++) {
            if (generation != 0) fputc(',', trace);
            write_float_vector(
                trace,
                generations + (size_t)generation * root_width,
                root_width
            );
        }
        fputs("],\"composed_pullback_mixed\":", trace);
        write_float_vector(trace, composed, root_width);
        fputs(",\"composition_defect\":", trace);
        write_float_vector(trace, composition_defect, root_width);
        fputs("}\n", trace);
        fflush(trace);
    }
    free_pullback(&pullback_environment);
    free(composition_defect);
    free(composed);
    free(generations);
    return defect_l2;
}

static double vector_max_abs(const float *values, int width) {
    double maximum = 0.0;
    for (int index = 0; index < width; index++) {
        double magnitude = fabs((double)values[index]);
        if (magnitude > maximum) maximum = magnitude;
    }
    return maximum;
}

static ActionTraceState allocate_action_trace_state(int root_width) {
    return (ActionTraceState){
        .root_width = root_width,
        .first_torsor_visible = checked_calloc(
            (size_t)root_width,
            sizeof(float)
        ),
        .previous_torsor_visible = checked_calloc(
            (size_t)root_width,
            sizeof(float)
        ),
        .last_delta_tau = checked_calloc(
            (size_t)root_width,
            sizeof(float)
        ),
        .telescoping_sum = checked_calloc(
            (size_t)root_width,
            sizeof(double)
        )
    };
}

static void free_action_trace_state(ActionTraceState *state) {
    free(state->telescoping_sum);
    free(state->last_delta_tau);
    free(state->previous_torsor_visible);
    free(state->first_torsor_visible);
    memset(state, 0, sizeof(*state));
}

static double record_action_transition(
    FILE *trace,
    ActionTraceState *state,
    const ActionMeasurement *measurement
) {
    if (measurement->root_width != state->root_width) {
        fail("grammatical action root width changed between boundaries");
    }
    if (!state->has_previous) {
        memcpy(
            state->first_torsor_visible,
            measurement->torsor_visible,
            (size_t)state->root_width * sizeof(float)
        );
        memcpy(
            state->previous_torsor_visible,
            measurement->torsor_visible,
            (size_t)state->root_width * sizeof(float)
        );
        state->previous_boundary_index = measurement->boundary_index;
        state->previous_layer = measurement->layer;
        state->previous_phase = measurement->phase;
        state->previous_boundary = measurement->boundary;
        state->has_previous = true;
        return 0.0;
    }

    for (int index = 0; index < state->root_width; index++) {
        state->last_delta_tau[index] = measurement->torsor_visible[index] -
            state->previous_torsor_visible[index];
        state->telescoping_sum[index] += state->last_delta_tau[index];
    }
    double delta_l2 = vector_l2(state->last_delta_tau, state->root_width);
    state->last_from_boundary_index = state->previous_boundary_index;
    state->last_to_boundary_index = measurement->boundary_index;
    if (trace != NULL) {
        fprintf(
            trace,
            "{\"kind\":\"grammatical_action_transition\","
            "\"from_boundary_index\":%d,\"to_boundary_index\":%d,"
            "\"from_layer\":%d,\"to_layer\":%d,\"from_phase\":",
            state->previous_boundary_index,
            measurement->boundary_index,
            state->previous_layer,
            measurement->layer
        );
        fprint_json_string(trace, state->previous_phase);
        fputs(",\"to_phase\":", trace);
        fprint_json_string(trace, measurement->phase);
        fputs(",\"from_boundary\":", trace);
        fprint_json_string(trace, state->previous_boundary);
        fputs(",\"to_boundary\":", trace);
        fprint_json_string(trace, measurement->boundary);
        fprintf(
            trace,
            ",\"root_width\":%d,\"delta_tau_l2\":%.17g,"
            "\"delta_tau_is_vector_difference_not_norm_difference\":true,"
            "\"delta_tau\":",
            state->root_width,
            delta_l2
        );
        write_float_vector(trace, state->last_delta_tau, state->root_width);
        fputs("}\n", trace);
        fflush(trace);
    }
    memcpy(
        state->previous_torsor_visible,
        measurement->torsor_visible,
        (size_t)state->root_width * sizeof(float)
    );
    state->previous_boundary_index = measurement->boundary_index;
    state->previous_layer = measurement->layer;
    state->previous_phase = measurement->phase;
    state->previous_boundary = measurement->boundary;
    state->transition_count++;
    return delta_l2;
}

static double write_action_telescoping_check(
    FILE *trace,
    const ActionTraceState *state,
    double *maximum_absolute_defect
) {
    float *residual = checked_calloc(
        (size_t)state->root_width,
        sizeof(float)
    );
    for (int index = 0; index < state->root_width; index++) {
        double endpoint = (double)state->previous_torsor_visible[index] -
            state->first_torsor_visible[index];
        residual[index] = (float)(state->telescoping_sum[index] - endpoint);
    }
    double l2 = vector_l2(residual, state->root_width);
    *maximum_absolute_defect = vector_max_abs(residual, state->root_width);
    if (trace != NULL) {
        fprintf(
            trace,
            "{\"kind\":\"grammatical_action_telescoping_check\","
            "\"transition_count\":%d,\"root_width\":%d,"
            "\"vector_l2_defect\":%.17g,"
            "\"maximum_absolute_defect\":%.17g,\"residual\":",
            state->transition_count,
            state->root_width,
            l2,
            *maximum_absolute_defect
        );
        write_float_vector(trace, residual, state->root_width);
        fputs("}\n", trace);
        fflush(trace);
    }
    free(residual);
    return l2;
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
    ActionTraceState *trace_state,
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
    write_action_measurement(trace, &measurement);
    double delta_tau_l2 = record_action_transition(
        trace,
        trace_state,
        &measurement
    );
    printf(
        "boundary=%d layer=%d %-9s %-28s "
        "local_mixed=%.8g local_commutator=%.8g "
        "torsor_visible=%.8g delta_tau=%.8g pullback_mixed=%.8g\n",
        *boundary_index,
        layer,
        phase,
        boundary,
        measurement.local_mixed_l2,
        measurement.local_commutator_l2,
        measurement.torsor_visible_l2,
        delta_tau_l2,
        measurement.pullback_mixed_l2
    );
    fflush(stdout);
    free_action_measurement(&measurement);
    (*boundary_index)++;
}

static QkCausalMeasurement measure_qk_causal(
    const GrammarTerm *term,
    GrammarRootScope scope,
    LayerRuntime *runtime,
    FrontierMap qk_map,
    Continuation qk_suffix,
    int qkv_boundary_index,
    int qk_boundary_index,
    const float *const qkv_states[ACTION_CONTEXT_COUNT],
    const float *const qk_states[ACTION_CONTEXT_COUNT],
    const ActionTraceState *trace_state
) {
    int frontier_width = frontier_width_for(runtime);
    int kv_frontier_width = kv_frontier_width_for(runtime);
    int score_width = attention_table_width_for(runtime);
    int output_prefix_width = frontier_width + kv_frontier_width;
    int root_width = action_root_width(term, scope);
    if (qk_map.input_width != qkv_state_width(runtime) ||
        qk_map.output_width != score_state_width(runtime) ||
        trace_state->last_from_boundary_index != qkv_boundary_index ||
        trace_state->last_to_boundary_index != qk_boundary_index) {
        fail("QK causal measurement boundary mismatch");
    }
    QkCausalMeasurement measurement = {
        .layer = runtime->layer,
        .qkv_boundary_index = qkv_boundary_index,
        .qk_boundary_index = qk_boundary_index,
        .score_width = score_width,
        .root_width = root_width,
        .measured_score_defect = checked_calloc(
            (size_t)score_width,
            sizeof(float)
        ),
        .directed_a_query_b_key = checked_calloc(
            (size_t)score_width,
            sizeof(float)
        ),
        .directed_b_query_a_key = checked_calloc(
            (size_t)score_width,
            sizeof(float)
        ),
        .analytic_cross_terms = checked_calloc(
            (size_t)score_width,
            sizeof(float)
        ),
        .score_reconstruction_defect = checked_calloc(
            (size_t)score_width,
            sizeof(float)
        ),
        .exact_delta_tau = checked_calloc(
            (size_t)root_width,
            sizeof(float)
        ),
        .transition_delta_tau = checked_calloc(
            (size_t)root_width,
            sizeof(float)
        ),
        .cross_removed_root = checked_calloc(
            (size_t)root_width,
            sizeof(float)
        )
    };

    float *independent_input = checked_calloc(
        (size_t)qk_map.input_width,
        sizeof(float)
    );
    float *coupled_output = checked_calloc(
        (size_t)qk_map.output_width,
        sizeof(float)
    );
    float *affine_output = checked_calloc(
        (size_t)qk_map.output_width,
        sizeof(float)
    );
    float *cross_removed_output = checked_calloc(
        (size_t)qk_map.output_width,
        sizeof(float)
    );
    torsor_completion(
        qkv_states[ACTION_X],
        qkv_states[ACTION_AX],
        qkv_states[ACTION_BX],
        independent_input,
        qk_map.input_width
    );
    qk_map.apply(qk_map.environment, independent_input, coupled_output);
    torsor_completion(
        qk_states[ACTION_X],
        qk_states[ACTION_AX],
        qk_states[ACTION_BX],
        affine_output,
        qk_map.output_width
    );
    memcpy(
        cross_removed_output,
        coupled_output,
        (size_t)qk_map.output_width * sizeof(float)
    );

    double copied_prefix_square = 0.0;
    for (int index = 0; index < output_prefix_width; index++) {
        double defect = (double)coupled_output[index] - affine_output[index];
        copied_prefix_square += defect * defect;
    }
    measurement.copied_prefix_l2 = sqrt(copied_prefix_square);

    int dim = runtime->transformer->config.dim;
    int kv_dim = dim * runtime->transformer->config.n_kv_heads /
        runtime->transformer->config.n_heads;
    int kv_mul = runtime->transformer->config.n_heads /
        runtime->transformer->config.n_kv_heads;
    int heads = runtime->transformer->config.n_heads;
    int head_size = dim / heads;
    int positions = runtime->positions;
    const float *qx = qkv_states[ACTION_X] + frontier_width;
    const float *qa = qkv_states[ACTION_AX] + frontier_width;
    const float *qb = qkv_states[ACTION_BX] + frontier_width;
    const float *kx = qx + frontier_width;
    const float *ka = qa + frontier_width;
    const float *kb = qb + frontier_width;
    double scale = 1.0 / sqrt((double)head_size);
    for (int position = 0; position < positions; position++) {
        for (int head = 0; head < heads; head++) {
            int query_offset = position * dim + head * head_size;
            int row_offset = (position * heads + head) * positions;
            for (int key_position = 0;
                 key_position <= position;
                 key_position++) {
                int key_offset = key_position * kv_dim +
                    (head / kv_mul) * head_size;
                double a_query_b_key = 0.0;
                double b_query_a_key = 0.0;
                for (int lane = 0; lane < head_size; lane++) {
                    double delta_a_q = (double)qa[query_offset + lane] -
                        qx[query_offset + lane];
                    double delta_b_q = (double)qb[query_offset + lane] -
                        qx[query_offset + lane];
                    double delta_a_k = (double)ka[key_offset + lane] -
                        kx[key_offset + lane];
                    double delta_b_k = (double)kb[key_offset + lane] -
                        kx[key_offset + lane];
                    a_query_b_key += delta_a_q * delta_b_k;
                    b_query_a_key += delta_b_q * delta_a_k;
                }
                int score_index = row_offset + key_position;
                measurement.directed_a_query_b_key[score_index] =
                    (float)(a_query_b_key * scale);
                measurement.directed_b_query_a_key[score_index] =
                    (float)(b_query_a_key * scale);
                measurement.analytic_cross_terms[score_index] =
                    measurement.directed_a_query_b_key[score_index] +
                    measurement.directed_b_query_a_key[score_index];
            }
        }
    }
    for (int index = 0; index < score_width; index++) {
        measurement.measured_score_defect[index] =
            coupled_output[output_prefix_width + index] -
            affine_output[output_prefix_width + index];
        measurement.score_reconstruction_defect[index] =
            measurement.measured_score_defect[index] -
            measurement.analytic_cross_terms[index];
        cross_removed_output[output_prefix_width + index] -=
            measurement.analytic_cross_terms[index];
    }
    measurement.measured_score_defect_l2 = vector_l2(
        measurement.measured_score_defect,
        score_width
    );
    measurement.directed_a_query_b_key_l2 = vector_l2(
        measurement.directed_a_query_b_key,
        score_width
    );
    measurement.directed_b_query_a_key_l2 = vector_l2(
        measurement.directed_b_query_a_key,
        score_width
    );
    measurement.analytic_cross_terms_l2 = vector_l2(
        measurement.analytic_cross_terms,
        score_width
    );
    measurement.score_reconstruction_l2_defect = vector_l2(
        measurement.score_reconstruction_defect,
        score_width
    );
    measurement.score_reconstruction_relative_defect =
        measurement.measured_score_defect_l2 == 0.0 ? 0.0 :
        measurement.score_reconstruction_l2_defect /
            measurement.measured_score_defect_l2;
    measurement.score_reconstruction_max_abs_defect = vector_max_abs(
        measurement.score_reconstruction_defect,
        score_width
    );

    float *whole_root = checked_calloc(
        (size_t)term->frontier_width,
        sizeof(float)
    );
    float *coupled_root = checked_calloc((size_t)root_width, sizeof(float));
    float *affine_root = checked_calloc((size_t)root_width, sizeof(float));
    float *removed_root = checked_calloc((size_t)root_width, sizeof(float));
    apply_root_observation(
        term,
        scope,
        qk_suffix,
        coupled_output,
        whole_root,
        coupled_root
    );
    apply_root_observation(
        term,
        scope,
        qk_suffix,
        affine_output,
        whole_root,
        affine_root
    );
    apply_root_observation(
        term,
        scope,
        qk_suffix,
        cross_removed_output,
        whole_root,
        removed_root
    );
    for (int index = 0; index < root_width; index++) {
        measurement.exact_delta_tau[index] =
            coupled_root[index] - affine_root[index];
        measurement.transition_delta_tau[index] =
            trace_state->last_delta_tau[index];
        measurement.cross_removed_root[index] =
            removed_root[index] - affine_root[index];
    }
    measurement.exact_delta_tau_l2 = vector_l2(
        measurement.exact_delta_tau,
        root_width
    );
    measurement.transition_delta_tau_l2 = vector_l2(
        measurement.transition_delta_tau,
        root_width
    );
    measurement.transition_identity_l2_defect = difference_l2(
        measurement.exact_delta_tau,
        measurement.transition_delta_tau,
        root_width
    );
    measurement.transition_identity_relative_defect =
        measurement.exact_delta_tau_l2 == 0.0 ? 0.0 :
        measurement.transition_identity_l2_defect /
            measurement.exact_delta_tau_l2;
    measurement.cross_removed_root_l2 = vector_l2(
        measurement.cross_removed_root,
        root_width
    );
    measurement.cross_removed_root_relative =
        measurement.exact_delta_tau_l2 == 0.0 ? 0.0 :
        measurement.cross_removed_root_l2 /
            measurement.exact_delta_tau_l2;

    free(removed_root);
    free(affine_root);
    free(coupled_root);
    free(whole_root);
    free(cross_removed_output);
    free(affine_output);
    free(coupled_output);
    free(independent_input);
    return measurement;
}

static void free_qk_causal_measurement(QkCausalMeasurement *measurement) {
    free(measurement->cross_removed_root);
    free(measurement->transition_delta_tau);
    free(measurement->exact_delta_tau);
    free(measurement->score_reconstruction_defect);
    free(measurement->analytic_cross_terms);
    free(measurement->directed_b_query_a_key);
    free(measurement->directed_a_query_b_key);
    free(measurement->measured_score_defect);
    memset(measurement, 0, sizeof(*measurement));
}

static void write_qk_causal_measurement(
    FILE *trace,
    const QkCausalMeasurement *measurement
) {
    if (trace == NULL) return;
    fprintf(
        trace,
        "{\"kind\":\"grammatical_qk_causal\",\"layer\":%d,"
        "\"qkv_boundary_index\":%d,\"qk_boundary_index\":%d,"
        "\"score_width\":%d,\"root_width\":%d,"
        "\"torsor_input\":\"s=a+b-x_at_qkv_boundary\","
        "\"measured_map_defect\":\"F(s)-(F(a)+F(b)-F(x))\","
        "\"analytic_cross_terms_formula\":"
        "\"delta_a_Q_delta_b_Kt_plus_delta_b_Q_delta_a_Kt\","
        "\"copied_prefix_l2\":%.17g,"
        "\"measured_score_defect_l2\":%.17g,"
        "\"directed_a_query_b_key_l2\":%.17g,"
        "\"directed_b_query_a_key_l2\":%.17g,"
        "\"analytic_cross_terms_l2\":%.17g,"
        "\"score_reconstruction_l2_defect\":%.17g,"
        "\"score_reconstruction_relative_defect\":%.17g,"
        "\"score_reconstruction_max_abs_defect\":%.17g,"
        "\"exact_delta_tau_l2\":%.17g,"
        "\"transition_delta_tau_l2\":%.17g,"
        "\"transition_identity_l2_defect\":%.17g,"
        "\"transition_identity_relative_defect\":%.17g,"
        "\"cross_removed_root_l2\":%.17g,"
        "\"cross_removed_root_relative\":%.17g,"
        "\"measured_score_defect\":",
        measurement->layer,
        measurement->qkv_boundary_index,
        measurement->qk_boundary_index,
        measurement->score_width,
        measurement->root_width,
        measurement->copied_prefix_l2,
        measurement->measured_score_defect_l2,
        measurement->directed_a_query_b_key_l2,
        measurement->directed_b_query_a_key_l2,
        measurement->analytic_cross_terms_l2,
        measurement->score_reconstruction_l2_defect,
        measurement->score_reconstruction_relative_defect,
        measurement->score_reconstruction_max_abs_defect,
        measurement->exact_delta_tau_l2,
        measurement->transition_delta_tau_l2,
        measurement->transition_identity_l2_defect,
        measurement->transition_identity_relative_defect,
        measurement->cross_removed_root_l2,
        measurement->cross_removed_root_relative
    );
    write_float_vector(
        trace,
        measurement->measured_score_defect,
        measurement->score_width
    );
    fputs(",\"directed_a_query_b_key\":", trace);
    write_float_vector(
        trace,
        measurement->directed_a_query_b_key,
        measurement->score_width
    );
    fputs(",\"directed_b_query_a_key\":", trace);
    write_float_vector(
        trace,
        measurement->directed_b_query_a_key,
        measurement->score_width
    );
    fputs(",\"analytic_cross_terms\":", trace);
    write_float_vector(
        trace,
        measurement->analytic_cross_terms,
        measurement->score_width
    );
    fputs(",\"score_reconstruction_defect\":", trace);
    write_float_vector(
        trace,
        measurement->score_reconstruction_defect,
        measurement->score_width
    );
    fputs(",\"exact_delta_tau\":", trace);
    write_float_vector(
        trace,
        measurement->exact_delta_tau,
        measurement->root_width
    );
    fputs(",\"transition_delta_tau\":", trace);
    write_float_vector(
        trace,
        measurement->transition_delta_tau,
        measurement->root_width
    );
    fputs(",\"cross_removed_root\":", trace);
    write_float_vector(
        trace,
        measurement->cross_removed_root,
        measurement->root_width
    );
    fputs("}\n", trace);
    fflush(trace);
}

static void report_qk_causal_measurement(
    FILE *trace,
    const GrammarTerm *term,
    GrammarRootScope scope,
    LayerRuntime *runtime,
    FrontierMap qk_map,
    Continuation qk_suffix,
    int qkv_boundary_index,
    int qk_boundary_index,
    const float *const qkv_states[ACTION_CONTEXT_COUNT],
    const float *const qk_states[ACTION_CONTEXT_COUNT],
    const ActionTraceState *trace_state
) {
    QkCausalMeasurement measurement = measure_qk_causal(
        term,
        scope,
        runtime,
        qk_map,
        qk_suffix,
        qkv_boundary_index,
        qk_boundary_index,
        qkv_states,
        qk_states,
        trace_state
    );
    printf(
        "  qk_causal layer=%d score=%.8g aQ_bK=%.8g bQ_aK=%.8g "
        "cross=%.8g "
        "reconstruction=%.8g relative=%.8g delta_tau=%.8g "
        "removed_root=%.8g remaining=%.8g\n",
        measurement.layer,
        measurement.measured_score_defect_l2,
        measurement.directed_a_query_b_key_l2,
        measurement.directed_b_query_a_key_l2,
        measurement.analytic_cross_terms_l2,
        measurement.score_reconstruction_l2_defect,
        measurement.score_reconstruction_relative_defect,
        measurement.exact_delta_tau_l2,
        measurement.cross_removed_root_l2,
        measurement.cross_removed_root_relative
    );
    fflush(stdout);
    write_qk_causal_measurement(trace, &measurement);
    free_qk_causal_measurement(&measurement);
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

static void run_behavior_only(
    FILE *trace,
    const GrammarTerm *term,
    GrammarRootScope scope,
    Transformer *transformer,
    EncodedContext contexts[ACTION_CONTEXT_COUNT],
    ContextFrontiers captures[ACTION_CONTEXT_COUNT]
) {
    int root_width = action_root_width(term, scope);
    float *final_outputs[ACTION_CONTEXT_COUNT];
    float *roots[ACTION_CONTEXT_COUNT];
    float *whole_root = checked_calloc(
        (size_t)term->frontier_width,
        sizeof(*whole_root)
    );
    for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
        final_outputs[role] = checked_calloc(
            (size_t)term->frontier_width,
            sizeof(*final_outputs[role])
        );
        roots[role] = checked_calloc(
            (size_t)root_width,
            sizeof(*roots[role])
        );
        term->final_rms_map.apply(
            term->final_rms_map.environment,
            layer_frontier(&captures[role], term->layers),
            final_outputs[role]
        );
        apply_root_observation(
            term,
            scope,
            term->root_identity,
            final_outputs[role],
            whole_root,
            roots[role]
        );
    }

    float *mixed = checked_calloc((size_t)root_width, sizeof(*mixed));
    float *commutator = checked_calloc(
        (size_t)root_width,
        sizeof(*commutator)
    );
    affine_mixed(
        roots[ACTION_X],
        roots[ACTION_AX],
        roots[ACTION_BX],
        roots[ACTION_ABX],
        mixed,
        root_width
    );
    subtract_vectors(
        roots[ACTION_ABX],
        roots[ACTION_BAX],
        commutator,
        root_width
    );
    if (trace != NULL) {
        fprintf(
            trace,
            "{\"kind\":\"grammatical_behavior_root\","
            "\"root_width\":%d,\"mixed_l2\":%.17g,"
            "\"commutator_l2\":%.17g,\"corners\":{",
            root_width,
            vector_l2(mixed, root_width),
            vector_l2(commutator, root_width)
        );
        for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
            if (role != 0) fputc(',', trace);
            fprint_json_string(trace, action_role_name((ActionRole)role));
            fputc(':', trace);
            write_float_vector(trace, roots[role], root_width);
        }
        fputs("},\"mixed\":", trace);
        write_float_vector(trace, mixed, root_width);
        fputs(",\"commutator\":", trace);
        write_float_vector(trace, commutator, root_width);
        fputs("}\n", trace);
        fflush(trace);
    }

    double maximum_reference_hidden_defect = 0.0;
    double maximum_reference_hidden_relative_defect = 0.0;
    for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
        double relative_defect = 0.0;
        double defect = check_reference_hidden_frontier(
            transformer,
            &contexts[role],
            final_outputs[role],
            &relative_defect
        );
        if (defect > maximum_reference_hidden_defect) {
            maximum_reference_hidden_defect = defect;
        }
        if (relative_defect > maximum_reference_hidden_relative_defect) {
            maximum_reference_hidden_relative_defect = relative_defect;
        }
        if (trace != NULL) {
            fprintf(
                trace,
                "{\"kind\":\"grammatical_behavior_reference_check\","
                "\"role\":\"%s\",\"llama2c_hidden_l2_defect\":%.17g,"
                "\"llama2c_hidden_relative_defect\":%.17g}\n",
                action_role_name((ActionRole)role),
                defect,
                relative_defect
            );
            fflush(trace);
        }
    }
    printf(
        "behavior_root mixed=%.8g commutator=%.8g "
        "maximum_llama2c_hidden_relative_defect=%.8g\n",
        vector_l2(mixed, root_width),
        vector_l2(commutator, root_width),
        maximum_reference_hidden_relative_defect
    );
    if (trace != NULL) {
        fprintf(
            trace,
            "{\"kind\":\"grammatical_behavior_check\","
            "\"maximum_llama2c_hidden_l2_defect\":%.17g,"
            "\"maximum_llama2c_hidden_relative_defect\":%.17g}\n",
            maximum_reference_hidden_defect,
            maximum_reference_hidden_relative_defect
        );
        fflush(trace);
    }

    free(commutator);
    free(mixed);
    for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
        free(roots[role]);
        free(final_outputs[role]);
    }
    free(whole_root);
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
    ActionTraceState *trace_state,
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
        int from_boundary_index = trace_state->previous_boundary_index;
        int to_boundary_index = *boundary_index;
        report_action_boundary(
            trace,
            term,
            scope,
            trace_state,
            boundary_index,
            layer,
            phase,
            maps[stage].name,
            suffixes[stage + 1],
            next_const,
            maps[stage].output_width
        );
        if (maps[stage].apply == qk_stage_apply) {
            report_qk_causal_measurement(
                trace,
                term,
                scope,
                maps[stage].environment,
                maps[stage],
                suffixes[stage + 1],
                from_boundary_index,
                to_boundary_index,
                current,
                next_const,
                trace_state
            );
        }
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

#ifndef CPS_GRAMMAR_ACTIONS_NO_MAIN
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
    GrammarTerm term = {0};
    build_grammar_term(&term, &transformer, positions);

    FILE *trace = NULL;
    if (options.trace_path != NULL) {
        trace = fopen(options.trace_path, "wb");
        if (trace == NULL) fail("could not create grammatical action trace");
        if (options.behavior_only) {
            fprintf(
                trace,
                "{\"kind\":\"grammatical_behavior_meta\","
                "\"schema_version\":1,"
                "\"semantics\":\"future_company_root_behavior\","
                "\"mixed_operator\":\"(U_a-I)(U_b-I)k\","
                "\"root_observer\":\"post_final_rms_hidden\","
                "\"root_scope\":\"%s\",\"positions\":%d,\"dim\":%d,"
                "\"layers\":%d,\"factorized_token_square\":%s,"
                "\"constructors_commute_on_x\":%s,"
                "\"corner_roots_retained\":true,"
                "\"stock_forward_hidden_parity\":true,"
                "\"norms_are_diagnostics_not_scores\":true}\n",
                grammar_root_scope_name(options.root_scope),
                positions,
                term.dim,
                term.layers,
                factorized_square ? "true" : "false",
                constructors_commute ? "true" : "false"
            );
        } else {
            fprintf(
                trace,
                "{\"kind\":\"grammatical_action_meta\","
                "\"schema_version\":5,"
                "\"semantics\":\"exact_constructor_pullbacks\","
                "\"mixed_operator\":\"(U_a-I)(U_b-I)k\","
                "\"commutator_operator\":\"(U_aU_b-U_bU_a)k\","
                "\"constructor_order\":{\"abx\":\"b_after_a_on_x\","
                "\"bax\":\"a_after_b_on_x\"},"
                "\"root_observer\":\"post_final_rms_hidden\","
                "\"root_scope\":\"%s\",\"positions\":%d,\"dim\":%d,"
                "\"layers\":%d,\"factorized_token_square\":%s,"
                "\"constructors_commute_on_x\":%s,"
                "\"delta_tau\":\"tau_next_minus_tau_previous_as_vector\","
                "\"qk_causal_reconstruction\":true,"
                "\"qk_directed_cross_terms_retained\":true,"
                "\"block_pullback_pairs_retained\":true,"
                "\"block_pullback_depth\":%d,"
                "\"stock_forward_hidden_parity\":true,"
                "\"norms_are_diagnostics_not_scores\":true}\n",
                grammar_root_scope_name(options.root_scope),
                positions,
                term.dim,
                term.layers,
                factorized_square ? "true" : "false",
                constructors_commute ? "true" : "false",
                options.pullback_depth
            );
        }
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
    if (options.behavior_only) {
        run_behavior_only(
            trace,
            &term,
            options.root_scope,
            &transformer,
            contexts,
            captures
        );
        if (trace != NULL && fclose(trace) != 0) {
            fail("could not close grammatical behavior trace");
        }
        for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
            free_frontiers(&captures[role]);
            free_context(&contexts[role]);
        }
        free_grammar_term(&term);
        free_tokenizer(&tokenizer);
        free_transformer(&transformer);
        return EXIT_SUCCESS;
    }

    int boundary_index = 0;
    ActionTraceState trace_state = allocate_action_trace_state(
        action_root_width(&term, options.root_scope)
    );
    double maximum_chain_output_defect = 0.0;
    double maximum_block_pullback_composition_defect = 0.0;
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
                &trace_state,
                &boundary_index,
                layer,
                "layer",
                "token_embedding_frontier",
                term.layer_suffixes[layer],
                layer_inputs,
                term.frontier_width
            );
        }

        double attention_pullback_defect = report_block_pullback(
            trace,
            &term,
            options.root_scope,
            layer,
            "attention",
            term.attention_maps[layer],
            term.post_attention_suffixes[layer],
            layer_inputs,
            post_attention,
            options.pullback_depth
        );
        if (attention_pullback_defect >
            maximum_block_pullback_composition_defect) {
            maximum_block_pullback_composition_defect =
                attention_pullback_defect;
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
            &trace_state,
            &boundary_index,
            layer,
            "attention",
            attention_stages,
            attention_count,
            term.post_attention_suffixes[layer],
            layer_inputs,
            post_attention
        );
        if (attention_defect > maximum_chain_output_defect) {
            maximum_chain_output_defect = attention_defect;
        }

        double ffn_pullback_defect = report_block_pullback(
            trace,
            &term,
            options.root_scope,
            layer,
            "ffn",
            term.ffn_maps[layer],
            term.layer_suffixes[layer + 1],
            post_attention,
            layer_outputs,
            options.pullback_depth
        );
        if (ffn_pullback_defect >
            maximum_block_pullback_composition_defect) {
            maximum_block_pullback_composition_defect =
                ffn_pullback_defect;
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
            &trace_state,
            &boundary_index,
            layer,
            "ffn",
            ffn_stages,
            ffn_count,
            term.layer_suffixes[layer + 1],
            post_attention,
            layer_outputs
        );
        if (ffn_defect > maximum_chain_output_defect) {
            maximum_chain_output_defect = ffn_defect;
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
        &trace_state,
        &boundary_index,
        term.layers,
        "root",
        "final_rms",
        term.root_identity,
        final_output_const,
        term.frontier_width
    );

    double maximum_reference_hidden_defect = 0.0;
    double maximum_reference_hidden_relative_defect = 0.0;
    for (int role = 0; role < ACTION_CONTEXT_COUNT; role++) {
        double relative_defect = 0.0;
        double defect = check_reference_hidden_frontier(
            &transformer,
            &contexts[role],
            final_outputs[role],
            &relative_defect
        );
        if (defect > maximum_reference_hidden_defect) {
            maximum_reference_hidden_defect = defect;
        }
        if (relative_defect > maximum_reference_hidden_relative_defect) {
            maximum_reference_hidden_relative_defect = relative_defect;
        }
        if (trace != NULL) {
            fprintf(
                trace,
                "{\"kind\":\"grammatical_action_reference_check\","
                "\"role\":\"%s\",\"llama2c_hidden_l2_defect\":%.17g,"
                "\"llama2c_hidden_relative_defect\":%.17g}\n",
                action_role_name((ActionRole)role),
                defect,
                relative_defect
            );
            fflush(trace);
        }
    }

    double telescoping_maximum_absolute_defect = 0.0;
    double telescoping_l2_defect = write_action_telescoping_check(
        trace,
        &trace_state,
        &telescoping_maximum_absolute_defect
    );

    printf(
        "boundaries=%d maximum_typed_chain_output_l2_defect=%.8g "
        "maximum_llama2c_hidden_l2_defect=%.8g "
        "maximum_llama2c_hidden_relative_defect=%.8g "
        "maximum_block_pullback_composition_l2_defect=%.8g "
        "telescoping_l2_defect=%.8g telescoping_max_abs=%.8g\n",
        boundary_index,
        maximum_chain_output_defect,
        maximum_reference_hidden_defect,
        maximum_reference_hidden_relative_defect,
        maximum_block_pullback_composition_defect,
        telescoping_l2_defect,
        telescoping_maximum_absolute_defect
    );
    if (trace != NULL) {
        fprintf(
            trace,
            "{\"kind\":\"grammatical_action_check\","
            "\"boundaries\":%d,"
            "\"maximum_typed_chain_output_l2_defect\":%.17g,"
            "\"maximum_llama2c_hidden_l2_defect\":%.17g,"
            "\"maximum_llama2c_hidden_relative_defect\":%.17g,"
            "\"maximum_block_pullback_composition_l2_defect\":%.17g,"
            "\"telescoping_l2_defect\":%.17g,"
            "\"telescoping_maximum_absolute_defect\":%.17g}\n",
            boundary_index,
            maximum_chain_output_defect,
            maximum_reference_hidden_defect,
            maximum_reference_hidden_relative_defect,
            maximum_block_pullback_composition_defect,
            telescoping_l2_defect,
            telescoping_maximum_absolute_defect
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
    free_action_trace_state(&trace_state);
    free_grammar_term(&term);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return EXIT_SUCCESS;
}
#endif
