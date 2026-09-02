/*
 * Observer-relative fixed points of individual residual additions.
 *
 * This is a measurement, not a completion selector or a layer-skip policy.
 * At one attention/FFN residual boundary, retain the actual input X and
 * actual output Y. For row i, omission replaces Y[i] by X[i], leaving every
 * other row of Y unchanged. This removes exactly that residual addition;
 * it does not recompute other rows against a mutated attention input.
 *
 * The observer is the untouched suffix followed by the trained output head
 * at the last position. We record p(o(x)) and p(x) for:
 *   - each corpus-target versus vocabulary-token contrast;
 *   - each such contrast's three-way sign (ties included);
 *   - the complete vocabulary argmax set.
 * All logits are retained, not only the displayed candidates. Exact equality
 * means equality of the recorded floating-point values, with no tolerance.
 * A sampled equality is not a proof on unseen inputs or other observers.
 *
 * Also omit, together, the rows that individually preserved the argmax set.
 * This explicitly checks whether pointwise fixedness survived composition.
 * No operation is removed from an inference executable on this evidence.
 */
#define CPS_GRAMMAR_ACTIONS_NO_MAIN
#include "cps_grammar_actions.c"

#include <stdint.h>
#include <time.h>

typedef struct {
    const char *corpus;
    const char *trace;
    const char *logits;
    int positions;
    int samples;
    int start_story;
    int position_stride;
} ObserverOptions;

typedef struct {
    Transformer *transformer;
    Continuation suffix;
    float *frontier;
    int positions;
} VocabularyPullback;

typedef struct { int token; float logit; } RankedToken;

typedef struct {
    uint64_t operations;
    uint64_t structurally_unreachable;
    uint64_t codata_fixed;
    uint64_t target_order_fixed;
    uint64_t choice_fixed;
    uint64_t joint_groups;
    uint64_t joint_choice_changed;
} FixedCounts;

static int observer_integer(const char *text, bool positive) {
    errno = 0;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (errno || end == text || *end || value < (positive ? 1 : 0) ||
        value > INT_MAX) fail("invalid observer measurement integer");
    return (int)value;
}

static ObserverOptions observer_options(int argc, char **argv) {
    ObserverOptions o = {.positions = 32, .samples = 4,
                         .start_story = 1, .position_stride = 1};
    if (argc < 3) fail("usage: cps_observer_fixed_points MODEL TOKENIZER "
        "--corpus FILE --trace FILE --logits FILE [--positions N] "
        "[--samples N] [--start-story N] [--position-stride N]");
    for (int i = 3; i < argc; i += 2) {
        if (i + 1 == argc) fail("observer option needs a value");
        if (!strcmp(argv[i], "--corpus")) o.corpus = argv[i + 1];
        else if (!strcmp(argv[i], "--trace")) o.trace = argv[i + 1];
        else if (!strcmp(argv[i], "--logits")) o.logits = argv[i + 1];
        else if (!strcmp(argv[i], "--positions"))
            o.positions = observer_integer(argv[i + 1], true);
        else if (!strcmp(argv[i], "--samples"))
            o.samples = observer_integer(argv[i + 1], true);
        else if (!strcmp(argv[i], "--start-story"))
            o.start_story = observer_integer(argv[i + 1], false);
        else if (!strcmp(argv[i], "--position-stride"))
            o.position_stride = observer_integer(argv[i + 1], true);
        else fail("unknown observer measurement option");
    }
    if (!o.corpus || !o.trace || !o.logits)
        fail("--corpus, --trace and --logits are required");
    if (!strcmp(o.trace, o.logits) || !strcmp(o.corpus, o.trace) ||
        !strcmp(o.corpus, o.logits)) fail("input/output paths must differ");
    return o;
}

/* Keep paragraph/context provenance; never splice separate stories together. */
static char *next_story(FILE *file) {
    char *line = NULL, *story = NULL;
    size_t line_capacity = 0, size = 0;
    ssize_t length;
    while ((length = getline(&line, &line_capacity, file)) >= 0) {
        if (!strncmp(line, "<|endoftext|>", 13)) break;
        if (size > SIZE_MAX - (size_t)length - 1) fail("story too large");
        char *grown = realloc(story, size + (size_t)length + 1);
        if (!grown) fail("could not grow corpus story");
        story = grown;
        memcpy(story + size, line, (size_t)length);
        size += (size_t)length;
        story[size] = '\0';
    }
    free(line);
    if (ferror(file)) fail("corpus read failed");
    if (!story && !feof(file)) story = strdup("");
    return story;
}

/* Byte-token names stay legible and JSON-safe; ordinary tokens are decoded. */
static const char *observer_piece(Tokenizer *tokenizer, int previous, int token) {
    const char *raw = tokenizer->vocab[token];
    if (!strncmp(raw, "<0x", 3)) return raw;
    return decode(tokenizer, previous, token);
}

static char *decoded_prefix(Tokenizer *tokenizer, const EncodedContext *context) {
    size_t length = 0;
    for (int i = 1; i < context->count; i++)
        length += strlen(observer_piece(tokenizer, context->tokens[i - 1],
                                        context->tokens[i]));
    char *text = checked_calloc(length + 1, 1);
    size_t offset = 0;
    for (int i = 1; i < context->count; i++) {
        const char *piece = observer_piece(tokenizer, context->tokens[i - 1],
                                           context->tokens[i]);
        size_t bytes = strlen(piece);
        memcpy(text + offset, piece, bytes);
        offset += bytes;
    }
    return text;
}

static void vocabulary_pullback_apply(void *environment,
                                      const float *input, float *result) {
    VocabularyPullback *p = environment;
    p->suffix.apply(p->suffix.environment, input, p->frontier);
    Config *c = &p->transformer->config;
    matmul(result, p->frontier + (size_t)(p->positions - 1) * c->dim,
           p->transformer->weights.wcls, c->dim, c->vocab_size);
    for (int i = 0; i < c->vocab_size; i++)
        if (!isfinite(result[i])) fail("non-finite observed logit");
}

static uint64_t retain_logits(FILE *file, const float *values, int count) {
    off_t offset = ftello(file);
    if (offset < 0 || fwrite(values, sizeof(float), (size_t)count, file) !=
        (size_t)count || fflush(file)) fail("could not flush full logits");
    return (uint64_t)offset;
}

static int rank_compare(const void *a, const void *b) {
    const RankedToken *x = a, *y = b;
    if (x->logit != y->logit) return x->logit > y->logit ? -1 : 1;
    return (x->token > y->token) - (x->token < y->token);
}

static void rank_vocabulary(RankedToken *rank, const float *q, int count) {
    for (int t = 0; t < count; t++)
        rank[t] = (RankedToken){.token = t, .logit = q[t]};
    qsort(rank, (size_t)count, sizeof(*rank), rank_compare);
}

static int sign_of(double x) { return (x > 0) - (x < 0); }

static bool same_choice(const float *before, const float *after, int count) {
    float a = before[0], b = after[0];
    for (int t = 1; t < count; t++) { a = fmaxf(a, before[t]); b = fmaxf(b, after[t]); }
    for (int t = 0; t < count; t++)
        if ((before[t] == a) != (after[t] == b)) return false;
    return true;
}

static void write_comparison(FILE *trace, Tokenizer *tokenizer, int previous,
                             int target, int rival, const float *on,
                             const float *off) {
    double before = (double)on[target] - on[rival];
    double after = (double)off[target] - off[rival];
    fprintf(trace, "{\"alternative\":%d,\"piece\":", rival);
    fprint_json_string(trace, observer_piece(tokenizer, previous, rival));
    fprintf(trace, ",\"with_update\":%.17g,\"without_update\":%.17g,"
        "\"contrast_fixed\":%s,\"ordering_fixed\":%s}", before, after,
        before == after ? "true" : "false",
        sign_of(before) == sign_of(after) ? "true" : "false");
}

static void write_top(FILE *trace, Tokenizer *tokenizer, int previous,
                       const RankedToken *rank, int count) {
    fputc('[', trace);
    for (int i = 0; i < count && i < 5; i++) {
        if (i) fputc(',', trace);
        fprintf(trace, "{\"token\":%d,\"piece\":", rank[i].token);
        fprint_json_string(trace, observer_piece(tokenizer, previous, rank[i].token));
        fprintf(trace, ",\"logit\":%.9g}", rank[i].logit);
    }
    fputc(']', trace);
}

static bool write_observation(FILE *trace, FILE *binary, Tokenizer *tokenizer,
    const EncodedContext *context, int target, int sample, int layer,
    const char *operation, int position, const bool *joint, int joint_count,
    const float *on, const float *off, const RankedToken *on_rank,
    RankedToken *off_rank, int vocabulary, bool unreachable, FixedCounts *counts) {
    int previous = context->tokens[context->count - 1];
    int exact = 0, order_fixed = 0;
    for (int t = 0; t < vocabulary; t++) {
        if (t == target) continue;
        double a = (double)on[target] - on[t];
        double b = (double)off[target] - off[t];
        exact += a == b;
        order_fixed += sign_of(a) == sign_of(b);
    }
    bool choice_fixed = same_choice(on, off, vocabulary);
    uint64_t offset = retain_logits(binary, off, vocabulary);
    rank_vocabulary(off_rank, off, vocabulary);
    fprintf(trace, "{\"kind\":\"%s\",\"sample\":%d,\"layer\":%d,"
        "\"operation\":\"%s\",\"position\":%d,\"term_piece\":",
        joint ? "joint_omission" : "fixed_observation", sample, layer,
        operation, position);
    if (position < 0) fputs("null", trace);
    else fprint_json_string(trace, observer_piece(tokenizer,
        position ? context->tokens[position - 1] : 0, context->tokens[position]));
    fprintf(trace, ",\"logits_offset\":%llu,\"codata_fixed\":%s,"
        "\"target_contrast_fixed_count\":%d,\"target_order_fixed_count\":%d,"
        "\"target_order_changed_count\":%d,\"choice_fixed\":%s,"
        "\"structurally_unreachable\":%s,\"top_without_update\":",
        (unsigned long long)offset, exact == vocabulary - 1 ? "true" : "false",
        exact, order_fixed, vocabulary - 1 - order_fixed,
        choice_fixed ? "true" : "false", unreachable ? "true" : "false");
    write_top(trace, tokenizer, previous, off_rank, vocabulary);
    fputs(",\"baseline_top_comparisons\":[", trace);
    int written = 0;
    for (int i = 0; i < vocabulary && written < 5; i++) {
        int t = on_rank[i].token;
        if (t == target) continue;
        if (written++) fputc(',', trace);
        write_comparison(trace, tokenizer, previous, target, t, on, off);
    }
    fputs("],\"changed_comparisons\":[", trace);
    written = 0;
    for (int i = 0; i < vocabulary && written < 5; i++) {
        int t = on_rank[i].token;
        if (t == target || sign_of((double)on[target] - on[t]) ==
            sign_of((double)off[target] - off[t])) continue;
        if (written++) fputc(',', trace);
        write_comparison(trace, tokenizer, previous, target, t, on, off);
    }
    fputc(']', trace);
    if (joint) {
        fprintf(trace, ",\"individually_choice_fixed_count\":%d,\"omitted_positions\":[",
                joint_count);
        written = 0;
        for (int i = 0; i < context->count; i++) if (joint[i]) {
            if (written++) fputc(',', trace);
            fprintf(trace, "%d", i);
        }
        fputc(']', trace);
        counts->joint_groups++;
        counts->joint_choice_changed += !choice_fixed;
    } else {
        counts->operations++;
        counts->structurally_unreachable += unreachable;
        counts->codata_fixed += exact == vocabulary - 1;
        counts->target_order_fixed += order_fixed == vocabulary - 1;
        counts->choice_fixed += choice_fixed;
    }
    fputs("}\n", trace);
    if (fflush(trace) || ferror(trace)) fail("could not flush observation trace");
    return choice_fixed;
}

static void measure_story(GrammarTerm *term, Tokenizer *tokenizer,
    const EncodedContext *context, int target, int sample, int story_index,
    const char *story, const ObserverOptions *options,
    FILE *trace, FILE *binary, FixedCounts *counts) {
    int layers = term->layers, dim = term->dim, width = term->frontier_width;
    int vocabulary = term->transformer->config.vocab_size;
    ContextFrontiers capture = allocate_frontiers(layers, width);
    capture_context_frontiers(term->transformer, context, term->runtimes, &capture);
    float *input = checked_calloc((size_t)width, sizeof(float));
    float *root = checked_calloc((size_t)width, sizeof(float));
    float *on = checked_calloc((size_t)vocabulary, sizeof(float));
    float *off = checked_calloc((size_t)vocabulary, sizeof(float));
    RankedToken *on_rank = checked_calloc((size_t)vocabulary, sizeof(*on_rank));
    RankedToken *off_rank = checked_calloc((size_t)vocabulary, sizeof(*off_rank));
    bool *fixed = checked_calloc((size_t)context->count, sizeof(*fixed));
    VocabularyPullback environment = {.transformer = term->transformer,
        .suffix = term->layer_suffixes[layers], .frontier = root,
        .positions = context->count};
    Continuation p = {.input_width = width, .result_width = vocabulary,
        .apply = vocabulary_pullback_apply, .environment = &environment};
    p.apply(p.environment, layer_frontier(&capture, layers), on);
    uint64_t baseline_offset = retain_logits(binary, on, vocabulary);
    rank_vocabulary(on_rank, on, vocabulary);
    char *prefix = decoded_prefix(tokenizer, context);
    fprintf(trace, "{\"kind\":\"sample\",\"sample\":%d,\"story_index\":%d,"
        "\"positions\":%d,\"text\":", sample, story_index, context->count);
    fprint_json_string(trace, prefix);
    fputs(",\"source_story\":", trace);
    fprint_json_string(trace, story);
    fprintf(trace, ",\"target\":%d,\"target_piece\":", target);
    fprint_json_string(trace, observer_piece(tokenizer,
        context->tokens[context->count - 1], target));
    fprintf(trace, ",\"logits_offset\":%llu,\"top_with_update\":",
            (unsigned long long)baseline_offset);
    write_top(trace, tokenizer, context->tokens[context->count - 1], on_rank, vocabulary);
    fputs(",\"tokens\":[", trace);
    for (int i = 0; i < context->count; i++) {
        if (i) fputc(',', trace);
        fprintf(trace, "%d", context->tokens[i]);
    }
    fputs("]}\n", trace);
    fflush(trace);
    printf("sample=%d story=%d context=", sample, story_index);
    fprint_json_string(stdout, prefix);
    fputs(" target=", stdout);
    fprint_json_string(stdout, observer_piece(tokenizer,
        context->tokens[context->count - 1], target));
    putchar('\n');
    fflush(stdout);
    free(prefix);

    for (int layer = 0; layer < layers; layer++) {
        for (int phase = 0; phase < 2; phase++) {
            const float *before = phase ? post_attention_frontier(&capture, layer)
                                        : layer_frontier(&capture, layer);
            const float *after = phase ? layer_frontier(&capture, layer + 1)
                                       : post_attention_frontier(&capture, layer);
            environment.suffix = phase ? term->layer_suffixes[layer + 1]
                                       : term->post_attention_suffixes[layer];
            const char *name = phase ? "ffn_residual" : "attention_residual";
            /* Establish the identical unmodified observer, not AR equivalence. */
            p.apply(p.environment, after, off);
            if (memcmp(on, off, (size_t)vocabulary * sizeof(float)))
                fail("untouched suffix did not reproduce baseline observation");
            memset(fixed, 0, (size_t)context->count * sizeof(*fixed));
            int fixed_count = 0;
            for (int position = 0; position < context->count; position++) {
                if (position % options->position_stride && position != context->count - 1)
                    continue;
                memcpy(input, after, (size_t)width * sizeof(float));
                memcpy(input + (size_t)position * dim,
                       before + (size_t)position * dim, (size_t)dim * sizeof(float));
                p.apply(p.environment, input, off);
                bool unreachable = layer == layers - 1 && position < context->count - 1;
                if (unreachable && memcmp(on, off, (size_t)vocabulary * sizeof(float)))
                    fail("last-layer non-root residual changed the root observation");
                fixed[position] = write_observation(trace, binary, tokenizer,
                    context, target, sample, layer, name, position, NULL, 0,
                    on, off, on_rank, off_rank, vocabulary, unreachable, counts);
                fixed_count += fixed[position];
            }
            if (fixed_count >= 2) {
                memcpy(input, after, (size_t)width * sizeof(float));
                for (int i = 0; i < context->count; i++) if (fixed[i])
                    memcpy(input + (size_t)i * dim, before + (size_t)i * dim,
                           (size_t)dim * sizeof(float));
                p.apply(p.environment, input, off);
                (void)write_observation(trace, binary, tokenizer, context,
                    target, sample, layer, name, -1, fixed, fixed_count,
                    on, off, on_rank, off_rank, vocabulary, false, counts);
            }
        }
    }
    free(fixed); free(off_rank); free(on_rank); free(off); free(on);
    free(root); free(input); free_frontiers(&capture);
}

int main(int argc, char **argv) {
    ObserverOptions options = observer_options(argc, argv);
    Transformer transformer;
    build_transformer(&transformer, argv[1]);
    if (options.positions > transformer.config.seq_len)
        fail("requested context exceeds model context length");
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, argv[2], transformer.config.vocab_size);
    FILE *corpus = fopen(options.corpus, "rb");
    if (!corpus) fail("could not open corpus");
    /* Exclusive creation preserves previous evidence and protects input files. */
    FILE *trace = fopen(options.trace, "wx");
    if (!trace) fail("could not create trace (existing files are not overwritten)");
    FILE *binary = fopen(options.logits, "wbx");
    if (!binary) fail("could not create full-logit sidecar");
    setvbuf(trace, NULL, _IOLBF, 0);
    fprintf(trace, "{\"kind\":\"meta\",\"schema_version\":1,"
        "\"measurement\":\"individual_residual_observer_fixed_points\","
        "\"model\":");
    fprint_json_string(trace, argv[1]);
    fputs(",\"corpus\":", trace); fprint_json_string(trace, options.corpus);
    fputs(",\"logits_file\":", trace); fprint_json_string(trace, options.logits);
    uint32_t endian = 1;
    fprintf(trace, ",\"logits_dtype\":\"float32\",\"byte_order\":\"%s\","
        "\"layers\":%d,\"dim\":%d,\"vocabulary\":%d,\"positions\":%d,"
        "\"position_stride\":%d,\"requested_samples\":%d,\"start_story\":%d,"
        "\"observer\":\"corpus-target contrasts, their signs, and full argmax set\","
        "\"equality\":\"exact recorded values; no tolerance\","
        "\"grammar_labels_supplied\":false,\"optimization_enabled\":false}\n",
        *(unsigned char *)&endian ? "little" : "big", transformer.config.n_layers,
        transformer.config.dim, transformer.config.vocab_size, options.positions,
        options.position_stride, options.samples, options.start_story);
    for (int t = 0; t < tokenizer.vocab_size; t++) {
        fprintf(trace, "{\"kind\":\"vocabulary\",\"token\":%d,\"piece\":", t);
        fprint_json_string(trace, observer_piece(&tokenizer, 0, t));
        fputs("}\n", trace);
    }
    fflush(trace);
    GrammarTerm term = {0};
    build_grammar_term(&term, &transformer, options.positions);
    FixedCounts counts = {0};
    int sample = 0, story_index = 0;
    double started = (double)clock() / CLOCKS_PER_SEC;
    char *story;
    while (sample < options.samples && (story = next_story(corpus)) != NULL) {
        int index = story_index++;
        if (index < options.start_story || !*story) { free(story); continue; }
        EncodedContext encoded = encode_context(&tokenizer, story);
        if (encoded.count > options.positions) {
            int target = encoded.tokens[options.positions];
            EncodedContext prefix = {.text = story, .tokens = encoded.tokens,
                                     .count = options.positions};
            measure_story(&term, &tokenizer, &prefix, target, sample++, index,
                          story, &options, trace, binary, &counts);
        }
        free_context(&encoded);
        free(story);
    }
    fprintf(trace, "{\"kind\":\"summary\",\"samples\":%d,\"operations\":%llu,"
        "\"structurally_unreachable\":%llu,\"codata_fixed\":%llu,"
        "\"target_order_fixed\":%llu,\"choice_fixed\":%llu,"
        "\"joint_groups\":%llu,\"joint_choice_changed\":%llu,"
        "\"cpu_seconds\":%.6f}\n", sample,
        (unsigned long long)counts.operations,
        (unsigned long long)counts.structurally_unreachable,
        (unsigned long long)counts.codata_fixed,
        (unsigned long long)counts.target_order_fixed,
        (unsigned long long)counts.choice_fixed,
        (unsigned long long)counts.joint_groups,
        (unsigned long long)counts.joint_choice_changed,
        (double)clock() / CLOCKS_PER_SEC - started);
    printf("samples=%d operations=%llu codata_fixed=%llu choice_fixed=%llu "
           "joint_choice_changed=%llu/%llu\n", sample,
           (unsigned long long)counts.operations, (unsigned long long)counts.codata_fixed,
           (unsigned long long)counts.choice_fixed,
           (unsigned long long)counts.joint_choice_changed,
           (unsigned long long)counts.joint_groups);
    if (fclose(binary) || fclose(trace) || fclose(corpus)) fail("could not close evidence files");
    free_grammar_term(&term); free_tokenizer(&tokenizer); free_transformer(&transformer);
    if (sample != options.samples) fail("corpus exhausted before requested coverage");
    return 0;
}
