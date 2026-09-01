/*
 * Torsor-safe sampled fixed continuations for the exact CPS transformer term.
 *
 * For an endomorphic operation F and its mechanically composed suffix k, each
 * sampled constructor state x contributes two root points
 *
 *     after  = k(F(x))
 *     bypass = k(x).
 *
 * The fixed calculation uses only the displacement after - bypass.  The two
 * points are also retained together in a mapped file whose header declares an
 * implicit homogeneous coordinate.  They must never be pseudoinverted as bare
 * vectors of points.
 */

#define CPS_FIXED_POINTS_NO_MAIN
#include "cps_fixed_points.c"

#if !defined(__APPLE__)
#error "cps_affine_spectrum currently requires macOS Accelerate"
#endif

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <dirent.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef enum {
    SPECTRUM_ATTENTION,
    SPECTRUM_FFN,
    SPECTRUM_LAYER,
    SPECTRUM_FINAL_RMS
} SpectrumOperation;

typedef struct {
    const char *corpus_directory;
    const char *matrix_path;
    const char *basis_path;
    const char *trace_path;
    int positions;
    int samples;
    int layer;
    SpectrumOperation operation;
    bool resume;
} SpectrumOptions;

typedef struct {
    char **items;
    size_t count;
    size_t capacity;
} PathList;

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t sample_capacity;
    uint32_t samples_written;
    uint32_t root_width;
    uint32_t positions;
    uint32_t layer;
    uint32_t operation;
    uint32_t point_count_per_sample;
    uint32_t implicit_homogeneous_coordinate;
    uint32_t reserved[5];
} AffinePairHeader;

typedef struct {
    int descriptor;
    size_t mapping_bytes;
    void *mapping;
    AffinePairHeader *header;
    float *pairs;
} AffinePairFile;

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t samples;
    uint32_t root_width;
    uint32_t displacement_rank;
    uint32_t displacement_nullity;
    uint32_t affine_constant_dimension;
    uint32_t singular_value_count;
    uint32_t right_mode_count;
    float rank_tolerance;
    float largest_singular_value;
    float smallest_computed_singular_value;
    float maximum_basis_residual;
    uint32_t reserved[3];
} FixedBasisHeader;

typedef struct {
    Transformer *transformer;
    int layers;
    int positions;
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
} SpectrumTerm;

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

static int parse_nonnegative_integer(const char *text, const char *name) {
    errno = 0;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 0 ||
        value > INT_MAX) {
        fprintf(stderr, "invalid %s: %s\n", name, text);
        exit(EXIT_FAILURE);
    }
    return (int)value;
}

static SpectrumOperation parse_operation(const char *text) {
    if (strcmp(text, "attention") == 0) return SPECTRUM_ATTENTION;
    if (strcmp(text, "ffn") == 0) return SPECTRUM_FFN;
    if (strcmp(text, "layer") == 0) return SPECTRUM_LAYER;
    if (strcmp(text, "final-rms") == 0) return SPECTRUM_FINAL_RMS;
    fail("operation must be attention, ffn, layer, or final-rms");
    return SPECTRUM_LAYER;
}

static const char *operation_name(SpectrumOperation operation) {
    switch (operation) {
        case SPECTRUM_ATTENTION: return "attention";
        case SPECTRUM_FFN: return "ffn";
        case SPECTRUM_LAYER: return "layer";
        case SPECTRUM_FINAL_RMS: return "final-rms";
    }
    fail("invalid spectrum operation");
    return "invalid";
}

static SpectrumOptions parse_spectrum_options(int argc, char **argv) {
    SpectrumOptions options = {
        .positions = 3,
        .samples = 0,
        .layer = 0,
        .operation = SPECTRUM_ATTENTION
    };
    for (int index = 3; index < argc;) {
        if (strcmp(argv[index], "--corpus-dir") == 0 && index + 1 < argc) {
            options.corpus_directory = argv[index + 1];
            index += 2;
        } else if (strcmp(argv[index], "--matrix") == 0 &&
                   index + 1 < argc) {
            options.matrix_path = argv[index + 1];
            index += 2;
        } else if (strcmp(argv[index], "--basis") == 0 &&
                   index + 1 < argc) {
            options.basis_path = argv[index + 1];
            index += 2;
        } else if (strcmp(argv[index], "--trace") == 0 &&
                   index + 1 < argc) {
            options.trace_path = argv[index + 1];
            index += 2;
        } else if (strcmp(argv[index], "--positions") == 0 &&
                   index + 1 < argc) {
            options.positions = parse_positive_integer(
                argv[index + 1],
                "positions"
            );
            index += 2;
        } else if (strcmp(argv[index], "--samples") == 0 &&
                   index + 1 < argc) {
            options.samples = parse_positive_integer(
                argv[index + 1],
                "samples"
            );
            index += 2;
        } else if (strcmp(argv[index], "--layer") == 0 &&
                   index + 1 < argc) {
            options.layer = parse_nonnegative_integer(
                argv[index + 1],
                "layer"
            );
            index += 2;
        } else if (strcmp(argv[index], "--operation") == 0 &&
                   index + 1 < argc) {
            options.operation = parse_operation(argv[index + 1]);
            index += 2;
        } else if (strcmp(argv[index], "--resume") == 0) {
            options.resume = true;
            index++;
        } else {
            fail("unrecognized cps_affine_spectrum option");
        }
    }
    if (options.corpus_directory == NULL || options.matrix_path == NULL ||
        options.basis_path == NULL) {
        fail("--corpus-dir, --matrix, and --basis are required");
    }
    return options;
}

static void path_list_add(PathList *list, const char *path) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity == 0 ? 32 : list->capacity * 2;
        char **items = realloc(list->items, capacity * sizeof(*items));
        if (items == NULL) fail("path-list allocation failed");
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count] = strdup(path);
    if (list->items[list->count] == NULL) fail("path copy failed");
    list->count++;
}

static bool has_prompt_basename(const char *path) {
    const char *slash = strrchr(path, '/');
    const char *name = slash == NULL ? path : slash + 1;
    return strcmp(name, "prompt.txt") == 0;
}

static void collect_prompt_paths(PathList *list, const char *directory) {
    DIR *stream = opendir(directory);
    if (stream == NULL) {
        fprintf(stderr, "could not open corpus directory %s\n", directory);
        exit(EXIT_FAILURE);
    }
    struct dirent *entry;
    while ((entry = readdir(stream)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        size_t path_length = strlen(directory) + strlen(entry->d_name) + 2;
        char *path = checked_calloc(path_length, sizeof(*path));
        snprintf(path, path_length, "%s/%s", directory, entry->d_name);
        struct stat status;
        if (lstat(path, &status) != 0) {
            fprintf(stderr, "could not stat corpus path %s\n", path);
            exit(EXIT_FAILURE);
        }
        if (S_ISDIR(status.st_mode)) {
            collect_prompt_paths(list, path);
        } else if (S_ISREG(status.st_mode) && has_prompt_basename(path)) {
            path_list_add(list, path);
        }
        free(path);
    }
    if (closedir(stream) != 0) fail("could not close corpus directory");
}

static int compare_paths(const void *left, const void *right) {
    const char *const *left_path = left;
    const char *const *right_path = right;
    return strcmp(*left_path, *right_path);
}

static void free_path_list(PathList *list) {
    for (size_t index = 0; index < list->count; index++) {
        free(list->items[index]);
    }
    free(list->items);
    memset(list, 0, sizeof(*list));
}

static char *read_text_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) fail("could not open corpus text");
    if (fseek(file, 0, SEEK_END) != 0) fail("could not seek corpus text");
    long length = ftell(file);
    if (length < 0) fail("could not size corpus text");
    if (fseek(file, 0, SEEK_SET) != 0) fail("could not rewind corpus text");
    char *text = checked_calloc((size_t)length + 1, sizeof(*text));
    if (length > 0 && fread(text, 1, (size_t)length, file) !=
                      (size_t)length) {
        fail("could not read corpus text");
    }
    if (fclose(file) != 0) fail("could not close corpus text");
    return text;
}

static size_t checked_pair_mapping_bytes(int samples, int width) {
    size_t values = (size_t)samples * 2U * (size_t)width;
    if (width <= 0 || samples <= 0 || values >
        (SIZE_MAX - sizeof(AffinePairHeader)) / sizeof(float)) {
        fail("affine pair matrix size overflow");
    }
    return sizeof(AffinePairHeader) + values * sizeof(float);
}

static bool affine_header_matches(
    const AffinePairHeader *header,
    const SpectrumOptions *options,
    int width
) {
    return memcmp(header->magic, "CPSAFF1", 8) == 0 &&
        header->version == 1 &&
        header->header_bytes == sizeof(*header) &&
        header->sample_capacity == (uint32_t)options->samples &&
        header->root_width == (uint32_t)width &&
        header->positions == (uint32_t)options->positions &&
        header->layer == (uint32_t)options->layer &&
        header->operation == (uint32_t)options->operation &&
        header->point_count_per_sample == 2 &&
        header->implicit_homogeneous_coordinate == 1;
}

static AffinePairFile open_pair_file(
    const SpectrumOptions *options,
    int width
) {
    AffinePairFile file = {.descriptor = -1};
    file.mapping_bytes = checked_pair_mapping_bytes(options->samples, width);
    int flags = O_RDWR | O_CREAT;
    if (!options->resume) flags |= O_TRUNC;
    file.descriptor = open(options->matrix_path, flags, 0644);
    if (file.descriptor < 0) fail("could not open affine pair matrix");
    struct stat status;
    if (fstat(file.descriptor, &status) != 0) {
        fail("could not stat affine pair matrix");
    }
    bool existing = options->resume &&
        (size_t)status.st_size == file.mapping_bytes;
    if (!existing && ftruncate(file.descriptor, (off_t)file.mapping_bytes) != 0) {
        fail("could not size affine pair matrix");
    }
    file.mapping = mmap(
        NULL,
        file.mapping_bytes,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        file.descriptor,
        0
    );
    if (file.mapping == MAP_FAILED) fail("could not map affine pair matrix");
    file.header = file.mapping;
    file.pairs = (float *)((unsigned char *)file.mapping +
        sizeof(*file.header));
    if (existing) {
        if (!affine_header_matches(file.header, options, width) ||
            file.header->samples_written > file.header->sample_capacity) {
            fail("resume matrix metadata does not match this run");
        }
    } else {
        memset(file.mapping, 0, file.mapping_bytes);
        memcpy(file.header->magic, "CPSAFF1", 8);
        file.header->version = 1;
        file.header->header_bytes = sizeof(*file.header);
        file.header->sample_capacity = (uint32_t)options->samples;
        file.header->root_width = (uint32_t)width;
        file.header->positions = (uint32_t)options->positions;
        file.header->layer = (uint32_t)options->layer;
        file.header->operation = (uint32_t)options->operation;
        file.header->point_count_per_sample = 2;
        file.header->implicit_homogeneous_coordinate = 1;
        if (msync(file.mapping, sizeof(*file.header), MS_SYNC) != 0) {
            fail("could not initialize affine pair matrix");
        }
    }
    return file;
}

static void append_pair_row(
    AffinePairFile *file,
    const float *after,
    const float *bypass,
    int width
) {
    uint32_t row = file->header->samples_written;
    if (row >= file->header->sample_capacity) {
        fail("affine pair matrix capacity exceeded");
    }
    float *record = file->pairs + (size_t)row * 2U * width;
    memcpy(record, after, (size_t)width * sizeof(*record));
    memcpy(record + width, bypass, (size_t)width * sizeof(*record));
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) fail("could not determine mapping page size");
    uintptr_t record_address = (uintptr_t)record;
    uintptr_t page_address = record_address -
        record_address % (uintptr_t)page_size;
    size_t prefix = (size_t)(record_address - page_address);
    size_t record_bytes = (size_t)2U * width * sizeof(*record);
    size_t flush_bytes = prefix + record_bytes;
    if (msync((void *)page_address, flush_bytes, MS_SYNC) != 0) {
        fail("could not flush affine pair values");
    }
    file->header->samples_written = row + 1;
    if (msync(file->mapping, sizeof(*file->header), MS_SYNC) != 0) {
        fail("could not flush affine pair header");
    }
}

static void close_pair_file(AffinePairFile *file) {
    if (msync(file->mapping, file->mapping_bytes, MS_SYNC) != 0) {
        fail("could not finalize affine pair matrix");
    }
    if (munmap(file->mapping, file->mapping_bytes) != 0) {
        fail("could not unmap affine pair matrix");
    }
    if (close(file->descriptor) != 0) fail("could not close affine pair matrix");
    memset(file, 0, sizeof(*file));
    file->descriptor = -1;
}

static SpectrumTerm build_spectrum_term(
    Transformer *transformer,
    int positions
) {
    SpectrumTerm term = {
        .transformer = transformer,
        .layers = transformer->config.n_layers,
        .positions = positions,
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

static void free_spectrum_term(SpectrumTerm *term) {
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

static FrontierMap selected_map(
    SpectrumTerm *term,
    SpectrumOperation operation,
    int layer
) {
    switch (operation) {
        case SPECTRUM_ATTENTION: return term->attention_maps[layer];
        case SPECTRUM_FFN: return term->ffn_maps[layer];
        case SPECTRUM_LAYER: return term->layer_maps[layer];
        case SPECTRUM_FINAL_RMS: return term->final_rms_map;
    }
    fail("invalid selected map");
    return term->layer_maps[0];
}

static Continuation selected_suffix(
    SpectrumTerm *term,
    SpectrumOperation operation,
    int layer
) {
    switch (operation) {
        case SPECTRUM_ATTENTION:
            return term->post_attention_suffixes[layer];
        case SPECTRUM_FFN:
        case SPECTRUM_LAYER:
            return term->layer_suffixes[layer + 1];
        case SPECTRUM_FINAL_RMS:
            return term->root_identity;
    }
    fail("invalid selected continuation");
    return term->root_identity;
}

static const float *selected_input(
    ContextFrontiers *capture,
    SpectrumOperation operation,
    int layer,
    int layers
) {
    switch (operation) {
        case SPECTRUM_ATTENTION:
        case SPECTRUM_LAYER:
            return layer_frontier(capture, layer);
        case SPECTRUM_FFN:
            return post_attention_frontier(capture, layer);
        case SPECTRUM_FINAL_RMS:
            return layer_frontier(capture, layers);
    }
    fail("invalid selected input");
    return NULL;
}

static bool token_window_seen(
    const int *seen,
    int seen_count,
    int positions,
    const int *tokens
) {
    for (int sample = 0; sample < seen_count; sample++) {
        if (memcmp(
                seen + (size_t)sample * positions,
                tokens,
                (size_t)positions * sizeof(*tokens)
            ) == 0) {
            return true;
        }
    }
    return false;
}

static void write_sample_trace(
    FILE *trace,
    Tokenizer *tokenizer,
    int sample,
    const char *path,
    int start,
    const int *tokens,
    int positions,
    double displacement
) {
    if (trace == NULL) return;
    fprintf(
        trace,
        "{\"kind\":\"affine_sample\",\"sample\":%d,\"source\":",
        sample
    );
    fprint_json_string(trace, path);
    fprintf(trace, ",\"token_start\":%d,\"tokens\":[", start);
    for (int position = 0; position < positions; position++) {
        if (position != 0) fputc(',', trace);
        fprintf(trace, "%d", tokens[position]);
    }
    fputs("],\"pieces\":[", trace);
    for (int position = 0; position < positions; position++) {
        if (position != 0) fputc(',', trace);
        int previous = position == 0 ? 0 : tokens[position - 1];
        fprint_json_string(
            trace,
            decode(tokenizer, previous, tokens[position])
        );
    }
    fprintf(
        trace,
        "],\"root_displacement_l2\":%.17g}\n",
        displacement
    );
    fflush(trace);
}

static double maximum_basis_residual(
    const AffinePairFile *pairs,
    const float *right_vectors,
    int rank,
    int width
) {
    int rows = (int)pairs->header->samples_written;
    int nullity = width - rank;
    double maximum = 0.0;
    for (int mode = 0; mode < nullity; mode++) {
        int vector_row = rank + mode;
        double square = 0.0;
        for (int sample = 0; sample < rows; sample++) {
            const float *record = pairs->pairs +
                (size_t)sample * 2U * width;
            double value = 0.0;
            for (int coordinate = 0; coordinate < width; coordinate++) {
                float covector = right_vectors[
                    vector_row + (size_t)coordinate * width
                ];
                value += ((double)record[coordinate] -
                    record[width + coordinate]) * covector;
            }
            square += value * value;
        }
        double residual = sqrt(square);
        if (residual > maximum) maximum = residual;
    }
    return maximum;
}

static void write_fixed_basis(
    const char *path,
    const AffinePairFile *pairs,
    const float *right_vectors,
    const float *singular_values,
    int singular_count,
    int rank,
    float tolerance,
    double maximum_residual
) {
    int width = (int)pairs->header->root_width;
    int nullity = width - rank;
    FixedBasisHeader header = {0};
    memcpy(header.magic, "CPSBAS2", 8);
    header.version = 2;
    header.header_bytes = sizeof(header);
    header.samples = pairs->header->samples_written;
    header.root_width = (uint32_t)width;
    header.displacement_rank = (uint32_t)rank;
    header.displacement_nullity = (uint32_t)nullity;
    header.affine_constant_dimension = 1;
    header.singular_value_count = (uint32_t)singular_count;
    header.right_mode_count = (uint32_t)width;
    header.rank_tolerance = tolerance;
    header.largest_singular_value = singular_count == 0 ? 0.0f :
        singular_values[0];
    header.smallest_computed_singular_value = singular_count == 0 ? 0.0f :
        singular_values[singular_count - 1];
    header.maximum_basis_residual = (float)maximum_residual;
    FILE *file = fopen(path, "wb");
    if (file == NULL) fail("could not create fixed basis file");
    if (fwrite(&header, sizeof(header), 1, file) != 1) {
        fail("could not write fixed basis header");
    }
    if (fwrite(
            singular_values,
            sizeof(*singular_values),
            (size_t)singular_count,
            file
        ) != (size_t)singular_count) {
        fail("could not write displacement singular values");
    }
    float *row = checked_calloc((size_t)width, sizeof(*row));
    for (int vector_row = 0; vector_row < width; vector_row++) {
        for (int coordinate = 0; coordinate < width; coordinate++) {
            row[coordinate] = right_vectors[
                vector_row + (size_t)coordinate * width
            ];
        }
        if (fwrite(row, sizeof(*row), (size_t)width, file) !=
            (size_t)width) {
            fail("could not write fixed basis vector");
        }
        fflush(file);
    }
    free(row);
    if (fclose(file) != 0) fail("could not close fixed basis file");
}

static void analyze_fixed_displacements(
    AffinePairFile *pairs,
    const SpectrumOptions *options,
    FILE *trace
) {
    int rows = (int)pairs->header->samples_written;
    int width = (int)pairs->header->root_width;
    if (rows <= 0) fail("no affine samples were recorded");
    __LAPACK_int m = rows;
    __LAPACK_int n = width;
    __LAPACK_int lda = m;
    __LAPACK_int ldu = 1;
    __LAPACK_int ldvt = n;
    int singular_count = rows < width ? rows : width;
    float *displacements = checked_calloc(
        (size_t)rows * width,
        sizeof(*displacements)
    );
    for (int sample = 0; sample < rows; sample++) {
        const float *record = pairs->pairs +
            (size_t)sample * 2U * width;
        for (int coordinate = 0; coordinate < width; coordinate++) {
            displacements[sample + (size_t)coordinate * rows] =
                record[coordinate] - record[width + coordinate];
        }
    }
    float *singular_values = checked_calloc(
        (size_t)singular_count,
        sizeof(*singular_values)
    );
    float *right_vectors = checked_calloc(
        (size_t)width * width,
        sizeof(*right_vectors)
    );
    float unused_u = 0.0f;
    char jobu = 'N';
    char jobvt = 'A';
    __LAPACK_int lwork = -1;
    __LAPACK_int info = 0;
    float work_query = 0.0f;
    sgesvd_(
        &jobu,
        &jobvt,
        &m,
        &n,
        displacements,
        &lda,
        singular_values,
        &unused_u,
        &ldu,
        right_vectors,
        &ldvt,
        &work_query,
        &lwork,
        &info
    );
    if (info != 0) fail("LAPACK workspace query failed");
    lwork = (__LAPACK_int)ceilf(work_query);
    float *work = checked_calloc((size_t)lwork, sizeof(*work));
    sgesvd_(
        &jobu,
        &jobvt,
        &m,
        &n,
        displacements,
        &lda,
        singular_values,
        &unused_u,
        &ldu,
        right_vectors,
        &ldvt,
        work,
        &lwork,
        &info
    );
    if (info != 0) fail("LAPACK SVD failed");
    float largest = singular_count == 0 ? 0.0f : singular_values[0];
    float tolerance = fmaxf(rows, width) * FLT_EPSILON * largest;
    int rank = 0;
    while (rank < singular_count && singular_values[rank] > tolerance) {
        rank++;
    }
    int nullity = width - rank;
    double maximum_residual = maximum_basis_residual(
        pairs,
        right_vectors,
        rank,
        width
    );
    write_fixed_basis(
        options->basis_path,
        pairs,
        right_vectors,
        singular_values,
        singular_count,
        rank,
        tolerance,
        maximum_residual
    );
    printf(
        "fixed_affine_space samples=%d root_width=%d displacement_rank=%d "
        "covector_nullity=%d constant_dimension=1 total_dimension=%d "
        "rank_tolerance=%.8g singular_range=[%.8g,%.8g] "
        "max_basis_residual=%.8g coverage_limited=%s\n",
        rows,
        width,
        rank,
        nullity,
        nullity + 1,
        tolerance,
        singular_count == 0 ? 0.0f :
            singular_values[singular_count - 1],
        largest,
        maximum_residual,
        rows < width ? "true" : "false"
    );
    if (trace != NULL) {
        fprintf(
            trace,
            "{\"kind\":\"fixed_affine_space\",\"samples\":%d,"
            "\"root_width\":%d,\"displacement_rank\":%d,"
            "\"covector_nullity\":%d,\"constant_dimension\":1,"
            "\"total_dimension\":%d,\"rank_tolerance\":%.17g,"
            "\"largest_singular_value\":%.17g,"
            "\"smallest_computed_singular_value\":%.17g,"
            "\"maximum_basis_residual\":%.17g,"
            "\"coverage_limited\":%s,\"singular_tail\":[",
            rows,
            width,
            rank,
            nullity,
            nullity + 1,
            tolerance,
            largest,
            singular_count == 0 ? 0.0f :
                singular_values[singular_count - 1],
            maximum_residual,
            rows < width ? "true" : "false"
        );
        int tail_count = singular_count < 16 ? singular_count : 16;
        for (int tail = tail_count; tail > 0; tail--) {
            if (tail != tail_count) fputc(',', trace);
            fprintf(trace, "%.17g", singular_values[singular_count - tail]);
        }
        fputs("]}\n", trace);
        fflush(trace);
    }
    free(work);
    free(right_vectors);
    free(singular_values);
    free(displacements);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(
            stderr,
            "usage: %s CHECKPOINT TOKENIZER --corpus-dir DIR --matrix PATH "
            "--basis PATH [--positions N] [--samples N] [--layer N] "
            "[--operation attention|ffn|layer|final-rms] [--trace PATH] "
            "[--resume]\n",
            argv[0]
        );
        return EXIT_FAILURE;
    }
    SpectrumOptions options = parse_spectrum_options(argc, argv);
    Transformer transformer;
    build_transformer(&transformer, argv[1]);
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, argv[2], transformer.config.vocab_size);
    if (options.positions > transformer.config.seq_len) {
        fail("spectral context exceeds model sequence length");
    }
    if (options.operation != SPECTRUM_FINAL_RMS &&
        options.layer >= transformer.config.n_layers) {
        fail("spectral layer is outside the model");
    }
    if (options.operation == SPECTRUM_FINAL_RMS) {
        options.layer = transformer.config.n_layers;
    }
    SpectrumTerm term = build_spectrum_term(&transformer, options.positions);
    if (options.samples == 0) {
        options.samples = term.frontier_width + 64;
    }
    FrontierMap map = selected_map(&term, options.operation, options.layer);
    Continuation suffix = selected_suffix(
        &term,
        options.operation,
        options.layer
    );
    if (map.input_width != map.output_width ||
        map.output_width != suffix.input_width) {
        fail("selected fixed-spectrum term is not endomorphic");
    }
    AffinePairFile pairs = open_pair_file(&options, term.frontier_width);
    FILE *trace = NULL;
    if (options.trace_path != NULL) {
        trace = fopen(options.trace_path, options.resume ? "ab" : "wb");
        if (trace == NULL) fail("could not open affine trace");
        if (!options.resume) {
            fprintf(
                trace,
                "{\"kind\":\"affine_meta\",\"schema_version\":1,"
                "\"semantics\":\"torsor_displacements_with_homogeneous_points\","
                "\"operation\":\"%s\",\"layer\":%d,"
                "\"positions\":%d,\"root_width\":%d,"
                "\"sample_capacity\":%d}\n",
                operation_name(options.operation),
                options.layer,
                options.positions,
                term.frontier_width,
                options.samples
            );
            fflush(trace);
        }
    }

    PathList paths = {0};
    collect_prompt_paths(&paths, options.corpus_directory);
    qsort(paths.items, paths.count, sizeof(*paths.items), compare_paths);
    if (paths.count == 0) fail("corpus directory contained no prompt.txt files");
    int *seen_tokens = checked_calloc(
        (size_t)options.samples * options.positions,
        sizeof(*seen_tokens)
    );
    int unique_windows = 0;
    float *mapped = checked_calloc(
        (size_t)term.frontier_width,
        sizeof(*mapped)
    );
    float *after = checked_calloc(
        (size_t)term.frontier_width,
        sizeof(*after)
    );
    float *bypass = checked_calloc(
        (size_t)term.frontier_width,
        sizeof(*bypass)
    );
    uint32_t resume_rows = pairs.header->samples_written;
    printf(
        "affine_sampling operation=%s layer=%d positions=%d root_width=%d "
        "target_samples=%d resume_rows=%u corpus_files=%zu\n",
        operation_name(options.operation),
        options.layer,
        options.positions,
        term.frontier_width,
        options.samples,
        resume_rows,
        paths.count
    );
    fflush(stdout);
    for (size_t path_index = 0;
         path_index < paths.count && unique_windows < options.samples;
         path_index++) {
        char *text = read_text_file(paths.items[path_index]);
        size_t token_capacity = strlen(text) + 3U;
        int *tokens = checked_calloc(token_capacity, sizeof(*tokens));
        int token_count = 0;
        encode(&tokenizer, text, 1, 0, tokens, &token_count);
        free(text);
        for (int start = 1;
             start + options.positions <= token_count &&
                 unique_windows < options.samples;
             start++) {
            const int *window = tokens + start;
            if (token_window_seen(
                    seen_tokens,
                    unique_windows,
                    options.positions,
                    window
                )) {
                continue;
            }
            memcpy(
                seen_tokens + (size_t)unique_windows * options.positions,
                window,
                (size_t)options.positions * sizeof(*window)
            );
            int sample = unique_windows++;
            if ((uint32_t)sample < resume_rows) continue;
            EncodedContext context = {
                .text = paths.items[path_index],
                .tokens = (int *)window,
                .count = options.positions
            };
            ContextFrontiers capture = allocate_frontiers(
                term.layers,
                term.frontier_width
            );
            capture_context_frontiers(
                &transformer,
                &context,
                term.runtimes,
                &capture
            );
            const float *input = selected_input(
                &capture,
                options.operation,
                options.layer,
                term.layers
            );
            map.apply(map.environment, input, mapped);
            suffix.apply(suffix.environment, mapped, after);
            suffix.apply(suffix.environment, input, bypass);
            double displacement = difference_l2(
                after,
                bypass,
                term.frontier_width
            );
            append_pair_row(
                &pairs,
                after,
                bypass,
                term.frontier_width
            );
            write_sample_trace(
                trace,
                &tokenizer,
                sample,
                paths.items[path_index],
                start,
                window,
                options.positions,
                displacement
            );
            free_frontiers(&capture);
            if ((sample + 1) % 32 == 0 || sample + 1 == options.samples) {
                printf(
                    "affine_progress samples=%d/%d latest_displacement=%.8g\n",
                    sample + 1,
                    options.samples,
                    displacement
                );
                fflush(stdout);
            }
        }
        free(tokens);
    }
    free(bypass);
    free(after);
    free(mapped);
    free(seen_tokens);
    free_path_list(&paths);
    if ((int)pairs.header->samples_written != options.samples) {
        fail("corpus did not supply enough unique token windows");
    }
    analyze_fixed_displacements(&pairs, &options, trace);
    if (trace != NULL && fclose(trace) != 0) fail("could not close affine trace");
    close_pair_file(&pairs);
    free_spectrum_term(&term);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return EXIT_SUCCESS;
}
