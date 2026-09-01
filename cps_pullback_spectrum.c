/*
 * Root-reachable continuation spectrum for the exact CPS transformer term.
 *
 * Let F be one endomorphic transformer operation and let k be its mechanically
 * composed suffix to the complete post-final-RMS hidden frontier.  The scalar
 * coordinate continuations of k generate the block-Krylov dictionary
 *
 *   1, k, U_F k, ..., U_F^(p-1) k,       U_F(g) = g . F.
 *
 * Rows in the evaluation file are contextual evaluations of those functions:
 * hidden states remain points.  Linear algebra is performed only between
 * columns of functions, with an explicit homogeneous constant.  A fitted
 * finite operator is not called closed: closure is measured again on held-out
 * contexts, and every eigenmode records its held-out eigen-equation residual.
 */

#define CPS_AFFINE_SPECTRUM_NO_MAIN
#include "cps_affine_spectrum.c"

typedef enum {
    ROOT_SCOPE_ALL,
    ROOT_SCOPE_LAST
} RootScope;

typedef struct {
    const char *corpus_directory;
    const char *evaluation_path;
    const char *spectrum_path;
    const char *trace_path;
    int positions;
    int samples;
    int fit_samples;
    int pullback_depth;
    int dictionary_depth;
    int layer;
    SpectrumOperation operation;
    RootScope root_scope;
    bool resume;
    bool analysis_only;
} PullbackOptions;

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t sample_capacity;
    uint32_t samples_written;
    uint32_t positions;
    uint32_t frontier_width;
    uint32_t pullback_depth;
    uint32_t layer;
    uint32_t operation;
    uint32_t root_points_per_sample;
    uint32_t implicit_homogeneous_coordinate;
    uint64_t values_per_sample;
    uint32_t reserved[6];
} PullbackEvaluationHeader;

typedef struct {
    int descriptor;
    size_t mapping_bytes;
    void *mapping;
    PullbackEvaluationHeader *header;
    float *values;
} PullbackEvaluationFile;

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t root_scope;
    uint32_t positions;
    uint32_t observation_width;
    uint32_t pullback_depth;
    uint32_t dictionary_columns;
    uint32_t total_samples;
    uint32_t fit_samples;
    uint32_t validation_samples;
    uint32_t sampled_rank;
    uint32_t fixed_dimension;
    uint32_t singular_value_count;
    uint32_t eigenvalue_count;
    uint32_t dictionary_basis_rows;
    uint32_t dictionary_basis_columns;
    uint32_t storage_order;
    double rank_tolerance;
    double largest_singular_value;
    double smallest_retained_singular_value;
    double fit_representation_relative;
    double fit_descent_relative;
    double validation_dictionary_relative;
    double fixed_tolerance;
    uint32_t reserved[6];
} PullbackSpectrumHeader;

typedef struct {
    int index;
    int pair_index;
    double real;
    double imaginary;
    double magnitude;
    double distance_to_one;
    double fit_relative_residual;
    double validation_relative_residual;
    double constant_variation;
} ModeDiagnostic;

static RootScope parse_root_scope(const char *text) {
    if (strcmp(text, "all") == 0) return ROOT_SCOPE_ALL;
    if (strcmp(text, "last") == 0) return ROOT_SCOPE_LAST;
    fail("root scope must be all or last");
    return ROOT_SCOPE_ALL;
}

static const char *root_scope_name(RootScope scope) {
    switch (scope) {
        case ROOT_SCOPE_ALL: return "all";
        case ROOT_SCOPE_LAST: return "last";
    }
    fail("invalid root scope");
    return "invalid";
}

static PullbackOptions parse_pullback_options(int argc, char **argv) {
    PullbackOptions options = {
        .operation = SPECTRUM_ATTENTION,
        .root_scope = ROOT_SCOPE_ALL
    };
    for (int index = 3; index < argc;) {
        if (strcmp(argv[index], "--corpus-dir") == 0 && index + 1 < argc) {
            options.corpus_directory = argv[index + 1];
            index += 2;
        } else if (strcmp(argv[index], "--evaluations") == 0 &&
                   index + 1 < argc) {
            options.evaluation_path = argv[index + 1];
            index += 2;
        } else if (strcmp(argv[index], "--spectrum") == 0 &&
                   index + 1 < argc) {
            options.spectrum_path = argv[index + 1];
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
        } else if (strcmp(argv[index], "--fit-samples") == 0 &&
                   index + 1 < argc) {
            options.fit_samples = parse_positive_integer(
                argv[index + 1],
                "fit samples"
            );
            index += 2;
        } else if (strcmp(argv[index], "--pullback-depth") == 0 &&
                   index + 1 < argc) {
            options.pullback_depth = parse_positive_integer(
                argv[index + 1],
                "pullback depth"
            );
            index += 2;
        } else if (strcmp(argv[index], "--dictionary-depth") == 0 &&
                   index + 1 < argc) {
            options.dictionary_depth = parse_positive_integer(
                argv[index + 1],
                "dictionary depth"
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
        } else if (strcmp(argv[index], "--root") == 0 &&
                   index + 1 < argc) {
            options.root_scope = parse_root_scope(argv[index + 1]);
            index += 2;
        } else if (strcmp(argv[index], "--resume") == 0) {
            options.resume = true;
            index++;
        } else if (strcmp(argv[index], "--analysis-only") == 0) {
            options.resume = true;
            options.analysis_only = true;
            index++;
        } else {
            fail("unrecognized cps_pullback_spectrum option");
        }
    }
    if (options.corpus_directory == NULL ||
        options.evaluation_path == NULL ||
        options.spectrum_path == NULL ||
        options.positions <= 0 || options.samples <= 0 ||
        options.fit_samples <= 0 || options.pullback_depth <= 0) {
        fail("--corpus-dir, --evaluations, --spectrum, --positions, "
             "--samples, --fit-samples, and --pullback-depth are required");
    }
    if (options.fit_samples >= options.samples) {
        fail("fit samples must leave at least one held-out sample");
    }
    if (options.dictionary_depth == 0) {
        options.dictionary_depth = options.pullback_depth;
    }
    if (options.dictionary_depth > options.pullback_depth) {
        fail("dictionary depth cannot exceed recorded pullback depth");
    }
    return options;
}

static size_t checked_evaluation_mapping_bytes(
    int samples,
    int frontier_width,
    int pullback_depth
) {
    size_t root_points = (size_t)pullback_depth + 1U;
    if (samples <= 0 || frontier_width <= 0 ||
        root_points > SIZE_MAX / (size_t)frontier_width) {
        fail("pullback evaluation matrix size overflow");
    }
    size_t values_per_sample = root_points * (size_t)frontier_width;
    if ((size_t)samples >
        (SIZE_MAX - sizeof(PullbackEvaluationHeader)) /
            sizeof(float) / values_per_sample) {
        fail("pullback evaluation matrix size overflow");
    }
    return sizeof(PullbackEvaluationHeader) +
        (size_t)samples * values_per_sample * sizeof(float);
}

static bool evaluation_header_matches(
    const PullbackEvaluationHeader *header,
    const PullbackOptions *options,
    int frontier_width
) {
    return memcmp(header->magic, "CPSKRY1", 8) == 0 &&
        header->version == 1 &&
        header->header_bytes == sizeof(*header) &&
        header->positions == (uint32_t)options->positions &&
        header->frontier_width == (uint32_t)frontier_width &&
        header->pullback_depth == (uint32_t)options->pullback_depth &&
        header->layer == (uint32_t)options->layer &&
        header->operation == (uint32_t)options->operation &&
        header->root_points_per_sample ==
            (uint32_t)options->pullback_depth + 1U &&
        header->implicit_homogeneous_coordinate == 1 &&
        header->values_per_sample ==
            ((uint64_t)options->pullback_depth + 1U) *
                (uint64_t)frontier_width;
}

static PullbackEvaluationFile open_evaluation_file(
    const PullbackOptions *options,
    int frontier_width
) {
    PullbackEvaluationFile file = {.descriptor = -1};
    file.mapping_bytes = checked_evaluation_mapping_bytes(
        options->samples,
        frontier_width,
        options->pullback_depth
    );
    int flags = O_RDWR | O_CREAT;
    if (!options->resume) flags |= O_TRUNC;
    file.descriptor = open(options->evaluation_path, flags, 0644);
    if (file.descriptor < 0) fail("could not open pullback evaluations");
    struct stat status;
    if (fstat(file.descriptor, &status) != 0) {
        fail("could not stat pullback evaluations");
    }
    bool existing = false;
    PullbackEvaluationHeader prior_header = {0};
    if (options->resume) {
        if ((size_t)status.st_size < sizeof(prior_header) ||
            pread(
                file.descriptor,
                &prior_header,
                sizeof(prior_header),
                0
            ) != sizeof(prior_header)) {
            fail("resume evaluation file has no complete header");
        }
        if (!evaluation_header_matches(
                &prior_header,
                options,
                frontier_width
            )) {
            fail("resume evaluation metadata does not match this run");
        }
        size_t prior_bytes = checked_evaluation_mapping_bytes(
            (int)prior_header.sample_capacity,
            frontier_width,
            options->pullback_depth
        );
        if ((size_t)status.st_size != prior_bytes ||
            prior_header.samples_written > prior_header.sample_capacity) {
            fail("resume evaluation file has inconsistent size");
        }
        if (options->samples < (int)prior_header.sample_capacity) {
            fail("pullback evaluation capacity may grow but not shrink");
        }
        existing = true;
    }
    if ((!existing || file.mapping_bytes != (size_t)status.st_size) &&
        ftruncate(file.descriptor, (off_t)file.mapping_bytes) != 0) {
        fail("could not size pullback evaluations");
    }
    file.mapping = mmap(
        NULL,
        file.mapping_bytes,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        file.descriptor,
        0
    );
    if (file.mapping == MAP_FAILED) fail("could not map pullback evaluations");
    file.header = file.mapping;
    file.values = (float *)((unsigned char *)file.mapping +
        sizeof(*file.header));
    if (existing) {
        if (file.header->sample_capacity != (uint32_t)options->samples) {
            file.header->sample_capacity = (uint32_t)options->samples;
            if (msync(file.mapping, sizeof(*file.header), MS_SYNC) != 0) {
                fail("could not persist grown evaluation capacity");
            }
        }
    } else {
        memset(file.mapping, 0, file.mapping_bytes);
        memcpy(file.header->magic, "CPSKRY1", 8);
        file.header->version = 1;
        file.header->header_bytes = sizeof(*file.header);
        file.header->sample_capacity = (uint32_t)options->samples;
        file.header->positions = (uint32_t)options->positions;
        file.header->frontier_width = (uint32_t)frontier_width;
        file.header->pullback_depth =
            (uint32_t)options->pullback_depth;
        file.header->layer = (uint32_t)options->layer;
        file.header->operation = (uint32_t)options->operation;
        file.header->root_points_per_sample =
            (uint32_t)options->pullback_depth + 1U;
        file.header->implicit_homogeneous_coordinate = 1;
        file.header->values_per_sample =
            ((uint64_t)options->pullback_depth + 1U) *
                (uint64_t)frontier_width;
        if (msync(file.mapping, sizeof(*file.header), MS_SYNC) != 0) {
            fail("could not initialize pullback evaluation header");
        }
    }
    return file;
}

static void append_evaluation_row(
    PullbackEvaluationFile *file,
    const float *values
) {
    uint32_t row = file->header->samples_written;
    if (row >= file->header->sample_capacity) {
        fail("pullback evaluation capacity exceeded");
    }
    size_t count = (size_t)file->header->values_per_sample;
    float *destination = file->values + (size_t)row * count;
    memcpy(destination, values, count * sizeof(*destination));
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) fail("could not determine mapping page size");
    uintptr_t address = (uintptr_t)destination;
    uintptr_t page = address - address % (uintptr_t)page_size;
    size_t prefix = (size_t)(address - page);
    if (msync(
            (void *)page,
            prefix + count * sizeof(*destination),
            MS_SYNC
        ) != 0) {
        fail("could not flush pullback evaluation values");
    }
    file->header->samples_written = row + 1U;
    if (msync(file->mapping, sizeof(*file->header), MS_SYNC) != 0) {
        fail("could not flush pullback evaluation header");
    }
}

static void close_evaluation_file(PullbackEvaluationFile *file) {
    if (msync(file->mapping, file->mapping_bytes, MS_SYNC) != 0) {
        fail("could not finalize pullback evaluations");
    }
    if (munmap(file->mapping, file->mapping_bytes) != 0) {
        fail("could not unmap pullback evaluations");
    }
    if (close(file->descriptor) != 0) {
        fail("could not close pullback evaluations");
    }
    memset(file, 0, sizeof(*file));
    file->descriptor = -1;
}

static void write_pullback_sample_trace(
    FILE *trace,
    Tokenizer *tokenizer,
    int sample,
    const char *path,
    int start,
    const int *tokens,
    int positions,
    const float *root_points,
    int frontier_width,
    int pullback_depth
) {
    if (trace == NULL) return;
    fprintf(
        trace,
        "{\"kind\":\"pullback_sample\",\"sample\":%d,\"source\":",
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
    fputs("],\"successive_root_displacement_l2\":[", trace);
    for (int depth = 0; depth < pullback_depth; depth++) {
        if (depth != 0) fputc(',', trace);
        const float *left = root_points + (size_t)depth * frontier_width;
        const float *right = left + frontier_width;
        fprintf(
            trace,
            "%.17g",
            difference_l2(left, right, frontier_width)
        );
    }
    fputs("]}\n", trace);
    fflush(trace);
}

static size_t checked_matrix_values(int rows, int columns) {
    if (rows <= 0 || columns <= 0 ||
        (size_t)rows > SIZE_MAX / (size_t)columns) {
        fail("continuation evaluation matrix size overflow");
    }
    return (size_t)rows * (size_t)columns;
}

static int observation_width(
    const PullbackOptions *options,
    const Transformer *transformer
) {
    if (options->root_scope == ROOT_SCOPE_LAST) {
        return transformer->config.dim;
    }
    return options->positions * transformer->config.dim;
}

static int observation_offset(
    const PullbackOptions *options,
    const Transformer *transformer
) {
    if (options->root_scope == ROOT_SCOPE_LAST) {
        return (options->positions - 1) * transformer->config.dim;
    }
    return 0;
}

static void build_dictionary_pair(
    const PullbackEvaluationFile *evaluations,
    const PullbackOptions *options,
    const Transformer *transformer,
    int sample_start,
    int rows,
    double *input_dictionary,
    double *pulled_dictionary
) {
    int frontier_width = (int)evaluations->header->frontier_width;
    int width = observation_width(options, transformer);
    int offset = observation_offset(options, transformer);
    int columns = 1 + options->dictionary_depth * width;
    size_t values_per_sample =
        (size_t)evaluations->header->values_per_sample;
    for (int row = 0; row < rows; row++) {
        const float *points = evaluations->values +
            (size_t)(sample_start + row) * values_per_sample;
        input_dictionary[row] = 1.0;
        pulled_dictionary[row] = 1.0;
        for (int depth = 0; depth < options->dictionary_depth; depth++) {
            const float *before = points +
                (size_t)depth * frontier_width + offset;
            const float *after = before + frontier_width;
            int first_column = 1 + depth * width;
            for (int coordinate = 0; coordinate < width; coordinate++) {
                size_t index = (size_t)row +
                    (size_t)(first_column + coordinate) * rows;
                input_dictionary[index] = before[coordinate];
                pulled_dictionary[index] = after[coordinate];
            }
        }
    }
    (void)columns;
}

static double squared_norm(const double *values, size_t count) {
    double result = 0.0;
    for (size_t index = 0; index < count; index++) {
        result += values[index] * values[index];
    }
    return result;
}

static void checked_fwrite(
    const void *values,
    size_t width,
    size_t count,
    FILE *file,
    const char *description
) {
    if (count != 0 && fwrite(values, width, count, file) != count) {
        fprintf(stderr, "could not write %s\n", description);
        exit(EXIT_FAILURE);
    }
}

static double real_matrix_eigen_residual(
    const double *matrix,
    int width,
    const double *right_vectors,
    int vector,
    double eigenvalue
) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (int row = 0; row < width; row++) {
        double image = 0.0;
        for (int column = 0; column < width; column++) {
            image += matrix[row + (size_t)column * width] *
                right_vectors[column + (size_t)vector * width];
        }
        double expected = eigenvalue *
            right_vectors[row + (size_t)vector * width];
        double difference = image - expected;
        numerator += difference * difference;
        denominator += image * image;
    }
    return denominator == 0.0 ? sqrt(numerator) :
        sqrt(numerator / denominator);
}

static double complex_matrix_eigen_residual(
    const double *matrix,
    int width,
    const double *right_vectors,
    int real_vector,
    double real_eigenvalue,
    double imaginary_eigenvalue
) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (int row = 0; row < width; row++) {
        double image_real = 0.0;
        double image_imaginary = 0.0;
        for (int column = 0; column < width; column++) {
            double entry = matrix[row + (size_t)column * width];
            image_real += entry * right_vectors[
                column + (size_t)real_vector * width
            ];
            image_imaginary += entry * right_vectors[
                column + (size_t)(real_vector + 1) * width
            ];
        }
        double vector_real = right_vectors[
            row + (size_t)real_vector * width
        ];
        double vector_imaginary = right_vectors[
            row + (size_t)(real_vector + 1) * width
        ];
        double expected_real = real_eigenvalue * vector_real -
            imaginary_eigenvalue * vector_imaginary;
        double expected_imaginary = imaginary_eigenvalue * vector_real +
            real_eigenvalue * vector_imaginary;
        double difference_real = image_real - expected_real;
        double difference_imaginary = image_imaginary - expected_imaginary;
        numerator += difference_real * difference_real +
            difference_imaginary * difference_imaginary;
        denominator += image_real * image_real +
            image_imaginary * image_imaginary;
    }
    return denominator == 0.0 ? sqrt(numerator) :
        sqrt(numerator / denominator);
}

static double real_function_eigen_residual(
    const double *input_basis,
    const double *pulled_basis,
    int rows,
    int rank,
    const double *right_vectors,
    int vector,
    double eigenvalue,
    double *constant_variation
) {
    double numerator = 0.0;
    double denominator = 0.0;
    double mean = 0.0;
    double input_square = 0.0;
    for (int row = 0; row < rows; row++) {
        double input = 0.0;
        double pulled = 0.0;
        for (int coordinate = 0; coordinate < rank; coordinate++) {
            double coefficient = right_vectors[
                coordinate + (size_t)vector * rank
            ];
            input += input_basis[row + (size_t)coordinate * rows] *
                coefficient;
            pulled += pulled_basis[row + (size_t)coordinate * rows] *
                coefficient;
        }
        double difference = pulled - eigenvalue * input;
        numerator += difference * difference;
        denominator += pulled * pulled;
        input_square += input * input;
        mean += input;
    }
    mean /= rows;
    double centered = 0.0;
    for (int row = 0; row < rows; row++) {
        double input = 0.0;
        for (int coordinate = 0; coordinate < rank; coordinate++) {
            input += input_basis[row + (size_t)coordinate * rows] *
                right_vectors[coordinate + (size_t)vector * rank];
        }
        double difference = input - mean;
        centered += difference * difference;
    }
    *constant_variation = input_square == 0.0 ? INFINITY :
        sqrt(centered / input_square);
    return denominator == 0.0 ? sqrt(numerator) :
        sqrt(numerator / denominator);
}

static double complex_function_eigen_residual(
    const double *input_basis,
    const double *pulled_basis,
    int rows,
    int rank,
    const double *right_vectors,
    int real_vector,
    double real_eigenvalue,
    double imaginary_eigenvalue,
    double *constant_variation
) {
    double numerator = 0.0;
    double denominator = 0.0;
    double mean_real = 0.0;
    double mean_imaginary = 0.0;
    double input_square = 0.0;
    for (int row = 0; row < rows; row++) {
        double input_real = 0.0;
        double input_imaginary = 0.0;
        double pulled_real = 0.0;
        double pulled_imaginary = 0.0;
        for (int coordinate = 0; coordinate < rank; coordinate++) {
            double basis_input = input_basis[
                row + (size_t)coordinate * rows
            ];
            double basis_pulled = pulled_basis[
                row + (size_t)coordinate * rows
            ];
            double coefficient_real = right_vectors[
                coordinate + (size_t)real_vector * rank
            ];
            double coefficient_imaginary = right_vectors[
                coordinate + (size_t)(real_vector + 1) * rank
            ];
            input_real += basis_input * coefficient_real;
            input_imaginary += basis_input * coefficient_imaginary;
            pulled_real += basis_pulled * coefficient_real;
            pulled_imaginary += basis_pulled * coefficient_imaginary;
        }
        double expected_real = real_eigenvalue * input_real -
            imaginary_eigenvalue * input_imaginary;
        double expected_imaginary = imaginary_eigenvalue * input_real +
            real_eigenvalue * input_imaginary;
        double difference_real = pulled_real - expected_real;
        double difference_imaginary = pulled_imaginary - expected_imaginary;
        numerator += difference_real * difference_real +
            difference_imaginary * difference_imaginary;
        denominator += pulled_real * pulled_real +
            pulled_imaginary * pulled_imaginary;
        input_square += input_real * input_real +
            input_imaginary * input_imaginary;
        mean_real += input_real;
        mean_imaginary += input_imaginary;
    }
    mean_real /= rows;
    mean_imaginary /= rows;
    double centered = 0.0;
    for (int row = 0; row < rows; row++) {
        double input_real = 0.0;
        double input_imaginary = 0.0;
        for (int coordinate = 0; coordinate < rank; coordinate++) {
            double basis_input = input_basis[
                row + (size_t)coordinate * rows
            ];
            input_real += basis_input * right_vectors[
                coordinate + (size_t)real_vector * rank
            ];
            input_imaginary += basis_input * right_vectors[
                coordinate + (size_t)(real_vector + 1) * rank
            ];
        }
        double difference_real = input_real - mean_real;
        double difference_imaginary = input_imaginary - mean_imaginary;
        centered += difference_real * difference_real +
            difference_imaginary * difference_imaginary;
    }
    *constant_variation = input_square == 0.0 ? INFINITY :
        sqrt(centered / input_square);
    return denominator == 0.0 ? sqrt(numerator) :
        sqrt(numerator / denominator);
}

static int compare_mode_diagnostics(const void *left, const void *right) {
    const ModeDiagnostic *a = left;
    const ModeDiagnostic *b = right;
    if (a->distance_to_one < b->distance_to_one) return -1;
    if (a->distance_to_one > b->distance_to_one) return 1;
    if (a->validation_relative_residual <
        b->validation_relative_residual) return -1;
    if (a->validation_relative_residual >
        b->validation_relative_residual) return 1;
    return a->index - b->index;
}

static int fixed_dimension_of_operator(
    const double *operator_matrix,
    int rank,
    double *singular_values,
    double *tolerance
) {
    double *difference = checked_calloc(
        checked_matrix_values(rank, rank),
        sizeof(*difference)
    );
    memcpy(
        difference,
        operator_matrix,
        (size_t)rank * rank * sizeof(*difference)
    );
    for (int coordinate = 0; coordinate < rank; coordinate++) {
        difference[coordinate + (size_t)coordinate * rank] -= 1.0;
    }
    char jobu = 'N';
    char jobvt = 'N';
    __LAPACK_int m = rank;
    __LAPACK_int n = rank;
    __LAPACK_int lda = rank;
    __LAPACK_int ldu = 1;
    __LAPACK_int ldvt = 1;
    __LAPACK_int lwork = -1;
    __LAPACK_int info = 0;
    double unused_u = 0.0;
    double unused_vt = 0.0;
    double work_query = 0.0;
    dgesvd_(
        &jobu,
        &jobvt,
        &m,
        &n,
        difference,
        &lda,
        singular_values,
        &unused_u,
        &ldu,
        &unused_vt,
        &ldvt,
        &work_query,
        &lwork,
        &info
    );
    if (info != 0) fail("fixed-space SVD workspace query failed");
    lwork = (__LAPACK_int)ceil(work_query);
    double *work = checked_calloc((size_t)lwork, sizeof(*work));
    dgesvd_(
        &jobu,
        &jobvt,
        &m,
        &n,
        difference,
        &lda,
        singular_values,
        &unused_u,
        &ldu,
        &unused_vt,
        &ldvt,
        work,
        &lwork,
        &info
    );
    if (info != 0) fail("fixed-space SVD failed");
    double largest = rank == 0 ? 0.0 : singular_values[0];
    *tolerance = rank * (double)FLT_EPSILON * largest;
    int fixed_dimension = 0;
    for (int index = 0; index < rank; index++) {
        if (singular_values[index] <= *tolerance) fixed_dimension++;
    }
    free(work);
    free(difference);
    return fixed_dimension;
}

static void diagonalize_operator(
    const double *operator_matrix,
    int rank,
    double *real_eigenvalues,
    double *imaginary_eigenvalues,
    double *right_vectors
) {
    double *copy = checked_calloc(
        checked_matrix_values(rank, rank),
        sizeof(*copy)
    );
    memcpy(
        copy,
        operator_matrix,
        (size_t)rank * rank * sizeof(*copy)
    );
    char jobvl = 'N';
    char jobvr = 'V';
    __LAPACK_int n = rank;
    __LAPACK_int lda = rank;
    __LAPACK_int ldvl = 1;
    __LAPACK_int ldvr = rank;
    __LAPACK_int lwork = -1;
    __LAPACK_int info = 0;
    double unused_left = 0.0;
    double work_query = 0.0;
    dgeev_(
        &jobvl,
        &jobvr,
        &n,
        copy,
        &lda,
        real_eigenvalues,
        imaginary_eigenvalues,
        &unused_left,
        &ldvl,
        right_vectors,
        &ldvr,
        &work_query,
        &lwork,
        &info
    );
    if (info != 0) fail("operator eigensolver workspace query failed");
    lwork = (__LAPACK_int)ceil(work_query);
    double *work = checked_calloc((size_t)lwork, sizeof(*work));
    dgeev_(
        &jobvl,
        &jobvr,
        &n,
        copy,
        &lda,
        real_eigenvalues,
        imaginary_eigenvalues,
        &unused_left,
        &ldvl,
        right_vectors,
        &ldvr,
        work,
        &lwork,
        &info
    );
    if (info != 0) fail("operator eigensolver failed");
    free(work);
    free(copy);
}

static void write_spectrum_file(
    const PullbackOptions *options,
    int observation_width_value,
    int columns,
    int validation_samples,
    int rank,
    int fixed_dimension,
    int singular_count,
    double rank_tolerance,
    double fit_representation_relative,
    double fit_descent_relative,
    double validation_dictionary_relative,
    double fixed_tolerance,
    const double *singular_values,
    const double *right_dictionary_basis,
    int right_dictionary_leading_dimension,
    const double *operator_matrix,
    const double *real_eigenvalues,
    const double *imaginary_eigenvalues,
    const double *right_eigenvectors,
    const double *fixed_singular_values
) {
    PullbackSpectrumHeader header = {0};
    memcpy(header.magic, "CPSKOP1", 8);
    header.version = 2;
    header.header_bytes = sizeof(header);
    header.root_scope = (uint32_t)options->root_scope;
    header.positions = (uint32_t)options->positions;
    header.observation_width = (uint32_t)observation_width_value;
    header.pullback_depth = (uint32_t)options->dictionary_depth;
    header.dictionary_columns = (uint32_t)columns;
    header.total_samples = (uint32_t)options->samples;
    header.fit_samples = (uint32_t)options->fit_samples;
    header.validation_samples = (uint32_t)validation_samples;
    header.sampled_rank = (uint32_t)rank;
    header.fixed_dimension = (uint32_t)fixed_dimension;
    header.singular_value_count = (uint32_t)singular_count;
    header.eigenvalue_count = (uint32_t)rank;
    header.dictionary_basis_rows = (uint32_t)rank;
    header.dictionary_basis_columns = (uint32_t)columns;
    header.storage_order = 1;
    header.rank_tolerance = rank_tolerance;
    header.largest_singular_value = singular_count == 0 ? 0.0 :
        singular_values[0];
    header.smallest_retained_singular_value = rank == 0 ? 0.0 :
        singular_values[rank - 1];
    header.fit_representation_relative = fit_representation_relative;
    header.fit_descent_relative = fit_descent_relative;
    header.validation_dictionary_relative = validation_dictionary_relative;
    header.fixed_tolerance = fixed_tolerance;
    FILE *file = fopen(options->spectrum_path, "wb");
    if (file == NULL) fail("could not create pullback spectrum file");
    checked_fwrite(
        &header,
        sizeof(header),
        1,
        file,
        "pullback spectrum header"
    );
    checked_fwrite(
        singular_values,
        sizeof(*singular_values),
        (size_t)singular_count,
        file,
        "dictionary singular values"
    );
    double *basis_row = checked_calloc((size_t)columns, sizeof(*basis_row));
    for (int row = 0; row < rank; row++) {
        for (int column = 0; column < columns; column++) {
            basis_row[column] = right_dictionary_basis[
                row + (size_t)column * right_dictionary_leading_dimension
            ];
        }
        checked_fwrite(
            basis_row,
            sizeof(*basis_row),
            (size_t)columns,
            file,
            "dictionary right basis"
        );
        fflush(file);
    }
    free(basis_row);
    checked_fwrite(
        operator_matrix,
        sizeof(*operator_matrix),
        (size_t)rank * rank,
        file,
        "reduced pullback operator"
    );
    checked_fwrite(
        real_eigenvalues,
        sizeof(*real_eigenvalues),
        (size_t)rank,
        file,
        "real eigenvalues"
    );
    checked_fwrite(
        imaginary_eigenvalues,
        sizeof(*imaginary_eigenvalues),
        (size_t)rank,
        file,
        "imaginary eigenvalues"
    );
    checked_fwrite(
        right_eigenvectors,
        sizeof(*right_eigenvectors),
        (size_t)rank * rank,
        file,
        "right eigenvectors"
    );
    checked_fwrite(
        fixed_singular_values,
        sizeof(*fixed_singular_values),
        (size_t)rank,
        file,
        "fixed-space singular values"
    );
    if (fclose(file) != 0) fail("could not close pullback spectrum file");
}

static void analyze_pullback_spectrum(
    const PullbackEvaluationFile *evaluations,
    const PullbackOptions *options,
    const Transformer *transformer,
    FILE *trace
) {
    int fit_rows = options->fit_samples;
    int validation_rows = options->samples - fit_rows;
    int width = observation_width(options, transformer);
    int columns = 1 + options->dictionary_depth * width;
    int singular_count = fit_rows < columns ? fit_rows : columns;
    size_t fit_values = checked_matrix_values(fit_rows, columns);
    size_t validation_values = checked_matrix_values(
        validation_rows,
        columns
    );
    double *fit_input = checked_calloc(fit_values, sizeof(*fit_input));
    double *fit_pulled = checked_calloc(fit_values, sizeof(*fit_pulled));
    double *validation_input = checked_calloc(
        validation_values,
        sizeof(*validation_input)
    );
    double *validation_pulled = checked_calloc(
        validation_values,
        sizeof(*validation_pulled)
    );
    build_dictionary_pair(
        evaluations,
        options,
        transformer,
        0,
        fit_rows,
        fit_input,
        fit_pulled
    );
    build_dictionary_pair(
        evaluations,
        options,
        transformer,
        fit_rows,
        validation_rows,
        validation_input,
        validation_pulled
    );

    double *singular_values = checked_calloc(
        (size_t)singular_count,
        sizeof(*singular_values)
    );
    double *left_vectors = checked_calloc(
        checked_matrix_values(fit_rows, singular_count),
        sizeof(*left_vectors)
    );
    double *right_dictionary_basis = checked_calloc(
        checked_matrix_values(singular_count, columns),
        sizeof(*right_dictionary_basis)
    );
    char jobz = 'S';
    __LAPACK_int m = fit_rows;
    __LAPACK_int n = columns;
    __LAPACK_int lda = fit_rows;
    __LAPACK_int ldu = fit_rows;
    __LAPACK_int ldvt = singular_count;
    __LAPACK_int lwork = -1;
    __LAPACK_int info = 0;
    double work_query = 0.0;
    __LAPACK_int *integer_work = checked_calloc(
        (size_t)8 * singular_count,
        sizeof(*integer_work)
    );
    dgesdd_(
        &jobz,
        &m,
        &n,
        fit_input,
        &lda,
        singular_values,
        left_vectors,
        &ldu,
        right_dictionary_basis,
        &ldvt,
        &work_query,
        &lwork,
        integer_work,
        &info
    );
    if (info != 0) fail("dictionary SVD workspace query failed");
    lwork = (__LAPACK_int)ceil(work_query);
    double *work = checked_calloc((size_t)lwork, sizeof(*work));
    dgesdd_(
        &jobz,
        &m,
        &n,
        fit_input,
        &lda,
        singular_values,
        left_vectors,
        &ldu,
        right_dictionary_basis,
        &ldvt,
        work,
        &lwork,
        integer_work,
        &info
    );
    if (info != 0) fail("dictionary SVD failed");
    double rank_tolerance = fmax(fit_rows, columns) *
        (double)FLT_EPSILON * singular_values[0];
    int rank = 0;
    while (rank < singular_count &&
           singular_values[rank] > rank_tolerance) {
        rank++;
    }
    if (rank == 0) fail("pullback dictionary has zero sampled rank");

    double *left_pulled = checked_calloc(
        checked_matrix_values(rank, columns),
        sizeof(*left_pulled)
    );
    cblas_dgemm(
        CblasColMajor,
        CblasTrans,
        CblasNoTrans,
        rank,
        columns,
        fit_rows,
        1.0,
        left_vectors,
        fit_rows,
        fit_pulled,
        fit_rows,
        0.0,
        left_pulled,
        rank
    );
    double fit_pulled_square = squared_norm(fit_pulled, fit_values);
    double projected_square = squared_norm(
        left_pulled,
        (size_t)rank * columns
    );
    double fit_residual_square = fit_pulled_square - projected_square;
    if (fit_residual_square < 0.0 &&
        fabs(fit_residual_square) < fit_pulled_square * 1e-10) {
        fit_residual_square = 0.0;
    }
    double fit_representation_relative = fit_pulled_square == 0.0 ? 0.0 :
        sqrt(fmax(0.0, fit_residual_square) / fit_pulled_square);

    double *fit_pulled_basis = checked_calloc(
        checked_matrix_values(fit_rows, rank),
        sizeof(*fit_pulled_basis)
    );
    cblas_dgemm(
        CblasColMajor,
        CblasNoTrans,
        CblasTrans,
        fit_rows,
        rank,
        columns,
        1.0,
        fit_pulled,
        fit_rows,
        right_dictionary_basis,
        singular_count,
        0.0,
        fit_pulled_basis,
        fit_rows
    );
    double fit_row_projection_square = squared_norm(
        fit_pulled_basis,
        (size_t)fit_rows * rank
    );
    double fit_descent_square = fit_pulled_square -
        fit_row_projection_square;
    if (fit_descent_square < 0.0 &&
        fabs(fit_descent_square) < fit_pulled_square * 1e-10) {
        fit_descent_square = 0.0;
    }
    double fit_descent_relative = fit_pulled_square == 0.0 ? 0.0 :
        sqrt(fmax(0.0, fit_descent_square) / fit_pulled_square);

    double *coefficient_map = checked_calloc(
        checked_matrix_values(rank, columns),
        sizeof(*coefficient_map)
    );
    memcpy(
        coefficient_map,
        left_pulled,
        (size_t)rank * columns * sizeof(*coefficient_map)
    );
    for (int row = 0; row < rank; row++) {
        double inverse = 1.0 / singular_values[row];
        for (int column = 0; column < columns; column++) {
            coefficient_map[row + (size_t)column * rank] *= inverse;
        }
    }
    double *operator_matrix = checked_calloc(
        checked_matrix_values(rank, rank),
        sizeof(*operator_matrix)
    );
    cblas_dgemm(
        CblasColMajor,
        CblasNoTrans,
        CblasTrans,
        rank,
        rank,
        columns,
        1.0,
        coefficient_map,
        rank,
        right_dictionary_basis,
        singular_count,
        0.0,
        operator_matrix,
        rank
    );
    for (int column = 0; column < rank; column++) {
        double inverse = 1.0 / singular_values[column];
        for (int row = 0; row < rank; row++) {
            operator_matrix[row + (size_t)column * rank] *=
                singular_values[row] * inverse;
        }
    }

    double *validation_input_basis = checked_calloc(
        checked_matrix_values(validation_rows, rank),
        sizeof(*validation_input_basis)
    );
    double *validation_pulled_basis = checked_calloc(
        checked_matrix_values(validation_rows, rank),
        sizeof(*validation_pulled_basis)
    );
    cblas_dgemm(
        CblasColMajor,
        CblasNoTrans,
        CblasTrans,
        validation_rows,
        rank,
        columns,
        1.0,
        validation_input,
        validation_rows,
        right_dictionary_basis,
        singular_count,
        0.0,
        validation_input_basis,
        validation_rows
    );
    cblas_dgemm(
        CblasColMajor,
        CblasNoTrans,
        CblasTrans,
        validation_rows,
        rank,
        columns,
        1.0,
        validation_pulled,
        validation_rows,
        right_dictionary_basis,
        singular_count,
        0.0,
        validation_pulled_basis,
        validation_rows
    );
    double *validation_prediction = checked_calloc(
        validation_values,
        sizeof(*validation_prediction)
    );
    cblas_dgemm(
        CblasColMajor,
        CblasNoTrans,
        CblasNoTrans,
        validation_rows,
        columns,
        rank,
        1.0,
        validation_input_basis,
        validation_rows,
        coefficient_map,
        rank,
        0.0,
        validation_prediction,
        validation_rows
    );
    double validation_difference_square = 0.0;
    double validation_pulled_square = 0.0;
    for (size_t index = 0; index < validation_values; index++) {
        double difference = validation_prediction[index] -
            validation_pulled[index];
        validation_difference_square += difference * difference;
        validation_pulled_square +=
            validation_pulled[index] * validation_pulled[index];
    }
    double validation_dictionary_relative =
        validation_pulled_square == 0.0 ?
        sqrt(validation_difference_square) :
        sqrt(validation_difference_square / validation_pulled_square);
    for (int column = 0; column < rank; column++) {
        double inverse = 1.0 / singular_values[column];
        for (int row = 0; row < validation_rows; row++) {
            validation_input_basis[row + (size_t)column * validation_rows] *=
                inverse;
            validation_pulled_basis[row + (size_t)column * validation_rows] *=
                inverse;
        }
    }

    double *real_eigenvalues = checked_calloc(
        (size_t)rank,
        sizeof(*real_eigenvalues)
    );
    double *imaginary_eigenvalues = checked_calloc(
        (size_t)rank,
        sizeof(*imaginary_eigenvalues)
    );
    double *right_eigenvectors = checked_calloc(
        checked_matrix_values(rank, rank),
        sizeof(*right_eigenvectors)
    );
    diagonalize_operator(
        operator_matrix,
        rank,
        real_eigenvalues,
        imaginary_eigenvalues,
        right_eigenvectors
    );
    double *fixed_singular_values = checked_calloc(
        (size_t)rank,
        sizeof(*fixed_singular_values)
    );
    double fixed_tolerance = 0.0;
    int fixed_dimension = fixed_dimension_of_operator(
        operator_matrix,
        rank,
        fixed_singular_values,
        &fixed_tolerance
    );

    ModeDiagnostic *diagnostics = checked_calloc(
        (size_t)rank,
        sizeof(*diagnostics)
    );
    for (int mode = 0; mode < rank; mode++) {
        diagnostics[mode].index = mode;
        diagnostics[mode].pair_index = -1;
        diagnostics[mode].real = real_eigenvalues[mode];
        diagnostics[mode].imaginary = imaginary_eigenvalues[mode];
        diagnostics[mode].magnitude = hypot(
            real_eigenvalues[mode],
            imaginary_eigenvalues[mode]
        );
        diagnostics[mode].distance_to_one = hypot(
            real_eigenvalues[mode] - 1.0,
            imaginary_eigenvalues[mode]
        );
    }
    for (int mode = 0; mode < rank; mode++) {
        if (imaginary_eigenvalues[mode] < 0.0) continue;
        if (imaginary_eigenvalues[mode] == 0.0) {
            diagnostics[mode].fit_relative_residual =
                real_matrix_eigen_residual(
                    operator_matrix,
                    rank,
                    right_eigenvectors,
                    mode,
                    real_eigenvalues[mode]
                );
            diagnostics[mode].validation_relative_residual =
                real_function_eigen_residual(
                    validation_input_basis,
                    validation_pulled_basis,
                    validation_rows,
                    rank,
                    right_eigenvectors,
                    mode,
                    real_eigenvalues[mode],
                    &diagnostics[mode].constant_variation
                );
        } else {
            if (mode + 1 >= rank || imaginary_eigenvalues[mode + 1] >= 0.0) {
                fail("eigensolver returned an invalid complex pair");
            }
            double fit_residual = complex_matrix_eigen_residual(
                operator_matrix,
                rank,
                right_eigenvectors,
                mode,
                real_eigenvalues[mode],
                imaginary_eigenvalues[mode]
            );
            double constant_variation = 0.0;
            double validation_residual = complex_function_eigen_residual(
                validation_input_basis,
                validation_pulled_basis,
                validation_rows,
                rank,
                right_eigenvectors,
                mode,
                real_eigenvalues[mode],
                imaginary_eigenvalues[mode],
                &constant_variation
            );
            diagnostics[mode].pair_index = mode + 1;
            diagnostics[mode + 1].pair_index = mode;
            diagnostics[mode].fit_relative_residual = fit_residual;
            diagnostics[mode + 1].fit_relative_residual = fit_residual;
            diagnostics[mode].validation_relative_residual =
                validation_residual;
            diagnostics[mode + 1].validation_relative_residual =
                validation_residual;
            diagnostics[mode].constant_variation = constant_variation;
            diagnostics[mode + 1].constant_variation = constant_variation;
            mode++;
        }
    }

    write_spectrum_file(
        options,
        width,
        columns,
        validation_rows,
        rank,
        fixed_dimension,
        singular_count,
        rank_tolerance,
        fit_representation_relative,
        fit_descent_relative,
        validation_dictionary_relative,
        fixed_tolerance,
        singular_values,
        right_dictionary_basis,
        singular_count,
        operator_matrix,
        real_eigenvalues,
        imaginary_eigenvalues,
        right_eigenvectors,
        fixed_singular_values
    );

    printf(
        "pullback_spectrum root=%s positions=%d depth=%d "
        "dictionary_columns=%d fit=%d validation=%d rank=%d "
        "fixed_dimension=%d fit_representation=%.8g fit_descent=%.8g "
        "validation_dictionary=%.8g "
        "rank_tolerance=%.8g fixed_tolerance=%.8g\n",
        root_scope_name(options->root_scope),
        options->positions,
        options->dictionary_depth,
        columns,
        fit_rows,
        validation_rows,
        rank,
        fixed_dimension,
        fit_representation_relative,
        fit_descent_relative,
        validation_dictionary_relative,
        rank_tolerance,
        fixed_tolerance
    );
    fflush(stdout);
    if (trace != NULL) {
        fprintf(
            trace,
            "{\"kind\":\"pullback_spectrum\",\"root_scope\":\"%s\","
            "\"positions\":%d,\"pullback_depth\":%d,"
            "\"dictionary_columns\":%d,\"fit_samples\":%d,"
            "\"validation_samples\":%d,\"sampled_rank\":%d,"
            "\"fixed_dimension\":%d,\"rank_tolerance\":%.17g,"
            "\"fixed_tolerance\":%.17g,"
            "\"fit_representation_relative\":%.17g,"
            "\"fit_descent_relative\":%.17g,"
            "\"validation_dictionary_relative\":%.17g,"
            "\"sample_coverage_limited\":%s}\n",
            root_scope_name(options->root_scope),
            options->positions,
            options->dictionary_depth,
            columns,
            fit_rows,
            validation_rows,
            rank,
            fixed_dimension,
            rank_tolerance,
            fixed_tolerance,
            fit_representation_relative,
            fit_descent_relative,
            validation_dictionary_relative,
            fit_rows < columns ? "true" : "false"
        );
        fflush(trace);
        qsort(
            diagnostics,
            (size_t)rank,
            sizeof(*diagnostics),
            compare_mode_diagnostics
        );
        for (int order = 0; order < rank; order++) {
            const ModeDiagnostic *mode = &diagnostics[order];
            fprintf(
                trace,
                "{\"kind\":\"pullback_eigenmode\",\"order_by_fixed_distance\":%d,"
                "\"operator_index\":%d,\"pair_index\":%d,"
                "\"eigenvalue_real\":%.17g,\"eigenvalue_imaginary\":%.17g,"
                "\"eigenvalue_magnitude\":%.17g,\"distance_to_one\":%.17g,"
                "\"fit_relative_residual\":%.17g,"
                "\"validation_relative_residual\":%.17g,"
                "\"constant_variation\":%.17g}\n",
                order,
                mode->index,
                mode->pair_index,
                mode->real,
                mode->imaginary,
                mode->magnitude,
                mode->distance_to_one,
                mode->fit_relative_residual,
                mode->validation_relative_residual,
                mode->constant_variation
            );
            fflush(trace);
        }
    }

    free(diagnostics);
    free(fixed_singular_values);
    free(right_eigenvectors);
    free(imaginary_eigenvalues);
    free(real_eigenvalues);
    free(validation_prediction);
    free(validation_pulled_basis);
    free(validation_input_basis);
    free(operator_matrix);
    free(coefficient_map);
    free(fit_pulled_basis);
    free(left_pulled);
    free(work);
    free(integer_work);
    free(right_dictionary_basis);
    free(left_vectors);
    free(singular_values);
    free(validation_pulled);
    free(validation_input);
    free(fit_pulled);
    free(fit_input);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(
            stderr,
            "usage: %s CHECKPOINT TOKENIZER --corpus-dir DIR "
            "--evaluations PATH --spectrum PATH --positions N --samples N "
            "--fit-samples N --pullback-depth N "
            "[--dictionary-depth N] "
            "[--layer N] [--operation attention|ffn|layer|final-rms] "
            "[--root all|last] [--trace PATH] [--resume] "
            "[--analysis-only]\n",
            argv[0]
        );
        return EXIT_FAILURE;
    }
    PullbackOptions options = parse_pullback_options(argc, argv);
    Transformer transformer;
    build_transformer(&transformer, argv[1]);
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, argv[2], transformer.config.vocab_size);
    if (options.positions > transformer.config.seq_len) {
        fail("pullback context exceeds model sequence length");
    }
    if (options.operation != SPECTRUM_FINAL_RMS &&
        options.layer >= transformer.config.n_layers) {
        fail("pullback layer is outside the model");
    }
    if (options.operation == SPECTRUM_FINAL_RMS) {
        options.layer = transformer.config.n_layers;
    }
    SpectrumTerm term = build_spectrum_term(&transformer, options.positions);
    FrontierMap map = selected_map(&term, options.operation, options.layer);
    Continuation suffix = selected_suffix(
        &term,
        options.operation,
        options.layer
    );
    if (map.input_width != map.output_width ||
        map.output_width != suffix.input_width ||
        suffix.result_width != term.frontier_width) {
        fail("selected pullback term has incompatible boundaries");
    }
    PullbackEvaluationFile evaluations = open_evaluation_file(
        &options,
        term.frontier_width
    );
    FILE *trace = NULL;
    if (options.trace_path != NULL) {
        struct stat trace_status;
        bool trace_was_empty = stat(options.trace_path, &trace_status) != 0 ||
            trace_status.st_size == 0;
        trace = fopen(
            options.trace_path,
            options.resume && !options.analysis_only ? "ab" : "wb"
        );
        if (trace == NULL) fail("could not open pullback trace");
        if (!options.resume || options.analysis_only || trace_was_empty) {
            fprintf(
                trace,
                "{\"kind\":\"pullback_meta\",\"schema_version\":1,"
                "\"semantics\":\"function_evaluations_with_homogeneous_constant\","
                "\"operator\":\"U_F(k)=k_after_F\","
                "\"operation\":\"%s\",\"layer\":%d,"
                "\"positions\":%d,\"frontier_width\":%d,"
                "\"root_scope\":\"%s\",\"observation_width\":%d,"
                "\"recorded_pullback_depth\":%d,"
                "\"dictionary_depth\":%d,\"sample_capacity\":%d,"
                "\"fit_samples\":%d,\"validation_samples\":%d,"
                "\"resume_rows\":%u}\n",
                operation_name(options.operation),
                options.layer,
                options.positions,
                term.frontier_width,
                root_scope_name(options.root_scope),
                observation_width(&options, &transformer),
                options.pullback_depth,
                options.dictionary_depth,
                options.samples,
                options.fit_samples,
                options.samples - options.fit_samples,
                evaluations.header->samples_written
            );
            fflush(trace);
        }
    }

    if (!options.analysis_only) {
        PathList paths = {0};
        collect_prompt_paths(&paths, options.corpus_directory);
        qsort(paths.items, paths.count, sizeof(*paths.items), compare_paths);
        if (paths.count == 0) {
            fail("corpus directory contained no prompt.txt files");
        }
        int *seen_tokens = checked_calloc(
            (size_t)options.samples * options.positions,
            sizeof(*seen_tokens)
        );
        size_t root_value_count =
            ((size_t)options.pullback_depth + 1U) * term.frontier_width;
        float *root_points = checked_calloc(
            root_value_count,
            sizeof(*root_points)
        );
        float *state_a = checked_calloc(
            (size_t)term.frontier_width,
            sizeof(*state_a)
        );
        float *state_b = checked_calloc(
            (size_t)term.frontier_width,
            sizeof(*state_b)
        );
        uint32_t resume_rows = evaluations.header->samples_written;
        int unique_windows = 0;
        printf(
            "pullback_sampling operation=%s layer=%d positions=%d "
            "frontier_width=%d depth=%d target_samples=%d fit_samples=%d "
            "root=%s resume_rows=%u corpus_files=%zu\n",
            operation_name(options.operation),
            options.layer,
            options.positions,
            term.frontier_width,
            options.pullback_depth,
            options.samples,
            options.fit_samples,
            root_scope_name(options.root_scope),
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
                memcpy(
                    state_a,
                    input,
                    (size_t)term.frontier_width * sizeof(*state_a)
                );
                float *current = state_a;
                float *next = state_b;
                for (int depth = 0;
                     depth <= options.pullback_depth;
                     depth++) {
                    suffix.apply(
                        suffix.environment,
                        current,
                        root_points + (size_t)depth * term.frontier_width
                    );
                    if (depth < options.pullback_depth) {
                        map.apply(map.environment, current, next);
                        float *swap = current;
                        current = next;
                        next = swap;
                    }
                }
                append_evaluation_row(&evaluations, root_points);
                write_pullback_sample_trace(
                    trace,
                    &tokenizer,
                    sample,
                    paths.items[path_index],
                    start,
                    window,
                    options.positions,
                    root_points,
                    term.frontier_width,
                    options.pullback_depth
                );
                free_frontiers(&capture);
                if ((sample + 1) % 32 == 0 ||
                    sample + 1 == options.samples) {
                    printf(
                        "pullback_progress samples=%d/%d latest_step_l2=%.8g\n",
                        sample + 1,
                        options.samples,
                        difference_l2(
                            root_points +
                                (size_t)(options.pullback_depth - 1) *
                                    term.frontier_width,
                            root_points +
                                (size_t)options.pullback_depth *
                                    term.frontier_width,
                            term.frontier_width
                        )
                    );
                    fflush(stdout);
                }
            }
            free(tokens);
        }
        free(state_b);
        free(state_a);
        free(root_points);
        free(seen_tokens);
        free_path_list(&paths);
    }
    if ((int)evaluations.header->samples_written != options.samples) {
        fail("corpus did not supply enough unique pullback contexts");
    }
    analyze_pullback_spectrum(
        &evaluations,
        &options,
        &transformer,
        trace
    );
    if (trace != NULL && fclose(trace) != 0) {
        fail("could not close pullback trace");
    }
    close_evaluation_file(&evaluations);
    free_spectrum_term(&term);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return EXIT_SUCCESS;
}
