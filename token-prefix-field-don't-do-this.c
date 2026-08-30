/*
 * DO NOT USE AS ESCARDO/ATKEY INFERENCE.
 *
 * This evaluator gives the selection product the carrier `int token`, then
 * calls `support_probe`/`sample_field` to obtain model logits at every forced
 * prefix.  Consequently the observer is terminalized inside each recursive
 * selection frame and every learned filler is applied repeatedly.  The
 * Stories260K two-token diagnostic reports 322 applications of 48 fillers
 * (maximum seven applications for one filler), despite zero calls before the
 * outer `run_pcont`.  Delaying those calls until `run_pcont` did not compose
 * them and did not make the weights one-shot.
 *
 * It is retained only as an auditable record of that rejected organization.
 */

#include "atkey_term_c.h"

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum { SEQUENCE_DELIMITER = 1 };

_Noreturn static void fail(const char *message) {
    fprintf(stderr, "atkey term: %s\n", message);
    exit(EXIT_FAILURE);
}

static void *checked_calloc(size_t count, size_t width) {
    void *memory = calloc(count, width);
    if (memory == NULL) fail("out of memory");
    return memory;
}

typedef struct ArenaBlock ArenaBlock;
struct ArenaBlock {
    ArenaBlock *next;
    size_t used;
    size_t capacity;
    size_t allocation_size;
    max_align_t alignment;
    unsigned char bytes[];
};

typedef struct {
    ArenaBlock *head;
    size_t default_capacity;
    bool file_backed;
    int backing_fd;
    off_t backing_size;
    const char *backing_path;
} Arena;

static size_t align_size(size_t size) {
    size_t alignment = _Alignof(max_align_t);
    return (size + alignment - 1) & ~(alignment - 1);
}

static off_t align_offset(off_t value, off_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

static void arena_enable_file_backing(Arena *arena, const char *path) {
    if (arena->head != NULL || arena->file_backed || path == NULL) {
        fail("invalid arena backing initialization");
    }
    int fd = open(path, O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        fprintf(
            stderr,
            "atkey term: could not create mapped candidate store %s: %s\n",
            path,
            strerror(errno)
        );
        exit(EXIT_FAILURE);
    }
    arena->file_backed = true;
    arena->backing_fd = fd;
    arena->backing_path = path;
}

static ArenaBlock *arena_new_block(Arena *arena, size_t capacity) {
    size_t allocation_size = sizeof(ArenaBlock) + capacity;
    ArenaBlock *block;
    if (!arena->file_backed) {
        block = checked_calloc(1, allocation_size);
    } else {
        long raw_page_size = sysconf(_SC_PAGESIZE);
        if (raw_page_size <= 0) fail("could not determine page size");
        off_t page_size = (off_t)raw_page_size;
        off_t offset = align_offset(arena->backing_size, page_size);
        if (offset < arena->backing_size ||
            allocation_size > (size_t)(INT64_MAX - offset)) {
            fail("mapped candidate store size overflow");
        }
        off_t finish = offset + (off_t)allocation_size;
        if (ftruncate(arena->backing_fd, finish) != 0) {
            fail("could not grow mapped candidate store");
        }
        void *mapping = mmap(
            NULL,
            allocation_size,
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            arena->backing_fd,
            offset
        );
        if (mapping == MAP_FAILED) fail("could not map candidate store block");
        block = mapping;
        arena->backing_size = finish;
    }
    block->capacity = capacity;
    block->allocation_size = allocation_size;
    return block;
}

static void *arena_alloc(Arena *arena, size_t size) {
    size = align_size(size == 0 ? 1 : size);
    ArenaBlock *block = arena->head;
    if (block == NULL || block->capacity - block->used < size) {
        size_t capacity = arena->default_capacity;
        if (capacity < size) capacity = size;
        ArenaBlock *next = arena_new_block(arena, capacity);
        next->next = block;
        arena->head = next;
        block = next;
    }
    void *result = block->bytes + block->used;
    block->used += size;
    return result;
}

static void arena_free(Arena *arena) {
    ArenaBlock *block = arena->head;
    while (block != NULL) {
        ArenaBlock *next = block->next;
        if (arena->file_backed) {
            if (munmap(block, block->allocation_size) != 0) {
                fail("could not unmap candidate store block");
            }
        } else {
            free(block);
        }
        block = next;
    }
    arena->head = NULL;
    if (arena->file_backed) {
        if (close(arena->backing_fd) != 0) {
            fail("could not close mapped candidate store");
        }
        arena->backing_fd = -1;
    }
}

typedef struct {
    int dim;
    int hidden_dim;
    int layers;
    int heads;
    int kv_heads;
    int vocab;
    int sequence_length;
} TermConfig;

static int kv_dim(const TermConfig *config) {
    return config->dim * config->kv_heads / config->heads;
}

static int head_size(const TermConfig *config) {
    return config->dim / config->heads;
}

typedef struct Prefix Prefix;
typedef struct PrefixChild PrefixChild;

struct PrefixChild {
    int token;
    Prefix *prefix;
    PrefixChild *next;
};

struct Prefix {
    uint32_t id;
    int depth;
    int token;
    bool terminated;
    Prefix *parent;
    PrefixChild *children;
};

typedef struct {
    Arena *arena;
    uint32_t next_id;
    Prefix *root;
} PrefixSpace;

static Prefix *new_prefix(
    PrefixSpace *space,
    Prefix *parent,
    int token
) {
    Prefix *prefix = arena_alloc(space->arena, sizeof(*prefix));
    memset(prefix, 0, sizeof(*prefix));
    prefix->id = space->next_id++;
    prefix->parent = parent;
    prefix->token = token;
    prefix->depth = parent == NULL ? 0 : parent->depth + 1;
    prefix->terminated = parent != NULL &&
        (parent->terminated || token == SEQUENCE_DELIMITER);
    return prefix;
}

static void init_prefix_space(PrefixSpace *space, Arena *arena) {
    memset(space, 0, sizeof(*space));
    space->arena = arena;
    space->root = new_prefix(space, NULL, -1);
}

static PrefixChild *prefix_child_entry(
    PrefixSpace *space,
    Prefix *parent,
    int token
) {
    for (PrefixChild *child = parent->children;
         child != NULL;
         child = child->next) {
        if (child->token == token) return child;
    }
    PrefixChild *child = arena_alloc(space->arena, sizeof(*child));
    *child = (PrefixChild){
        .token = token,
        .prefix = new_prefix(space, parent, token),
        .next = parent->children,
    };
    parent->children = child;
    return child;
}

static Prefix *prefix_child(PrefixSpace *space, Prefix *parent, int token) {
    return prefix_child_entry(space, parent, token)->prefix;
}

static Prefix *prefix_at_depth(Prefix *prefix, int depth) {
    if (depth < 0 || prefix->depth < depth) {
        fail("completion shorter than field dependency");
    }
    while (prefix->depth > depth) prefix = prefix->parent;
    return prefix;
}

static int prefix_token_at(Prefix *prefix, int index) {
    Prefix *position = prefix_at_depth(prefix, index + 1);
    return position->token;
}

typedef struct {
    int width;
    float values[];
} Vec;

static Vec *new_vec(Arena *arena, int width) {
    if (width <= 0) fail("invalid vector width");
    Vec *vector = arena_alloc(
        arena,
        sizeof(*vector) + (size_t)width * sizeof(float)
    );
    vector->width = width;
    return vector;
}

typedef struct {
    uint32_t field_id;
    uint32_t prefix_id;
    void *value;
} MemoEntry;

typedef struct {
    MemoEntry *entries;
    size_t capacity;
    size_t count;
} MemoTable;

static uint64_t mix64(uint64_t value) {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31;
    return value;
}

static size_t memo_slot(
    const MemoTable *table,
    uint32_t field_id,
    uint32_t prefix_id
) {
    uint64_t key = ((uint64_t)field_id << 32) | prefix_id;
    return (size_t)mix64(key) & (table->capacity - 1);
}

static void memo_init(MemoTable *table) {
    table->capacity = 1024;
    table->entries = checked_calloc(table->capacity, sizeof(*table->entries));
}

static void memo_insert_raw(
    MemoTable *table,
    uint32_t field_id,
    uint32_t prefix_id,
    void *value
) {
    size_t slot = memo_slot(table, field_id, prefix_id);
    while (table->entries[slot].value != NULL) {
        slot = (slot + 1) & (table->capacity - 1);
    }
    table->entries[slot] = (MemoEntry){field_id, prefix_id, value};
    table->count++;
}

static void memo_grow(MemoTable *table) {
    MemoEntry *old_entries = table->entries;
    size_t old_capacity = table->capacity;
    table->capacity *= 2;
    table->count = 0;
    table->entries = checked_calloc(table->capacity, sizeof(*table->entries));
    for (size_t index = 0; index < old_capacity; index++) {
        MemoEntry entry = old_entries[index];
        if (entry.value != NULL) {
            memo_insert_raw(
                table,
                entry.field_id,
                entry.prefix_id,
                entry.value
            );
        }
    }
    free(old_entries);
}

static void *memo_find(
    const MemoTable *table,
    uint32_t field_id,
    uint32_t prefix_id
) {
    size_t slot = memo_slot(table, field_id, prefix_id);
    while (table->entries[slot].value != NULL) {
        MemoEntry entry = table->entries[slot];
        if (entry.field_id == field_id && entry.prefix_id == prefix_id) {
            return entry.value;
        }
        slot = (slot + 1) & (table->capacity - 1);
    }
    return NULL;
}

static void memo_put(
    MemoTable *table,
    uint32_t field_id,
    uint32_t prefix_id,
    void *value
) {
    if (value == NULL) fail("cannot memoize null");
    if ((table->count + 1) * 10 >= table->capacity * 7) memo_grow(table);
    memo_insert_raw(table, field_id, prefix_id, value);
}

typedef struct Evaluator Evaluator;
typedef struct Field Field;
typedef void *(*FieldCompute)(Evaluator *evaluator, Field *field, Prefix *prefix);

struct Field {
    uint32_t id;
    int dependency;
    FieldCompute compute;
    void *environment;
};

struct Evaluator {
    AtkeyRuntime *runtime;
    TermConfig config;
    Arena term_arena;
    Arena run_arena;
    MemoTable memo;
    PrefixSpace prefixes;
    uint32_t next_field_id;
};

static Field *new_field(
    Evaluator *evaluator,
    int dependency,
    FieldCompute compute,
    void *environment
) {
    if (dependency < 0) fail("negative field dependency");
    Field *field = arena_alloc(&evaluator->term_arena, sizeof(*field));
    field->id = evaluator->next_field_id++;
    field->dependency = dependency;
    field->compute = compute;
    field->environment = environment;
    return field;
}

static void *sample_field(Evaluator *evaluator, Field *field, Prefix *prefix) {
    Prefix *canonical = prefix_at_depth(prefix, field->dependency);
    void *cached = memo_find(&evaluator->memo, field->id, canonical->id);
    if (cached != NULL) return cached;
    void *value = field->compute(evaluator, field, canonical);
    memo_put(&evaluator->memo, field->id, canonical->id, value);
    return value;
}

typedef struct {
    int count;
    Field **items;
} FieldFamily;

/* A product of independent companies lifted through the same filler. The
 * outer index is observational company, not another occurrence of a learned
 * map. Companies may have different position counts. */
typedef struct {
    int count;
    FieldFamily *items;
} CompanyBatch;

static FieldFamily new_family(Evaluator *evaluator, int count) {
    if (count < 0) fail("negative family size");
    FieldFamily family;
    family.count = count;
    family.items = arena_alloc(
        &evaluator->term_arena,
        (size_t)count * sizeof(*family.items)
    );
    return family;
}

static CompanyBatch new_company_batch(Evaluator *evaluator, int count) {
    if (count <= 0) fail("empty company batch");
    CompanyBatch batch;
    batch.count = count;
    batch.items = arena_alloc(
        &evaluator->term_arena,
        (size_t)count * sizeof(*batch.items)
    );
    return batch;
}

typedef enum {
    FILLER_EMBEDDING,
    FILLER_RMS,
    FILLER_MATMUL,
} FillerKind;

typedef struct {
    FillerKind kind;
    AtkeyRuntime *runtime;
    int id;
    const float *weights;
    int input_width;
    int output_width;
    int vocab;
} LearnedFiller;

typedef struct {
    LearnedFiller attention_norm;
    LearnedFiller query;
    LearnedFiller key;
    LearnedFiller value;
    LearnedFiller attention_output;
    LearnedFiller ffn_norm;
    LearnedFiller ffn_gate;
    LearnedFiller ffn_up;
    LearnedFiller ffn_down;
} LayerFillers;

typedef struct {
    LearnedFiller embedding;
    LayerFillers *layers;
    LearnedFiller final_norm;
    LearnedFiller output;
} ModelFillers;

static LearnedFiller make_rms_filler(
    AtkeyRuntime *runtime,
    int id,
    const float *weights,
    int width
) {
    return (LearnedFiller){
        .kind = FILLER_RMS,
        .runtime = runtime,
        .id = id,
        .weights = weights,
        .input_width = width,
        .output_width = width,
    };
}

static LearnedFiller make_matmul_filler(
    AtkeyRuntime *runtime,
    int id,
    const float *weights,
    int input_width,
    int output_width
) {
    return (LearnedFiller){
        .kind = FILLER_MATMUL,
        .runtime = runtime,
        .id = id,
        .weights = weights,
        .input_width = input_width,
        .output_width = output_width,
    };
}

static ModelFillers build_fillers(Evaluator *evaluator) {
    AtkeyRuntime *runtime = evaluator->runtime;
    TermConfig *config = &evaluator->config;
    ModelFillers model = {0};
    model.embedding = (LearnedFiller){
        .kind = FILLER_EMBEDDING,
        .runtime = runtime,
        .id = atkey_embedding_filler_id(),
        .weights = atkey_embedding_weight(runtime),
        .output_width = config->dim,
        .vocab = config->vocab,
    };
    model.layers = arena_alloc(
        &evaluator->term_arena,
        (size_t)config->layers * sizeof(*model.layers)
    );
    for (int layer = 0; layer < config->layers; layer++) {
        LayerFillers *fillers = &model.layers[layer];
        fillers->attention_norm = make_rms_filler(
            runtime,
            atkey_layer_filler_id(layer, 0),
            atkey_attention_rms_weight(runtime, layer),
            config->dim
        );
        fillers->query = make_matmul_filler(
            runtime,
            atkey_layer_filler_id(layer, 1),
            atkey_query_weight(runtime, layer),
            config->dim,
            config->dim
        );
        fillers->key = make_matmul_filler(
            runtime,
            atkey_layer_filler_id(layer, 2),
            atkey_key_weight(runtime, layer),
            config->dim,
            kv_dim(config)
        );
        fillers->value = make_matmul_filler(
            runtime,
            atkey_layer_filler_id(layer, 3),
            atkey_value_weight(runtime, layer),
            config->dim,
            kv_dim(config)
        );
        fillers->attention_output = make_matmul_filler(
            runtime,
            atkey_layer_filler_id(layer, 4),
            atkey_attention_output_weight(runtime, layer),
            config->dim,
            config->dim
        );
        fillers->ffn_norm = make_rms_filler(
            runtime,
            atkey_layer_filler_id(layer, 5),
            atkey_ffn_rms_weight(runtime, layer),
            config->dim
        );
        fillers->ffn_gate = make_matmul_filler(
            runtime,
            atkey_layer_filler_id(layer, 6),
            atkey_ffn_gate_weight(runtime, layer),
            config->dim,
            config->hidden_dim
        );
        fillers->ffn_up = make_matmul_filler(
            runtime,
            atkey_layer_filler_id(layer, 7),
            atkey_ffn_up_weight(runtime, layer),
            config->dim,
            config->hidden_dim
        );
        fillers->ffn_down = make_matmul_filler(
            runtime,
            atkey_layer_filler_id(layer, 8),
            atkey_ffn_down_weight(runtime, layer),
            config->hidden_dim,
            config->dim
        );
    }
    model.final_norm = make_rms_filler(
        runtime,
        atkey_final_rms_filler_id(runtime),
        atkey_final_rms_weight(runtime),
        config->dim
    );
    model.output = make_matmul_filler(
        runtime,
        atkey_output_filler_id(runtime),
        atkey_output_weight(runtime),
        config->dim,
        config->vocab
    );
    return model;
}

static Vec *apply_hidden_filler(
    Evaluator *evaluator,
    const LearnedFiller *filler,
    const Vec *input
) {
    if (filler->kind == FILLER_EMBEDDING) fail("hidden embedding filler");
    if (input->width != filler->input_width) fail("filler input width mismatch");
    Vec *output = new_vec(&evaluator->run_arena, filler->output_width);
    if (filler->kind == FILLER_RMS) {
        atkey_rms_apply(
            filler->runtime,
            filler->id,
            output->values,
            input->values,
            filler->weights,
            filler->input_width
        );
    } else {
        atkey_matmul_apply(
            filler->runtime,
            filler->id,
            output->values,
            input->values,
            filler->weights,
            filler->input_width,
            filler->output_width
        );
    }
    return output;
}

typedef enum {
    TOKEN_CONSTANT,
    TOKEN_COMPLETION_POSITION,
} TokenFieldKind;

typedef struct {
    int dependency;
    TokenFieldKind kind;
    int value;
} TokenField;

static int sample_token_field(const TokenField *field, Prefix *prefix) {
    Prefix *canonical = prefix_at_depth(prefix, field->dependency);
    if (field->kind == TOKEN_CONSTANT) return field->value;
    return prefix_token_at(canonical, field->value);
}

typedef struct {
    const TokenField *input;
    const LearnedFiller *filler;
} EmbeddingEnvironment;

static void *compute_embedding(
    Evaluator *evaluator,
    Field *field,
    Prefix *prefix
) {
    EmbeddingEnvironment *environment = field->environment;
    const LearnedFiller *filler = environment->filler;
    int token = sample_token_field(environment->input, prefix);
    if (token < 0 || token >= filler->vocab) fail("token outside vocabulary");
    Vec *output = new_vec(&evaluator->run_arena, filler->output_width);
    atkey_embedding_apply(
        filler->runtime,
        filler->id,
        output->values,
        token,
        filler->weights,
        filler->vocab,
        filler->output_width
    );
    return output;
}

static FieldFamily embedding_context(
    Evaluator *evaluator,
    const TokenField *inputs,
    int count,
    const LearnedFiller *filler
) {
    FieldFamily output = new_family(evaluator, count);
    for (int position = 0; position < count; position++) {
        EmbeddingEnvironment *environment = arena_alloc(
            &evaluator->term_arena,
            sizeof(*environment)
        );
        *environment = (EmbeddingEnvironment){&inputs[position], filler};
        output.items[position] = new_field(
            evaluator,
            inputs[position].dependency,
            compute_embedding,
            environment
        );
    }
    return output;
}

static CompanyBatch embedding_company_context(
    Evaluator *evaluator,
    TokenField *const *inputs,
    const int *counts,
    int company_count,
    const LearnedFiller *filler
) {
    CompanyBatch output = new_company_batch(evaluator, company_count);
    for (int company = 0; company < company_count; company++) {
        output.items[company] = embedding_context(
            evaluator,
            inputs[company],
            counts[company],
            filler
        );
    }
    return output;
}

typedef struct {
    Field *input;
    const LearnedFiller *filler;
} MapEnvironment;

static void *compute_learned_map(
    Evaluator *evaluator,
    Field *field,
    Prefix *prefix
) {
    MapEnvironment *environment = field->environment;
    Vec *input = sample_field(evaluator, environment->input, prefix);
    return apply_hidden_filler(evaluator, environment->filler, input);
}

/* Context S -> (A -> B) -> T: filler is supplied once; family multiplicity
 * remains in the output field company. */
static FieldFamily family_context(
    Evaluator *evaluator,
    FieldFamily input,
    const LearnedFiller *filler
) {
    FieldFamily output = new_family(evaluator, input.count);
    for (int position = 0; position < input.count; position++) {
        MapEnvironment *environment = arena_alloc(
            &evaluator->term_arena,
            sizeof(*environment)
        );
        *environment = (MapEnvironment){input.items[position], filler};
        output.items[position] = new_field(
            evaluator,
            input.items[position]->dependency,
            compute_learned_map,
            environment
        );
    }
    return output;
}

static CompanyBatch company_family_context(
    Evaluator *evaluator,
    CompanyBatch input,
    const LearnedFiller *filler
) {
    CompanyBatch output = new_company_batch(evaluator, input.count);
    for (int company = 0; company < input.count; company++) {
        output.items[company] = family_context(
            evaluator,
            input.items[company],
            filler
        );
    }
    return output;
}

typedef struct {
    Field *left;
    Field *right;
} BinaryEnvironment;

static void *compute_add(Evaluator *evaluator, Field *field, Prefix *prefix) {
    BinaryEnvironment *environment = field->environment;
    Vec *left = sample_field(evaluator, environment->left, prefix);
    Vec *right = sample_field(evaluator, environment->right, prefix);
    if (left->width != right->width) fail("residual width mismatch");
    Vec *output = new_vec(&evaluator->run_arena, left->width);
    atkey_add(output->values, left->values, right->values, left->width);
    return output;
}

static FieldFamily add_families(
    Evaluator *evaluator,
    FieldFamily left,
    FieldFamily right
) {
    if (left.count != right.count) fail("residual family size mismatch");
    FieldFamily output = new_family(evaluator, left.count);
    for (int position = 0; position < left.count; position++) {
        BinaryEnvironment *environment = arena_alloc(
            &evaluator->term_arena,
            sizeof(*environment)
        );
        *environment = (BinaryEnvironment){
            left.items[position],
            right.items[position],
        };
        int dependency = left.items[position]->dependency;
        if (right.items[position]->dependency > dependency) {
            dependency = right.items[position]->dependency;
        }
        output.items[position] = new_field(
            evaluator,
            dependency,
            compute_add,
            environment
        );
    }
    return output;
}

static CompanyBatch add_company_batches(
    Evaluator *evaluator,
    CompanyBatch left,
    CompanyBatch right
) {
    if (left.count != right.count) fail("residual company count mismatch");
    CompanyBatch output = new_company_batch(evaluator, left.count);
    for (int company = 0; company < left.count; company++) {
        output.items[company] = add_families(
            evaluator,
            left.items[company],
            right.items[company]
        );
    }
    return output;
}

typedef struct {
    Vec *query;
    Vec *key;
} RopePair;

typedef struct {
    Field *query;
    Field *key;
    int position;
} RopeEnvironment;

static void *compute_rope(Evaluator *evaluator, Field *field, Prefix *prefix) {
    RopeEnvironment *environment = field->environment;
    Vec *query = sample_field(evaluator, environment->query, prefix);
    Vec *key = sample_field(evaluator, environment->key, prefix);
    RopePair *pair = arena_alloc(&evaluator->run_arena, sizeof(*pair));
    pair->query = new_vec(&evaluator->run_arena, evaluator->config.dim);
    pair->key = new_vec(&evaluator->run_arena, kv_dim(&evaluator->config));
    atkey_rope(
        pair->query->values,
        pair->key->values,
        query->values,
        key->values,
        environment->position,
        evaluator->config.dim,
        kv_dim(&evaluator->config),
        head_size(&evaluator->config)
    );
    return pair;
}

typedef struct {
    Field *pair;
    bool query;
} RopeProjectionEnvironment;

static void *compute_rope_projection(
    Evaluator *evaluator,
    Field *field,
    Prefix *prefix
) {
    RopeProjectionEnvironment *environment = field->environment;
    RopePair *pair = sample_field(evaluator, environment->pair, prefix);
    return environment->query ? pair->query : pair->key;
}

static void rotate_families(
    Evaluator *evaluator,
    FieldFamily queries,
    FieldFamily keys,
    FieldFamily *rotated_queries,
    FieldFamily *rotated_keys
) {
    if (queries.count != keys.count) fail("Q/K family size mismatch");
    *rotated_queries = new_family(evaluator, queries.count);
    *rotated_keys = new_family(evaluator, keys.count);
    for (int position = 0; position < queries.count; position++) {
        RopeEnvironment *rope_environment = arena_alloc(
            &evaluator->term_arena,
            sizeof(*rope_environment)
        );
        *rope_environment = (RopeEnvironment){
            queries.items[position],
            keys.items[position],
            position,
        };
        int dependency = queries.items[position]->dependency;
        if (keys.items[position]->dependency > dependency) {
            dependency = keys.items[position]->dependency;
        }
        Field *pair = new_field(
            evaluator,
            dependency,
            compute_rope,
            rope_environment
        );
        RopeProjectionEnvironment *query_environment = arena_alloc(
            &evaluator->term_arena,
            sizeof(*query_environment)
        );
        RopeProjectionEnvironment *key_environment = arena_alloc(
            &evaluator->term_arena,
            sizeof(*key_environment)
        );
        *query_environment = (RopeProjectionEnvironment){pair, true};
        *key_environment = (RopeProjectionEnvironment){pair, false};
        rotated_queries->items[position] = new_field(
            evaluator,
            dependency,
            compute_rope_projection,
            query_environment
        );
        rotated_keys->items[position] = new_field(
            evaluator,
            dependency,
            compute_rope_projection,
            key_environment
        );
    }
}

static void rotate_company_batches(
    Evaluator *evaluator,
    CompanyBatch queries,
    CompanyBatch keys,
    CompanyBatch *rotated_queries,
    CompanyBatch *rotated_keys
) {
    if (queries.count != keys.count) fail("Q/K company count mismatch");
    *rotated_queries = new_company_batch(evaluator, queries.count);
    *rotated_keys = new_company_batch(evaluator, keys.count);
    for (int company = 0; company < queries.count; company++) {
        rotate_families(
            evaluator,
            queries.items[company],
            keys.items[company],
            &rotated_queries->items[company],
            &rotated_keys->items[company]
        );
    }
}

typedef struct {
    Field *query;
    Field **keys;
    Field **values;
    int count;
} AttentionEnvironment;

static void *compute_attention(
    Evaluator *evaluator,
    Field *field,
    Prefix *prefix
) {
    AttentionEnvironment *environment = field->environment;
    Vec *query = sample_field(evaluator, environment->query, prefix);
    const float *key_pointers[environment->count];
    const float *value_pointers[environment->count];
    for (int index = 0; index < environment->count; index++) {
        Vec *key = sample_field(evaluator, environment->keys[index], prefix);
        Vec *value = sample_field(evaluator, environment->values[index], prefix);
        key_pointers[index] = key->values;
        value_pointers[index] = value->values;
    }
    Vec *output = new_vec(&evaluator->run_arena, evaluator->config.dim);
    atkey_attention(
        output->values,
        query->values,
        key_pointers,
        value_pointers,
        environment->count,
        evaluator->config.dim,
        evaluator->config.heads,
        evaluator->config.kv_heads
    );
    return output;
}

static FieldFamily causal_attention_context(
    Evaluator *evaluator,
    FieldFamily queries,
    FieldFamily keys,
    FieldFamily values
) {
    if (queries.count != keys.count || keys.count != values.count) {
        fail("Q/K/V family size mismatch");
    }
    FieldFamily output = new_family(evaluator, queries.count);
    for (int position = 0; position < queries.count; position++) {
        AttentionEnvironment *environment = arena_alloc(
            &evaluator->term_arena,
            sizeof(*environment)
        );
        environment->query = queries.items[position];
        environment->count = position + 1;
        environment->keys = arena_alloc(
            &evaluator->term_arena,
            (size_t)environment->count * sizeof(*environment->keys)
        );
        environment->values = arena_alloc(
            &evaluator->term_arena,
            (size_t)environment->count * sizeof(*environment->values)
        );
        int dependency = environment->query->dependency;
        for (int index = 0; index < environment->count; index++) {
            environment->keys[index] = keys.items[index];
            environment->values[index] = values.items[index];
            if (keys.items[index]->dependency > dependency) {
                dependency = keys.items[index]->dependency;
            }
            if (values.items[index]->dependency > dependency) {
                dependency = values.items[index]->dependency;
            }
        }
        output.items[position] = new_field(
            evaluator,
            dependency,
            compute_attention,
            environment
        );
    }
    return output;
}

static CompanyBatch causal_attention_company_context(
    Evaluator *evaluator,
    CompanyBatch queries,
    CompanyBatch keys,
    CompanyBatch values
) {
    if (queries.count != keys.count || keys.count != values.count) {
        fail("Q/K/V company count mismatch");
    }
    CompanyBatch output = new_company_batch(evaluator, queries.count);
    for (int company = 0; company < queries.count; company++) {
        output.items[company] = causal_attention_context(
            evaluator,
            queries.items[company],
            keys.items[company],
            values.items[company]
        );
    }
    return output;
}

static void *compute_swiglu(Evaluator *evaluator, Field *field, Prefix *prefix) {
    BinaryEnvironment *environment = field->environment;
    Vec *gate = sample_field(evaluator, environment->left, prefix);
    Vec *up = sample_field(evaluator, environment->right, prefix);
    if (gate->width != up->width) fail("SwiGLU width mismatch");
    Vec *output = new_vec(&evaluator->run_arena, gate->width);
    atkey_swiglu(
        output->values,
        gate->values,
        gate->values,
        up->values,
        gate->width
    );
    return output;
}

static FieldFamily swiglu_context(
    Evaluator *evaluator,
    FieldFamily gates,
    FieldFamily ups
) {
    if (gates.count != ups.count) fail("SwiGLU family size mismatch");
    FieldFamily output = new_family(evaluator, gates.count);
    for (int position = 0; position < gates.count; position++) {
        BinaryEnvironment *environment = arena_alloc(
            &evaluator->term_arena,
            sizeof(*environment)
        );
        *environment = (BinaryEnvironment){
            gates.items[position],
            ups.items[position],
        };
        int dependency = gates.items[position]->dependency;
        if (ups.items[position]->dependency > dependency) {
            dependency = ups.items[position]->dependency;
        }
        output.items[position] = new_field(
            evaluator,
            dependency,
            compute_swiglu,
            environment
        );
    }
    return output;
}

static CompanyBatch swiglu_company_context(
    Evaluator *evaluator,
    CompanyBatch gates,
    CompanyBatch ups
) {
    if (gates.count != ups.count) fail("SwiGLU company count mismatch");
    CompanyBatch output = new_company_batch(evaluator, gates.count);
    for (int company = 0; company < gates.count; company++) {
        output.items[company] = swiglu_context(
            evaluator,
            gates.items[company],
            ups.items[company]
        );
    }
    return output;
}

/* A filled layer is the filler at the next scale. Every observational company
 * is lifted through each learned filler together; no field is sampled here. */
static CompanyBatch layer_company(
    Evaluator *evaluator,
    const LayerFillers *fillers,
    CompanyBatch input
) {
    CompanyBatch normalized = company_family_context(
        evaluator,
        input,
        &fillers->attention_norm
    );
    CompanyBatch queries = company_family_context(
        evaluator,
        normalized,
        &fillers->query
    );
    CompanyBatch keys = company_family_context(
        evaluator,
        normalized,
        &fillers->key
    );
    CompanyBatch values = company_family_context(
        evaluator,
        normalized,
        &fillers->value
    );
    CompanyBatch rotated_queries;
    CompanyBatch rotated_keys;
    rotate_company_batches(
        evaluator,
        queries,
        keys,
        &rotated_queries,
        &rotated_keys
    );
    CompanyBatch attended = causal_attention_company_context(
        evaluator,
        rotated_queries,
        rotated_keys,
        values
    );
    CompanyBatch projected = company_family_context(
        evaluator,
        attended,
        &fillers->attention_output
    );
    CompanyBatch attention_residual = add_company_batches(
        evaluator,
        input,
        projected
    );
    CompanyBatch ffn_input = company_family_context(
        evaluator,
        attention_residual,
        &fillers->ffn_norm
    );
    CompanyBatch gate = company_family_context(
        evaluator,
        ffn_input,
        &fillers->ffn_gate
    );
    CompanyBatch up = company_family_context(
        evaluator,
        ffn_input,
        &fillers->ffn_up
    );
    CompanyBatch gated = swiglu_company_context(evaluator, gate, up);
    CompanyBatch down = company_family_context(
        evaluator,
        gated,
        &fillers->ffn_down
    );
    return add_company_batches(evaluator, attention_residual, down);
}

static CompanyBatch network_company(
    Evaluator *evaluator,
    const ModelFillers *fillers,
    CompanyBatch hidden,
    CompanyBatch *scales
) {
    scales[0] = hidden;
    for (int layer = 0; layer < evaluator->config.layers; layer++) {
        hidden = layer_company(
            evaluator,
            &fillers->layers[layer],
            hidden
        );
        scales[layer + 1] = hidden;
    }
    return hidden;
}

typedef struct {
    int horizon;
    Field **logits;
} ModelTerm;

/* Compose every learned field before selection is constructed. */
static ModelTerm model_fields_term(
    Evaluator *evaluator,
    const ModelFillers *fillers,
    const int *prompt,
    int prompt_count,
    int horizon
) {
    int position_count = prompt_count + horizon - 1;
    TokenField *tokens = arena_alloc(
        &evaluator->term_arena,
        (size_t)position_count * sizeof(*tokens)
    );
    for (int position = 0; position < prompt_count; position++) {
        tokens[position] = (TokenField){
            .dependency = 0,
            .kind = TOKEN_CONSTANT,
            .value = prompt[position],
        };
    }
    for (int index = 0; index < horizon - 1; index++) {
        tokens[prompt_count + index] = (TokenField){
            .dependency = index + 1,
            .kind = TOKEN_COMPLETION_POSITION,
            .value = index,
        };
    }
    TokenField *token_companies[] = {tokens};
    int token_company_counts[] = {position_count};
    CompanyBatch companies = embedding_company_context(
        evaluator,
        token_companies,
        token_company_counts,
        1,
        &fillers->embedding
    );
    CompanyBatch *scales = arena_alloc(
        &evaluator->term_arena,
        (size_t)(evaluator->config.layers + 1) * sizeof(*scales)
    );
    companies = network_company(
        evaluator,
        fillers,
        companies,
        scales
    );
    FieldFamily hidden = companies.items[0];
    FieldFamily normalized = family_context(
        evaluator,
        hidden,
        &fillers->final_norm
    );
    FieldFamily all_logits = family_context(
        evaluator,
        normalized,
        &fillers->output
    );
    ModelTerm term;
    term.horizon = horizon;
    term.logits = arena_alloc(
        &evaluator->term_arena,
        (size_t)horizon * sizeof(*term.logits)
    );
    for (int index = 0; index < horizon; index++) {
        term.logits[index] = all_logits.items[prompt_count - 1 + index];
        if (term.logits[index]->dependency != index) {
            fail("causal logit dependency invariant failed");
        }
    }
    return term;
}

typedef struct Completion Completion;
typedef struct Outcome Outcome;
typedef struct Search Search;
typedef struct SuffixThunk SuffixThunk;

typedef struct {
    bool selected;
    Completion *completion;
    double score;
} ProductSelection;

typedef struct {
    int token;
    Outcome *outcome;
} CandidateAudit;

struct Completion {
    int token;
    Prefix *before;
    CandidateAudit *audits;
    int audit_count;
    Completion *direct_tail;
    SuffixThunk *lazy_tail;
};

typedef struct {
    Outcome *(*apply)(
        void *environment,
        Prefix *prefix,
        Completion *completion
    );
    void *environment;
} Observer;

struct Outcome {
    Search *search;
    Prefix *prefix;
    Completion *completion;
};

typedef struct {
    Vec *logits;
    int token;
    double reward;
} RewardStep;

typedef struct ProductFrame ProductFrame;

typedef enum {
    SAMPLE_STOP_NONE,
    SAMPLE_STOP_DEADLINE,
    SAMPLE_STOP_DEMAND_LIMIT,
    SAMPLE_STOP_SUPPORT_COMPLETE,
} SampleStopReason;

struct SuffixThunk {
    bool forced;
    Completion *value;
    Search *search;
    Prefix *history;
    int remaining;
    Observer observer;
    bool sample_budget_assigned;
    uint64_t sample_demand_budget;
    uint64_t owner_frame_id;
    int head_token;
};

struct Search {
    Evaluator *evaluator;
    ModelTerm *model;
    const char *prompt_text;
    int prompt_last_token;
    int top_k;
    bool bounded;
    bool threshold_enabled;
    double reward_threshold;
    bool sampling_enabled;
    int sample_milliseconds;
    int sample_demand_limit;
    struct timespec sample_deadline;
    bool sample_deadline_armed;
    bool sample_budget_exhausted;
    SampleStopReason sample_stop_reason;
    uint64_t sampled_candidate_demands;
    uint64_t sampled_forced_demands;
    uint64_t sampled_truncated_frames;
    bool trace;
    FILE *audit_stream;
    FILE *strength_stream;
    uint64_t next_strength_event_id;
    uint64_t next_frame_id;
    size_t completion_cells;
    size_t outcomes;
};

static void strength_log(Search *search, const char *format, ...) {
    if (search->strength_stream == NULL) return;
    fprintf(
        search->strength_stream,
        "strength_event id=%" PRIu64 " ",
        search->next_strength_event_id++
    );
    va_list arguments;
    va_start(arguments, format);
    vfprintf(search->strength_stream, format, arguments);
    va_end(arguments);
    fputc('\n', search->strength_stream);
    fflush(search->strength_stream);
}

static struct timespec add_milliseconds(
    struct timespec start,
    int milliseconds
) {
    if (milliseconds < 0) fail("negative sampling duration");
    start.tv_sec += milliseconds / 1000;
    start.tv_nsec += (long)(milliseconds % 1000) * 1000000L;
    if (start.tv_nsec >= 1000000000L) {
        start.tv_sec++;
        start.tv_nsec -= 1000000000L;
    }
    return start;
}

static bool time_reached(struct timespec now, struct timespec deadline) {
    if (now.tv_sec > deadline.tv_sec) return true;
    if (now.tv_sec < deadline.tv_sec) return false;
    return now.tv_nsec >= deadline.tv_nsec;
}

static const char *sample_stop_name(SampleStopReason reason) {
    switch (reason) {
        case SAMPLE_STOP_NONE: return "none";
        case SAMPLE_STOP_DEADLINE: return "deadline";
        case SAMPLE_STOP_DEMAND_LIMIT: return "demand_limit";
        case SAMPLE_STOP_SUPPORT_COMPLETE: return "support_complete";
    }
    return "invalid";
}

static ProductSelection history_product_select(
    Search *search,
    Prefix *history,
    int remaining,
    Observer observer,
    uint64_t sample_demand_budget,
    bool bounded_demand,
    double cutoff,
    uint64_t parent_frame_id,
    int via_token
);

static Completion *new_completion(
    Search *search,
    Prefix *before,
    int token,
    Completion *direct_tail,
    SuffixThunk *lazy_tail,
    CandidateAudit *audits,
    int audit_count
) {
    if (before == NULL) fail("completion cell missing history");
    if (token < 0 || token >= search->evaluator->config.vocab) {
        fail("completion token outside vocabulary");
    }
    if ((direct_tail == NULL) == (lazy_tail == NULL) &&
        before->depth + 1 < search->model->horizon) {
        fail("completion tail must be exactly one of direct or lazy");
    }
    Completion *completion = arena_alloc(
        &search->evaluator->run_arena,
        sizeof(*completion)
    );
    search->completion_cells++;
    *completion = (Completion){
        .token = token,
        .before = before,
        .audits = audits,
        .audit_count = audit_count,
        .direct_tail = direct_tail,
        .lazy_tail = lazy_tail,
    };
    return completion;
}

static Completion *force_suffix(SuffixThunk *thunk) {
    if (!thunk->forced) {
        strength_log(
            thunk->search,
            "kind=suffix_force_begin owner_frame=%" PRIu64
            " token=%d prefix=%" PRIu32 " depth=%d remaining=%d mode=exact",
            thunk->owner_frame_id,
            thunk->head_token,
            thunk->history->id,
            thunk->history->depth,
            thunk->remaining
        );
        ProductSelection selection = history_product_select(
            thunk->search,
            thunk->history,
            thunk->remaining,
            thunk->observer,
            thunk->sample_budget_assigned ?
                thunk->sample_demand_budget : UINT64_MAX,
            false,
            -INFINITY,
            thunk->owner_frame_id,
            thunk->head_token
        );
        if (!selection.selected) {
            fail("exact suffix demand was rejected");
        }
        thunk->value = selection.completion;
        thunk->forced = true;
        strength_log(
            thunk->search,
            "kind=suffix_force_end owner_frame=%" PRIu64
            " token=%d selected=1 score=%.17g",
            thunk->owner_frame_id,
            thunk->head_token,
            selection.score
        );
    } else {
        strength_log(
            thunk->search,
            "kind=suffix_reuse owner_frame=%" PRIu64 " token=%d",
            thunk->owner_frame_id,
            thunk->head_token
        );
    }
    return thunk->value;
}

static ProductSelection force_suffix_above(
    SuffixThunk *thunk,
    double cutoff
) {
    if (thunk->forced) {
        strength_log(
            thunk->search,
            "kind=suffix_reuse owner_frame=%" PRIu64
            " token=%d mode=bounded cutoff=%.17g",
            thunk->owner_frame_id,
            thunk->head_token,
            cutoff
        );
        return (ProductSelection){
            .selected = true,
            .completion = thunk->value,
            .score = NAN,
        };
    }
    strength_log(
        thunk->search,
        "kind=suffix_force_begin owner_frame=%" PRIu64
        " token=%d prefix=%" PRIu32 " depth=%d remaining=%d "
        "mode=bounded cutoff=%.17g",
        thunk->owner_frame_id,
        thunk->head_token,
        thunk->history->id,
        thunk->history->depth,
        thunk->remaining,
        cutoff
    );
    ProductSelection selection = history_product_select(
        thunk->search,
        thunk->history,
        thunk->remaining,
        thunk->observer,
        thunk->sample_budget_assigned ?
            thunk->sample_demand_budget : UINT64_MAX,
        true,
        cutoff,
        thunk->owner_frame_id,
        thunk->head_token
    );
    if (selection.selected) {
        thunk->value = selection.completion;
        thunk->forced = true;
    }
    strength_log(
        thunk->search,
        "kind=suffix_force_end owner_frame=%" PRIu64
        " token=%d selected=%d score=%.17g",
        thunk->owner_frame_id,
        thunk->head_token,
        selection.selected ? 1 : 0,
        selection.score
    );
    return selection;
}

static Completion *completion_tail(Completion *completion) {
    if (completion == NULL) return NULL;
    if (completion->lazy_tail != NULL) {
        completion->direct_tail = force_suffix(completion->lazy_tail);
        completion->lazy_tail = NULL;
    }
    return completion->direct_tail;
}

static Completion *completion_at(Outcome *outcome, int depth) {
    if (depth < outcome->prefix->depth) return NULL;
    Completion *completion = outcome->completion;
    for (int index = outcome->prefix->depth; index < depth; index++) {
        if (completion == NULL || completion->token == SEQUENCE_DELIMITER) {
            return NULL;
        }
        completion = completion_tail(completion);
    }
    return completion;
}

static bool completion_at_above(
    Outcome *outcome,
    int depth,
    double cutoff,
    Completion **result,
    double *upper_bound
) {
    if (depth < outcome->prefix->depth) {
        *result = NULL;
        return true;
    }
    Completion *completion = outcome->completion;
    for (int index = outcome->prefix->depth; index < depth; index++) {
        if (completion == NULL || completion->token == SEQUENCE_DELIMITER) {
            *result = NULL;
            return true;
        }
        if (completion->lazy_tail != NULL) {
            ProductSelection selection = force_suffix_above(
                completion->lazy_tail,
                cutoff
            );
            if (!selection.selected) {
                *upper_bound = selection.score;
                *result = NULL;
                return false;
            }
            completion->direct_tail = selection.completion;
            completion->lazy_tail = NULL;
        }
        completion = completion->direct_tail;
    }
    *result = completion;
    return true;
}

static double token_log_probability(const Vec *logits, int token) {
    if (token < 0 || token >= logits->width) fail("scored token outside logits");
    float maximum = -FLT_MAX;
    for (int index = 0; index < logits->width; index++) {
        if (logits->values[index] > maximum) maximum = logits->values[index];
    }
    double partition = 0.0;
    for (int index = 0; index < logits->width; index++) {
        partition += exp((double)logits->values[index] - (double)maximum);
    }
    double reward = (double)logits->values[token] -
        (double)maximum - log(partition);
    if (!isfinite(reward) || reward > 0.0) {
        fail("model produced a non-monotone token reward");
    }
    return reward;
}

/* Compare variable-length outcomes on the same horizon. Without this
 * normalization an early delimiter gets every unobserved suffix position for
 * free: its raw negative log-probability sum is compared with a full-length
 * sum. Full-horizon outcomes are unchanged. */
static double horizon_equivalent_reward(
    const Search *search,
    double log_probability_sum,
    int scored_tokens
) {
    if (scored_tokens <= 0 || scored_tokens > search->model->horizon) {
        fail("invalid scored token count");
    }
    return log_probability_sum *
        (double)search->model->horizon / (double)scored_tokens;
}

static bool reward_meets_cutoff(
    const Search *search,
    double reward,
    double cutoff
) {
    return search->threshold_enabled ? reward >= cutoff : reward > cutoff;
}

static bool partial_cannot_recover(
    const Search *search,
    double partial_reward,
    double cutoff
) {
    return search->threshold_enabled ?
        partial_reward < cutoff : partial_reward <= cutoff;
}

static bool outcome_step(Outcome *outcome, int depth, RewardStep *step) {
    Search *search = outcome->search;
    if (depth < 0 || depth >= search->model->horizon) return false;
    Prefix *history;
    int token;
    if (depth < outcome->prefix->depth) {
        history = prefix_at_depth(outcome->prefix, depth);
        token = prefix_token_at(outcome->prefix, depth);
    } else {
        Completion *completion = completion_at(outcome, depth);
        if (completion == NULL) return false;
        if (completion->before->depth != depth) {
            fail("completion history depth invariant failed");
        }
        history = completion->before;
        token = completion->token;
    }
    Field *field = search->model->logits[depth];
    Vec *logits = sample_field(
        search->evaluator,
        field,
        history
    );
    *step = (RewardStep){
        .logits = logits,
        .token = token,
        .reward = token_log_probability(logits, token),
    };
    return true;
}

typedef enum {
    REWARD_STEP_END,
    REWARD_STEP_READY,
    REWARD_STEP_REJECTED,
} RewardStepStatus;

static RewardStepStatus outcome_step_above(
    Outcome *outcome,
    int depth,
    double cutoff,
    RewardStep *step,
    double *upper_bound
) {
    Search *search = outcome->search;
    if (depth < 0 || depth >= search->model->horizon) {
        return REWARD_STEP_END;
    }
    Prefix *history;
    int token;
    if (depth < outcome->prefix->depth) {
        history = prefix_at_depth(outcome->prefix, depth);
        token = prefix_token_at(outcome->prefix, depth);
    } else {
        Completion *completion = NULL;
        if (!completion_at_above(
                outcome,
                depth,
                cutoff,
                &completion,
                upper_bound
            )) {
            return REWARD_STEP_REJECTED;
        }
        if (completion == NULL) return REWARD_STEP_END;
        if (completion->before->depth != depth) {
            fail("completion history depth invariant failed");
        }
        history = completion->before;
        token = completion->token;
    }
    Field *field = search->model->logits[depth];
    Vec *logits = sample_field(search->evaluator, field, history);
    *step = (RewardStep){
        .logits = logits,
        .token = token,
        .reward = token_log_probability(logits, token),
    };
    return REWARD_STEP_READY;
}

static double outcome_log_probability_sum(
    Outcome *outcome,
    int *scored_token_count
) {
    double total = 0.0;
    int scored_tokens = 0;
    for (int depth = 0; depth < outcome->search->model->horizon; depth++) {
        RewardStep step;
        if (!outcome_step(outcome, depth, &step)) break;
        total += step.reward;
        scored_tokens++;
        if (step.token == SEQUENCE_DELIMITER) break;
    }
    *scored_token_count = scored_tokens;
    return total;
}

static double outcome_reward(Outcome *outcome) {
    int scored_tokens = 0;
    double total = outcome_log_probability_sum(outcome, &scored_tokens);
    return horizon_equivalent_reward(
        outcome->search,
        total,
        scored_tokens
    );
}

static double outcome_selection_score(Outcome *outcome) {
    return outcome_reward(outcome);
}

static bool outcome_reward_above(
    Outcome *outcome,
    double incumbent,
    double *score
) {
    double total = 0.0;
    int scored_tokens = 0;
    for (int depth = 0; depth < outcome->search->model->horizon; depth++) {
        RewardStep step;
        double forced_upper_bound = -INFINITY;
        RewardStepStatus status = outcome_step_above(
            outcome,
            depth,
            incumbent,
            &step,
            &forced_upper_bound
        );
        if (status == REWARD_STEP_REJECTED) {
            *score = forced_upper_bound;
            return false;
        }
        if (status == REWARD_STEP_END) {
            *score = horizon_equivalent_reward(
                outcome->search,
                total,
                scored_tokens
            );
            return reward_meets_cutoff(
                outcome->search,
                *score,
                incumbent
            );
        }
        total += step.reward;
        scored_tokens++;
        if (step.token == SEQUENCE_DELIMITER) {
            *score = horizon_equivalent_reward(
                outcome->search,
                total,
                scored_tokens
            );
            return reward_meets_cutoff(
                outcome->search,
                *score,
                incumbent
            );
        }
        /* Every future log-probability is nonpositive. The raw partial sum is
         * therefore an upper bound on the final horizon-equivalent reward:
         * the best hypothetical suffix adds zero and reaches the horizon. */
        if (partial_cannot_recover(
                outcome->search,
                total,
                incumbent
            )) {
            *score = total;
            return false;
        }
    }
    *score = total;
    return true;
}

typedef struct {
    Search *search;
} ModelObserverEnvironment;

/* The observer only pairs an already-composed model field family with a lazy
 * completion. It performs no model/layer/filler construction. */
static Outcome *observe_model_fields(
    void *raw_environment,
    Prefix *prefix,
    Completion *completion
) {
    ModelObserverEnvironment *environment = raw_environment;
    Outcome *outcome = arena_alloc(
        &environment->search->evaluator->run_arena,
        sizeof(*outcome)
    );
    environment->search->outcomes++;
    *outcome = (Outcome){
        .search = environment->search,
        .prefix = prefix,
        .completion = completion,
    };
    return outcome;
}

typedef struct SuffixEntry SuffixEntry;
struct SuffixEntry {
    int token;
    SuffixThunk thunk;
    SuffixEntry *next;
};

struct ProductFrame {
    Search *search;
    Prefix *history;
    int remaining;
    Observer observer;
    uint64_t sample_demand_budget;
    uint64_t sample_random_state;
    SuffixEntry *suffixes;
    uint64_t audit_id;
    uint64_t parent_frame_id;
    int via_token;
};

static void audit_escaped(FILE *stream, const char *text);
static int audit_prefix_text(FILE *stream, Search *search, Prefix *prefix);

static void strength_log_token_role(
    Search *search,
    const char *role,
    uint64_t frame_id,
    uint64_t source_frame_id,
    Prefix *before,
    Prefix *after,
    int token
) {
    FILE *stream = search->strength_stream;
    if (stream == NULL) return;
    if (before == NULL || after == NULL || after->parent != before ||
        after->token != token) {
        fail("strength token-role occurrence is not a prefix edge");
    }
    fprintf(
        stream,
        "strength_event id=%" PRIu64 " kind=token_role role=%s ",
        search->next_strength_event_id++,
        role
    );
    if (frame_id == UINT64_MAX) {
        fputs("frame=unit ", stream);
    } else {
        fprintf(stream, "frame=%" PRIu64 " ", frame_id);
    }
    if (source_frame_id == UINT64_MAX) {
        fputs("source_frame=none ", stream);
    } else {
        fprintf(stream, "source_frame=%" PRIu64 " ", source_frame_id);
    }
    fprintf(
        stream,
        "token=%d occurrence_prefix=%" PRIu32
        " position=%d piece=\"",
        token,
        after->id,
        before->depth
    );
    if (token == SEQUENCE_DELIMITER) {
        audit_escaped(stream, "<EOS>");
    } else {
        int previous = before->depth == 0 ? search->prompt_last_token :
            prefix_token_at(before, before->depth - 1);
        audit_escaped(
            stream,
            atkey_decode(search->evaluator->runtime, previous, token)
        );
    }
    fputs("\" context_before=\"", stream);
    audit_prefix_text(stream, search, before);
    fputs("\" context_after=\"", stream);
    audit_prefix_text(stream, search, after);
    fputs("\"\n", stream);
    fflush(stream);
}

static SuffixThunk *product_suffix(ProductFrame *frame, int token) {
    for (SuffixEntry *entry = frame->suffixes;
         entry != NULL;
         entry = entry->next) {
        if (entry->token == token) {
            strength_log(
                frame->search,
                "kind=suffix_lookup frame=%" PRIu64
                " token=%d result=reuse forced=%d prefix=%" PRIu32,
                frame->audit_id,
                token,
                entry->thunk.forced ? 1 : 0,
                entry->thunk.history->id
            );
            return &entry->thunk;
        }
    }
    Evaluator *evaluator = frame->search->evaluator;
    SuffixEntry *entry = arena_alloc(&evaluator->run_arena, sizeof(*entry));
    Prefix *next_history = prefix_child(
        &evaluator->prefixes,
        frame->history,
        token
    );
    entry->token = token;
    entry->thunk = (SuffixThunk){
        .forced = false,
        .value = NULL,
        .search = frame->search,
        .history = next_history,
        .remaining = frame->remaining - 1,
        .observer = frame->observer,
        .owner_frame_id = frame->audit_id,
        .head_token = token,
    };
    entry->next = frame->suffixes;
    frame->suffixes = entry;
    strength_log(
        frame->search,
        "kind=suffix_bind frame=%" PRIu64
        " token=%d child_prefix=%" PRIu32 " child_depth=%d remaining=%d",
        frame->audit_id,
        token,
        next_history->id,
        next_history->depth,
        entry->thunk.remaining
    );
    return &entry->thunk;
}

static Outcome *observe_candidate(
    ProductFrame *frame,
    int token,
    const char *purpose,
    uint64_t suffix_demand_budget
) {
    strength_log(
        frame->search,
        "kind=observer_apply frame=%" PRIu64
        " token=%d purpose=%s prefix=%" PRIu32,
        frame->audit_id,
        token,
        purpose,
        frame->history->id
    );
    SuffixThunk *suffix = product_suffix(frame, token);
    if (strcmp(purpose, "candidate") == 0) {
        if (frame->search->sampling_enabled) {
            if (suffix->sample_budget_assigned &&
                suffix->sample_demand_budget != suffix_demand_budget) {
                fail("sampled suffix received inconsistent demand budgets");
            }
            suffix->sample_budget_assigned = true;
            suffix->sample_demand_budget = suffix_demand_budget;
            strength_log(
                frame->search,
                "kind=suffix_budget frame=%" PRIu64
                " token=%d demand_budget=%" PRIu64,
                frame->audit_id,
                token,
                suffix_demand_budget
            );
        }
        strength_log_token_role(
            frame->search,
            "island",
            frame->audit_id,
            UINT64_MAX,
            frame->history,
            suffix->history,
            token
        );
    }
    Completion *candidate = new_completion(
        frame->search,
        frame->history,
        token,
        NULL,
        suffix,
        NULL,
        0
    );
    Outcome *outcome = frame->observer.apply(
        frame->observer.environment,
        frame->history,
        candidate
    );
    strength_log(
        frame->search,
        "kind=observer_return frame=%" PRIu64
        " token=%d purpose=%s outcome_prefix=%" PRIu32,
        frame->audit_id,
        token,
        purpose,
        outcome->prefix->id
    );
    return outcome;
}

static int compare_local_candidate(
    float left_logit,
    int left_token,
    float right_logit,
    int right_token
) {
    if (left_logit > right_logit) return -1;
    if (left_logit < right_logit) return 1;
    if (left_token < right_token) return -1;
    if (left_token > right_token) return 1;
    return 0;
}

static void top_k_tokens(const Vec *logits, int count, int *tokens) {
    if (count <= 0 || count > logits->width) fail("invalid top-k");
    int filled = 0;
    for (int token = 0; token < logits->width; token++) {
        int insertion = filled;
        while (insertion > 0) {
            int previous = tokens[insertion - 1];
            if (compare_local_candidate(
                    logits->values[previous],
                    previous,
                    logits->values[token],
                    token
                ) <= 0) {
                break;
            }
            insertion--;
        }
        if (insertion < count) {
            int last = filled < count ? filled : count - 1;
            for (int index = last; index > insertion; index--) {
                tokens[index] = tokens[index - 1];
            }
            tokens[insertion] = token;
            if (filled < count) filled++;
        }
    }
    if (filled != count) fail("top-k support incomplete");
}

static void audit_escaped(FILE *stream, const char *text) {
    for (const unsigned char *cursor = (const unsigned char *)text;
         *cursor != '\0';
         cursor++) {
        unsigned char byte = *cursor;
        if (byte == '\\' || byte == '"') {
            fputc('\\', stream);
            fputc(byte, stream);
        } else if (byte == '\n') {
            fputs("\\n", stream);
        } else if (byte == '\r') {
            fputs("\\r", stream);
        } else if (byte == '\t') {
            fputs("\\t", stream);
        } else if (byte < 0x20 || byte == 0x7f) {
            fprintf(stream, "\\u%04x", byte);
        } else {
            fputc(byte, stream);
        }
    }
}

static int audit_prefix_text(FILE *stream, Search *search, Prefix *prefix) {
    audit_escaped(stream, search->prompt_text);
    int previous = search->prompt_last_token;
    for (int index = 0; index < prefix->depth; index++) {
        int token = prefix_token_at(prefix, index);
        if (token == SEQUENCE_DELIMITER) {
            audit_escaped(stream, "<EOS>");
            return token;
        }
        audit_escaped(
            stream,
            atkey_decode(search->evaluator->runtime, previous, token)
        );
        previous = token;
    }
    return previous;
}

static void audit_frame(ProductFrame *frame) {
    FILE *stream = frame->search->audit_stream;
    if (stream == NULL) return;
    fprintf(
        stream,
        "frame id=%" PRIu64 " depth=%d remaining=%d prefix=\"",
        frame->audit_id,
        frame->history->depth,
        frame->remaining
    );
    audit_prefix_text(stream, frame->search, frame->history);
    fputs("\"\n", stream);
    fflush(stream);
}

static void audit_append(
    FILE *stream,
    Search *search,
    Prefix *prefix,
    int token
) {
    fputc('"', stream);
    int previous = search->prompt_last_token;
    if (prefix->depth > 0) {
        previous = prefix_token_at(prefix, prefix->depth - 1);
    }
    if (token == SEQUENCE_DELIMITER) {
        audit_escaped(stream, "<EOS>");
    } else {
        audit_escaped(
            stream,
            atkey_decode(search->evaluator->runtime, previous, token)
        );
    }
    fputc('"', stream);
}

static void audit_continuation(
    FILE *stream,
    ProductFrame *frame,
    Outcome *outcome
) {
    Completion *candidate = completion_at(
        outcome,
        frame->history->depth
    );
    fputc('"', stream);
    if (candidate == NULL || candidate->token == SEQUENCE_DELIMITER) {
        fputc('"', stream);
        return;
    }
    int previous = candidate->token;
    Completion *completion = completion_tail(candidate);
    for (int index = frame->history->depth + 1;
         completion != NULL && index < frame->search->model->horizon;
         index++) {
        int token = completion->token;
        if (token == SEQUENCE_DELIMITER) {
            audit_escaped(stream, "<EOS>");
            break;
        }
        audit_escaped(
            stream,
            atkey_decode(frame->search->evaluator->runtime, previous, token)
        );
        previous = token;
        completion = completion_tail(completion);
    }
    fputc('"', stream);
}

static double outcome_partial_reward(Outcome *outcome, int final_depth) {
    double total = 0.0;
    for (int depth = 0; depth <= final_depth; depth++) {
        RewardStep step;
        if (!outcome_step(outcome, depth, &step)) break;
        total += step.reward;
        if (step.token == SEQUENCE_DELIMITER) break;
    }
    return total;
}

static void audit_candidate_partial(
    ProductFrame *frame,
    int local_rank,
    const Vec *logits,
    int token,
    Outcome *outcome
) {
    FILE *stream = frame->search->audit_stream;
    if (stream == NULL) return;
    fprintf(
        stream,
        "candidate frame=%" PRIu64 " rank=%d append=",
        frame->audit_id,
        local_rank
    );
    audit_append(
        stream,
        frame->search,
        frame->history,
        token
    );
    fprintf(
        stream,
        " local_logit=%.9g "
        "local_log_probability=%.17g partial_score=%.17g\n",
        logits->values[token],
        token_log_probability(logits, token),
        outcome_partial_reward(outcome, frame->history->depth)
    );
    fflush(stream);
}

static void audit_candidate_result(
    ProductFrame *frame,
    int local_rank,
    Outcome *outcome,
    double score,
    bool exact
) {
    FILE *stream = frame->search->audit_stream;
    if (stream == NULL) return;
    fprintf(
        stream,
        "result frame=%" PRIu64 " rank=%d %s=%.17g",
        frame->audit_id,
        local_rank,
        exact ? "exact_score" : "pruned_upper_bound",
        score
    );
    if (exact) {
        fputs(" continuation=", stream);
        audit_continuation(stream, frame, outcome);
    }
    fputc('\n', stream);
    fflush(stream);
}

static void audit_selected(ProductFrame *frame, int rank, double score) {
    FILE *stream = frame->search->audit_stream;
    if (stream == NULL) return;
    fprintf(
        stream,
        "selected frame=%" PRIu64 " rank=%d exact_score=%.17g\n",
        frame->audit_id,
        rank,
        score
    );
    fflush(stream);
}

static void audit_rejected(
    ProductFrame *frame,
    double cutoff,
    double upper_bound
) {
    FILE *stream = frame->search->audit_stream;
    if (stream == NULL) return;
    fprintf(
        stream,
        "rejected frame=%" PRIu64 " cutoff=%.17g upper_bound=%.17g\n",
        frame->audit_id,
        cutoff,
        upper_bound
    );
    fflush(stream);
}

#define SAMPLE_RANDOM_SEED UINT64_C(0x4d595df4d0f33173)

static uint64_t sample_random_mix(uint64_t value) {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31;
    return value;
}

/* Sampling order belongs to a history selection, not to the evaluator's
 * depth-first traversal. Deriving it from the prompt and prefix makes the
 * demanded support at each Select stable as a resource budget grows. */
static uint64_t sample_frame_random_state(
    const Search *search,
    Prefix *history
) {
    uint64_t hash = SAMPLE_RANDOM_SEED;
    for (const unsigned char *byte =
             (const unsigned char *)search->prompt_text;
         *byte != '\0';
         byte++) {
        hash ^= (uint64_t)*byte;
        hash *= UINT64_C(0x100000001b3);
    }
    for (int index = 0; index < history->depth; index++) {
        hash ^= (uint64_t)(uint32_t)prefix_token_at(history, index);
        hash *= UINT64_C(0x100000001b3);
    }
    hash ^= (uint64_t)(uint32_t)history->depth;
    hash = sample_random_mix(hash);
    return hash == 0 ? SAMPLE_RANDOM_SEED : hash;
}

static uint64_t sample_random_next(uint64_t *state) {
    uint64_t value = *state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    return value * UINT64_C(2685821657736338717);
}

static double sample_random_unit(uint64_t *state) {
    return (double)(sample_random_next(state) >> 11) *
        (1.0 / 9007199254740992.0);
}

/* A wall-clock deadline may stop only between observer applications. Once a
 * candidate continuation has been demanded, the recursive Select product is
 * allowed to finish that demand and return its backed value. A newly entered
 * frame is also allowed one demand so every sampled local Select remains
 * total. Fixed demand budgets are divided structurally by
 * history_product_select instead of being polled globally here. */
static bool sample_time_allows_new_demand(
    Search *search,
    bool frame_already_demanded
) {
    if (!search->sampling_enabled) return true;
    if (search->sample_deadline_armed) {
        struct timespec now;
        if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
            fail("could not read monotonic sampling clock");
        }
        if (time_reached(now, search->sample_deadline)) {
            search->sample_budget_exhausted = true;
            search->sample_stop_reason = SAMPLE_STOP_DEADLINE;
        }
    }
    if (search->sample_stop_reason != SAMPLE_STOP_DEADLINE) return true;
    if (!frame_already_demanded) {
        search->sampled_forced_demands++;
        return true;
    }
    search->sampled_truncated_frames++;
    return false;
}

/* Draw the next local continuation without replacement from the model's
 * top-k logits.  This changes only which observer applications are demanded;
 * the observer reward used by Select is untouched. */
static int sample_next_local_rank(
    ProductFrame *frame,
    const Vec *logits,
    const int *ranked_support,
    const bool *already_sampled,
    double *support_probability,
    double *draw
) {
    Search *search = frame->search;
    double maximum = -DBL_MAX;
    int remaining = 0;
    for (int rank = 0; rank < search->top_k; rank++) {
        if (already_sampled[rank]) continue;
        int token = ranked_support[rank];
        if (token < 0 || token >= logits->width) {
            fail("sampled local support token is outside the vocabulary");
        }
        double logit = logits->values[token];
        if (logit > maximum) maximum = logit;
        remaining++;
    }
    if (remaining == 0) fail("sampled an empty local continuation support");

    double mass = 0.0;
    for (int rank = 0; rank < search->top_k; rank++) {
        if (already_sampled[rank]) continue;
        mass += exp(
            (double)logits->values[ranked_support[rank]] - maximum
        );
    }
    if (!(mass > 0.0) || !isfinite(mass)) {
        fail("sampled local continuation support has invalid mass");
    }

    *draw = sample_random_unit(&frame->sample_random_state);
    double target = *draw * mass;
    double cumulative = 0.0;
    int selected = -1;
    for (int rank = 0; rank < search->top_k; rank++) {
        if (already_sampled[rank]) continue;
        cumulative += exp(
            (double)logits->values[ranked_support[rank]] - maximum
        );
        selected = rank;
        if (target < cumulative) break;
    }
    if (selected < 0) fail("failed to sample a local continuation");
    *support_probability = exp(
        (double)logits->values[ranked_support[selected]] - maximum
    ) / mass;
    return selected;
}

static void audit_sampled_continuation_demand(
    ProductFrame *frame,
    int ordinal,
    int local_rank,
    int token,
    double support_probability,
    double draw,
    uint64_t suffix_demand_budget
) {
    Search *search = frame->search;
    strength_log(
        search,
        "kind=continuation_sample frame=%" PRIu64
        " ordinal=%d local_rank=%d token=%d "
        "support_probability=%.17g draw=%.17g suffix_demand_budget=%" PRIu64,
        frame->audit_id,
        ordinal,
        local_rank,
        token,
        support_probability,
        draw,
        suffix_demand_budget
    );
    if (search->audit_stream == NULL) return;
    fprintf(
        search->audit_stream,
        "continuation_sample frame=%" PRIu64
        " ordinal=%d local_rank=%d support_probability=%.17g "
        "draw=%.17g suffix_demand_budget=%" PRIu64 " append=",
        frame->audit_id,
        ordinal,
        local_rank,
        support_probability,
        draw,
        suffix_demand_budget
    );
    audit_append(search->audit_stream, search, frame->history, token);
    fputc('\n', search->audit_stream);
    fflush(search->audit_stream);
}

/* Escardo's dependent product specialized to token histories. The candidate
 * suffix function is the ProductFrame's memoized suffix table, corresponding
 * to Escardo's where-bound function tree. */
static ProductSelection history_product_select(
    Search *search,
    Prefix *history,
    int remaining,
    Observer observer,
    uint64_t sample_demand_budget,
    bool bounded_demand,
    double cutoff,
    uint64_t parent_frame_id,
    int via_token
) {
    if (remaining == 0) {
        if (parent_frame_id != UINT64_MAX) {
            strength_log_token_role(
                search,
                "bridge",
                UINT64_MAX,
                parent_frame_id,
                history->parent,
                history,
                via_token
            );
        }
        strength_log(
            search,
            "kind=unit parent_frame=%" PRIu64
            " via_token=%d prefix=%" PRIu32 " depth=%d",
            parent_frame_id,
            via_token,
            history->id,
            history->depth
        );
        return (ProductSelection){
            .selected = true,
            .completion = NULL,
            .score = NAN,
        };
    }
    if (remaining != search->model->horizon - history->depth) {
        fail("selection suffix depth invariant failed");
    }
    ProductFrame *frame = arena_alloc(
        &search->evaluator->run_arena,
        sizeof(*frame)
    );
    *frame = (ProductFrame){
        .search = search,
        .history = history,
        .remaining = remaining,
        .observer = observer,
        .sample_demand_budget = sample_demand_budget,
        .sample_random_state = sample_frame_random_state(search, history),
        .suffixes = NULL,
        .audit_id = search->next_frame_id++,
        .parent_frame_id = parent_frame_id,
        .via_token = via_token,
    };
    if (frame->parent_frame_id != UINT64_MAX) {
        strength_log_token_role(
            search,
            "bridge",
            frame->audit_id,
            frame->parent_frame_id,
            history->parent,
            history,
            frame->via_token
        );
    }
    if (frame->parent_frame_id == UINT64_MAX) {
        strength_log(
            search,
            "kind=select_enter frame=%" PRIu64
            " parent_frame=root via_token=none prefix=%" PRIu32
            " depth=%d remaining=%d bounded_demand=%d cutoff=%.17g",
            frame->audit_id,
            history->id,
            history->depth,
            remaining,
            bounded_demand ? 1 : 0,
            cutoff
        );
    } else {
        strength_log(
            search,
            "kind=select_enter frame=%" PRIu64 " parent_frame=%" PRIu64
            " via_token=%d prefix=%" PRIu32 " depth=%d remaining=%d "
            "bounded_demand=%d cutoff=%.17g",
            frame->audit_id,
            frame->parent_frame_id,
            frame->via_token,
            history->id,
            history->depth,
            remaining,
            bounded_demand ? 1 : 0,
            cutoff
        );
    }
    audit_frame(frame);

    if (history->terminated) {
        if (search->audit_stream != NULL) {
            fprintf(
                search->audit_stream,
                "unit frame=%" PRIu64 " append=\"<EOS>\"\n",
                frame->audit_id
            );
            fflush(search->audit_stream);
        }
        strength_log(
            search,
            "kind=deterministic_unit frame=%" PRIu64
            " token=%d reason=terminated_history",
            frame->audit_id,
            SEQUENCE_DELIMITER
        );
        Completion *selected = new_completion(
            search,
            history,
            SEQUENCE_DELIMITER,
            NULL,
            product_suffix(frame, SEQUENCE_DELIMITER),
            NULL,
            0
        );
        strength_log(
            search,
            "kind=compose_return frame=%" PRIu64
            " token=%d suffix=lazy score=nan",
            frame->audit_id,
            SEQUENCE_DELIMITER
        );
        return (ProductSelection){
            .selected = true,
            .completion = selected,
            .score = NAN,
        };
    }

    Outcome *probe = observe_candidate(
        frame,
        0,
        "support_probe",
        UINT64_MAX
    );
    RewardStep current_step;
    if (!outcome_step(probe, history->depth, &current_step)) {
        fail("selector could not observe current logits");
    }
    int support[search->top_k];
    memset(support, 0, sizeof(support));
    top_k_tokens(current_step.logits, search->top_k, support);
    bool sampled_ranks[search->top_k];
    memset(sampled_ranks, 0, sizeof(sampled_ranks));

    CandidateAudit *audits = NULL;
    if (search->trace) {
        audits = arena_alloc(
            &search->evaluator->run_arena,
            (size_t)search->top_k * sizeof(*audits)
        );
    }

    bool found = false;
    int best_token = -1;
    int best_rank = -1;
    double best_score = -INFINITY;
    double best_upper_bound = -INFINITY;
    int demanded_count = 0;

    int demand_cap = search->top_k;
    uint64_t child_budget_base = UINT64_MAX;
    uint64_t child_budget_remainder = 0;
    if (search->sampling_enabled && sample_demand_budget != UINT64_MAX) {
        if (sample_demand_budget < (uint64_t)remaining) {
            fail("sampled subtree demand budget cannot produce one completion");
        }
        uint64_t possible_current_demands =
            sample_demand_budget / (uint64_t)remaining;
        if (possible_current_demands < (uint64_t)demand_cap) {
            demand_cap = (int)possible_current_demands;
        }
        if (demand_cap <= 0) {
            fail("sampled subtree has no current continuation demand");
        }
        uint64_t unused_budget = 0;
        if (demand_cap < search->top_k) {
            /* Admit complete candidate subtrees in fixed sampled order. Until
             * every local alternative has one complete path, surplus below
             * the next admission threshold stays unused. This makes the
             * recursively demanded support monotone in the root budget. */
            child_budget_base = (uint64_t)(remaining - 1);
            child_budget_remainder = 0;
            uint64_t used_budget =
                (uint64_t)demand_cap * (uint64_t)remaining;
            unused_budget = sample_demand_budget - used_budget;
        } else {
            uint64_t child_budget_total =
                sample_demand_budget - (uint64_t)demand_cap;
            child_budget_base = child_budget_total /
                (uint64_t)demand_cap;
            child_budget_remainder = child_budget_total %
                (uint64_t)demand_cap;
        }
        if (remaining > 1 &&
            child_budget_base < (uint64_t)(remaining - 1)) {
            fail("sampled child budget cannot produce one suffix");
        }
        strength_log(
            search,
            "kind=demand_budget_split frame=%" PRIu64
            " subtree_budget=%" PRIu64 " demand_cap=%d"
            " child_budget_base=%" PRIu64
            " child_budget_remainder=%" PRIu64
            " unused_budget=%" PRIu64,
            frame->audit_id,
            sample_demand_budget,
            demand_cap,
            child_budget_base,
            child_budget_remainder,
            unused_budget
        );
        if (demand_cap < search->top_k) {
            search->sample_budget_exhausted = true;
            if (search->sample_stop_reason != SAMPLE_STOP_DEADLINE) {
                search->sample_stop_reason = SAMPLE_STOP_DEMAND_LIMIT;
            }
            search->sampled_truncated_frames++;
            strength_log(
                search,
                "kind=select_truncate_support frame=%" PRIu64
                " retained=%d support=%d stop_reason=demand_limit",
                frame->audit_id,
                demand_cap,
                search->top_k
            );
        }
    }

    for (int ordinal = 0; ordinal < demand_cap; ordinal++) {
        if (!sample_time_allows_new_demand(
                search,
                demanded_count > 0
            )) {
            strength_log(
                search,
                "kind=select_truncate frame=%" PRIu64
                " demanded=%d stop_reason=%s",
                frame->audit_id,
                demanded_count,
                sample_stop_name(search->sample_stop_reason)
            );
            break;
        }

        uint64_t suffix_demand_budget = UINT64_MAX;
        if (search->sampling_enabled &&
            sample_demand_budget != UINT64_MAX) {
            suffix_demand_budget = child_budget_base +
                ((uint64_t)ordinal < child_budget_remainder ? 1 : 0);
        }

        int candidate_rank = ordinal;
        if (search->sampling_enabled) {
            double support_probability = 0.0;
            double draw = 0.0;
            candidate_rank = sample_next_local_rank(
                frame,
                current_step.logits,
                support,
                sampled_ranks,
                &support_probability,
                &draw
            );
            sampled_ranks[candidate_rank] = true;
            audit_sampled_continuation_demand(
                frame,
                ordinal,
                candidate_rank,
                support[candidate_rank],
                support_probability,
                draw,
                suffix_demand_budget
            );
            search->sampled_candidate_demands++;
        }
        int candidate = support[candidate_rank];
        int audit_index = demanded_count++;
        Outcome *candidate_outcome = observe_candidate(
            frame,
            candidate,
            "candidate",
            suffix_demand_budget
        );
        audit_candidate_partial(
            frame,
            candidate_rank,
            current_step.logits,
            candidate,
            candidate_outcome
        );
        double candidate_score = 0.0;
        bool exact;
        const char *demand_mode;
        double demand_cutoff;
        if (search->threshold_enabled) {
            demand_mode = "threshold";
            demand_cutoff = search->reward_threshold;
        } else if (!search->bounded || (!bounded_demand && !found)) {
            demand_mode = "exact";
            demand_cutoff = -INFINITY;
        } else {
            demand_mode = "bounded";
            demand_cutoff = found ? best_score : cutoff;
        }
        strength_log(
            search,
            "kind=bellman_demand frame=%" PRIu64
            " rank=%d token=%d mode=%s cutoff=%.17g",
            frame->audit_id,
            candidate_rank,
            candidate,
            demand_mode,
            demand_cutoff
        );
        if (search->threshold_enabled) {
            candidate_score = 0.0;
            exact = outcome_reward_above(
                candidate_outcome,
                search->reward_threshold,
                &candidate_score
            );
        } else if (!search->bounded || (!bounded_demand && !found)) {
            candidate_score = outcome_reward(candidate_outcome);
            exact = true;
        } else {
            double incumbent = found ? best_score : cutoff;
            exact = outcome_reward_above(
                candidate_outcome,
                incumbent,
                &candidate_score
            );
        }
        audit_candidate_result(
            frame,
            candidate_rank,
            candidate_outcome,
            candidate_score,
            exact
        );
        strength_log(
            search,
            "kind=bellman_return frame=%" PRIu64
            " rank=%d token=%d status=%s score=%.17g",
            frame->audit_id,
            candidate_rank,
            candidate,
            exact ? "exact" : "upper_bound",
            candidate_score
        );
        if (audits != NULL) {
            audits[audit_index] = (CandidateAudit){
                candidate,
                candidate_outcome,
            };
        }
        if (!exact) {
            if (candidate_score > best_upper_bound) {
                best_upper_bound = candidate_score;
            }
            continue;
        }
        if (search->threshold_enabled) {
            audit_selected(frame, candidate_rank, candidate_score);
            strength_log(
                search,
                "kind=select_choose frame=%" PRIu64
                " rank=%d token=%d backed_reward=%.17g reason=threshold",
                frame->audit_id,
                candidate_rank,
                candidate,
                candidate_score
            );
            Completion *selected = new_completion(
                search,
                history,
                candidate,
                NULL,
                product_suffix(frame, candidate),
                audits,
                audits != NULL ? demanded_count : 0
            );
            strength_log(
                search,
                "kind=compose_return frame=%" PRIu64
                " token=%d suffix=lazy score=%.17g",
                frame->audit_id,
                candidate,
                candidate_score
            );
            return (ProductSelection){
                .selected = true,
                .completion = selected,
                .score = candidate_score,
            };
        }
        bool improves = !found;
        if (found) {
            improves = candidate_score > best_score ||
                (candidate_score == best_score &&
                 candidate_rank < best_rank);
        }
        if (improves) {
            found = true;
            best_token = candidate;
            best_rank = candidate_rank;
            best_score = candidate_score;
            strength_log(
                search,
                "kind=incumbent_update frame=%" PRIu64
                " rank=%d token=%d backed_reward=%.17g",
                frame->audit_id,
                candidate_rank,
                candidate,
                candidate_score
            );
        }
    }

    if (!found) {
        if (best_upper_bound == -INFINITY) best_upper_bound = cutoff;
        audit_rejected(frame, cutoff, best_upper_bound);
        strength_log(
            search,
            "kind=select_reject frame=%" PRIu64
            " cutoff=%.17g upper_bound=%.17g",
            frame->audit_id,
            cutoff,
            best_upper_bound
        );
        return (ProductSelection){
            .selected = false,
            .completion = NULL,
            .score = best_upper_bound,
        };
    }

    audit_selected(frame, best_rank, best_score);
    strength_log(
        search,
        "kind=select_choose frame=%" PRIu64
        " rank=%d token=%d backed_reward=%.17g reason=maximum",
        frame->audit_id,
        best_rank,
        best_token,
        best_score
    );

    Completion *selected = new_completion(
        search,
        history,
        best_token,
        NULL,
        product_suffix(frame, best_token),
        audits,
        audits != NULL ? demanded_count : 0
    );
    strength_log(
        search,
        "kind=compose_return frame=%" PRIu64
        " token=%d suffix=lazy score=%.17g",
        frame->audit_id,
        best_token,
        best_score
    );
    return (ProductSelection){
        .selected = true,
        .completion = selected,
        .score = best_score,
    };
}

typedef struct {
    Search search;
    ModelObserverEnvironment model_observer;
    Observer observer;
} AtkeyProgram;

static void compose_program(
    AtkeyProgram *program,
    Evaluator *evaluator,
    ModelTerm *model,
    const char *prompt_text,
    int prompt_last_token,
    int top_k,
    bool bounded,
    bool threshold_enabled,
    double reward_threshold,
    int sample_milliseconds,
    int sample_demand_limit,
    bool trace,
    FILE *audit_stream,
    FILE *strength_stream
) {
    memset(program, 0, sizeof(*program));
    program->search = (Search){
        .evaluator = evaluator,
        .model = model,
        .prompt_text = prompt_text,
        .prompt_last_token = prompt_last_token,
        .top_k = top_k,
        .bounded = bounded,
        .threshold_enabled = threshold_enabled,
        .reward_threshold = reward_threshold,
        .sampling_enabled = sample_milliseconds > 0 ||
            sample_demand_limit > 0,
        .sample_milliseconds = sample_milliseconds,
        .sample_demand_limit = sample_demand_limit,
        .trace = trace,
        .audit_stream = audit_stream,
        .strength_stream = strength_stream,
        .next_strength_event_id = 0,
        .next_frame_id = 0,
    };
    program->model_observer.search = &program->search;
    program->observer = (Observer){
        observe_model_fields,
        &program->model_observer,
    };
}

/* J_R -> K_R: p(selection e p), followed by the sole final run. */
static Outcome *run_pcont(AtkeyProgram *program) {
    Search *search = &program->search;
    strength_log(
        search,
        "kind=tau_begin mode=%s horizon=%d top_k=%d",
        search->sampling_enabled ? "sampled_recursive" : "exact",
        search->model->horizon,
        search->top_k
    );
    if (search->sample_milliseconds > 0) {
        struct timespec start;
        if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
            fail("could not start monotonic sampling clock");
        }
        search->sample_deadline = add_milliseconds(
            start,
            search->sample_milliseconds
        );
        search->sample_deadline_armed = true;
    }
    double cutoff = program->search.threshold_enabled ?
        program->search.reward_threshold : -INFINITY;
    ProductSelection selection = history_product_select(
        &program->search,
        program->search.evaluator->prefixes.root,
        program->search.model->horizon,
        program->observer,
        program->search.sample_demand_limit > 0 ?
            (uint64_t)program->search.sample_demand_limit : UINT64_MAX,
        false,
        cutoff,
        UINT64_MAX,
        -1
    );
    if (!selection.selected) {
        if (program->search.threshold_enabled) {
            fail("no completion satisfies the reward threshold");
        }
        fail("root selection was rejected");
    }
    if (search->sampling_enabled && !search->sample_budget_exhausted) {
        search->sample_stop_reason = SAMPLE_STOP_SUPPORT_COMPLETE;
    }
    strength_log(
        search,
        "kind=tau_selection_return selected=1 score=%.17g",
        selection.score
    );
    Outcome *outcome = program->observer.apply(
        program->observer.environment,
        program->search.evaluator->prefixes.root,
        selection.completion
    );
    strength_log(
        search,
        "kind=tau_observer_return outcome_prefix=%" PRIu32,
        outcome->prefix->id
    );
    return outcome;
}

typedef struct {
    const char *checkpoint;
    const char *tokenizer;
    const char *prompt;
    const char *candidate_store_path;
    int length;
    int top_k;
    bool bounded;
    bool threshold_enabled;
    double reward_threshold;
    int sample_milliseconds;
    int sample_demand_limit;
    bool trace;
    const char *audit_path;
    const char *strength_path;
} Options;

static int parse_integer(const char *text, const char *flag) {
    errno = 0;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value < INT32_MIN || value > INT32_MAX) {
        fprintf(stderr, "atkey term: %s expects an integer\n", flag);
        exit(EXIT_FAILURE);
    }
    return (int)value;
}

static double parse_real(const char *text, const char *flag) {
    errno = 0;
    char *end = NULL;
    double value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value)) {
        fprintf(stderr, "atkey term: %s expects a finite number\n", flag);
        exit(EXIT_FAILURE);
    }
    return value;
}

static Options parse_options(int argc, char **argv) {
    if (argc < 2) fail("missing checkpoint");
    Options options = {
        .checkpoint = argv[1],
        .tokenizer = "tokenizer.bin",
        .prompt = "Once upon a time",
        .candidate_store_path = NULL,
        .length = 16,
        .top_k = 2,
        .bounded = true,
        .threshold_enabled = false,
        .reward_threshold = -INFINITY,
        .sample_milliseconds = 0,
        .sample_demand_limit = 0,
        .trace = false,
        .audit_path = NULL,
        .strength_path = NULL,
    };
    for (int index = 2; index < argc; index++) {
        const char *flag = argv[index];
        if (index + 1 >= argc) fail("incomplete option");
        const char *value = argv[++index];
        if (strcmp(flag, "-z") == 0) {
            options.tokenizer = value;
        } else if (strcmp(flag, "-n") == 0) {
            options.length = parse_integer(value, flag);
        } else if (strcmp(flag, "-k") == 0) {
            options.top_k = parse_integer(value, flag);
        } else if (strcmp(flag, "-i") == 0) {
            options.prompt = value;
        } else if (strcmp(flag, "-b") == 0) {
            int parsed = parse_integer(value, flag);
            if (parsed != 0 && parsed != 1) fail("-b expects 0 or 1");
            options.bounded = parsed == 1;
        } else if (strcmp(flag, "-d") == 0) {
            int parsed = parse_integer(value, flag);
            if (parsed != 0 && parsed != 1) fail("-d expects 0 or 1");
            options.trace = parsed == 1;
        } else if (strcmp(flag, "-s") == 0) {
            options.threshold_enabled = true;
            options.reward_threshold = parse_real(value, flag);
        } else if (strcmp(flag, "--sample-ms") == 0) {
            options.sample_milliseconds = parse_integer(value, flag);
        } else if (strcmp(flag, "--sample-demands") == 0) {
            options.sample_demand_limit = parse_integer(value, flag);
        } else if (strcmp(flag, "-a") == 0) {
            options.audit_path = value;
        } else if (strcmp(flag, "--strength-log") == 0) {
            options.strength_path = value;
        } else if (strcmp(flag, "-m") == 0) {
            options.candidate_store_path = value;
        } else {
            fprintf(stderr, "atkey term: unknown option: %s\n", flag);
            exit(EXIT_FAILURE);
        }
    }
    if (options.length <= 0) fail("completion length must be positive");
    if (options.top_k <= 0) fail("top-k must be positive");
    if (options.sample_milliseconds < 0) {
        fail("--sample-ms must be nonnegative");
    }
    if (options.sample_demand_limit < 0) {
        fail("--sample-demands must be nonnegative");
    }
    if (options.sample_demand_limit > 0 &&
        options.sample_demand_limit < options.length) {
        fail("--sample-demands must be at least the completion length");
    }
    bool sampling_enabled = options.sample_milliseconds > 0 ||
        options.sample_demand_limit > 0;
    if (sampling_enabled && options.threshold_enabled) {
        fail("-s cannot be combined with a truncated demand support");
    }
    if (options.strength_path != NULL && options.audit_path != NULL &&
        strcmp(options.strength_path, options.audit_path) == 0) {
        fail("candidate audit and strength log paths must differ");
    }
    return options;
}

static size_t total_filler_applications(AtkeyRuntime *runtime) {
    size_t total = 0;
    int count = atkey_filler_count(runtime);
    for (int filler = 0; filler < count; filler++) {
        total += atkey_filler_calls(runtime, filler);
    }
    return total;
}

static void print_weight_evidence(AtkeyRuntime *runtime) {
    int count = atkey_filler_count(runtime);
    size_t total_calls = 0;
    size_t total_reads = 0;
    size_t maximum_calls = 0;
    for (int filler = 0; filler < count; filler++) {
        size_t calls = atkey_filler_calls(runtime, filler);
        size_t reads = atkey_filler_scalar_reads(runtime, filler);
        total_calls += calls;
        total_reads += reads;
        if (calls > maximum_calls) maximum_calls = calls;
    }
    printf(
        "learned_filler_count=%d learned_filler_applications=%zu "
        "learned_scalar_accesses=%zu max_applications_per_filler=%zu\n",
        count,
        total_calls,
        total_reads,
        maximum_calls
    );
    puts("filler_applications:");
    for (int filler = 0; filler < count; filler++) {
        printf(
            "  filler=%d applications=%zu scalar_accesses=%zu\n",
            filler,
            atkey_filler_calls(runtime, filler),
            atkey_filler_scalar_reads(runtime, filler)
        );
    }
}

static int collect_tokens(Outcome *outcome, int *tokens) {
    int count = 0;
    for (int index = 0;
         index < outcome->prefix->depth &&
         count < outcome->search->model->horizon;
         index++) {
        int token = prefix_token_at(outcome->prefix, index);
        tokens[count++] = token;
        if (token == SEQUENCE_DELIMITER) return count;
    }
    Completion *completion = outcome->completion;
    while (completion != NULL && count < outcome->search->model->horizon) {
        tokens[count++] = completion->token;
        if (completion->token == SEQUENCE_DELIMITER) break;
        completion = completion_tail(completion);
    }
    return count;
}

static void print_token_array(const int *tokens, int count) {
    putchar('[');
    for (int index = 0; index < count; index++) {
        if (index != 0) putchar(',');
        printf("%d", tokens[index]);
    }
    putchar(']');
}

static void print_completion_text(
    AtkeyRuntime *runtime,
    int previous,
    const int *tokens,
    int count
) {
    for (int index = 0; index < count; index++) {
        int token = tokens[index];
        if (token == SEQUENCE_DELIMITER) break;
        atkey_print_piece(runtime, previous, token);
        previous = token;
    }
    putchar('\n');
}

static void print_local_top_k(
    const Vec *logits,
    int top_k,
    const int *support
) {
    putchar('[');
    for (int index = 0; index < top_k; index++) {
        int token = support[index];
        if (index != 0) putchar(',');
        printf(
            "(%d,%.9g,%.17g)",
            token,
            logits->values[token],
            token_log_probability(logits, token)
        );
    }
    putchar(']');
}

static void print_decision_trace(Outcome *outcome) {
    Search *search = outcome->search;
    puts("selection_trace:");
    double cumulative = 0.0;
    Completion *completion = outcome->completion;
    for (int depth = 0;
         depth < search->model->horizon && completion != NULL;
         depth++) {
        RewardStep step;
        if (!outcome_step(outcome, depth, &step)) break;
        cumulative += step.reward;
        int support[search->top_k];
        top_k_tokens(step.logits, search->top_k, support);
        int rank = -1;
        for (int index = 0; index < search->top_k; index++) {
            if (support[index] == step.token) rank = index;
        }
        if (rank < 0) fail("selected token outside local support");
        printf(
            "  depth=%d selected_token=%d selected_local_rank=%d "
            "selected_logit=%.9g selected_log_probability=%.17g "
            "cumulative_reward=%.17g\n",
            depth,
            step.token,
            rank,
            step.logits->values[step.token],
            step.reward,
            cumulative
        );
        fputs("    local_top_k=", stdout);
        print_local_top_k(step.logits, search->top_k, support);
        putchar('\n');
        puts("    continuation_candidates:");
        for (int index = 0; index < completion->audit_count; index++) {
            CandidateAudit audit = completion->audits[index];
            int tokens[search->model->horizon];
            int count = collect_tokens(audit.outcome, tokens);
            printf(
                "      candidate_token=%d score_status=exact "
                "whole_completion_score=%.17g whole_completion_tokens=",
                audit.token,
                outcome_selection_score(audit.outcome)
            );
            print_token_array(tokens, count);
            putchar('\n');
        }
        if (step.token == SEQUENCE_DELIMITER) break;
        completion = completion_tail(completion);
    }
}

static double elapsed_seconds(struct timespec start, struct timespec finish) {
    return (double)(finish.tv_sec - start.tv_sec) +
        (double)(finish.tv_nsec - start.tv_nsec) / 1000000000.0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    Options options = parse_options(argc, argv);
    AtkeyRuntime *runtime = atkey_runtime_new(
        options.checkpoint,
        options.tokenizer
    );
    if (runtime == NULL) fail("failed to load model");

    Evaluator evaluator = {0};
    evaluator.runtime = runtime;
    evaluator.config = (TermConfig){
        .dim = atkey_dim(runtime),
        .hidden_dim = atkey_hidden_dim(runtime),
        .layers = atkey_layer_count(runtime),
        .heads = atkey_head_count(runtime),
        .kv_heads = atkey_kv_head_count(runtime),
        .vocab = atkey_vocab_size(runtime),
        .sequence_length = atkey_sequence_length(runtime),
    };
    evaluator.term_arena.default_capacity = 1 << 20;
    evaluator.run_arena.default_capacity = 4 << 20;
    if (options.candidate_store_path != NULL) {
        arena_enable_file_backing(
            &evaluator.run_arena,
            options.candidate_store_path
        );
    }
    memo_init(&evaluator.memo);
    init_prefix_space(&evaluator.prefixes, &evaluator.run_arena);

    int prompt_count = 0;
    int *prompt = atkey_encode(runtime, options.prompt, &prompt_count);
    if (prompt == NULL || prompt_count <= 0) fail("tokenizer failed");
    if (options.top_k > evaluator.config.vocab) fail("top-k exceeds vocabulary");
    if (prompt_count + options.length > evaluator.config.sequence_length) {
        fail("prompt and completion exceed model context");
    }

    ModelFillers fillers = build_fillers(&evaluator);
    ModelTerm model = model_fields_term(
        &evaluator,
        &fillers,
        prompt,
        prompt_count,
        options.length
    );
    FILE *audit_stream = NULL;
    if (options.audit_path != NULL) {
        audit_stream = fopen(options.audit_path, "w");
        if (audit_stream == NULL) fail("could not open candidate audit file");
        fprintf(
            audit_stream,
            "candidate_audit prompt=%s horizon=%d top_k=%d bounded=%d "
            "goal=%s threshold=%.17g sample_ms=%d "
            "sample_demands=%d "
            "score_mode=log_probability\n",
            options.prompt,
            options.length,
            options.top_k,
            options.bounded ? 1 : 0,
            options.threshold_enabled ? "satisfy" :
                (options.sample_milliseconds > 0 ||
                 options.sample_demand_limit > 0) ?
                    "sampled_recursive_maximize" : "maximize",
            options.reward_threshold,
            options.sample_milliseconds,
            options.sample_demand_limit
        );
        fflush(audit_stream);
    }
    FILE *strength_stream = NULL;
    if (options.strength_path != NULL) {
        strength_stream = fopen(options.strength_path, "w");
        if (strength_stream == NULL) fail("could not open strength log file");
        fputs("strength_log version=1 prompt=\"", strength_stream);
        audit_escaped(strength_stream, options.prompt);
        fprintf(
            strength_stream,
            "\" horizon=%d top_k=%d bounded=%d sample_ms=%d "
            "sample_demands=%d score_mode=log_probability\n",
            options.length,
            options.top_k,
            options.bounded ? 1 : 0,
            options.sample_milliseconds,
            options.sample_demand_limit
        );
        fflush(strength_stream);
    }
    AtkeyProgram program;
    compose_program(
        &program,
        &evaluator,
        &model,
        options.prompt,
        prompt[prompt_count - 1],
        options.top_k,
        options.bounded,
        options.threshold_enabled,
        options.reward_threshold,
        options.sample_milliseconds,
        options.sample_demand_limit,
        options.trace,
        audit_stream,
        strength_stream
    );

    size_t calls_before_run = total_filler_applications(runtime);
    if (calls_before_run != 0) fail("a learned kernel ran before run_pcont");

    struct timespec started;
    struct timespec finished;
    clock_gettime(CLOCK_MONOTONIC, &started);
    Outcome *result = run_pcont(&program);
    int scored_token_count = 0;
    double log_probability_sum = outcome_log_probability_sum(
        result,
        &scored_token_count
    );
    double reward = horizon_equivalent_reward(
        &program.search,
        log_probability_sum,
        scored_token_count
    );
    strength_log(
        &program.search,
        "kind=reward_observed score=%.17g log_probability_sum=%.17g "
        "scored_tokens=%d",
        reward,
        log_probability_sum,
        scored_token_count
    );
    clock_gettime(CLOCK_MONOTONIC, &finished);

    int selected_tokens[options.length];
    int selected_count = collect_tokens(result, selected_tokens);
    if (strength_stream != NULL) {
        strength_log(
            &program.search,
            "kind=run_end selected_tokens=%d selection_frames=%" PRIu64,
            selected_count,
            program.search.next_frame_id
        );
        program.search.strength_stream = NULL;
    }
    bool terminated = selected_count > 0 &&
        selected_tokens[selected_count - 1] == SEQUENCE_DELIMITER;

    printf("prompt: %s\n", options.prompt);
    printf("prompt_token_count=%d\n", prompt_count);
    fputs("completion: ", stdout);
    print_completion_text(
        runtime,
        prompt[prompt_count - 1],
        selected_tokens,
        selected_count
    );
    fputs("selected_tokens=", stdout);
    print_token_array(selected_tokens, selected_count);
    putchar('\n');
    printf(
        "termination_token=%s\n",
        terminated ? "1" : "none"
    );
    printf("selected_reward=%.17g\n", reward);
    printf(
        "selected_log_probability_sum=%.17g\n"
        "selected_scored_tokens=%d\n",
        log_probability_sum,
        scored_token_count
    );
    puts("selected_reward_kind=horizon_equivalent_mean_log_probability");
    if (program.search.sampling_enabled) {
        puts("selection_goal=maximum_over_recursively_demanded_support");
        printf(
            "sampling_budget_ms=%d sampling_demand_limit=%d "
            "sampling_stop_reason=%s\n",
            program.search.sample_milliseconds,
            program.search.sample_demand_limit,
            sample_stop_name(program.search.sample_stop_reason)
        );
        printf(
            "sampled_candidate_demands=%" PRIu64
            " forced_after_budget=%" PRIu64
            " truncated_selection_frames=%" PRIu64
            " sampled_unique_prefixes=%" PRIu32 "\n",
            program.search.sampled_candidate_demands,
            program.search.sampled_forced_demands,
            program.search.sampled_truncated_frames,
            evaluator.prefixes.next_id - 1
        );
        printf(
            "sampling_policy=local_top_k_without_replacement_recursive_select "
            "sampling_seed=%" PRIu64 "\n",
            SAMPLE_RANDOM_SEED
        );
    } else if (options.threshold_enabled) {
        printf(
            "selection_goal=first_satisfying_completion "
            "reward_threshold=%.17g\n",
            options.reward_threshold
        );
    } else {
        puts("selection_goal=exact_maximum");
    }
    printf("learned_kernel_calls_before_run=%zu\n", calls_before_run);
    printf("elapsed_seconds=%.9f\n", elapsed_seconds(started, finished));
    if (evaluator.run_arena.file_backed) {
        printf(
            "candidate_store=%s mapped_bytes=%jd\n",
            evaluator.run_arena.backing_path,
            (intmax_t)evaluator.run_arena.backing_size
        );
    }
    if (options.trace) {
        print_decision_trace(result);
    }
    uint64_t selection_nodes = program.search.next_frame_id;
    printf(
        "selection_function_nodes=%" PRIu64 " prefix_nodes=%" PRIu32 " "
        "completion_cells=%zu outcomes=%zu "
        "memoized_field_values=%zu\n",
        selection_nodes,
        evaluator.prefixes.next_id,
        program.search.completion_cells,
        program.search.outcomes,
        evaluator.memo.count
    );
    print_weight_evidence(runtime);

    if (strength_stream != NULL) {
        printf(
            "strength_log=%s events=%" PRIu64 "\n",
            options.strength_path,
            program.search.next_strength_event_id
        );
    }

    if (audit_stream != NULL) fclose(audit_stream);
    if (strength_stream != NULL) fclose(strength_stream);
    atkey_free_tokens(prompt);
    free(evaluator.memo.entries);
    arena_free(&evaluator.run_arena);
    arena_free(&evaluator.term_arena);
    atkey_runtime_free(runtime);
    return 0;
}
