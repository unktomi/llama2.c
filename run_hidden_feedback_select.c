/*
 * Hidden-state-feedback plus deferred Escardo selection, copied from
 * llama2.c's run.c.
 *
 * Prompt tokens are embedded normally. After the final prompt token, the
 * final RMS-normalized hidden state crosses the autoregressive boundary
 * without selecting a token. The default crossing is the identity: the output
 * hidden state becomes the next position's input hidden state directly. Every
 * output hidden state is retained, and no selected token is fed back into the
 * recurrence. The retained tape supplies only the sampled carrier. Every
 * demanded hypothetical token occurrence is then assembled into one causal
 * company and each learned filler is applied once to that complete family.
 * The model callback returns the candidate-owned context opinions for every
 * completed branch. Memoized selection strength passes that complete outcome
 * unchanged through every local Select and orders outcomes lexicographically
 * from their least approving token upward. Only the root projects the retained
 * outcome to a witness; no path sum or single-token reward is used.
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <errno.h>
#include <fcntl.h>
#include "llama_company.h"
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

float* forward_internal(
    Transformer* transformer,
    int token,
    int pos,
    int use_token_embedding,
    int project_logits
) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    if (use_token_embedding) {
        // Prompt positions enter through the ordinary token constructor.
        float* content_row = w->token_embedding_table + token * dim;
        memcpy(x, content_row, dim*sizeof(*x));
    }
    // Otherwise x still contains the preceding position's final normalized
    // hidden state. It becomes this position's residual-stream input directly.

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    if (project_logits) {
        matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
        return s->logits;
    }
    return x;
}

float* forward(Transformer* transformer, int token, int pos) {
    return forward_internal(transformer, token, pos, 1, 1);
}

float* forward_token_hidden(Transformer* transformer, int token, int pos) {
    return forward_internal(transformer, token, pos, 1, 0);
}

typedef enum {
    FEEDBACK_IDENTITY,
    FEEDBACK_AFFINE_TOKEN_BARYCENTER,
} FeedbackBoundary;

static const char *feedback_boundary_name(FeedbackBoundary boundary) {
    return boundary == FEEDBACK_IDENTITY
        ? "identity"
        : "affine_token_barycenter";
}

/*
 * The tied or untied output classifier supplies token-indexed observations;
 * the input embedding table turns their normalized distribution into an
 * affine combination of token constructors. No token is selected here.
 */
static void apply_affine_token_barycenter(Transformer *transformer) {
    Config *config = &transformer->config;
    TransformerWeights *weights = &transformer->weights;
    RunState *state = &transformer->state;
    matmul(
        state->logits,
        state->x,
        weights->wcls,
        config->dim,
        config->vocab_size
    );
    softmax(state->logits, config->vocab_size);
    memset(state->xb, 0, (size_t)config->dim * sizeof(*state->xb));
    for (int token = 0; token < config->vocab_size; token++) {
        const float *embedding = weights->token_embedding_table +
            (size_t)token * config->dim;
        float coefficient = state->logits[token];
        for (int lane = 0; lane < config->dim; lane++) {
            state->xb[lane] += coefficient * embedding[lane];
        }
    }
    memcpy(
        state->x,
        state->xb,
        (size_t)config->dim * sizeof(*state->x)
    );
}

float* forward_feedback_hidden(
    Transformer* transformer,
    int pos,
    FeedbackBoundary boundary
) {
    if (boundary == FEEDBACK_AFFINE_TOKEN_BARYCENTER) {
        apply_affine_token_barycenter(transformer);
    }
    return forward_internal(transformer, 0, pos, 0, 0);
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// deferred projection as a product of selection functions

typedef struct {
    int token;
    int local_rank;
    float logit;
    double log_probability;
    double backed_observer_rating;
} ProjectionCandidate;

typedef struct ProjectionSelectFrame ProjectionSelectFrame;

struct ProjectionSelectFrame {
    int position;
    int candidate_count;
    ProjectionCandidate *candidates;
    int selected_index;
};

typedef struct {
    unsigned long long strength_nodes;
    unsigned long long candidate_ratings;
    unsigned long long observer_applications;
    unsigned long long structured_leaf_outcomes;
    unsigned long long root_terminalizations;
    unsigned long long company_rows;
    unsigned long long family_filler_calls;
    unsigned long long maximum_calls_per_filler;
    unsigned long long family_scalar_reads;
    unsigned long long strength_filler_calls;
    unsigned long long strength_scalar_reads;
    unsigned long long company_model_nanoseconds;
    unsigned long long strength_nanoseconds;
    FILE *trace;
    Tokenizer *tokenizer;
} ProjectionStrengthCounters;

static unsigned long long projection_monotonic_nanoseconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (unsigned long long)now.tv_sec * 1000000000ULL +
        (unsigned long long)now.tv_nsec;
}

typedef enum {
    PROJECTION_OBSERVER_LOGIT_STRENGTH,
    PROJECTION_OBSERVER_COMPANY_STRENGTH,
    PROJECTION_OBSERVER_HIDDEN_DISPLACEMENT,
} ProjectionObserverKind;

static const char *projection_observer_name(ProjectionObserverKind kind) {
    switch (kind) {
        case PROJECTION_OBSERVER_LOGIT_STRENGTH:
            return "escardo_position_logit_vector";
        case PROJECTION_OBSERVER_COMPANY_STRENGTH:
            return "candidate_owned_context_opinions";
        case PROJECTION_OBSERVER_HIDDEN_DISPLACEMENT:
            return "ordered_hidden_displacement_cosine";
    }
    return "invalid_projection_observer";
}

static const char *projection_score_role(ProjectionObserverKind kind) {
    switch (kind) {
        case PROJECTION_OBSERVER_LOGIT_STRENGTH:
            return "first_variable_logit_coordinate";
        case PROJECTION_OBSERVER_COMPANY_STRENGTH:
            return "leximin_complete_company_outcome";
        case PROJECTION_OBSERVER_HIDDEN_DISPLACEMENT:
            return "whole_path_scalar";
    }
    return "invalid_projection_score";
}

static void projection_json_string(FILE *stream, const char *value) {
    fputc('"', stream);
    if (value != NULL) {
        for (const unsigned char *cursor = (const unsigned char *)value;
             *cursor != '\0'; cursor++) {
            unsigned char byte = *cursor;
            if (byte == '"') fputs("\\\"", stream);
            else if (byte == '\\') fputs("\\\\", stream);
            else if (byte == '\n') fputs("\\n", stream);
            else if (byte == '\r') fputs("\\r", stream);
            else if (byte == '\t') fputs("\\t", stream);
            else if (byte < 0x20) {
                fprintf(stream, "\\u%04x", byte);
            } else {
                fputc(byte, stream);
            }
        }
    }
    fputc('"', stream);
}

static void projection_json_piece(
    FILE *stream,
    Tokenizer *tokenizer,
    int token
) {
    char *piece = decode(tokenizer, 0, token);
    if (piece != NULL && piece[0] != '\0' && piece[1] == '\0' &&
        (unsigned char)piece[0] >= 0x80) {
        fprintf(stream, "\"\\u%04x\"", (unsigned char)piece[0]);
        return;
    }
    projection_json_string(stream, piece);
}

static double projection_log_partition(float *logits, int vocab_size) {
    double maximum = -INFINITY;
    for (int token = 0; token < vocab_size; token++) {
        if ((double)logits[token] > maximum) maximum = logits[token];
    }
    double total = 0.0;
    for (int token = 0; token < vocab_size; token++) {
        total += exp((double)logits[token] - maximum);
    }
    if (!isfinite(maximum) || !(total > 0.0) || !isfinite(total)) {
        fprintf(stderr, "invalid deferred projection logits\n");
        exit(EXIT_FAILURE);
    }
    return maximum + log(total);
}

static int projection_candidate_compare(const void *left_value, const void *right_value) {
    const ProjectionCandidate *left = left_value;
    const ProjectionCandidate *right = right_value;
    if (left->logit > right->logit) return -1;
    if (left->logit < right->logit) return 1;
    if (left->token < right->token) return -1;
    if (left->token > right->token) return 1;
    return 0;
}

/*
 * A local Select ranges over the complete model vocabulary.  Ranking is kept
 * only for diagnostics and deterministic tie-breaking; it does not truncate
 * the carrier.
 */
static void projection_full_vocabulary(
    float *logits,
    int vocab_size,
    ProjectionCandidate *candidates
) {
    for (int token = 0; token < vocab_size; token++) {
        candidates[token] = (ProjectionCandidate){
            .token = token,
            .logit = logits[token],
        };
    }
    qsort(
        candidates,
        (size_t)vocab_size,
        sizeof(*candidates),
        projection_candidate_compare
    );

    double partition = projection_log_partition(logits, vocab_size);
    for (int rank = 0; rank < vocab_size; rank++) {
        candidates[rank].local_rank = rank + 1;
        candidates[rank].log_probability =
            (double)candidates[rank].logit - partition;
    }
}

typedef enum {
    PROJECTION_SELECTION_UNFORCED,
    PROJECTION_SELECTION_FORCING,
    PROJECTION_SELECTION_FORCED,
} ProjectionSelectionState;

typedef struct {
    Transformer observer_transformer;
    AtkeyRuntime *company_runtime;
    ProjectionSelectFrame *frames;
    int frame_count;
    int dim;
    int first_observed_position;
    int observed_position_count;
    float *target_displacements;
    float *candidate_previous;
    double target_norm;
    ProjectionObserverKind observer_kind;
    int local_support;
    unsigned long long sample_seed;
    unsigned long long sample_state;
    unsigned long long next_frame_id;
    double first_variable_selection_rating;
    int first_variable_selection_rating_set;
    ProjectionStrengthCounters *counters;
} ProjectionProduct;

static void projection_json_fragment(FILE *stream, const char *value) {
    if (value == NULL) return;
    for (const unsigned char *cursor = (const unsigned char *)value;
         *cursor != '\0'; cursor++) {
        unsigned char byte = *cursor;
        if (byte == '"') fputs("\\\"", stream);
        else if (byte == '\\') fputs("\\\\", stream);
        else if (byte == '\n') fputs("\\n", stream);
        else if (byte == '\r') fputs("\\r", stream);
        else if (byte == '\t') fputs("\\t", stream);
        else if (byte < 0x20 || byte >= 0x80) {
            fprintf(stream, "\\u%04x", byte);
        } else {
            fputc(byte, stream);
        }
    }
}

static void projection_trace_path(
    FILE *stream,
    ProjectionProduct *product,
    const int *path
) {
    fputc('"', stream);
    int previous = product->frames[0].candidates[path[0]].token;
    projection_json_fragment(
        stream,
        decode(product->counters->tokenizer, 0, previous)
    );
    for (int position = 1; position < product->frame_count; position++) {
        int token = product->frames[position].candidates[path[position]].token;
        projection_json_fragment(
            stream,
            decode(product->counters->tokenizer, previous, token)
        );
        previous = token;
    }
    fputc('"', stream);
}

static void projection_trace_carrier_candidate(
    FILE *stream,
    Tokenizer *tokenizer,
    const ProjectionSelectFrame *frame,
    const ProjectionCandidate *candidate
) {
    if (stream == NULL) return;
    fprintf(
        stream,
        "{\"event\":\"carrier_candidate\",\"position\":%d,"
        "\"unit\":%s,\"token\":%d,\"local_rank\":%d,"
        "\"logit\":%.9g,\"local_log_probability\":%.17g,"
        "\"piece\":",
        frame->position,
        frame->candidate_count == 1 ? "true" : "false",
        candidate->token,
        candidate->local_rank,
        candidate->logit,
        candidate->log_probability
    );
    projection_json_piece(stream, tokenizer, candidate->token);
    fputs("}\n", stream);
    fflush(stream);
}

static double projection_vector_norm(const float *values, size_t count) {
    double squares = 0.0;
    for (size_t index = 0; index < count; index++) {
        squares += (double)values[index] * values[index];
    }
    return sqrt(squares);
}

static void projection_trace_whole_path_result(
    ProjectionProduct *product,
    const int *path,
    int root_terminalization,
    double rating,
    double candidate_norm,
    double company_log_probability,
    double independent_log_probability
) {
    if (product->counters->trace == NULL) return;
    (void)company_log_probability;
    (void)independent_log_probability;
    FILE *stream = product->counters->trace;
    fprintf(
        stream,
        "{\"event\":\"whole_path_observed\"," 
        "\"root_terminalization\":%s,\"observer\":\"%s\","
        "\"rating\":%.17g",
        root_terminalization ? "true" : "false",
        projection_observer_name(product->observer_kind),
        rating
    );
    if (product->observer_kind ==
            PROJECTION_OBSERVER_COMPANY_STRENGTH) {
        fputs(
            ",\"rating_role\":"
            "\"diagnostic_worst_coordinate_after_leximin_selection\"",
            stream
        );
    } else if (product->observer_kind ==
            PROJECTION_OBSERVER_LOGIT_STRENGTH) {
        fputs(
            ",\"rating_role\":\"first_variable_logit_coordinate\""
            ",\"position_logit_coordinates\":[",
            stream
        );
        for (int position = product->first_observed_position;
             position < product->frame_count; position++) {
            if (position != product->first_observed_position) fputc(',', stream);
            const ProjectionSelectFrame *frame = &product->frames[position];
            int candidate_index = path[position];
            if (candidate_index < 0 ||
                candidate_index >= frame->candidate_count) {
                fprintf(stderr, "logit strength retained an invalid path\n");
                exit(EXIT_FAILURE);
            }
            fprintf(stream, "%.9g", frame->candidates[candidate_index].logit);
        }
        fputc(']', stream);
    } else {
        fprintf(
            stream,
            ",\"candidate_displacement_norm\":%.17g",
            candidate_norm
        );
    }
    fputs(",\"text\":", stream);
    projection_trace_path(stream, product, path);
    fputs("}\n", stream);
    fflush(stream);
}

/*
 * Scalar whole-path observation retained for the displacement audit. Company
 * selection uses projection_company_product_select instead: its outcome is a
 * token-indexed vector and is terminalized only after backward induction.
 * The density-ratio branch below is retained only as a numerical diagnostic.
 */
static double projection_observe_whole_path(
    ProjectionProduct *product,
    const int *path,
    int root_terminalization
) {
    if (product->observer_kind !=
            PROJECTION_OBSERVER_HIDDEN_DISPLACEMENT) {
        fprintf(stderr, "scalar path observer used outside displacement audit\n");
        exit(EXIT_FAILURE);
    }
    double dot = 0.0;
    double candidate_squares = 0.0;
    int observed_position = 0;
    for (int position = 0; position < product->frame_count; position++) {
        int candidate_index = path[position];
        ProjectionSelectFrame *frame = &product->frames[position];
        if (candidate_index < 0 || candidate_index >= frame->candidate_count) {
            fprintf(stderr, "selection observer received an invalid path\n");
            exit(EXIT_FAILURE);
        }
        int token = frame->candidates[candidate_index].token;
        float *hidden = forward_token_hidden(
            &product->observer_transformer,
            token,
            position
        );
        if (product->observer_kind ==
                PROJECTION_OBSERVER_HIDDEN_DISPLACEMENT &&
            position == product->first_observed_position - 1) {
            memcpy(
                product->candidate_previous,
                hidden,
                (size_t)product->dim * sizeof(*product->candidate_previous)
            );
        }
        if (product->observer_kind ==
                PROJECTION_OBSERVER_HIDDEN_DISPLACEMENT &&
            position >= product->first_observed_position) {
            const float *target = product->target_displacements +
                (size_t)observed_position * product->dim;
            for (int lane = 0; lane < product->dim; lane++) {
                double candidate_displacement =
                    (double)hidden[lane] - product->candidate_previous[lane];
                dot += (double)target[lane] * candidate_displacement;
                candidate_squares +=
                    candidate_displacement * candidate_displacement;
            }
            memcpy(
                product->candidate_previous,
                hidden,
                (size_t)product->dim * sizeof(*product->candidate_previous)
            );
            observed_position++;
        }
    }
    if (product->observer_kind ==
            PROJECTION_OBSERVER_HIDDEN_DISPLACEMENT &&
        observed_position != product->observed_position_count) {
        fprintf(stderr, "whole-path observer position mismatch\n");
        exit(EXIT_FAILURE);
    }
    double candidate_norm = sqrt(candidate_squares);
    if (!(product->target_norm > 0.0) || !(candidate_norm > 0.0) ||
        !isfinite(dot) || !isfinite(candidate_norm)) {
        fprintf(
            stderr,
            "invalid ordered hidden-displacement observation\n"
        );
        exit(EXIT_FAILURE);
    }
    double rating = dot / (product->target_norm * candidate_norm);
    if (root_terminalization) {
        product->counters->root_terminalizations++;
    } else {
        product->counters->observer_applications++;
    }
    projection_trace_whole_path_result(
        product,
        path,
        root_terminalization,
        rating,
        candidate_norm,
        0.0,
        0.0
    );
    return rating;
}

static unsigned long long projection_mix(unsigned long long value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return value;
}

static double projection_random_unit(unsigned long long *state) {
    *state = projection_mix(*state + 0x9e3779b97f4a7c15ULL);
    return (double)(*state >> 11) * (1.0 / 9007199254740992.0);
}

static unsigned long long projection_prefix_seed(
    ProjectionProduct *product,
    const int *path,
    int position
) {
    unsigned long long state = product->sample_seed ^
        (unsigned long long)(position + 1);
    for (int index = 0; index < position; index++) {
        int selected = path[index];
        int token = product->frames[index].candidates[selected].token;
        state = projection_mix(state ^ (unsigned long long)(token + 1));
    }
    return state;
}

/* Sampling limits which arguments a Select demands; it never changes p. */
static void projection_demanded_candidates(
    ProjectionProduct *product,
    const ProjectionSelectFrame *frame,
    const int *path,
    int demand_count,
    int *indices
) {
    if (demand_count == frame->candidate_count) {
        for (int index = 0; index < demand_count; index++) indices[index] = index;
        return;
    }
    unsigned char *used = calloc(
        (size_t)frame->candidate_count,
        sizeof(*used)
    );
    if (used == NULL) {
        fprintf(stderr, "could not allocate sampled Select support\n");
        exit(EXIT_FAILURE);
    }
    unsigned long long state = projection_prefix_seed(
        product,
        path,
        frame->position
    );
    for (int demand = 0; demand < demand_count; demand++) {
        double maximum = -INFINITY;
        for (int index = 0; index < frame->candidate_count; index++) {
            if (!used[index] && frame->candidates[index].logit > maximum) {
                maximum = frame->candidates[index].logit;
            }
        }
        double mass = 0.0;
        for (int index = 0; index < frame->candidate_count; index++) {
            if (!used[index]) {
                mass += exp((double)frame->candidates[index].logit - maximum);
            }
        }
        double target = projection_random_unit(&state) * mass;
        double cumulative = 0.0;
        int selected = -1;
        for (int index = 0; index < frame->candidate_count; index++) {
            if (used[index]) continue;
            cumulative += exp(
                (double)frame->candidates[index].logit - maximum
            );
            selected = index;
            if (target < cumulative) break;
        }
        if (selected < 0) {
            fprintf(stderr, "could not demand a sampled Select argument\n");
            exit(EXIT_FAILURE);
        }
        used[selected] = 1;
        indices[demand] = selected;
    }
    free(used);
}

static void trace_projection_product_candidate(
    ProjectionProduct *product,
    unsigned long long frame_id,
    int demand_ordinal,
    const ProjectionSelectFrame *frame,
    const ProjectionCandidate *candidate,
    double observer_rating,
    int observer_leaf_node,
    unsigned long long observer_reachability,
    int observer_position,
    int observer_context_node,
    int observer_company_node,
    int observer_company_token,
    int observer_incoming_boundary,
    double observer_logit,
    double observer_log_partition,
    const int *path
) {
    if (product->counters->trace == NULL) return;
    FILE *stream = product->counters->trace;
    fprintf(
        stream,
        "{\"event\":\"candidate_rated\",\"frame\":%llu,"
        "\"position\":%d,\"unit\":%s,\"demand_ordinal\":%d,"
        "\"token\":%d,\"local_rank\":%d,\"logit\":%.9g,"
        "\"local_log_probability\":%.17g,"
        "\"observer_rating\":%.17g,",
        frame_id,
        frame->position,
        frame->candidate_count == 1 ? "true" : "false",
        demand_ordinal,
        candidate->token,
        candidate->local_rank,
        candidate->logit,
        candidate->log_probability,
        observer_rating
    );
    if (observer_leaf_node >= 0) {
        fprintf(
            stream,
            "\"observer\":\"candidate_owned_context_opinions\","
            "\"observer_leaf_node\":%d,"
            "\"observer_reachability\":%llu,"
            "\"observer_worst_position\":%d,"
            "\"observer_context_node\":%d,"
            "\"observer_company_node\":%d,"
            "\"observer_company_token\":%d,"
            "\"observer_direction\":\"%s\","
            "\"observer_logit\":%.17g,"
            "\"observer_log_partition\":%.17g,"
            "\"observer_order\":\"leximin_all_coordinates\","
            "\"observer_rating_role\":"
            "\"diagnostic_worst_coordinate_of_backed_outcome\",",
            observer_leaf_node,
            observer_reachability,
            observer_position,
            observer_context_node,
            observer_company_node,
            observer_company_token,
            observer_incoming_boundary
                ? "candidate_about_context" : "outgoing",
            observer_logit,
            observer_log_partition
        );
    } else if (product->observer_kind ==
            PROJECTION_OBSERVER_LOGIT_STRENGTH) {
        fputs(
            "\"observer\":\"escardo_position_logit_vector\","
            "\"observer_rating_role\":\"position_logit_coordinate\",",
            stream
        );
    } else {
        fprintf(
            stream,
            "\"observer\":\"%s\","
            "\"observer_rating_role\":\"whole_path_scalar\",",
            projection_observer_name(product->observer_kind)
        );
    }
    fputs("\"piece\":", stream);
    projection_json_piece(stream, product->counters->tokenizer, candidate->token);
    fputs(",\"text\":", stream);
    projection_trace_path(stream, product, path);
    fputs("}\n", stream);
    fflush(stream);
}

static void trace_projection_product_select(
    ProjectionProduct *product,
    unsigned long long frame_id,
    const ProjectionSelectFrame *frame,
    const ProjectionCandidate *candidate,
    double observer_rating,
    int observer_leaf_node,
    unsigned long long observer_reachability,
    int observer_position,
    int observer_context_node,
    int observer_company_node,
    int observer_company_token,
    int observer_incoming_boundary,
    double observer_logit,
    double observer_log_partition,
    const int *path
) {
    if (product->counters->trace == NULL) return;
    FILE *stream = product->counters->trace;
    fprintf(
        stream,
        "{\"event\":\"select\",\"frame\":%llu,\"position\":%d,"
        "\"unit\":%s,\"token\":%d,\"local_rank\":%d,"
        "\"observer_rating\":%.17g,",
        frame_id,
        frame->position,
        frame->candidate_count == 1 ? "true" : "false",
        candidate->token,
        candidate->local_rank,
        observer_rating
    );
    if (observer_leaf_node >= 0) {
        fprintf(
            stream,
            "\"observer\":\"candidate_owned_context_opinions\","
            "\"observer_leaf_node\":%d,"
            "\"observer_reachability\":%llu,"
            "\"observer_worst_position\":%d,"
            "\"observer_context_node\":%d,"
            "\"observer_company_node\":%d,"
            "\"observer_company_token\":%d,"
            "\"observer_direction\":\"%s\","
            "\"observer_logit\":%.17g,"
            "\"observer_log_partition\":%.17g,"
            "\"observer_order\":\"leximin_all_coordinates\","
            "\"observer_rating_role\":"
            "\"diagnostic_worst_coordinate_of_backed_outcome\",",
            observer_leaf_node,
            observer_reachability,
            observer_position,
            observer_context_node,
            observer_company_node,
            observer_company_token,
            observer_incoming_boundary
                ? "candidate_about_context" : "outgoing",
            observer_logit,
            observer_log_partition
        );
    } else if (product->observer_kind ==
            PROJECTION_OBSERVER_LOGIT_STRENGTH) {
        fputs(
            "\"observer\":\"escardo_position_logit_vector\","
            "\"observer_rating_role\":\"position_logit_coordinate\",",
            stream
        );
    } else {
        fprintf(
            stream,
            "\"observer\":\"%s\","
            "\"observer_rating_role\":\"whole_path_scalar\",",
            projection_observer_name(product->observer_kind)
        );
    }
    fputs("\"text\":", stream);
    projection_trace_path(stream, product, path);
    fputs("}\n", stream);
    fflush(stream);
}

/*
 * Escardo strength specialized to the fixed logit tape produced by the one
 * hidden-to-hidden model run.
 *
 * The outcome is not a scalar fold.  It is the position-indexed vector of
 * selected logit coordinates.  The Select at position i compares only the i
 * coordinate returned by its observer.  Thus raw logits from different rows
 * are never added or otherwise treated as sharing an origin.
 *
 * For the ordinary product equation
 *
 *   b(x)   = select the suffix under x
 *   a      = Select_i (x -> observe_i(x : b(x)))
 *   result = a : b(a)
 *
 * b(x) is extensionally the same suffix for every x here: all rows were
 * produced before token projection and no selected token is fed back into the
 * model.  We force that suffix once, bind it below this frame, test every
 * demanded x through the position-i observer, and reuse the bound suffix for
 * the selected a.  This is the function-as-tree specialization; it does not
 * construct token prefixes or call a model observer.
 */
static void projection_logit_strength_force(
    ProjectionProduct *product,
    int position,
    int *selected_path
) {
    if (position == product->frame_count) return;

    /* The where-bound b(x), shared because the fixed logit tape is prefix-free. */
    projection_logit_strength_force(product, position + 1, selected_path);

    ProjectionSelectFrame *frame = &product->frames[position];
    int best_index = -1;
    double best_coordinate = -INFINITY;
    product->counters->strength_nodes++;
    for (int candidate_index = 0;
         candidate_index < frame->candidate_count;
         candidate_index++) {
        ProjectionCandidate *candidate = &frame->candidates[candidate_index];
        double coordinate = candidate->logit;
        if (!isfinite(coordinate)) {
            fprintf(stderr, "logit strength observed a non-finite coordinate\n");
            exit(EXIT_FAILURE);
        }
        candidate->backed_observer_rating = coordinate;
        product->counters->candidate_ratings++;
        product->counters->observer_applications++;
        if (best_index < 0 || coordinate > best_coordinate ||
            (coordinate == best_coordinate &&
             candidate->local_rank <
                frame->candidates[best_index].local_rank)) {
            best_index = candidate_index;
            best_coordinate = coordinate;
        }
    }
    if (best_index < 0) {
        fprintf(stderr, "logit strength received an empty Select\n");
        exit(EXIT_FAILURE);
    }
    selected_path[position] = best_index;
    frame->selected_index = best_index;
    if (position == product->first_observed_position) {
        product->first_variable_selection_rating = best_coordinate;
        product->first_variable_selection_rating_set = 1;
    }
}

static double projection_logit_product_select(
    ProjectionProduct *product,
    int *selected_path
) {
    unsigned long long strength_start = projection_monotonic_nanoseconds();
    projection_logit_strength_force(product, 0, selected_path);
    product->counters->strength_nanoseconds =
        projection_monotonic_nanoseconds() - strength_start;
    if (!product->first_variable_selection_rating_set) {
        fprintf(stderr, "logit strength did not select a completion coordinate\n");
        exit(EXIT_FAILURE);
    }

    int *candidate_path = malloc(
        (size_t)product->frame_count * sizeof(*candidate_path)
    );
    if (candidate_path == NULL) {
        fprintf(stderr, "could not allocate logit-strength trace path\n");
        exit(EXIT_FAILURE);
    }

    /* Emit the actual backward order, with the finally selected context fixed. */
    for (int position = product->frame_count - 1; position >= 0; position--) {
        ProjectionSelectFrame *frame = &product->frames[position];
        unsigned long long frame_id = product->next_frame_id++;
        for (int candidate_index = 0;
             candidate_index < frame->candidate_count;
             candidate_index++) {
            memcpy(
                candidate_path,
                selected_path,
                (size_t)product->frame_count * sizeof(*candidate_path)
            );
            candidate_path[position] = candidate_index;
            ProjectionCandidate *candidate = &frame->candidates[candidate_index];
            trace_projection_product_candidate(
                product,
                frame_id,
                candidate_index,
                frame,
                candidate,
                candidate->backed_observer_rating,
                -1,
                0,
                -1,
                -1,
                -1,
                -1,
                0,
                candidate->logit,
                NAN,
                candidate_path
            );
        }
        ProjectionCandidate *selected =
            &frame->candidates[selected_path[position]];
        trace_projection_product_select(
            product,
            frame_id,
            frame,
            selected,
            selected->backed_observer_rating,
            -1,
            0,
            -1,
            -1,
            -1,
            -1,
            0,
            selected->logit,
            NAN,
            selected_path
        );
    }
    free(candidate_path);

    product->counters->structured_leaf_outcomes = 1;
    product->counters->root_terminalizations = 1;
    projection_trace_whole_path_result(
        product,
        selected_path,
        1,
        product->first_variable_selection_rating,
        0.0,
        0.0,
        0.0
    );
    return product->first_variable_selection_rating;
}

/*
 * Mechanical finite product of selection functions:
 *
 *   b(x) = delta_x(xs -> p(x : xs))
 *   a    = epsilon(x -> p(x : b(x)))
 *   result = a : b(a)
 *
 * `budget` limits the demanded local arguments. Every demanded x is rated by
 * the one whole-path p after its recursively selected b(x); no ratings fold.
 */
static double projection_product_select(
    ProjectionProduct *product,
    int position,
    unsigned long long budget,
    const int *prefix,
    int *selected_path
) {
    if (position == product->frame_count) {
        memcpy(
            selected_path,
            prefix,
            (size_t)product->frame_count * sizeof(*selected_path)
        );
        return projection_observe_whole_path(product, prefix, 0);
    }

    ProjectionSelectFrame *frame = &product->frames[position];
    int demand_count = frame->candidate_count;
    if (demand_count > product->local_support) {
        demand_count = product->local_support;
    }
    if (budget != ULLONG_MAX && budget < (unsigned long long)demand_count) {
        demand_count = (int)budget;
    }
    if (demand_count < 1) demand_count = 1;
    int *demands = malloc((size_t)demand_count * sizeof(*demands));
    int *branch_path = malloc(
        (size_t)product->frame_count * sizeof(*branch_path)
    );
    int *best_path = malloc(
        (size_t)product->frame_count * sizeof(*best_path)
    );
    if (demands == NULL || branch_path == NULL || best_path == NULL) {
        fprintf(stderr, "could not allocate selection-product frame\n");
        exit(EXIT_FAILURE);
    }
    projection_demanded_candidates(
        product,
        frame,
        prefix,
        demand_count,
        demands
    );

    unsigned long long frame_id = product->next_frame_id++;
    product->counters->strength_nodes++;
    int best_demand = -1;
    double best_rating = -INFINITY;
    for (int demand = 0; demand < demand_count; demand++) {
        int candidate_index = demands[demand];
        ProjectionCandidate *candidate = &frame->candidates[candidate_index];
        memcpy(
            branch_path,
            prefix,
            (size_t)product->frame_count * sizeof(*branch_path)
        );
        branch_path[position] = candidate_index;

        unsigned long long child_budget = budget;
        if (frame->candidate_count > 1 && budget != ULLONG_MAX) {
            unsigned long long quotient =
                budget / (unsigned long long)demand_count;
            unsigned long long remainder =
                budget % (unsigned long long)demand_count;
            child_budget = quotient +
                ((unsigned long long)demand < remainder ? 1ULL : 0ULL);
            if (child_budget == 0) child_budget = 1;
        }
        double rating = projection_product_select(
            product,
            position + 1,
            child_budget,
            branch_path,
            branch_path
        );
        candidate->backed_observer_rating = rating;
        product->counters->candidate_ratings++;
        trace_projection_product_candidate(
            product,
            frame_id,
            demand,
            frame,
            candidate,
            rating,
            -1,
            0,
            -1,
            -1,
            -1,
            -1,
            0,
            NAN,
            NAN,
            branch_path
        );
        if (best_demand < 0 || rating > best_rating ||
            (rating == best_rating &&
             candidate->local_rank <
                frame->candidates[demands[best_demand]].local_rank)) {
            best_demand = demand;
            best_rating = rating;
            memcpy(
                best_path,
                branch_path,
                (size_t)product->frame_count * sizeof(*best_path)
            );
        }
    }

    int best_candidate = demands[best_demand];
    memcpy(
        selected_path,
        best_path,
        (size_t)product->frame_count * sizeof(*selected_path)
    );
    trace_projection_product_select(
        product,
        frame_id,
        frame,
        &frame->candidates[best_candidate],
        best_rating,
        -1,
        0,
        -1,
        -1,
        -1,
        -1,
        0,
        NAN,
        NAN,
        best_path
    );
    free(best_path);
    free(branch_path);
    free(demands);
    return best_rating;
}

/*
 * Defunctionalized demanded support of the selection product. Node zero is a
 * synthetic prefix; every other node is one token occurrence in the causal
 * company. The support is fixed by the same local demand function used above,
 * before any learned observation is run.
 */
typedef struct {
    int coordinate_count;
    double *coordinates;
    double *leximin_coordinates;
    int *leximin_positions;
    int *context_nodes;
    int *company_nodes;
    int *company_tokens;
    unsigned char *incoming_boundary;
    double *logits;
    double *log_partitions;
} ProjectionTermOutcome;

typedef struct {
    int parent;
    int position;
    int candidate_index;
    int *children;
    int child_count;
    unsigned long long reachability;
    double row_log_partition;
    int selected_child;
    int selected_leaf;
    ProjectionTermOutcome *outcome;
    ProjectionSelectionState selection_state;
} ProjectionTermNode;

typedef struct {
    double coordinate;
    int position;
} ProjectionOrderedOpinion;

static int projection_ordered_opinion_compare(
    const void *left_value,
    const void *right_value
) {
    const ProjectionOrderedOpinion *left = left_value;
    const ProjectionOrderedOpinion *right = right_value;
    if (left->coordinate < right->coordinate) return -1;
    if (left->coordinate > right->coordinate) return 1;
    if (left->position < right->position) return -1;
    if (left->position > right->position) return 1;
    return 0;
}

typedef struct {
    ProjectionTermNode *nodes;
    int count;
    int capacity;
    unsigned long long leaf_count;
    unsigned long long path_demands;
} ProjectionTerm;

static int projection_term_add_node(
    ProjectionTerm *term,
    int parent,
    int position,
    int candidate_index
) {
    if (term->count == term->capacity) {
        int capacity = term->capacity == 0 ? 1024 : term->capacity * 2;
        if (capacity < term->capacity || capacity <= 0) {
            fprintf(stderr, "selection term exceeded integer capacity\n");
            exit(EXIT_FAILURE);
        }
        ProjectionTermNode *nodes = realloc(
            term->nodes,
            (size_t)capacity * sizeof(*nodes)
        );
        if (nodes == NULL) {
            fprintf(stderr, "could not grow demanded selection term\n");
            exit(EXIT_FAILURE);
        }
        term->nodes = nodes;
        term->capacity = capacity;
    }
    int index = term->count++;
    term->nodes[index] = (ProjectionTermNode){
        .parent = parent,
        .position = position,
        .candidate_index = candidate_index,
        .selected_child = -1,
        .selected_leaf = -1,
        .selection_state = PROJECTION_SELECTION_UNFORCED,
    };
    return index;
}

static int projection_term_find_child(
    const ProjectionTerm *term,
    int parent,
    int candidate_index
) {
    const ProjectionTermNode *node = &term->nodes[parent];
    for (int ordinal = 0; ordinal < node->child_count; ordinal++) {
        int child = node->children[ordinal];
        if (term->nodes[child].candidate_index == candidate_index) {
            return child;
        }
    }
    return -1;
}

static int projection_term_append_child(
    ProjectionTerm *term,
    int parent,
    int position,
    int candidate_index
) {
    int child = projection_term_add_node(
        term,
        parent,
        position,
        candidate_index
    );
    ProjectionTermNode *node = &term->nodes[parent];
    if (node->child_count == INT_MAX) {
        fprintf(stderr, "selection term child count overflow\n");
        exit(EXIT_FAILURE);
    }
    int child_count = node->child_count + 1;
    int *children = realloc(
        node->children,
        (size_t)child_count * sizeof(*children)
    );
    if (children == NULL) {
        fprintf(stderr, "could not grow sampled selection branches\n");
        exit(EXIT_FAILURE);
    }
    node->children = children;
    node->children[node->child_count] = child;
    node->child_count = child_count;
    return child;
}

/*
 * Demand one argument from the still-complete local vocabulary.  A positive
 * local_support is an explicit proposal truncation to the highest-ranked K
 * cells; zero leaves the proposal distribution over the entire vocabulary.
 * The draw chooses only which continuation is demanded.  It never supplies
 * the observer result or chooses the Select witness.
 */
static int projection_term_sample_candidate(
    ProjectionProduct *product,
    const ProjectionSelectFrame *frame
) {
    if (frame->candidate_count == 1) return 0;
    int support_count = frame->candidate_count;
    if (product->local_support > 0 &&
        support_count > product->local_support) {
        support_count = product->local_support;
    }
    if (support_count <= 0) {
        fprintf(stderr, "sampled Select has no proposal arguments\n");
        exit(EXIT_FAILURE);
    }
    double maximum = frame->candidates[0].logit;
    double mass = 0.0;
    for (int index = 0; index < support_count; index++) {
        mass += exp((double)frame->candidates[index].logit - maximum);
    }
    if (!(mass > 0.0) || !isfinite(mass)) {
        fprintf(stderr, "invalid sampled Select proposal mass\n");
        exit(EXIT_FAILURE);
    }
    double target = projection_random_unit(&product->sample_state) * mass;
    double cumulative = 0.0;
    int selected = support_count - 1;
    for (int index = 0; index < support_count; index++) {
        cumulative += exp(
            (double)frame->candidates[index].logit - maximum
        );
        if (target < cumulative) {
            selected = index;
            break;
        }
    }
    return selected;
}

/*
 * Construct the finite dependent demand tree extensionally by sampling
 * complete hypothetical paths.  Shared prefixes are represented once and
 * repeated demands increment reachability.  No path is selected here: after
 * the one family evaluation, projection_term_select supplies Escardo's
 * recursive b(x), a, and a:b(a) over exactly this observed subtree.
 */
static void projection_term_sample_paths(
    ProjectionProduct *product,
    ProjectionTerm *term,
    unsigned long long path_demands,
    int *path
) {
    if (path_demands == 0) {
        fprintf(stderr, "sampled selection requires a path demand\n");
        exit(EXIT_FAILURE);
    }
    int *path_nodes = malloc(
        (size_t)product->frame_count * sizeof(*path_nodes)
    );
    if (path_nodes == NULL) {
        fprintf(stderr, "could not allocate sampled selection path\n");
        exit(EXIT_FAILURE);
    }
    for (unsigned long long demand = 0; demand < path_demands; demand++) {
        int parent = 0;
        for (int position = 0; position < product->frame_count; position++) {
            ProjectionSelectFrame *frame = &product->frames[position];
            int candidate_index = projection_term_sample_candidate(
                product,
                frame
            );
            path[position] = candidate_index;
            int child = projection_term_find_child(
                term,
                parent,
                candidate_index
            );
            if (child < 0) {
                child = projection_term_append_child(
                    term,
                    parent,
                    position,
                    candidate_index
                );
            }
            path_nodes[position] = child;
            parent = child;
        }
        if (term->nodes[parent].reachability == 0) term->leaf_count++;
        if (term->nodes[0].reachability == ULLONG_MAX) {
            fprintf(stderr, "selection reachability multiplicity overflow\n");
            exit(EXIT_FAILURE);
        }
        term->nodes[0].reachability++;
        for (int position = 0; position < product->frame_count; position++) {
            ProjectionTermNode *node = &term->nodes[path_nodes[position]];
            if (node->reachability == ULLONG_MAX) {
                fprintf(stderr, "selection reachability multiplicity overflow\n");
                exit(EXIT_FAILURE);
            }
            node->reachability++;
        }
        term->path_demands++;
    }
    free(path_nodes);
}

static void projection_term_path(
    const ProjectionTerm *term,
    int leaf,
    int frame_count,
    int *path
) {
    for (int position = 0; position < frame_count; position++) {
        path[position] = -1;
    }
    for (int node = leaf; node != 0; node = term->nodes[node].parent) {
        const ProjectionTermNode *entry = &term->nodes[node];
        if (entry->position < 0 || entry->position >= frame_count) {
            fprintf(stderr, "selection term contains an invalid position\n");
            exit(EXIT_FAILURE);
        }
        path[entry->position] = entry->candidate_index;
    }
    for (int position = 0; position < frame_count; position++) {
        if (path[position] < 0) {
            fprintf(stderr, "selection term leaf lost a continuation\n");
            exit(EXIT_FAILURE);
        }
    }
}

static ProjectionTermOutcome *projection_term_observe_leaf(
    const ProjectionProduct *product,
    ProjectionTerm *term,
    const LlamaCompanyResult *result,
    int leaf_node,
    int *path_nodes
) {
    if (leaf_node <= 0 || leaf_node >= term->count ||
        term->nodes[leaf_node].child_count != 0) {
        fprintf(stderr, "invalid structured outcome leaf\n");
        exit(EXIT_FAILURE);
    }
    ProjectionTermNode *leaf = &term->nodes[leaf_node];
    if (leaf->outcome != NULL) return leaf->outcome;

    int frame_count = product->frame_count;
    for (int position = 0; position < frame_count; position++) {
        path_nodes[position] = -1;
    }
    for (int node = leaf_node; node != 0; node = term->nodes[node].parent) {
        int position = term->nodes[node].position;
        if (position < 0 || position >= frame_count) {
            fprintf(stderr, "structured outcome contains an invalid position\n");
            exit(EXIT_FAILURE);
        }
        path_nodes[position] = node;
    }

    ProjectionTermOutcome *outcome = calloc(1, sizeof(*outcome));
    if (outcome == NULL) {
        fprintf(stderr, "could not allocate structured outcome\n");
        exit(EXIT_FAILURE);
    }
    outcome->coordinate_count = frame_count;
    outcome->coordinates = malloc(
        (size_t)frame_count * sizeof(*outcome->coordinates)
    );
    outcome->leximin_coordinates = malloc(
        (size_t)frame_count * sizeof(*outcome->leximin_coordinates)
    );
    outcome->leximin_positions = malloc(
        (size_t)frame_count * sizeof(*outcome->leximin_positions)
    );
    outcome->context_nodes = malloc(
        (size_t)frame_count * sizeof(*outcome->context_nodes)
    );
    outcome->company_nodes = malloc(
        (size_t)frame_count * sizeof(*outcome->company_nodes)
    );
    outcome->company_tokens = malloc(
        (size_t)frame_count * sizeof(*outcome->company_tokens)
    );
    outcome->incoming_boundary = malloc(
        (size_t)frame_count * sizeof(*outcome->incoming_boundary)
    );
    outcome->logits = malloc(
        (size_t)frame_count * sizeof(*outcome->logits)
    );
    outcome->log_partitions = malloc(
        (size_t)frame_count * sizeof(*outcome->log_partitions)
    );
    if (outcome->coordinates == NULL ||
        outcome->leximin_coordinates == NULL ||
        outcome->leximin_positions == NULL ||
        outcome->context_nodes == NULL ||
        outcome->company_nodes == NULL || outcome->company_tokens == NULL ||
        outcome->incoming_boundary == NULL || outcome->logits == NULL ||
        outcome->log_partitions == NULL) {
        fprintf(stderr, "could not allocate structured outcome coordinates\n");
        exit(EXIT_FAILURE);
    }

    for (int position = 0; position < frame_count; position++) {
        int company_node = path_nodes[position];
        if (company_node <= 0 || company_node >= term->count) {
            fprintf(stderr, "structured outcome lost a token occurrence\n");
            exit(EXIT_FAILURE);
        }
        ProjectionTermNode *company = &term->nodes[company_node];
        ProjectionSelectFrame *frame = &product->frames[company->position];
        int company_token =
            frame->candidates[company->candidate_index].token;
        int context_node = company->parent;

        outcome->context_nodes[position] = context_node;
        outcome->company_nodes[position] = company_node;
        outcome->company_tokens[position] = company_token;
        outcome->incoming_boundary[position] = 1;

        /*
         * The first token has no causal company inside this term. Keep a
         * neutral boundary coordinate for the complete retained outcome; it
         * is never the root observer because every run has a completion.
         */
        if (context_node == 0) {
            outcome->coordinates[position] = 0.0;
            outcome->logits[position] = 0.0;
            outcome->log_partitions[position] = 0.0;
            continue;
        }
        if (context_node < 0 || context_node >= term->count ||
            company_token < 0 || company_token >= result->vocab_size) {
            fprintf(stderr, "structured outcome has no contextual boundary\n");
            exit(EXIT_FAILURE);
        }
        ProjectionTermNode *context = &term->nodes[context_node];
        if (!isfinite(context->row_log_partition)) {
            fprintf(stderr, "invalid company outcome coordinate\n");
            exit(EXIT_FAILURE);
        }
        const float *context_logits = result->logits +
            (size_t)(context_node - 1) * result->vocab_size;
        double logit = context_logits[company_token];
        double coordinate = logit - context->row_log_partition;
        if (!isfinite(coordinate)) {
            fprintf(stderr, "non-finite company outcome coordinate\n");
            exit(EXIT_FAILURE);
        }
        outcome->coordinates[position] = coordinate;
        outcome->logits[position] = logit;
        outcome->log_partitions[position] = context->row_log_partition;
    }

    ProjectionOrderedOpinion *ordered = malloc(
        (size_t)frame_count * sizeof(*ordered)
    );
    if (ordered == NULL) {
        fprintf(stderr, "could not order structured company opinions\n");
        exit(EXIT_FAILURE);
    }
    for (int position = 0; position < frame_count; position++) {
        ordered[position] = (ProjectionOrderedOpinion){
            .coordinate = outcome->coordinates[position],
            .position = position,
        };
    }
    qsort(
        ordered,
        (size_t)frame_count,
        sizeof(*ordered),
        projection_ordered_opinion_compare
    );
    for (int rank = 0; rank < frame_count; rank++) {
        outcome->leximin_coordinates[rank] = ordered[rank].coordinate;
        outcome->leximin_positions[rank] = ordered[rank].position;
    }
    free(ordered);
    leaf->outcome = outcome;
    return outcome;
}

/*
 * Outcomes are ordered by least acceptable company first.  This is the
 * literal arg-min-of-dislike order: maximize the worst token opinion, then the
 * second worst, and so on.  No coordinate is added to another and no scalar
 * reward replaces the complete callback result during strength.
 */
static int projection_term_outcome_compare(
    const ProjectionTermOutcome *left,
    const ProjectionTermOutcome *right
) {
    if (left == NULL || right == NULL ||
        left->coordinate_count != right->coordinate_count) {
        fprintf(stderr, "cannot compare incompatible company outcomes\n");
        exit(EXIT_FAILURE);
    }
    for (int rank = 0; rank < left->coordinate_count; rank++) {
        double left_coordinate = left->leximin_coordinates[rank];
        double right_coordinate = right->leximin_coordinates[rank];
        if (left_coordinate > right_coordinate) return 1;
        if (left_coordinate < right_coordinate) return -1;
    }
    return 0;
}

static double projection_term_worst_opinion(
    const ProjectionTermOutcome *outcome,
    int *position,
    int *context_node,
    int *company_node,
    int *company_token,
    int *incoming_boundary,
    double *observer_logit,
    double *observer_log_partition
) {
    if (outcome == NULL || outcome->coordinate_count <= 0) {
        fprintf(stderr, "selection lost its complete company outcome\n");
        exit(EXIT_FAILURE);
    }
    int worst_position = outcome->leximin_positions[0];
    if (worst_position < 0 ||
        worst_position >= outcome->coordinate_count) {
        fprintf(stderr, "selection outcome has no worst company position\n");
        exit(EXIT_FAILURE);
    }
    *position = worst_position;
    *context_node = outcome->context_nodes[worst_position];
    *company_node = outcome->company_nodes[worst_position];
    *company_token = outcome->company_tokens[worst_position];
    *incoming_boundary = outcome->incoming_boundary[worst_position];
    *observer_logit = outcome->logits[worst_position];
    *observer_log_partition = outcome->log_partitions[worst_position];
    return outcome->coordinates[worst_position];
}

/*
 * Literal memoized specialization of Escardo's product to the already
 * observed finite term:
 *
 *   b(x)   = force the selection rooted below x
 *   a      = Select(x -> p(x : b(x)))
 *   result = a : b(a)
 *
 * `selection_state` is the where-bound memo for b(x). The selected child's
 * stored suffix and complete outcome are the result; neither is recomputed
 * after a is chosen. The function contains no model operation: its only
 * learned input is the
 * immutable LlamaCompanyResult produced before strength starts. The runtime
 * counter check around it enforces that boundary.
 */
static int projection_term_select(
    ProjectionProduct *product,
    ProjectionTerm *term,
    int node_index
) {
    ProjectionTermNode *node = &term->nodes[node_index];
    if (node->selection_state == PROJECTION_SELECTION_FORCED) {
        return node->selected_leaf;
    }
    if (node->selection_state == PROJECTION_SELECTION_FORCING) {
        fprintf(stderr, "selection term contains a recursive cycle\n");
        exit(EXIT_FAILURE);
    }
    node->selection_state = PROJECTION_SELECTION_FORCING;
    if (node->child_count == 0) {
        if (node->outcome == NULL) {
            fprintf(stderr, "leaf outcome was not observed before strength\n");
            exit(EXIT_FAILURE);
        }
        node->selected_child = -1;
        node->selected_leaf = node_index;
        node->selection_state = PROJECTION_SELECTION_FORCED;
        return node->selected_leaf;
    }

    unsigned long long frame_id = product->next_frame_id++;
    product->counters->strength_nodes++;
    int best_child_ordinal = -1;
    ProjectionTermOutcome *best_outcome = NULL;
    double best_worst_opinion = -INFINITY;
    int *trace_path = malloc(
        (size_t)product->frame_count * sizeof(*trace_path)
    );
    if (trace_path == NULL) {
        fprintf(stderr, "could not allocate backed selection path\n");
        exit(EXIT_FAILURE);
    }
    for (int ordinal = 0; ordinal < node->child_count; ordinal++) {
        int child_index = node->children[ordinal];
        /* This is the single force of the memoized b(x). */
        int observer_leaf_node = projection_term_select(
            product,
            term,
            child_index
        );
        ProjectionTermNode *child = &term->nodes[child_index];
        ProjectionSelectFrame *frame = &product->frames[child->position];
        ProjectionCandidate *candidate =
            &frame->candidates[child->candidate_index];
        if (observer_leaf_node != child->selected_leaf ||
            observer_leaf_node <= 0 || observer_leaf_node >= term->count ||
            term->nodes[observer_leaf_node].outcome == NULL) {
            fprintf(stderr, "selection lost its structured leaf outcome\n");
            exit(EXIT_FAILURE);
        }
        ProjectionTermOutcome *outcome =
            term->nodes[observer_leaf_node].outcome;
        int observer_position = -1;
        int observer_context_node = -1;
        int observer_company_node = -1;
        int observer_company_token = -1;
        int observer_incoming_boundary = 0;
        double observer_logit = 0.0;
        double observer_log_partition = 0.0;
        double worst_opinion = projection_term_worst_opinion(
            outcome,
            &observer_position,
            &observer_context_node,
            &observer_company_node,
            &observer_company_token,
            &observer_incoming_boundary,
            &observer_logit,
            &observer_log_partition
        );
        candidate->backed_observer_rating = worst_opinion;
        product->counters->candidate_ratings++;
        projection_term_path(
            term,
            child->selected_leaf,
            product->frame_count,
            trace_path
        );
        trace_projection_product_candidate(
            product,
            frame_id,
            ordinal,
            frame,
            candidate,
            worst_opinion,
            observer_leaf_node,
            child->reachability,
            observer_position,
            observer_context_node,
            observer_company_node,
            observer_company_token,
            observer_incoming_boundary,
            observer_logit,
            observer_log_partition,
            trace_path
        );
        int outcome_order = best_outcome == NULL
            ? 1
            : projection_term_outcome_compare(outcome, best_outcome);
        if (best_child_ordinal < 0 || outcome_order > 0 ||
            (outcome_order == 0 &&
             candidate->local_rank <
                product->frames[
                    term->nodes[node->children[best_child_ordinal]].position
                ].candidates[
                    term->nodes[
                        node->children[best_child_ordinal]
                    ].candidate_index
                ].local_rank)) {
            best_child_ordinal = ordinal;
            best_outcome = outcome;
            best_worst_opinion = worst_opinion;
        }
    }
    int best_child_index = node->children[best_child_ordinal];
    ProjectionTermNode *best_child = &term->nodes[best_child_index];
    node->selected_child = best_child_index;
    node->selected_leaf = best_child->selected_leaf;
    projection_term_path(
        term,
        node->selected_leaf,
        product->frame_count,
        trace_path
    );
    ProjectionSelectFrame *best_frame =
        &product->frames[best_child->position];
    ProjectionCandidate *best_candidate =
        &best_frame->candidates[best_child->candidate_index];
    int best_observer_position = -1;
    int best_observer_context_node = -1;
    int best_observer_company_node = -1;
    int best_observer_company_token = -1;
    int best_observer_incoming_boundary = 0;
    double best_observer_logit = 0.0;
    double best_observer_log_partition = 0.0;
    double selected_worst_opinion = projection_term_worst_opinion(
        best_outcome,
        &best_observer_position,
        &best_observer_context_node,
        &best_observer_company_node,
        &best_observer_company_token,
        &best_observer_incoming_boundary,
        &best_observer_logit,
        &best_observer_log_partition
    );
    if (selected_worst_opinion != best_worst_opinion) {
        fprintf(stderr, "selection changed its retained company outcome\n");
        exit(EXIT_FAILURE);
    }
    trace_projection_product_select(
        product,
        frame_id,
        best_frame,
        best_candidate,
        selected_worst_opinion,
        node->selected_leaf,
        best_child->reachability,
        best_observer_position,
        best_observer_context_node,
        best_observer_company_node,
        best_observer_company_token,
        best_observer_incoming_boundary,
        best_observer_logit,
        best_observer_log_partition,
        trace_path
    );
    if (best_frame->position == product->first_observed_position) {
        product->first_variable_selection_rating = selected_worst_opinion;
        product->first_variable_selection_rating_set = 1;
    }
    node->selection_state = PROJECTION_SELECTION_FORCED;
    free(trace_path);
    return node->selected_leaf;
}

static void projection_term_free(ProjectionTerm *term) {
    for (int index = 0; index < term->count; index++) {
        free(term->nodes[index].children);
        ProjectionTermOutcome *outcome = term->nodes[index].outcome;
        if (outcome != NULL) {
            free(outcome->log_partitions);
            free(outcome->logits);
            free(outcome->leximin_positions);
            free(outcome->leximin_coordinates);
            free(outcome->incoming_boundary);
            free(outcome->company_tokens);
            free(outcome->company_nodes);
            free(outcome->context_nodes);
            free(outcome->coordinates);
            free(outcome);
        }
    }
    free(term->nodes);
    memset(term, 0, sizeof(*term));
}

static double projection_company_strength_select(
    ProjectionProduct *product,
    unsigned long long path_demands,
    int *scratch_path,
    int *selected_path
) {
    if (product->company_runtime == NULL) {
        fprintf(stderr, "company selection has no family runtime\n");
        exit(EXIT_FAILURE);
    }
    ProjectionTerm term = {0};
    int synthetic_root = projection_term_add_node(&term, -1, -1, -1);
    if (synthetic_root != 0) {
        fprintf(stderr, "selection term lost its synthetic root\n");
        exit(EXIT_FAILURE);
    }
    projection_term_sample_paths(
        product,
        &term,
        path_demands,
        scratch_path
    );
    int row_count = term.count - 1;
    if (row_count <= 0 || term.leaf_count <= 0) {
        fprintf(stderr, "demanded selection term is empty\n");
        exit(EXIT_FAILURE);
    }
    product->counters->company_rows = (unsigned long long)row_count;
    if (product->counters->trace != NULL) {
        fprintf(
            product->counters->trace,
            "{\"event\":\"selection_term_built\",\"rows\":%d,"
            "\"unique_leaves\":%llu,\"path_demands\":%llu,"
            "\"exact\":false,\"full_vocabulary_carrier\":true,"
            "\"proposal_top_k\":%d,"
            "\"root_reachability\":%llu}\n",
            row_count,
            term.leaf_count,
            term.path_demands,
            product->local_support,
            term.nodes[0].reachability
        );
        fflush(product->counters->trace);
    }

    int *tokens = malloc((size_t)row_count * sizeof(*tokens));
    int *positions = malloc((size_t)row_count * sizeof(*positions));
    int *parents = malloc((size_t)row_count * sizeof(*parents));
    if (tokens == NULL || positions == NULL || parents == NULL) {
        fprintf(stderr, "could not allocate causal company shape\n");
        exit(EXIT_FAILURE);
    }
    for (int index = 1; index < term.count; index++) {
        ProjectionTermNode *node = &term.nodes[index];
        ProjectionSelectFrame *frame = &product->frames[node->position];
        int row = index - 1;
        tokens[row] = frame->candidates[node->candidate_index].token;
        positions[row] = node->position;
        parents[row] = node->parent == 0 ? -1 : node->parent - 1;
    }
    LlamaCompanyShape shape = {
        .row_count = row_count,
        .tokens = tokens,
        .positions = positions,
        .parents = parents,
    };

    int filler_count = atkey_filler_count(product->company_runtime);
    size_t *calls_before = malloc(
        (size_t)filler_count * sizeof(*calls_before)
    );
    size_t *reads_before = malloc(
        (size_t)filler_count * sizeof(*reads_before)
    );
    if (calls_before == NULL || reads_before == NULL) {
        fprintf(stderr, "could not allocate family filler counters\n");
        exit(EXIT_FAILURE);
    }
    for (int filler = 0; filler < filler_count; filler++) {
        calls_before[filler] = atkey_filler_calls(
            product->company_runtime,
            filler
        );
        reads_before[filler] = atkey_filler_scalar_reads(
            product->company_runtime,
            filler
        );
    }
    LlamaCompanyResult result;
    unsigned long long company_model_start =
        projection_monotonic_nanoseconds();
    if (!llama_company_evaluate(
            product->company_runtime,
            &shape,
            false,
            &result
        )) {
        fprintf(stderr, "could not evaluate demanded causal company\n");
        exit(EXIT_FAILURE);
    }
    product->counters->company_model_nanoseconds =
        projection_monotonic_nanoseconds() - company_model_start;
    for (int filler = 0; filler < filler_count; filler++) {
        size_t calls = atkey_filler_calls(
            product->company_runtime,
            filler
        ) - calls_before[filler];
        size_t reads = atkey_filler_scalar_reads(
            product->company_runtime,
            filler
        ) - reads_before[filler];
        product->counters->family_filler_calls +=
            (unsigned long long)calls;
        product->counters->family_scalar_reads +=
            (unsigned long long)reads;
        if ((unsigned long long)calls >
            product->counters->maximum_calls_per_filler) {
            product->counters->maximum_calls_per_filler =
                (unsigned long long)calls;
        }
    }
    if (product->counters->trace != NULL) {
        fprintf(
            product->counters->trace,
            "{\"event\":\"company_run\",\"rows\":%d,"
            "\"family_filler_calls\":%llu,"
            "\"maximum_calls_per_filler\":%llu,"
            "\"family_scalar_reads\":%llu,"
            "\"model_ms\":%.9g}\n",
            row_count,
            product->counters->family_filler_calls,
            product->counters->maximum_calls_per_filler,
            product->counters->family_scalar_reads,
            (double)product->counters->company_model_nanoseconds / 1000000.0
        );
        fflush(product->counters->trace);
    }

    /* From this point onward strength is a pure computation over `result`. */
    for (int filler = 0; filler < filler_count; filler++) {
        calls_before[filler] = atkey_filler_calls(
            product->company_runtime,
            filler
        );
        reads_before[filler] = atkey_filler_scalar_reads(
            product->company_runtime,
            filler
        );
    }

    int *leaf_path = malloc(
        (size_t)product->frame_count * sizeof(*leaf_path)
    );
    int *leaf_path_nodes = malloc(
        (size_t)product->frame_count * sizeof(*leaf_path_nodes)
    );
    if (leaf_path == NULL || leaf_path_nodes == NULL) {
        fprintf(stderr, "could not allocate family-observed path\n");
        exit(EXIT_FAILURE);
    }
    for (int index = 1; index < term.count; index++) {
        ProjectionTermNode *node = &term.nodes[index];
        int row = index - 1;
        float *row_logits = result.logits +
            (size_t)row * result.vocab_size;
        node->row_log_partition = projection_log_partition(
            row_logits,
            result.vocab_size
        );
    }
    for (int index = 1; index < term.count; index++) {
        ProjectionTermNode *node = &term.nodes[index];
        if (node->child_count != 0) continue;
        ProjectionTermOutcome *outcome = projection_term_observe_leaf(
            product,
            &term,
            &result,
            index,
            leaf_path_nodes
        );
        product->counters->structured_leaf_outcomes++;
        projection_term_path(
            &term,
            index,
            product->frame_count,
            leaf_path
        );
        if (product->counters->trace != NULL) {
            FILE *stream = product->counters->trace;
            fprintf(
                stream,
                "{\"event\":\"company_outcome\",\"leaf_node\":%d,"
                "\"outcome\":\"candidate_owned_context_opinions\","
                "\"coordinates\":[",
                index
            );
            for (int position = 0; position < outcome->coordinate_count;
                 position++) {
                if (position != 0) fputc(',', stream);
                fprintf(stream, "%.17g", outcome->coordinates[position]);
            }
            fputs("],\"leximin\":[", stream);
            for (int rank = 0; rank < outcome->coordinate_count; rank++) {
                if (rank != 0) fputc(',', stream);
                fprintf(
                    stream,
                    "{\"rank\":%d,\"position\":%d,\"opinion\":%.17g}",
                    rank,
                    outcome->leximin_positions[rank],
                    outcome->leximin_coordinates[rank]
                );
            }
            fputs("],\"bindings\":[", stream);
            for (int position = 0; position < outcome->coordinate_count;
                 position++) {
                if (position != 0) fputc(',', stream);
                fprintf(
                    stream,
                    "{\"position\":%d,\"context_node\":%d,"
                    "\"company_node\":%d,\"company_token\":%d,"
                    "\"direction\":\"%s\",\"logit\":%.17g,"
                    "\"log_partition\":%.17g}",
                    position,
                    outcome->context_nodes[position],
                    outcome->company_nodes[position],
                    outcome->company_tokens[position],
                    outcome->context_nodes[position] == 0
                        ? "term_root" : "candidate_about_context",
                    outcome->logits[position],
                    outcome->log_partitions[position]
                );
            }
            fputs("],\"text\":", stream);
            projection_trace_path(stream, product, leaf_path);
            fputs("}\n", stream);
            fflush(stream);
        }
    }

    unsigned long long strength_start = projection_monotonic_nanoseconds();
    int selected_leaf = projection_term_select(product, &term, 0);
    product->counters->strength_nanoseconds =
        projection_monotonic_nanoseconds() - strength_start;
    if (selected_leaf != term.nodes[0].selected_leaf ||
        selected_leaf <= 0 || selected_leaf >= term.count) {
        fprintf(stderr, "root selection lost its retained outcome\n");
        exit(EXIT_FAILURE);
    }
    projection_term_path(
        &term,
        selected_leaf,
        product->frame_count,
        selected_path
    );
    ProjectionTermOutcome *selected_outcome =
        term.nodes[selected_leaf].outcome;
    if (selected_outcome == NULL) {
        fprintf(stderr, "root selection lost its complete outcome\n");
        exit(EXIT_FAILURE);
    }
    int worst_position = -1;
    int worst_context_node = -1;
    int worst_company_node = -1;
    int worst_company_token = -1;
    int worst_incoming_boundary = 0;
    double worst_logit = 0.0;
    double worst_log_partition = 0.0;
    double rating = projection_term_worst_opinion(
        selected_outcome,
        &worst_position,
        &worst_context_node,
        &worst_company_node,
        &worst_company_token,
        &worst_incoming_boundary,
        &worst_logit,
        &worst_log_partition
    );
    product->counters->root_terminalizations++;
    if (product->counters->trace != NULL) {
        FILE *stream = product->counters->trace;
        fprintf(
            stream,
            "{\"event\":\"root_terminalized\","
            "\"outcome\":\"candidate_owned_context_opinions\","
            "\"selected_outcome_leaf\":%d,"
            "\"order\":\"leximin_all_coordinates\","
            "\"worst_company_position\":%d,"
            "\"worst_company_opinion\":%.17g,"
            "\"worst_context_node\":%d,"
            "\"worst_company_node\":%d,"
            "\"worst_company_token\":%d,"
            "\"worst_direction\":\"%s\","
            "\"worst_logit\":%.17g,"
            "\"worst_log_partition\":%.17g,"
            "\"coordinates\":[",
            selected_leaf,
            worst_position,
            rating,
            worst_context_node,
            worst_company_node,
            worst_company_token,
            worst_incoming_boundary
                ? "candidate_about_context" : "outgoing",
            worst_logit,
            worst_log_partition
        );
        for (int position = 0;
             position < selected_outcome->coordinate_count; position++) {
            if (position != 0) fputc(',', stream);
            fprintf(stream, "%.17g", selected_outcome->coordinates[position]);
        }
        fputs("],\"text\":", stream);
        projection_trace_path(stream, product, selected_path);
        fputs("}\n", stream);
        fflush(stream);
    }

    for (int filler = 0; filler < filler_count; filler++) {
        size_t calls = atkey_filler_calls(
            product->company_runtime,
            filler
        ) - calls_before[filler];
        size_t reads = atkey_filler_scalar_reads(
            product->company_runtime,
            filler
        ) - reads_before[filler];
        product->counters->strength_filler_calls +=
            (unsigned long long)calls;
        product->counters->strength_scalar_reads +=
            (unsigned long long)reads;
    }
    if (product->counters->strength_filler_calls != 0 ||
        product->counters->strength_scalar_reads != 0) {
        fprintf(stderr, "strength invoked a learned model filler\n");
        exit(EXIT_FAILURE);
    }
    if (product->counters->trace != NULL) {
        fprintf(
            product->counters->trace,
            "{\"event\":\"strength_run\","
            "\"algorithm\":\"memoized_escardo_leximin_product\","
            "\"model_filler_calls\":%llu,"
            "\"model_scalar_reads\":%llu,"
            "\"milliseconds\":%.9g}\n",
            product->counters->strength_filler_calls,
            product->counters->strength_scalar_reads,
            (double)product->counters->strength_nanoseconds / 1000000.0
        );
        fflush(product->counters->trace);
    }

    free(leaf_path_nodes);
    free(leaf_path);
    free(reads_before);
    free(calls_before);
    llama_company_result_free(&result);
    free(parents);
    free(positions);
    free(tokens);
    projection_term_free(&term);
    return rating;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate_hidden_feedback_select(
    Transformer *transformer,
    AtkeyRuntime *company_runtime,
    Tokenizer *tokenizer,
    char *prompt,
    int steps,
    int local_support,
    unsigned long long leaf_limit,
    unsigned long long sample_seed,
    FeedbackBoundary feedback_boundary,
    ProjectionObserverKind observer_kind,
    FILE *trace
) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    if (num_prompt_tokens >= steps) {
        fprintf(stderr, "prompt leaves no completion positions\n");
        exit(EXIT_FAILURE);
    }

    Config *config = &transformer->config;
    if (local_support < 0 || local_support > config->vocab_size) {
        fprintf(stderr, "company proposal top-k exceeds the vocabulary\n");
        exit(EXIT_FAILURE);
    }
    if (observer_kind == PROJECTION_OBSERVER_COMPANY_STRENGTH &&
        leaf_limit == ULLONG_MAX) {
        fprintf(stderr, "company selection requires explicit -b path demands\n");
        exit(EXIT_FAILURE);
    }
    int output_count = steps - num_prompt_tokens;
    int frame_count = steps;
    size_t hidden_count =
        (size_t)(output_count + 1) * (size_t)config->dim;
    size_t prefill_hidden_count =
        (size_t)num_prompt_tokens * (size_t)config->dim;
    /* Every frame owns a full-vocabulary slab; Select-unit frames use slot 0. */
    size_t candidate_count =
        (size_t)frame_count * (size_t)config->vocab_size;
    if (config->dim <= 0 || output_count <= 0 ||
        frame_count != steps ||
        hidden_count / (size_t)config->dim != (size_t)(output_count + 1) ||
        prefill_hidden_count / (size_t)config->dim !=
            (size_t)num_prompt_tokens ||
        candidate_count / (size_t)config->vocab_size != (size_t)frame_count) {
        fprintf(stderr, "invalid hidden-state sequence size\n");
        exit(EXIT_FAILURE);
    }
    float *hidden_sequence = malloc(hidden_count * sizeof(*hidden_sequence));
    float *prefill_hidden_sequence = malloc(
        prefill_hidden_count * sizeof(*prefill_hidden_sequence)
    );
    ProjectionSelectFrame *selection_frames = calloc(
        (size_t)frame_count,
        sizeof(*selection_frames)
    );
    ProjectionCandidate *projection_candidates = calloc(
        candidate_count,
        sizeof(*projection_candidates)
    );
    float *target_displacements = malloc(
        (size_t)output_count * config->dim * sizeof(*target_displacements)
    );
    float *candidate_previous = malloc(
        (size_t)config->dim * sizeof(*candidate_previous)
    );
    int *partial_path = calloc((size_t)frame_count, sizeof(*partial_path));
    int *selected_path = calloc((size_t)frame_count, sizeof(*selected_path));
    if (hidden_sequence == NULL || prefill_hidden_sequence == NULL ||
        selection_frames == NULL ||
        projection_candidates == NULL || target_displacements == NULL ||
        candidate_previous == NULL || partial_path == NULL ||
        selected_path == NULL) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }

    if (trace != NULL) {
        fputs("{\"event\":\"run\",\"mode\":"
              "\"hidden_state_feedback_escardo_projection\",\"prompt\":",
              trace);
        projection_json_string(trace, prompt);
        fprintf(
            trace,
            ",\"steps\":%d,\"prefill_unit_count\":%d,"
            "\"output_count\":%d,\"selection_frame_count\":%d,"
            "\"proposal_vocabulary_size\":%d,"
            "\"proposal_top_k\":%d,\"path_demands\":%llu,"
            "\"feedback_boundary\":\"%s\","
            "\"observer\":\"%s\"}\n",
            steps,
            num_prompt_tokens,
            output_count,
            frame_count,
            config->vocab_size,
            local_support,
            leaf_limit,
            feedback_boundary_name(feedback_boundary),
            projection_observer_name(observer_kind)
        );
        fflush(trace);
    }

    long start = time_in_ms();

    /*
     * Numerically this is the ordinary deterministic prefill. Semantically
     * each token also occupies a singleton Select frame below; retaining every
     * hidden state lets its unit quantifier apply the learned observer later.
     */
    for (int pos = 0; pos < num_prompt_tokens; pos++) {
        float *hidden = forward_token_hidden(
            transformer,
            prompt_tokens[pos],
            pos
        );
        memcpy(
            prefill_hidden_sequence + (size_t)pos * config->dim,
            hidden,
            (size_t)config->dim * sizeof(*prefill_hidden_sequence)
        );
        if (pos == num_prompt_tokens - 1) {
            memcpy(
                hidden_sequence,
                hidden,
                (size_t)config->dim * sizeof(*hidden_sequence)
            );
        }
    }

    // The final prompt state predicts output zero. Retain one additional state
    // after every completion slot so the target and re-embedded path pools both
    // contain one contextual state per token frame.
    for (int index = 1; index <= output_count; index++) {
        int pos = num_prompt_tokens + index - 1;
        float *hidden = forward_feedback_hidden(
            transformer,
            pos,
            feedback_boundary
        );
        memcpy(
            hidden_sequence + (size_t)index * config->dim,
            hidden,
            (size_t)config->dim * sizeof(*hidden_sequence)
        );
    }

    long recurrence_end = time_in_ms();

    for (int index = 1; index <= output_count; index++) {
        const float *previous = hidden_sequence +
            (size_t)(index - 1) * config->dim;
        const float *current = hidden_sequence +
            (size_t)index * config->dim;
        float *displacement = target_displacements +
            (size_t)(index - 1) * config->dim;
        for (int lane = 0; lane < config->dim; lane++) {
            displacement[lane] = current[lane] - previous[lane];
        }
    }
    size_t target_value_count = (size_t)output_count * config->dim;
    double target_displacement_norm = projection_vector_norm(
        target_displacements,
        target_value_count
    );
    if (!(target_displacement_norm > 0.0) ||
        !isfinite(target_displacement_norm)) {
        fprintf(stderr, "invalid retained hidden-state displacements\n");
        exit(EXIT_FAILURE);
    }

    for (int position = 0; position < frame_count; position++) {
        ProjectionSelectFrame *frame = &selection_frames[position];
        *frame = (ProjectionSelectFrame){
            .position = position,
            .candidate_count = position < num_prompt_tokens
                ? 1
                : config->vocab_size,
            .candidates = projection_candidates +
                (size_t)position * config->vocab_size,
            .selected_index = -1,
        };
        if (position < num_prompt_tokens) {
            frame->candidates[0] = (ProjectionCandidate){
                .token = prompt_tokens[position],
                .local_rank = 1,
                .logit = position == 0 ? 0.0f : -INFINITY,
                .log_probability = position == 0 ? 0.0 : -INFINITY,
            };
        }
    }

    /* Record the ordinary learned observation of each forced prefill token. */
    for (int position = 1; position < num_prompt_tokens; position++) {
        matmul(
            transformer->state.logits,
            prefill_hidden_sequence + (size_t)(position - 1) * config->dim,
            transformer->weights.wcls,
            config->dim,
            config->vocab_size
        );
        ProjectionCandidate *unit = &selection_frames[position].candidates[0];
        double partition = projection_log_partition(
            transformer->state.logits,
            config->vocab_size
        );
        unit->logit = transformer->state.logits[unit->token];
        unit->log_probability = (double)unit->logit - partition;
    }

    for (int position = 0; position < num_prompt_tokens; position++) {
        projection_trace_carrier_candidate(
            trace,
            tokenizer,
            &selection_frames[position],
            &selection_frames[position].candidates[0]
        );
    }

    // Only now observe every retained completion hidden state through the
    // classifier and retain its complete vocabulary as the local Select carrier.
    for (int index = 0; index < output_count; index++) {
        matmul(
            transformer->state.logits,
            hidden_sequence + (size_t)index * config->dim,
            transformer->weights.wcls,
            config->dim,
            config->vocab_size
        );
        ProjectionSelectFrame *frame =
            &selection_frames[num_prompt_tokens + index];
        projection_full_vocabulary(
            transformer->state.logits,
            config->vocab_size,
            frame->candidates
        );
        for (int rank = 0; rank < frame->candidate_count; rank++) {
            projection_trace_carrier_candidate(
                trace,
                tokenizer,
                frame,
                &frame->candidates[rank]
            );
        }
    }

    long projection_end = time_in_ms();

    ProjectionStrengthCounters strength = {
        .trace = trace,
        .tokenizer = tokenizer,
    };

    Transformer observer_transformer = {0};
    int observer_state_allocated = 0;
    if (observer_kind == PROJECTION_OBSERVER_HIDDEN_DISPLACEMENT) {
        observer_transformer = (Transformer){
            .config = transformer->config,
            .weights = transformer->weights,
            .fd = -1,
            .data = MAP_FAILED,
        };
        malloc_run_state(
            &observer_transformer.state,
            &observer_transformer.config
        );
        observer_state_allocated = 1;
    }
    ProjectionProduct product = {
        .observer_transformer = observer_transformer,
        .company_runtime = company_runtime,
        .frames = selection_frames,
        .frame_count = frame_count,
        .dim = config->dim,
        .first_observed_position = num_prompt_tokens,
        .observed_position_count = output_count,
        .target_displacements = target_displacements,
        .candidate_previous = candidate_previous,
        .target_norm = target_displacement_norm,
        .observer_kind = observer_kind,
        .local_support = local_support,
        .sample_seed = sample_seed,
        .sample_state = projection_mix(
            sample_seed ^ 0x6a09e667f3bcc909ULL
        ),
        .counters = &strength,
    };

    long observer_start = time_in_ms();
    double selection_rating;
    double terminal_rating;
    if (observer_kind == PROJECTION_OBSERVER_LOGIT_STRENGTH) {
        selection_rating = projection_logit_product_select(
            &product,
            selected_path
        );
        terminal_rating = selection_rating;
    } else if (observer_kind == PROJECTION_OBSERVER_COMPANY_STRENGTH) {
        selection_rating = projection_company_strength_select(
            &product,
            leaf_limit,
            partial_path,
            selected_path
        );
        terminal_rating = selection_rating;
    } else {
        selection_rating = projection_product_select(
            &product,
            0,
            leaf_limit,
            partial_path,
            selected_path
        );
        terminal_rating = projection_observe_whole_path(
            &product,
            selected_path,
            1
        );
    }
    for (int position = 0; position < frame_count; position++) {
        selection_frames[position].selected_index = selected_path[position];
    }
    if (observer_kind == PROJECTION_OBSERVER_HIDDEN_DISPLACEMENT &&
        fabs(selection_rating - terminal_rating) > 1e-12) {
        fprintf(stderr, "composed observer and terminalization disagree\n");
        exit(EXIT_FAILURE);
    }

    long selection_end = time_in_ms();

    // Reconstruct the prompt exactly as stock generate() does, then decode the
    // selected deferred projections. Decoding does not feed them back to the
    // model.
    int previous = prompt_tokens[0];
    for (int index = 1; index < num_prompt_tokens; index++) {
        safe_printf(decode(tokenizer, previous, prompt_tokens[index]));
        previous = prompt_tokens[index];
    }
    int decoded_count = 0;
    for (int index = 0; index < output_count; index++) {
        ProjectionSelectFrame *frame =
            &selection_frames[num_prompt_tokens + index];
        int token = frame->candidates[frame->selected_index].token;
        if (token == 1) break;
        safe_printf(decode(tokenizer, previous, token));
        previous = token;
        decoded_count++;
    }
    printf("\n");
    fflush(stdout);

    fprintf(
        stderr,
        "mode: hidden_state_feedback_escardo_projection\n"
        "feedback_boundary: %s\n"
        "prompt_positions: %d\n"
        "prefill_unit_selections: %d\n"
        "feedback_positions: %d\n"
        "deferred_projections: %d\n"
        "proposal_vocabulary_size: %d\n"
        "proposal_top_k: %d\n"
        "path_demands: %llu\n"
        "decoded_tokens: %d\n"
        "selection_carrier: ProjectionTermOutcome\n"
        "outcome_kind: %s\n"
        "selected_coordinate_role: %s\n"
        "strength_nodes: %llu\n"
        "whole_path_observer_applications: %llu\n"
        "structured_leaf_outcomes: %llu\n"
        "strength_candidate_ratings: %llu\n"
        "root_terminalizations: %llu\n"
        "company_rows: %llu\n"
        "family_filler_calls: %llu\n"
        "maximum_calls_per_filler: %llu\n"
        "family_scalar_reads: %llu\n"
        "strength_model_filler_calls: %llu\n"
        "strength_model_scalar_reads: %llu\n"
        "company_model_ms: %.3f\n"
        "pure_strength_ms: %.3f\n"
        "selected_terminal_diagnostic: %.17g\n"
        "recurrence_ms: %ld\n"
        "projection_ms: %ld\n"
        "observer_product_ms: %ld\n",
        feedback_boundary_name(feedback_boundary),
        num_prompt_tokens,
        num_prompt_tokens,
        output_count,
        output_count,
        config->vocab_size,
        local_support,
        leaf_limit,
        decoded_count,
        projection_observer_name(observer_kind),
        projection_score_role(observer_kind),
        strength.strength_nodes,
        strength.observer_applications,
        strength.structured_leaf_outcomes,
        strength.candidate_ratings,
        strength.root_terminalizations,
        strength.company_rows,
        strength.family_filler_calls,
        strength.maximum_calls_per_filler,
        strength.family_scalar_reads,
        strength.strength_filler_calls,
        strength.strength_scalar_reads,
        (double)strength.company_model_nanoseconds / 1000000.0,
        (double)strength.strength_nanoseconds / 1000000.0,
        selection_rating,
        recurrence_end - start,
        projection_end - recurrence_end,
        selection_end - observer_start
    );
    if (trace != NULL) {
        fprintf(
            trace,
            "{\"event\":\"run_end\","
            "\"selected_terminal_diagnostic\":%.17g,"
            "\"selected_coordinate_role\":\"%s\","
            "\"structured_leaf_outcomes\":%llu,"
            "\"strength_nodes\":%llu,"
            "\"whole_path_observer_applications\":%llu,"
            "\"strength_candidate_ratings\":%llu,"
            "\"root_terminalizations\":%llu,"
            "\"company_rows\":%llu,"
            "\"family_filler_calls\":%llu,"
            "\"maximum_calls_per_filler\":%llu,"
            "\"family_scalar_reads\":%llu,"
            "\"strength_model_filler_calls\":%llu,"
            "\"strength_model_scalar_reads\":%llu,"
            "\"company_model_ms\":%.9g,"
            "\"pure_strength_ms\":%.9g",
            selection_rating,
            projection_score_role(observer_kind),
            strength.structured_leaf_outcomes,
            strength.strength_nodes,
            strength.observer_applications,
            strength.candidate_ratings,
            strength.root_terminalizations,
            strength.company_rows,
            strength.family_filler_calls,
            strength.maximum_calls_per_filler,
            strength.family_scalar_reads,
            strength.strength_filler_calls,
            strength.strength_scalar_reads,
            (double)strength.company_model_nanoseconds / 1000000.0,
            (double)strength.strength_nanoseconds / 1000000.0
        );
        fputs("}\n", trace);
        fflush(trace);
    }

    if (observer_state_allocated) {
        free_run_state(&product.observer_transformer.state);
    }
    free(selected_path);
    free(partial_path);
    free(candidate_previous);
    free(target_displacements);
    free(projection_candidates);
    free(selection_frames);
    free(prefill_hidden_sequence);
    free(hidden_sequence);
    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int prev_token;
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if (token == 2) { user_turn = 1; }

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}


// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run_hidden_feedback_select <checkpoint> [options]\n");
    fprintf(stderr, "Example: run_hidden_feedback_select model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -l <int>    exact completion positions; overrides total -n\n");
    fprintf(stderr, "  -g <string> feedback boundary: identity (default) or affine\n");
    fprintf(stderr, "  -r <string> observer: company (default) or logits\n");
    fprintf(stderr, "  -k <int>    proposal top-k; 0 = full vocabulary (default)\n");
    fprintf(stderr, "  -b <int>    complete hypothetical path demands (required)\n");
    fprintf(stderr, "  -s <int>    support sampling seed, default 42\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -o <string> optional flushed JSONL candidate trace\n");
    fprintf(stderr, "  -m <string> only generate is supported\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    int steps = 256;            // number of steps to run for
    int completion_length = -1;
    int local_support = 0;
    unsigned long long leaf_limit = ULLONG_MAX;
    unsigned long long sample_seed = 42;
    FeedbackBoundary feedback_boundary = FEEDBACK_IDENTITY;
    ProjectionObserverKind observer_kind =
        PROJECTION_OBSERVER_COMPANY_STRENGTH;
    char *prompt = NULL;        // prompt string
    char *trace_path = NULL;
    char *mode = "generate";    // generate|chat

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'l') {
            completion_length = atoi(argv[i + 1]);
            if (completion_length <= 0) error_usage();
        }
        else if (argv[i][1] == 'g') {
            if (strcmp(argv[i + 1], "affine") == 0) {
                feedback_boundary = FEEDBACK_AFFINE_TOKEN_BARYCENTER;
            } else if (strcmp(argv[i + 1], "identity") == 0) {
                feedback_boundary = FEEDBACK_IDENTITY;
            } else {
                error_usage();
            }
        }
        else if (argv[i][1] == 'r') {
            if (strcmp(argv[i + 1], "company") == 0) {
                observer_kind = PROJECTION_OBSERVER_COMPANY_STRENGTH;
            } else if (strcmp(argv[i + 1], "logits") == 0) {
                observer_kind = PROJECTION_OBSERVER_LOGIT_STRENGTH;
            } else {
                error_usage();
            }
        }
        else if (argv[i][1] == 'b' || argv[i][1] == 's' ||
                 argv[i][1] == 'k') {
            errno = 0;
            char *end = NULL;
            unsigned long long parsed = strtoull(argv[i + 1], &end, 10);
            if (errno != 0 || end == argv[i + 1] || *end != '\0') {
                error_usage();
            }
            if (argv[i][1] == 'b') {
                if (parsed == 0) error_usage();
                leaf_limit = parsed;
            } else if (argv[i][1] == 's') {
                if (parsed == 0) error_usage();
                sample_seed = parsed;
            } else {
                if (parsed > INT_MAX) error_usage();
                local_support = (int)parsed;
            }
        }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'o') { trace_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

    AtkeyRuntime *company_runtime = NULL;
    if (observer_kind == PROJECTION_OBSERVER_COMPANY_STRENGTH) {
        company_runtime = atkey_runtime_new(checkpoint_path, tokenizer_path);
        if (company_runtime == NULL) {
            fprintf(stderr, "could not initialize one-shot company runtime\n");
            exit(EXIT_FAILURE);
        }
    }

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    if (completion_length > 0) {
        const char *effective_prompt = prompt == NULL ? "" : prompt;
        size_t prompt_capacity = strlen(effective_prompt) + 3;
        if (prompt_capacity > (size_t)INT_MAX) {
            fprintf(stderr, "prompt is too large to tokenize\n");
            exit(EXIT_FAILURE);
        }
        int *counted_tokens = malloc(prompt_capacity * sizeof(*counted_tokens));
        if (counted_tokens == NULL) {
            fprintf(stderr, "could not allocate prompt token count\n");
            exit(EXIT_FAILURE);
        }
        int counted_prompt_tokens = 0;
        encode(
            &tokenizer,
            (char *)effective_prompt,
            1,
            0,
            counted_tokens,
            &counted_prompt_tokens
        );
        free(counted_tokens);
        if (counted_prompt_tokens > transformer.config.seq_len -
                completion_length) {
            fprintf(
                stderr,
                "prompt has %d tokens; %d completion positions exceed "
                "the %d-token model context\n",
                counted_prompt_tokens,
                completion_length,
                transformer.config.seq_len
            );
            exit(EXIT_FAILURE);
        }
        steps = counted_prompt_tokens + completion_length;
    }

    FILE *trace = NULL;
    if (trace_path != NULL) {
        trace = fopen(trace_path, "w");
        if (trace == NULL) {
            fprintf(stderr, "could not open trace output: %s\n", trace_path);
            exit(EXIT_FAILURE);
        }
    }

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate_hidden_feedback_select(
            &transformer,
            company_runtime,
            &tokenizer,
            prompt,
            steps,
            local_support,
            leaf_limit,
            sample_seed,
            feedback_boundary,
            observer_kind,
            trace
        );
    } else {
        fprintf(stderr, "hidden-state feedback supports generate mode only: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    if (trace != NULL) fclose(trace);
    atkey_runtime_free(company_runtime);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif
