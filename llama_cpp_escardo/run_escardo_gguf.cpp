/*
 * Exact finite product of model-backed selection functions over llama.cpp.
 *
 * There is one observer operation.  For a fixed selected suffix s it returns
 * the causal posterior frame
 *
 *   observe(left, s)[x'] = log P_model(x' ++ s | left).
 *
 * Every coordinate in one frame has exactly the same left and right company,
 * so its additive normalization is common.  This is the model-defined
 * posterior over the missing constructor, not a later next-token logit
 * repurposed as a retrospective score.  The empty suffix is exactly the
 * model's existing proposal covector (the selection-product unit law).
 *
 * Escardo's dependent product is evaluated literally.  For every x, first
 * obtain b(x) by recursively applying the suffix selection.  Then apply the
 * same observer to (prefix, b(x)).  A branch attains when x is the maximum
 * coordinate of its own observer covector over the common finite support.
 * The local searchable-set selection returns the first attaining branch, or
 * the final branch when none attains.  Values from distinct covector frames
 * are never compared or added.  The selected outcome retains the complete
 * position-indexed covector family, and only the root emits its token tuple.
 *
 * This deliberately contains no path-likelihood sum, scalar backup, UCB
 * bonus, or sampled AR rollout.  Model applications at a causal frontier and
 * bound-continuation observations are batched; batching changes numerical
 * scheduling, not the selection-product law.
 */

#include "llama.h"
#include "chat.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

struct Options {
    std::string model_path;
    std::string prompt;
    std::string trace_path;
    int length = -1;
    int top_k = 0;
    int gpu_layers = 999;
    int threads = 8;
    int context_size = 0;
    bool exact = false;
    bool allow_eog = false;
    bool use_chat_template = false;
    bool enable_thinking = true;
};

long parse_long(const char * text, const char * name) {
    errno = 0;
    char * end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        fail(std::string(name) + " must be an integer");
    }
    return value;
}

[[noreturn]] void usage(const char * program) {
    std::fprintf(
        stderr,
        "usage: %s MODEL.gguf --prompt TEXT --length N --exact --top-k K "
        "[--trace FILE] [--gpu-layers N] [--threads N] [--ctx-size N] "
        "[--allow-eog] [--chat --reasoning on|off]\n",
        program
    );
    std::exit(EXIT_FAILURE);
}

Options parse_options(int argc, char ** argv) {
    if (argc < 2) usage(argv[0]);
    Options options;
    options.model_path = argv[1];
    for (int index = 2; index < argc; ++index) {
        const char * flag = argv[index];
        if (std::strcmp(flag, "--exact") == 0) {
            options.exact = true;
            continue;
        }
        if (std::strcmp(flag, "--allow-eog") == 0) {
            options.allow_eog = true;
            continue;
        }
        if (std::strcmp(flag, "--chat") == 0) {
            options.use_chat_template = true;
            continue;
        }
        if (index + 1 >= argc) usage(argv[0]);
        const char * value = argv[++index];
        if (std::strcmp(flag, "--prompt") == 0) {
            options.prompt = value;
        } else if (std::strcmp(flag, "--length") == 0) {
            options.length = static_cast<int>(parse_long(value, "length"));
        } else if (std::strcmp(flag, "--top-k") == 0) {
            options.top_k = static_cast<int>(parse_long(value, "top-k"));
        } else if (std::strcmp(flag, "--trace") == 0) {
            options.trace_path = value;
        } else if (std::strcmp(flag, "--gpu-layers") == 0) {
            options.gpu_layers = static_cast<int>(
                parse_long(value, "gpu-layers")
            );
        } else if (std::strcmp(flag, "--threads") == 0) {
            options.threads = static_cast<int>(parse_long(value, "threads"));
        } else if (std::strcmp(flag, "--ctx-size") == 0) {
            options.context_size = static_cast<int>(
                parse_long(value, "ctx-size")
            );
        } else if (std::strcmp(flag, "--reasoning") == 0) {
            if (std::strcmp(value, "on") == 0) {
                options.enable_thinking = true;
            } else if (std::strcmp(value, "off") == 0) {
                options.enable_thinking = false;
            } else {
                fail("reasoning must be 'on' or 'off'");
            }
        } else {
            usage(argv[0]);
        }
    }
    if (options.prompt.empty() || options.length <= 0 ||
        options.top_k <= 0 || !options.exact || options.gpu_layers < 0 ||
        options.threads <= 0 || options.context_size < 0) {
        usage(argv[0]);
    }
    return options;
}

std::string format_chat_prompt(
    const llama_model * model,
    const std::string & user_prompt,
    bool enable_thinking
) {
    common_chat_templates_ptr templates = common_chat_templates_init(model, "");
    if (!templates) fail("model has no usable embedded chat template");
    common_chat_msg message;
    message.role = "user";
    message.content = user_prompt;
    common_chat_templates_inputs inputs;
    inputs.messages.push_back(std::move(message));
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;
    inputs.enable_thinking = enable_thinking;
    common_chat_params rendered = common_chat_templates_apply(
        templates.get(), inputs
    );
    if (rendered.prompt.empty()) fail("chat template rendered an empty prompt");
    return rendered.prompt;
}

void json_string(FILE * stream, const std::string & text) {
    std::fputc('"', stream);
    for (unsigned char byte : text) {
        switch (byte) {
            case '"': std::fputs("\\\"", stream); break;
            case '\\': std::fputs("\\\\", stream); break;
            case '\b': std::fputs("\\b", stream); break;
            case '\f': std::fputs("\\f", stream); break;
            case '\n': std::fputs("\\n", stream); break;
            case '\r': std::fputs("\\r", stream); break;
            case '\t': std::fputs("\\t", stream); break;
            default:
                if (byte < 0x20) std::fprintf(stream, "\\u%04x", byte);
                else std::fputc(byte, stream);
        }
    }
    std::fputc('"', stream);
}

std::vector<llama_token> tokenize(
    const llama_vocab * vocab,
    const std::string & text
) {
    int32_t count = llama_tokenize(
        vocab, text.data(), static_cast<int32_t>(text.size()),
        nullptr, 0, true, true
    );
    if (count == std::numeric_limits<int32_t>::min()) {
        fail("prompt token count overflow");
    }
    if (count > 0) fail("tokenizer unexpectedly accepted an empty buffer");
    std::vector<llama_token> tokens(static_cast<size_t>(-count));
    count = llama_tokenize(
        vocab, text.data(), static_cast<int32_t>(text.size()),
        tokens.data(), static_cast<int32_t>(tokens.size()), true, true
    );
    if (count < 0) fail("could not tokenize prompt");
    tokens.resize(static_cast<size_t>(count));
    return tokens;
}

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    char local[256];
    int32_t count = llama_token_to_piece(
        vocab, token, local, static_cast<int32_t>(sizeof(local)), 0, true
    );
    if (count >= 0) return std::string(local, static_cast<size_t>(count));
    std::string result(static_cast<size_t>(-count), '\0');
    count = llama_token_to_piece(
        vocab, token, result.data(), static_cast<int32_t>(result.size()), 0,
        true
    );
    if (count < 0) fail("could not decode token piece");
    result.resize(static_cast<size_t>(count));
    return result;
}

std::string decode_tokens(
    const llama_vocab * vocab,
    const std::vector<llama_token> & tokens
) {
    std::string result;
    for (llama_token token : tokens) result += token_piece(vocab, token);
    return result;
}

double log_partition(const std::vector<float> & logits) {
    float maximum = -std::numeric_limits<float>::max();
    for (float value : logits) maximum = std::max(maximum, value);
    double total = 0.0;
    for (float value : logits) {
        total += std::exp(static_cast<double>(value) - maximum);
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        fail("invalid vocabulary covector");
    }
    return static_cast<double>(maximum) + std::log(total);
}

struct Trace {
    FILE * stream = nullptr;
    ~Trace() {
        if (stream != nullptr) std::fclose(stream);
    }
    void flush() const {
        if (stream != nullptr) std::fflush(stream);
    }
};

struct Counters {
    uint64_t strength_nodes = 0;
    uint64_t candidate_observations = 0;
    uint64_t continuation_demands = 0;
    uint64_t model_decode_calls = 0;
    uint64_t model_decoded_terms = 0;
    uint64_t observer_decode_calls = 0;
    uint64_t sequence_copies = 0;
};

struct Value {
    llama_token token = LLAMA_TOKEN_NULL;
    int local_rank = 0;
    float proposal_logit = 0.0f;
    double proposal_log_probability = 0.0;
};

struct NodeState {
    int sequence = -1;
    int position = -1;
    std::vector<float> logits;
    std::vector<float> hidden;
};

struct CovectorFrame {
    uint64_t id = 0;
    std::vector<llama_token> support;
    std::vector<double> coordinates;
    int maximum_rank = -1;
};

struct Outcome {
    std::vector<Value> path;
    std::vector<CovectorFrame> observations;
};

struct Branch {
    Value value;
    Outcome outcome;
    CovectorFrame observation;
    bool attains = false;
};

class BatchOwner {
public:
    BatchOwner(int token_capacity, int embedding_width) :
        batch_(llama_batch_init(token_capacity, embedding_width, 1)) { }

    ~BatchOwner() {
        llama_batch_free(batch_);
    }

    BatchOwner(const BatchOwner &) = delete;
    BatchOwner & operator=(const BatchOwner &) = delete;

    llama_batch & reset() {
        batch_.n_tokens = 0;
        return batch_;
    }

private:
    llama_batch batch_{};
};

class SequencePool {
public:
    SequencePool(llama_memory_t memory, int count) : memory_(memory) {
        for (int sequence = count - 1; sequence >= 1; --sequence) {
            available_.push_back(sequence);
        }
    }

    int acquire() {
        if (available_.empty()) fail("recursive term exhausted sequence ids");
        int sequence = available_.back();
        available_.pop_back();
        if (!llama_memory_seq_rm(memory_, sequence, -1, -1)) {
            fail("could not clear acquired sequence");
        }
        return sequence;
    }

    void release(int sequence) {
        if (sequence <= 0) fail("attempted to release reserved sequence");
        if (!llama_memory_seq_rm(memory_, sequence, -1, -1)) {
            fail("could not release sequence");
        }
        available_.push_back(sequence);
    }

private:
    llama_memory_t memory_ = nullptr;
    std::vector<int> available_;
};

class ExactSelection {
public:
    ExactSelection(
        llama_model * model,
        const llama_vocab * vocab,
        const Options & options,
        std::vector<llama_token> prompt_tokens,
        Trace & trace,
        Counters & counters
    ) :
        model_(model),
        vocab_(vocab),
        options_(options),
        prompt_tokens_(std::move(prompt_tokens)),
        trace_(trace),
        counters_(counters),
        vocab_size_(llama_vocab_n_tokens(vocab)),
        embedding_in_(llama_model_n_embd_inp(model)),
        embedding_out_(llama_model_n_embd_out(model)),
        token_batch_(options.top_k, 0),
        observer_batch_(options.top_k, llama_model_n_embd_inp(model)) {
        if (embedding_in_ != embedding_out_) {
            fail("structured hidden feedback requires equal input/output widths");
        }
        if (options_.top_k > vocab_size_) {
            fail("top-k exceeds model vocabulary");
        }

        const size_t sequence_bound =
            static_cast<size_t>(options_.top_k) * options_.length + 3;
        if (sequence_bound > llama_max_parallel_sequences()) {
            fail("top-k and length exceed llama.cpp's sequence-id capacity");
        }
        const int sequence_count = static_cast<int>(sequence_bound);
        int required = static_cast<int>(prompt_tokens_.size()) +
            8 * options_.length + 256;
        int context_size = options_.context_size;
        if (context_size == 0) {
            context_size = std::max(1024, (required + 255) & ~255);
        }
        int training_context = llama_model_n_ctx_train(model_);
        if (context_size > training_context) context_size = training_context;
        if (required > context_size) {
            fail("prompt and recursive summaries exceed configured context");
        }

        llama_context_params params = llama_context_default_params();
        params.n_ctx = static_cast<uint32_t>(context_size);
        params.n_batch = static_cast<uint32_t>(std::max<size_t>(
            512, prompt_tokens_.size()
        ));
        params.n_ubatch = std::min<uint32_t>(params.n_batch, 512);
        params.n_seq_max = sequence_count;
        params.n_threads = options_.threads;
        params.n_threads_batch = options_.threads;
        params.embeddings = true;
        params.kv_unified = true;
        params.no_perf = false;

        llama_context * raw = llama_init_from_model(model_, params);
        if (raw == nullptr) fail("could not create recursive GGUF context");
        context_.reset(raw);
        memory_ = llama_get_memory(context_.get());
        if (memory_ == nullptr) fail("model exposes no sequence memory");
        sequence_pool_ = std::make_unique<SequencePool>(
            memory_, sequence_count
        );
    }

    Outcome run() {
        NodeState prompt = decode_prompt();
        return select(prompt, options_.length, 0);
    }

private:
    struct ContextDeleter {
        void operator()(llama_context * context) const {
            if (context != nullptr) llama_free(context);
        }
    };

    void add_token(
        llama_batch & batch,
        llama_token token,
        llama_pos position,
        llama_seq_id sequence,
        bool output
    ) const {
        int32_t index = batch.n_tokens++;
        batch.token[index] = token;
        batch.pos[index] = position;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = sequence;
        batch.logits[index] = output ? 1 : 0;
    }

    void add_observer_embedding(
        llama_batch & batch,
        const std::vector<float> & embedding,
        llama_pos position,
        llama_seq_id observer_sequence
    ) const {
        if (embedding.size() != static_cast<size_t>(embedding_in_)) {
            fail("observer received hidden state of wrong width");
        }
        int32_t index = batch.n_tokens++;
        std::memcpy(
            batch.embd + static_cast<size_t>(index) * embedding_in_,
            embedding.data(),
            embedding.size() * sizeof(float)
        );
        batch.pos[index] = position;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = observer_sequence;
        batch.logits[index] = 1;
    }

    NodeState copy_output(int output_index, int sequence, int position) {
        float * logits = llama_get_logits_ith(context_.get(), output_index);
        float * hidden = llama_get_embeddings_ith(
            context_.get(), output_index
        );
        if (logits == nullptr || hidden == nullptr) {
            fail("decoder did not expose logits and final hidden state");
        }
        NodeState state;
        state.sequence = sequence;
        state.position = position;
        state.logits.assign(logits, logits + vocab_size_);
        state.hidden.assign(hidden, hidden + embedding_out_);
        return state;
    }

    NodeState decode_prompt() {
        llama_batch batch = llama_batch_init(
            static_cast<int32_t>(prompt_tokens_.size()), 0, 1
        );
        for (size_t index = 0; index < prompt_tokens_.size(); ++index) {
            add_token(
                batch,
                prompt_tokens_[index],
                static_cast<llama_pos>(index),
                0,
                index + 1 == prompt_tokens_.size()
            );
        }
        int result = llama_decode(context_.get(), batch);
        llama_batch_free(batch);
        if (result != 0) {
            fail("prompt decode failed with code " + std::to_string(result));
        }
        counters_.model_decode_calls++;
        counters_.model_decoded_terms += prompt_tokens_.size();
        return copy_output(
            -1, 0, static_cast<int>(prompt_tokens_.size()) - 1
        );
    }

    std::vector<NodeState> decode_sibling_tokens(
        const std::vector<Value> & support,
        int position,
        const std::vector<int> & sequences
    ) {
        if (support.size() != sequences.size()) {
            fail("sibling support and sequence counts differ");
        }
        llama_batch & batch = token_batch_.reset();
        for (size_t index = 0; index < support.size(); ++index) {
            add_token(
                batch,
                support[index].token,
                position,
                sequences[index],
                true
            );
        }
        int result = llama_decode(context_.get(), batch);
        if (result != 0) {
            fail(
                "sibling branch decode failed with code " +
                std::to_string(result)
            );
        }
        counters_.model_decode_calls++;
        counters_.model_decoded_terms += support.size();

        std::vector<NodeState> states;
        states.reserve(support.size());
        for (size_t index = 0; index < support.size(); ++index) {
            states.push_back(copy_output(
                static_cast<int>(index), sequences[index], position
            ));
        }
        return states;
    }

    std::vector<CovectorFrame> observe_bound_continuations(
        const NodeState & history,
        const std::vector<Branch> & branches,
        int remaining
    ) {
        if (branches.empty()) fail("observer received no bound continuations");
        const int suffix_length = remaining - 1;
        std::vector<Value> support;
        support.reserve(branches.size());
        for (const Branch & branch : branches) {
            if (static_cast<int>(branch.outcome.path.size()) != remaining) {
                fail("bound continuation has the wrong horizon");
            }
            support.push_back(branch.value);
        }

        std::vector<CovectorFrame> observations;
        observations.reserve(branches.size());
        for (const Branch & demanded : branches) {
            CovectorFrame frame;
            frame.id = next_observer_frame_++;
            frame.maximum_rank = 0;
            for (const Value & alternative : support) {
                frame.support.push_back(alternative.token);
                frame.coordinates.push_back(
                    alternative.proposal_log_probability
                );
            }

            if (suffix_length > 0) {
                std::vector<int> sequences;
                sequences.reserve(support.size());
                for (size_t alternative = 0;
                     alternative < support.size(); ++alternative) {
                    int sequence = sequence_pool_->acquire();
                    copy_sequence(history.sequence, sequence);
                    sequences.push_back(sequence);
                }
                std::vector<NodeState> states = decode_sibling_tokens(
                    support,
                    history.position + 1,
                    sequences
                );
                counters_.observer_decode_calls++;

                for (int offset = 0; offset < suffix_length; ++offset) {
                    const llama_token suffix_token = demanded.outcome.path[
                        static_cast<size_t>(offset + 1)
                    ].token;
                    for (size_t alternative = 0;
                         alternative < support.size(); ++alternative) {
                        const NodeState & state = states[alternative];
                        frame.coordinates[alternative] +=
                            static_cast<double>(state.logits[
                                static_cast<size_t>(suffix_token)
                            ]) - log_partition(state.logits);
                    }
                    if (offset + 1 == suffix_length) break;

                    llama_batch & batch = token_batch_.reset();
                    for (size_t alternative = 0;
                         alternative < support.size(); ++alternative) {
                        add_token(
                            batch,
                            suffix_token,
                            history.position + 2 + offset,
                            sequences[alternative],
                            true
                        );
                    }
                    int result = llama_decode(context_.get(), batch);
                    if (result != 0) {
                        fail(
                            "posterior suffix decode failed with code " +
                            std::to_string(result)
                        );
                    }
                    counters_.model_decode_calls++;
                    counters_.model_decoded_terms += support.size();
                    counters_.observer_decode_calls++;
                    states.clear();
                    states.reserve(support.size());
                    for (size_t alternative = 0;
                         alternative < support.size(); ++alternative) {
                        states.push_back(copy_output(
                            static_cast<int>(alternative),
                            sequences[alternative],
                            history.position + 2 + offset
                        ));
                    }
                }
                for (int sequence : sequences) {
                    sequence_pool_->release(sequence);
                }
            }

            for (size_t coordinate = 1;
                 coordinate < frame.coordinates.size(); ++coordinate) {
                const size_t incumbent = static_cast<size_t>(
                    frame.maximum_rank
                );
                if (frame.coordinates[coordinate] >
                        frame.coordinates[incumbent]) {
                    frame.maximum_rank = static_cast<int>(coordinate);
                }
            }
            observations.push_back(std::move(frame));
        }
        return observations;
    }

    void copy_sequence(
        int source,
        int destination,
        int begin = -1,
        int end = -1
    ) {
        llama_memory_seq_cp(memory_, source, destination, begin, end);
        counters_.sequence_copies++;
    }

    std::vector<Value> local_support(const NodeState & history) const {
        std::vector<llama_token> tokens;
        tokens.reserve(static_cast<size_t>(vocab_size_));
        for (int token = 0; token < vocab_size_; ++token) {
            llama_token value = static_cast<llama_token>(token);
            if (!options_.allow_eog && llama_vocab_is_eog(vocab_, value)) {
                continue;
            }
            tokens.push_back(value);
        }
        if (tokens.size() < static_cast<size_t>(options_.top_k)) {
            fail("not enough selectable vocabulary coordinates");
        }
        std::partial_sort(
            tokens.begin(), tokens.begin() + options_.top_k, tokens.end(),
            [&](llama_token left, llama_token right) {
                float a = history.logits[static_cast<size_t>(left)];
                float b = history.logits[static_cast<size_t>(right)];
                return a != b ? a > b : left < right;
            }
        );
        tokens.resize(static_cast<size_t>(options_.top_k));
        double partition = log_partition(history.logits);
        std::vector<Value> values;
        values.reserve(tokens.size());
        for (size_t index = 0; index < tokens.size(); ++index) {
            llama_token token = tokens[index];
            float logit = history.logits[static_cast<size_t>(token)];
            values.push_back(Value{
                token,
                static_cast<int>(index + 1),
                logit,
                static_cast<double>(logit) - partition,
            });
        }
        return values;
    }

    std::vector<llama_token> path_tokens(const Outcome & outcome) const {
        std::vector<llama_token> tokens;
        tokens.reserve(outcome.path.size());
        for (const Value & value : outcome.path) tokens.push_back(value.token);
        return tokens;
    }

    std::string path_text(const Outcome & outcome) const {
        std::string completion = decode_tokens(vocab_, path_tokens(outcome));
        if (options_.use_chat_template) return completion;
        return options_.prompt + completion;
    }

    void trace_demand(
        uint64_t frame,
        int depth,
        int remaining,
        int ordinal,
        const Value & value
    ) {
        if (trace_.stream == nullptr) return;
        std::fprintf(
            trace_.stream,
            "{\"event\":\"continuation_demand\",\"frame\":%llu"
            ",\"depth\":%d,\"remaining\":%d,\"ordinal\":%d"
            ",\"token\":%d,\"local_rank\":%d,\"piece\":",
            static_cast<unsigned long long>(frame),
            depth,
            remaining,
            ordinal,
            value.token,
            value.local_rank
        );
        json_string(trace_.stream, token_piece(vocab_, value.token));
        std::fputs("}\n", trace_.stream);
        trace_.flush();
    }

    void trace_candidate(
        uint64_t frame,
        int depth,
        int remaining,
        const Branch & branch
    ) {
        if (trace_.stream == nullptr) return;
        std::fprintf(
            trace_.stream,
            "{\"event\":\"candidate\",\"frame\":%llu"
            ",\"depth\":%d,\"remaining\":%d,\"token\":%d"
            ",\"local_rank\":%d,\"logit\":%.9g"
            ",\"proposal_log_probability\":%.17g"
            ",\"observer_frame\":%llu,\"attains\":%s"
            ",\"observer_max_rank\":%d,\"observer_max_token\":%d"
            ",\"candidate_covector_logit\":%.9g,\"text\":",
            static_cast<unsigned long long>(frame),
            depth,
            remaining,
            branch.value.token,
            branch.value.local_rank,
            branch.value.proposal_logit,
            branch.value.proposal_log_probability,
            static_cast<unsigned long long>(branch.observation.id),
            branch.attains ? "true" : "false",
            branch.observation.maximum_rank + 1,
            branch.observation.support[static_cast<size_t>(
                branch.observation.maximum_rank
            )],
            branch.observation.coordinates[static_cast<size_t>(
                branch.value.local_rank - 1
            )]
        );
        json_string(trace_.stream, path_text(branch.outcome));
        std::fputs(",\"observer_support\":[", trace_.stream);
        for (size_t index = 0;
             index < branch.observation.support.size(); ++index) {
            if (index != 0) std::fputc(',', trace_.stream);
            std::fprintf(
                trace_.stream,
                "{\"rank\":%zu,\"token\":%d,\"piece\":",
                index + 1,
                branch.observation.support[index]
            );
            json_string(
                trace_.stream,
                token_piece(vocab_, branch.observation.support[index])
            );
            std::fprintf(
                trace_.stream,
                ",\"logit\":%.9g}",
                branch.observation.coordinates[index]
            );
        }
        std::fputs("]}\n", trace_.stream);
        trace_.flush();
    }

    void trace_select(
        uint64_t frame,
        int depth,
        int remaining,
        const Branch & branch
    ) {
        if (trace_.stream == nullptr) return;
        std::fprintf(
            trace_.stream,
            "{\"event\":\"select\",\"frame\":%llu"
            ",\"depth\":%d,\"remaining\":%d,\"token\":%d"
            ",\"local_rank\":%d,\"observer_frame\":%llu"
            ",\"attains\":%s"
            ",\"selection_rule\":\"first_attaining_else_last\""
            ",\"propagated\":\"complete_covector_family\""
            ",\"text\":",
            static_cast<unsigned long long>(frame),
            depth,
            remaining,
            branch.value.token,
            branch.value.local_rank,
            static_cast<unsigned long long>(branch.observation.id),
            branch.attains ? "true" : "false"
        );
        json_string(trace_.stream, path_text(branch.outcome));
        std::fputs("}\n", trace_.stream);
        trace_.flush();
    }

    Outcome select(
        const NodeState & history,
        int remaining,
        int depth
    ) {
        if (remaining <= 0) fail("selection received empty horizon");
        const uint64_t frame = next_frame_++;
        counters_.strength_nodes++;
        std::vector<Value> support = local_support(history);
        std::vector<Branch> branches;
        branches.reserve(support.size());

        for (size_t index = 0; index < support.size(); ++index) {
            trace_demand(
                frame,
                depth,
                remaining,
                static_cast<int>(index),
                support[index]
            );
            counters_.continuation_demands++;
        }

        std::vector<int> child_sequences;
        child_sequences.reserve(support.size());
        for (size_t index = 0; index < support.size(); ++index) {
            int child_sequence = sequence_pool_->acquire();
            copy_sequence(history.sequence, child_sequence);
            child_sequences.push_back(child_sequence);
        }
        std::vector<NodeState> children = decode_sibling_tokens(
            support, history.position + 1, child_sequences
        );

        for (size_t index = 0; index < support.size(); ++index) {
            const Value value = support[index];
            NodeState & child = children[index];
            Outcome outcome;
            if (remaining > 1) {
                outcome = select(child, remaining - 1, depth + 1);
            }
            if (child.sequence >= 0) {
                sequence_pool_->release(child.sequence);
                child.sequence = -1;
            }
            outcome.path.insert(outcome.path.begin(), value);
            Branch branch;
            branch.value = value;
            branch.outcome = std::move(outcome);
            branches.push_back(std::move(branch));
        }

        std::vector<CovectorFrame> observations = observe_bound_continuations(
            history,
            branches,
            remaining
        );

        size_t best = branches.size() - 1;
        bool found_attaining = false;
        for (size_t index = 0; index < branches.size(); ++index) {
            Branch & branch = branches[index];
            branch.observation = std::move(observations[index]);
            branch.attains = branch.observation.maximum_rank ==
                branch.value.local_rank - 1;
            branch.outcome.observations.insert(
                branch.outcome.observations.begin(),
                branch.observation
            );
            counters_.candidate_observations++;
            trace_candidate(frame, depth, remaining, branch);
            if (!found_attaining && branch.attains) {
                best = index;
                found_attaining = true;
            }
        }
        trace_select(frame, depth, remaining, branches[best]);
        return std::move(branches[best].outcome);
    }

    llama_model * model_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    const Options & options_;
    std::vector<llama_token> prompt_tokens_;
    Trace & trace_;
    Counters & counters_;
    int vocab_size_ = 0;
    int embedding_in_ = 0;
    int embedding_out_ = 0;
    BatchOwner token_batch_;
    BatchOwner observer_batch_;
    std::unique_ptr<llama_context, ContextDeleter> context_;
    llama_memory_t memory_ = nullptr;
    std::unique_ptr<SequencePool> sequence_pool_;
    uint64_t next_frame_ = 0;
    uint64_t next_observer_frame_ = 0;
};

void quiet_log(enum ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        std::fputs(text, stderr);
    }
}

} // namespace

int main(int argc, char ** argv) {
    try {
        Options options = parse_options(argc, argv);
        llama_log_set(quiet_log, nullptr);
        ggml_backend_load_all();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = options.gpu_layers;
        llama_model * raw_model = llama_model_load_from_file(
            options.model_path.c_str(), model_params
        );
        if (raw_model == nullptr) fail("could not load GGUF model");
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            raw_model, llama_model_free
        );

        const std::string user_prompt = options.prompt;
        std::string formatted_prompt = options.prompt;
        if (options.use_chat_template) {
            formatted_prompt = format_chat_prompt(
                model.get(), user_prompt, options.enable_thinking
            );
        }
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        std::vector<llama_token> prompt_tokens = tokenize(
            vocab, formatted_prompt
        );
        if (prompt_tokens.empty()) fail("prompt encoded to no tokens");
        if (prompt_tokens.size() + static_cast<size_t>(options.length) >
                static_cast<size_t>(llama_model_n_ctx_train(model.get()))) {
            fail("prompt and completion exceed model training context");
        }

        Trace trace;
        if (!options.trace_path.empty()) {
            trace.stream = std::fopen(options.trace_path.c_str(), "w");
            if (trace.stream == nullptr) fail("could not open trace output");
            std::fputs("{\"event\":\"run\",\"user_prompt\":", trace.stream);
            json_string(trace.stream, user_prompt);
            std::fputs(",\"formatted_prompt\":", trace.stream);
            json_string(trace.stream, formatted_prompt);
            std::fprintf(
                trace.stream,
                ",\"chat_template\":%s,\"enable_thinking\":%s"
                ",\"length\":%d,\"proposal_top_k\":%d"
                ",\"mode\":\"exact_select_product_masked_company\""
                ",\"observer_attention\":"
                "\"causal_posterior_over_fixed_bound_suffix\""
                ",\"selection_rule\":\"first_attaining_else_last\"}\n",
                options.use_chat_template ? "true" : "false",
                options.enable_thinking ? "true" : "false",
                options.length,
                options.top_k
            );
            trace.flush();
        }

        Counters counters;
        auto started = Clock::now();
        ExactSelection term(
            model.get(),
            vocab,
            options,
            prompt_tokens,
            trace,
            counters
        );
        Outcome selected = term.run();
        auto stopped = Clock::now();
        double seconds = std::chrono::duration<double>(
            stopped - started
        ).count();

        std::vector<llama_token> selected_tokens;
        selected_tokens.reserve(selected.path.size());
        for (const Value & value : selected.path) {
            selected_tokens.push_back(value.token);
        }
        std::string completion = decode_tokens(vocab, selected_tokens);
        std::string displayed = options.use_chat_template
            ? completion
            : user_prompt + completion;

        std::puts("completion:");
        std::puts(displayed.c_str());
        std::printf(
            "score_kind=bound_continuation_covector_attainment\n"
            "selection_carrier=token_path_with_covector_family\n"
            "selection_observer=causal_posterior_root_callback\n"
            "observer_attention="
            "causal_posterior_over_fixed_bound_suffix\n"
            "aggregate_path_score=none\n"
            "root_terminalizations=1\n"
            "strength_nodes=%llu\n"
            "candidate_observations=%llu\n"
            "continuation_demands=%llu\n"
            "model_decode_calls=%llu\n"
            "model_decoded_terms=%llu\n"
            "observer_decode_calls=%llu\n"
            "sequence_copies=%llu\n"
            "numerical_backend=llama.cpp/ggml\n"
            "chat_template=%s\n"
            "enable_thinking=%s\n"
            "search_seconds=%.6f\n",
            static_cast<unsigned long long>(counters.strength_nodes),
            static_cast<unsigned long long>(counters.candidate_observations),
            static_cast<unsigned long long>(counters.continuation_demands),
            static_cast<unsigned long long>(counters.model_decode_calls),
            static_cast<unsigned long long>(counters.model_decoded_terms),
            static_cast<unsigned long long>(counters.observer_decode_calls),
            static_cast<unsigned long long>(counters.sequence_copies),
            options.use_chat_template ? "embedded_jinja" : "raw",
            options.enable_thinking ? "true" : "false",
            seconds
        );

        if (trace.stream != nullptr) {
            std::fputs(
                "{\"event\":\"root_terminalization\",\"text\":",
                trace.stream
            );
            json_string(trace.stream, completion);
            std::fprintf(
                trace.stream,
                ",\"strength_nodes\":%llu"
                ",\"candidate_observations\":%llu}\n",
                static_cast<unsigned long long>(counters.strength_nodes),
                static_cast<unsigned long long>(
                    counters.candidate_observations
                )
            );
            trace.flush();
        }
        return EXIT_SUCCESS;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "escardo-gguf: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
