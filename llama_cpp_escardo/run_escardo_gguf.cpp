/*
 * Exact finite open-transformer term over llama.cpp.
 *
 * A selection outcome is not a scalar reward. It is a token path together
 * with one KV summary cell and the final hidden state produced by its
 * recursively composed continuation. At a Select, every demanded child is
 * forced first. Their summary cells are then installed in one temporary
 * sequence and observed by one common hidden-state query. The model produces
 * one vocabulary covector, and siblings are compared only at their token
 * coordinates in that covector. The selected token path and the observer's
 * own KV/hidden summary are propagated to the parent. Only the root emits.
 *
 * This deliberately contains no sampled AR path, path likelihood sum,
 * scalar backup, UCB bonus, or wall-clock search loop. The local carrier is
 * an explicit top-k support, and Escardo's product is evaluated exactly over
 * that finite carrier. Performance scheduling is a separate concern.
 *
 * llama.cpp requires a new causal input to follow the maximum cached
 * position. Child summaries therefore share the span endpoint, the observer
 * query sits immediately after that endpoint, and its KV cell is shifted back
 * to the endpoint before propagation. Native self-attention also includes the
 * observer's own cell. The run trace names both backend differences from the
 * manually assembled llama2.c observer explicitly.
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

struct Outcome {
    std::vector<Value> path;
    int summary_sequence = -1;
    int summary_position = -1;
    std::vector<float> final_hidden;
};

struct Branch {
    Value value;
    Outcome outcome;
    float shared_covector_logit = 0.0f;
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
        embedding_out_(llama_model_n_embd_out(model)) {
        if (embedding_in_ != embedding_out_) {
            fail("structured hidden feedback requires equal input/output widths");
        }
        if (options_.top_k > vocab_size_) {
            fail("top-k exceeds model vocabulary");
        }

        constexpr int sequence_count = 256;
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
        Outcome result = select(prompt, options_.length, 0);
        sequence_pool_->release(result.summary_sequence);
        result.summary_sequence = -1;
        return result;
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

    NodeState copy_last_output(int sequence, int position) {
        float * logits = llama_get_logits_ith(context_.get(), -1);
        float * hidden = llama_get_embeddings_ith(context_.get(), -1);
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
        return copy_last_output(
            0, static_cast<int>(prompt_tokens_.size()) - 1
        );
    }

    NodeState decode_token(
        llama_token token,
        int position,
        int sequence
    ) {
        llama_set_causal_attn(context_.get(), true);
        llama_batch batch = llama_batch_init(1, 0, 1);
        add_token(batch, token, position, sequence, true);
        int result = llama_decode(context_.get(), batch);
        llama_batch_free(batch);
        if (result != 0) {
            fail("branch decode failed with code " + std::to_string(result));
        }
        counters_.model_decode_calls++;
        counters_.model_decoded_terms++;
        return copy_last_output(sequence, position);
    }

    NodeState decode_observer(
        const std::vector<float> & hidden,
        int position,
        int observer_sequence
    ) {
        llama_set_causal_attn(context_.get(), true);
        llama_batch batch = llama_batch_init(1, embedding_in_, 1);
        add_observer_embedding(
            batch, hidden, position, observer_sequence
        );
        int result = llama_decode(context_.get(), batch);
        llama_batch_free(batch);
        if (result != 0) {
            fail("selection observer decode failed with code " +
                std::to_string(result));
        }
        counters_.model_decode_calls++;
        counters_.model_decoded_terms++;
        counters_.observer_decode_calls++;
        return copy_last_output(observer_sequence, position);
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
            ",\"shared_covector_logit\":%.9g,\"text\":",
            static_cast<unsigned long long>(frame),
            depth,
            remaining,
            branch.value.token,
            branch.value.local_rank,
            branch.value.proposal_logit,
            branch.value.proposal_log_probability,
            branch.shared_covector_logit
        );
        json_string(trace_.stream, path_text(branch.outcome));
        std::fputs("}\n", trace_.stream);
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
            ",\"local_rank\":%d,\"shared_covector_logit\":%.9g"
            ",\"propagated\":\"structured_kv_hidden_outcome\""
            ",\"text\":",
            static_cast<unsigned long long>(frame),
            depth,
            remaining,
            branch.value.token,
            branch.value.local_rank,
            branch.shared_covector_logit
        );
        json_string(trace_.stream, path_text(branch.outcome));
        std::fputs("}\n", trace_.stream);
        trace_.flush();
    }

    Outcome leaf_outcome(const NodeState & child) {
        int summary = sequence_pool_->acquire();
        copy_sequence(
            child.sequence,
            summary,
            child.position,
            child.position + 1
        );
        Outcome outcome;
        outcome.summary_sequence = summary;
        outcome.summary_position = child.position;
        outcome.final_hidden = child.hidden;
        return outcome;
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
            const Value value = support[index];
            trace_demand(
                frame, depth, remaining, static_cast<int>(index), value
            );
            counters_.continuation_demands++;

            int child_sequence = sequence_pool_->acquire();
            copy_sequence(history.sequence, child_sequence);
            NodeState child = decode_token(
                value.token, history.position + 1, child_sequence
            );
            Outcome outcome = remaining == 1
                ? leaf_outcome(child)
                : select(child, remaining - 1, depth + 1);
            sequence_pool_->release(child_sequence);
            outcome.path.insert(outcome.path.begin(), value);
            branches.push_back(Branch{value, std::move(outcome), 0.0f});
        }

        int observer_sequence = sequence_pool_->acquire();
        copy_sequence(history.sequence, observer_sequence);
        for (const Branch & branch : branches) {
            copy_sequence(
                branch.outcome.summary_sequence,
                observer_sequence,
                branch.outcome.summary_position,
                branch.outcome.summary_position + 1
            );
        }

        const int summary_position = history.position + remaining;
        const int query_position = summary_position + 1;
        NodeState observation = decode_observer(
            history.hidden,
            query_position,
            observer_sequence
        );

        size_t best = 0;
        for (size_t index = 0; index < branches.size(); ++index) {
            Branch & branch = branches[index];
            branch.shared_covector_logit = observation.logits[
                static_cast<size_t>(branch.value.token)
            ];
            counters_.candidate_observations++;
            trace_candidate(frame, depth, remaining, branch);
            if (index == 0) continue;
            const Branch & winner = branches[best];
            if (branch.shared_covector_logit >
                    winner.shared_covector_logit ||
                (branch.shared_covector_logit ==
                    winner.shared_covector_logit &&
                 branch.value.local_rank < winner.value.local_rank)) {
                best = index;
            }
        }
        trace_select(frame, depth, remaining, branches[best]);

        int summary_sequence = sequence_pool_->acquire();
        copy_sequence(
            observer_sequence,
            summary_sequence,
            query_position,
            query_position + 1
        );
        const int shift = summary_position - query_position;
        if (shift != 0) {
            if (!llama_memory_can_shift(memory_)) {
                fail("model KV memory cannot move a composed summary to its span");
            }
            llama_memory_seq_add(
                memory_, summary_sequence,
                query_position, query_position + 1, shift
            );
        }

        sequence_pool_->release(observer_sequence);
        for (Branch & branch : branches) {
            sequence_pool_->release(branch.outcome.summary_sequence);
            branch.outcome.summary_sequence = -1;
        }

        Outcome result;
        result.path = std::move(branches[best].outcome.path);
        result.summary_sequence = summary_sequence;
        result.summary_position = summary_position;
        result.final_hidden = std::move(observation.hidden);
        return result;
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
    std::unique_ptr<llama_context, ContextDeleter> context_;
    llama_memory_t memory_ = nullptr;
    std::unique_ptr<SequencePool> sequence_pool_;
    uint64_t next_frame_ = 0;
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
                ",\"mode\":\"exact_structured_kv_shared_covector\""
                ",\"observer_attention\":"
                "\"causal_prefix_children_at_span_endpoint_then_self\"}\n",
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
            "score_kind=shared_model_covector_per_select\n"
            "selection_carrier=structured_kv_hidden_outcome\n"
            "selection_observer=recursive_candidate_axis_attention\n"
            "observer_attention="
            "causal_prefix_children_at_span_endpoint_then_self\n"
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
