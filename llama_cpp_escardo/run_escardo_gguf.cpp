/*
 * Exact finite product of model-backed selection functions over llama.cpp.
 *
 * There is one observer operation. For a recursively completed branch
 * x : b(x), it teacher-forces the whole counterfactual and scans the frozen
 * model's native vocabulary covector at every causal boundary:
 *
 *   A_i(x : b(x)) = (sum_{j=i}^{N} z_j, N - i + 1)
 *   q_i(x : b(x)) = softmax(A_i.sum / A_i.mass).
 *
 * The carrier (sum,mass) merges componentwise and is therefore associative;
 * softmax turns the mean logits into the normalized geometric mean of the
 * boundary distributions. A local selector evaluates its constructor x in
 * q_i. No selected-token conditional probabilities are added into a path
 * score, and every scale retains its own covector in the result tuple.
 *
 * Escardo's dependent product is evaluated literally.  For every x, first
 * obtain b(x) by recursively applying the suffix selection.  Then apply the
 * same observer to (prefix, b(x)). A branch attains when x is the maximum
 * coordinate of its own scanned covector over the common finite support.
 * This fixed-point property is retained as an ambiguity diagnostic. Selection
 * itself is typed evaluation: maximize
 * log_softmax(boundary_scan(p(x)))[x]. Raw coordinates from distinct frames
 * are never compared or added. The selected outcome retains the complete
 * position-indexed covector family, and only the root emits its token tuple.
 *
 * This deliberately contains no path-likelihood sum, scalar backup, UCB
 * bonus, or sampled AR rollout.  Model applications at a causal frontier and
 * bound-continuation observations are batched; batching changes numerical
 * scheduling, not the selection-product law.
 */

#include "llama.h"
#include "chat.h"
#include "escardo_product_runtime.h"
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
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
namespace product = escardo_product;

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
    for (size_t index = 0; index < text.size();) {
        const unsigned char byte = static_cast<unsigned char>(text[index]);
        size_t utf8_length = 0;
        if (byte >= 0xc2 && byte <= 0xdf && index + 1 < text.size()) {
            const unsigned char b1 = static_cast<unsigned char>(text[index + 1]);
            if (b1 >= 0x80 && b1 <= 0xbf) utf8_length = 2;
        } else if (byte >= 0xe0 && byte <= 0xef &&
                   index + 2 < text.size()) {
            const unsigned char b1 = static_cast<unsigned char>(text[index + 1]);
            const unsigned char b2 = static_cast<unsigned char>(text[index + 2]);
            const bool first_valid =
                (byte == 0xe0 && b1 >= 0xa0 && b1 <= 0xbf) ||
                (byte == 0xed && b1 >= 0x80 && b1 <= 0x9f) ||
                ((byte >= 0xe1 && byte <= 0xec) &&
                 b1 >= 0x80 && b1 <= 0xbf) ||
                ((byte >= 0xee && byte <= 0xef) &&
                 b1 >= 0x80 && b1 <= 0xbf);
            if (first_valid && b2 >= 0x80 && b2 <= 0xbf) utf8_length = 3;
        } else if (byte >= 0xf0 && byte <= 0xf4 &&
                   index + 3 < text.size()) {
            const unsigned char b1 = static_cast<unsigned char>(text[index + 1]);
            const unsigned char b2 = static_cast<unsigned char>(text[index + 2]);
            const unsigned char b3 = static_cast<unsigned char>(text[index + 3]);
            const bool first_valid =
                (byte == 0xf0 && b1 >= 0x90 && b1 <= 0xbf) ||
                (byte == 0xf4 && b1 >= 0x80 && b1 <= 0x8f) ||
                ((byte >= 0xf1 && byte <= 0xf3) &&
                 b1 >= 0x80 && b1 <= 0xbf);
            if (first_valid && b2 >= 0x80 && b2 <= 0xbf &&
                b3 >= 0x80 && b3 <= 0xbf) {
                utf8_length = 4;
            }
        }
        if (utf8_length != 0) {
            std::fwrite(text.data() + index, 1, utf8_length, stream);
            index += utf8_length;
            continue;
        }
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
                else if (byte < 0x80) std::fputc(byte, stream);
                else std::fprintf(stream, "\\u00%02x", byte);
        }
        index++;
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

double log_partition(const std::vector<double> & logits) {
    double maximum = -std::numeric_limits<double>::max();
    for (double value : logits) maximum = std::max(maximum, value);
    double total = 0.0;
    for (double value : logits) total += std::exp(value - maximum);
    if (!(total > 0.0) || !std::isfinite(total)) {
        fail("invalid affine boundary covector");
    }
    return maximum + std::log(total);
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
    uint64_t attaining_alternatives = 0;
    uint64_t ambiguous_selection_nodes = 0;
    uint64_t zero_attaining_selection_nodes = 0;
};

struct NodeState {
    uint64_t id = 0;
    int sequence = -1;
    int position = -1;
    std::vector<float> logits;
    std::vector<float> hidden;
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
        available_.reserve(static_cast<size_t>(std::max(0, count - 1)));
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

int boundary_scan_batch_capacity(const Options & options) {
    return options.top_k;
}

class ExactSelection :
    public product::SelectionTerm,
    public product::RootObserver,
    public product::ProductEventSink {
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
        embedding_out_(llama_model_n_embd_out(model)),
        token_batch_(boundary_scan_batch_capacity(options), 0) {
        if (options_.top_k > vocab_size_) {
            fail("top-k exceeds model vocabulary");
        }

        const size_t k = static_cast<size_t>(options_.top_k);
        const size_t sequence_bound = k * options_.length + k + 3;
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
        params.n_batch = static_cast<uint32_t>(std::max<size_t>({
            512, prompt_tokens_.size(), k
        }));
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

    product::ProductResult run() {
        NodeState prompt = decode_prompt();
        register_state(prompt);
        product::ExactProduct composed(
            static_cast<size_t>(options_.length), *this, *this, this
        );
        try {
            product::ProductResult result = composed.run(state_ref(prompt));
            ensure_cleanup_ok();
            if (!active_frontiers_.empty() || states_.size() != 1 ||
                states_.find(prompt.id) == states_.end()) {
                fail("dynamic product leaked a model frontier");
            }
            counters_.strength_nodes = composed.counters().selection_nodes;
            counters_.candidate_observations =
                composed.counters().observer_demands;
            counters_.continuation_demands =
                composed.counters().observer_demands;
            counters_.attaining_alternatives =
                composed.counters().attaining_alternatives;
            counters_.ambiguous_selection_nodes =
                composed.counters().ambiguous_selection_nodes;
            counters_.zero_attaining_selection_nodes =
                composed.counters().zero_attaining_selection_nodes;
            unregister_state(prompt);
            return result;
        } catch (...) {
            auto found = states_.find(prompt.id);
            if (found != states_.end() && found->second == &prompt) {
                states_.erase(found);
            }
            throw;
        }
    }

private:
    struct ContextDeleter {
        void operator()(llama_context * context) const {
            if (context != nullptr) llama_free(context);
        }
    };

    struct ActiveFrontier {
        std::vector<int> sequences;
        std::vector<NodeState> children;
        size_t registered_count = 0;
    };

    void ensure_cleanup_ok() const {
        if (cleanup_failed_) fail("could not release a model frontier");
    }

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

    NodeState copy_output(int output_index, int sequence, int position) {
        float * logits = llama_get_logits_ith(context_.get(), output_index);
        float * hidden = llama_get_embeddings_ith(
            context_.get(), output_index
        );
        if (logits == nullptr || hidden == nullptr) {
            fail("decoder did not expose logits and final hidden state");
        }
        NodeState state;
        state.id = next_state_id_++;
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
        const std::vector<llama_token> & tokens,
        int position,
        const std::vector<int> & sequences
    ) {
        if (tokens.size() != sequences.size()) {
            fail("sibling support and sequence counts differ");
        }
        llama_batch & batch = token_batch_.reset();
        for (size_t index = 0; index < tokens.size(); ++index) {
            add_token(
                batch,
                tokens[index],
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
        counters_.model_decoded_terms += tokens.size();

        std::vector<NodeState> states;
        states.reserve(tokens.size());
        for (size_t index = 0; index < tokens.size(); ++index) {
            states.push_back(copy_output(
                static_cast<int>(index), sequences[index], position
            ));
        }
        return states;
    }

    product::StructuredOutcomeRef state_ref(NodeState & state) const {
        if (state.id == 0 || state.sequence < 0 || state.position < 0 ||
            state.logits.size() != static_cast<size_t>(vocab_size_) ||
            state.hidden.size() != static_cast<size_t>(embedding_out_)) {
            fail("cannot bind an incomplete model state");
        }
        product::StructuredOutcomeRef result;
        result.kv_summary.handle = state.id;
        result.kv_summary.position = state.position;
        result.final_hidden.handle = state.id;
        result.final_hidden.data = state.hidden.data();
        result.final_hidden.width = state.hidden.size();
        result.proposal.handle = state.id;
        result.proposal.data = state.logits.data();
        result.proposal.width = state.logits.size();
        return result;
    }

    void register_state(NodeState & state) {
        if (state.id == 0 || !states_.emplace(state.id, &state).second) {
            fail("model state id is zero or already registered");
        }
    }

    void unregister_state(NodeState & state) {
        auto found = states_.find(state.id);
        if (found == states_.end() || found->second != &state) {
            fail("attempted to release an unregistered model state");
        }
        states_.erase(found);
    }

    NodeState & state_from_ref(
        const product::StructuredOutcomeRef & reference
    ) const {
        const uint64_t id = reference.kv_summary.handle;
        if (id == 0 || reference.final_hidden.handle != id ||
            reference.proposal.handle != id) {
            fail("structured model-state handles disagree");
        }
        auto found = states_.find(id);
        if (found == states_.end()) fail("model state is no longer live");
        NodeState & state = *found->second;
        if (reference.kv_summary.position != state.position ||
            reference.final_hidden.data != state.hidden.data() ||
            reference.final_hidden.width != state.hidden.size() ||
            reference.proposal.data != state.logits.data() ||
            reference.proposal.width != state.logits.size()) {
            fail("structured model-state reference is stale");
        }
        return state;
    }

    product::ObserverId observer_id() const override {
        return 1;
    }

    product::ObservationBatchResult observe(
        const product::ObservationBatch & batch
    ) override {
        product::BoundaryScanSchedule schedule =
            product::make_boundary_scan_schedule(batch);
        product::ObservationBatchResult result;
        result.selection_id = batch.selection_id;
        result.observations.reserve(batch.demands.size());

        std::unordered_map<product::DemandId, size_t> demand_rows;
        demand_rows.reserve(batch.demands.size());
        for (const product::ObservationDemand & demand : batch.demands) {
            product::DemandObservation observation;
            observation.demand_id = demand.demand_id;
            observation.covector.position = batch.selecting_position;
            observation.covector.frame = product::ObservationFrame{
                observer_id(), next_observer_frame_++
            };
            if (!demand_rows.emplace(
                    demand.demand_id, result.observations.size()
                ).second) {
                fail("observer received duplicate demand ids");
            }
            result.observations.push_back(std::move(observation));
        }

        const size_t lane_count = schedule.lanes.size();
        if (lane_count != batch.demands.size()) {
            fail("boundary scan schedule lost an observer demand");
        }
        const size_t path_length =
            schedule.lanes.front().candidate_then_suffix.size();
        if (path_length == 0) fail("boundary scan lane has an empty path");
        const size_t boundary_count = path_length + 1;

        std::vector<int> sequences;
        std::vector<llama_token> tokens;
        std::vector<std::vector<double>> covector_sums(
            lane_count,
            std::vector<double>(static_cast<size_t>(vocab_size_), 0.0)
        );
        sequences.reserve(lane_count);
        tokens.reserve(lane_count);
        int model_position = -1;
        try {
            for (size_t lane_index = 0;
                 lane_index < lane_count; ++lane_index) {
                const product::BoundaryScanLane & lane =
                    schedule.lanes[lane_index];
                if (lane.candidate_then_suffix.size() != path_length) {
                    fail("boundary scan lanes disagree on path horizon");
                }
                NodeState & history = state_from_ref(
                    lane.history_before_candidate
                );
                if (model_position < 0) model_position = history.position + 1;
                if (model_position != history.position + 1) {
                    fail("boundary scan lanes disagree on causal position");
                }
                for (int token = 0; token < vocab_size_; ++token) {
                    covector_sums[lane_index][static_cast<size_t>(token)] =
                        static_cast<double>(history.logits[
                            static_cast<size_t>(token)
                        ]);
                }
                int sequence = sequence_pool_->acquire();
                sequences.push_back(sequence);
                copy_sequence(history.sequence, sequence);
                tokens.push_back(lane.candidate_then_suffix.front());
            }

            std::vector<NodeState> states = decode_sibling_tokens(
                tokens, model_position, sequences
            );
            counters_.observer_decode_calls++;
            for (size_t lane_index = 0;
                 lane_index < lane_count; ++lane_index) {
                for (int token = 0; token < vocab_size_; ++token) {
                    covector_sums[lane_index][static_cast<size_t>(token)] +=
                        static_cast<double>(states[lane_index].logits[
                            static_cast<size_t>(token)
                        ]);
                }
            }
            for (size_t offset = 1; offset < path_length; ++offset) {
                for (size_t lane_index = 0;
                     lane_index < lane_count; ++lane_index) {
                    tokens[lane_index] = schedule.lanes[
                        lane_index
                    ].candidate_then_suffix[offset];
                }
                states = decode_sibling_tokens(
                    tokens,
                    model_position + static_cast<int>(offset),
                    sequences
                );
                counters_.observer_decode_calls++;
                for (size_t lane_index = 0;
                     lane_index < lane_count; ++lane_index) {
                    for (int token = 0; token < vocab_size_; ++token) {
                        covector_sums[lane_index][
                            static_cast<size_t>(token)
                        ] += static_cast<double>(states[lane_index].logits[
                            static_cast<size_t>(token)
                        ]);
                    }
                }
            }

            for (size_t lane_index = 0;
                 lane_index < lane_count; ++lane_index) {
                const product::BoundaryScanLane & lane =
                    schedule.lanes[lane_index];
                const size_t row = demand_rows.at(lane.demand_id);
                const product::ObservationDemand & demand = batch.demands[row];
                product::PositionCovector & frame =
                    result.observations[row].covector;
                std::vector<double> & scan = covector_sums[lane_index];
                const double inverse_mass =
                    1.0 / static_cast<double>(boundary_count);
                for (double & coordinate : scan) {
                    coordinate *= inverse_mass;
                }
                frame.boundary_count = boundary_count;
                frame.vocabulary_log_partition = log_partition(scan);
                frame.coordinates.reserve(demand.common_support.size());
                for (product::Token token : demand.common_support) {
                    product::FramedCoordinate coordinate;
                    coordinate.token = token;
                    coordinate.frame = frame.frame;
                    coordinate.value = scan[static_cast<size_t>(token)];
                    frame.coordinates.push_back(coordinate);
                }
            }
        } catch (...) {
            for (int sequence : sequences) {
                sequence_pool_->release(sequence);
            }
            throw;
        }
        for (int sequence : sequences) sequence_pool_->release(sequence);
        return result;
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

    std::vector<product::BoundContinuation> local_support(
        const NodeState & history,
        size_t position
    ) {
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
        std::vector<product::BoundContinuation> values;
        values.reserve(tokens.size());
        for (size_t index = 0; index < tokens.size(); ++index) {
            llama_token token = tokens[index];
            float logit = history.logits[static_cast<size_t>(token)];
            product::BoundContinuation binding;
            binding.binding_id = next_binding_id_++;
            binding.token = token;
            binding.local_rank = static_cast<int32_t>(index + 1);
            binding.position = position;
            binding.proposal_logit = logit;
            binding.proposal_log_probability =
                static_cast<double>(logit) - partition;
            values.push_back(binding);
        }
        return values;
    }

    product::ProductNode demand(
        const product::StructuredOutcomeRef & history_reference,
        size_t position
    ) override {
        ensure_cleanup_ok();
        NodeState & history = state_from_ref(history_reference);
        product::ProductNode node;
        node.node_id = next_node_id_++;
        node.position = position;
        node.alternatives = local_support(history, position);
        if (!active_frontiers_.emplace(
                node.node_id, ActiveFrontier{}
            ).second) {
            fail("dynamic product repeated a node id");
        }
        return node;
    }

    void force(
        const product::StructuredOutcomeRef & history_reference,
        product::ProductNode & node
    ) override {
        ensure_cleanup_ok();
        NodeState & history = state_from_ref(history_reference);
        auto found = active_frontiers_.find(node.node_id);
        if (found == active_frontiers_.end()) {
            fail("cannot force an inactive selection frontier");
        }
        ActiveFrontier & frontier = found->second;
        if (!frontier.sequences.empty() || !frontier.children.empty()) {
            fail("selection frontier was forced twice");
        }

        std::vector<llama_token> tokens;
        tokens.reserve(node.alternatives.size());
        frontier.sequences.reserve(node.alternatives.size());
        for (const product::BoundContinuation & binding : node.alternatives) {
            int sequence = sequence_pool_->acquire();
            frontier.sequences.push_back(sequence);
            copy_sequence(history.sequence, sequence);
            tokens.push_back(binding.token);
        }
        frontier.children = decode_sibling_tokens(
            tokens, history.position + 1, frontier.sequences
        );
        if (frontier.children.size() != node.alternatives.size()) {
            fail("forced frontier returned the wrong child count");
        }
        for (NodeState & child : frontier.children) {
            register_state(child);
            frontier.registered_count++;
        }
        for (size_t index = 0; index < node.alternatives.size(); ++index) {
            node.alternatives[index].child_materialized = true;
            node.alternatives[index].outcome = state_ref(
                frontier.children[index]
            );
        }
    }

    void release(product::ProductNode & node) noexcept override {
        auto found = active_frontiers_.find(node.node_id);
        if (found == active_frontiers_.end()) {
            cleanup_failed_ = true;
            return;
        }
        ActiveFrontier & frontier = found->second;
        try {
            for (size_t index = 0;
                 index < frontier.registered_count; ++index) {
                unregister_state(frontier.children[index]);
            }
            for (int sequence : frontier.sequences) {
                sequence_pool_->release(sequence);
            }
        } catch (...) {
            cleanup_failed_ = true;
        }
        active_frontiers_.erase(found);
    }

    std::vector<llama_token> path_tokens(
        const product::BoundPath & path
    ) const {
        std::vector<llama_token> tokens;
        tokens.reserve(path.positions.size());
        for (const product::BoundValue & value : path.positions) {
            tokens.push_back(value.token);
        }
        return tokens;
    }

    std::string path_text(const product::BoundPath & path) const {
        std::string completion = decode_tokens(vocab_, path_tokens(path));
        if (options_.use_chat_template) return completion;
        return options_.prompt + completion;
    }

    size_t maximum_rank(const product::PositionCovector & covector) const {
        if (covector.coordinates.empty()) fail("trace received empty covector");
        size_t best = 0;
        for (size_t index = 1; index < covector.coordinates.size(); ++index) {
            if (covector.coordinates[index].frame != covector.frame) {
                fail("trace received a coordinate from another frame");
            }
            if (covector.coordinates[index].value >
                    covector.coordinates[best].value) {
                best = index;
            }
        }
        return best;
    }

    const product::FramedCoordinate & coordinate_for(
        const product::PositionCovector & covector,
        product::Token token
    ) const {
        for (const product::FramedCoordinate & coordinate :
             covector.coordinates) {
            if (coordinate.token == token) return coordinate;
        }
        fail("trace covector omitted the selected token");
    }

    double scan_evaluation(
        const product::PositionCovector & covector,
        product::Token token
    ) const {
        if (covector.coordinates.empty()) {
            fail("trace received empty boundary-scan covector");
        }
        for (const product::FramedCoordinate & coordinate :
             covector.coordinates) {
            if (coordinate.frame != covector.frame) {
                fail("trace received a coordinate from another frame");
            }
        }
        if (covector.boundary_count == 0) {
            fail("trace boundary-scan covector has zero mass");
        }
        if (!std::isfinite(covector.vocabulary_log_partition)) {
            fail("trace boundary-scan covector has no vocabulary partition");
        }
        return coordinate_for(covector, token).value -
            covector.vocabulary_log_partition;
    }

    size_t coordinate_rank(
        const product::PositionCovector & covector,
        product::Token token
    ) const {
        size_t token_index = covector.coordinates.size();
        for (size_t index = 0; index < covector.coordinates.size(); ++index) {
            if (covector.coordinates[index].token == token) {
                token_index = index;
                break;
            }
        }
        if (token_index == covector.coordinates.size()) {
            fail("trace covector omitted a path token");
        }
        const double value = covector.coordinates[token_index].value;
        size_t rank = 1;
        for (size_t index = 0; index < covector.coordinates.size(); ++index) {
            if (covector.coordinates[index].value > value ||
                (covector.coordinates[index].value == value &&
                 index < token_index)) {
                rank++;
            }
        }
        return rank;
    }

    void continuation_demanded(
        product::SelectionId selection_id,
        size_t ordinal,
        const product::BoundContinuation & value
    ) override {
        if (trace_.stream == nullptr) return;
        std::fprintf(
            trace_.stream,
            "{\"event\":\"continuation_demand\",\"frame\":%llu"
            ",\"depth\":%d,\"remaining\":%d,\"ordinal\":%d"
            ",\"token\":%d,\"local_rank\":%d,\"piece\":",
            static_cast<unsigned long long>(selection_id - 1),
            static_cast<int>(value.position),
            options_.length - static_cast<int>(value.position),
            static_cast<int>(ordinal),
            value.token,
            value.local_rank
        );
        json_string(trace_.stream, token_piece(vocab_, value.token));
        std::fputs("}\n", trace_.stream);
        trace_.flush();
    }

    void candidate_observed(
        product::SelectionId selection_id,
        const product::ObservedAlternative & alternative
    ) override {
        if (alternative.binding == nullptr ||
            alternative.complete_path == nullptr ||
            alternative.current_covector == nullptr ||
            alternative.observation_tuple == nullptr) {
            fail("trace received an incomplete observed alternative");
        }
        if (trace_.stream == nullptr) return;
        const product::BoundValue & value = *alternative.binding;
        const product::PositionCovector & covector =
            *alternative.current_covector;
        const size_t best = maximum_rank(covector);
        const product::FramedCoordinate & candidate = coordinate_for(
            covector, value.token
        );
        std::fprintf(
            trace_.stream,
            "{\"event\":\"candidate\",\"frame\":%llu"
            ",\"depth\":%d,\"remaining\":%d,\"token\":%d"
            ",\"local_rank\":%d,\"logit\":%.9g"
            ",\"proposal_log_probability\":%.17g"
            ",\"observer_frame\":%llu"
            ",\"attains_on_common_support\":%s"
            ",\"observer_max_rank\":%d,\"observer_max_token\":%d"
            ",\"boundary_count\":%zu"
            ",\"scan_covector_logit\":%.9g"
            ",\"scan_vocabulary_log_partition\":%.17g"
            ",\"scan_ev_log_probability\":%.17g,\"text\":",
            static_cast<unsigned long long>(selection_id - 1),
            static_cast<int>(value.position),
            options_.length - static_cast<int>(value.position),
            value.token,
            value.local_rank,
            value.proposal_logit,
            value.proposal_log_probability,
            static_cast<unsigned long long>(covector.frame.frame_id - 1),
            alternative.attains ? "true" : "false",
            static_cast<int>(best + 1),
            covector.coordinates[best].token,
            covector.boundary_count,
            candidate.value,
            covector.vocabulary_log_partition,
            alternative.scan_ev_log_probability
        );
        json_string(trace_.stream, path_text(*alternative.complete_path));
        std::fputs(",\"observer_support\":[", trace_.stream);
        for (size_t index = 0; index < covector.coordinates.size(); ++index) {
            if (index != 0) std::fputc(',', trace_.stream);
            const product::FramedCoordinate & coordinate =
                covector.coordinates[index];
            std::fprintf(
                trace_.stream,
                "{\"proposal_rank\":%zu,\"token\":%d,\"piece\":",
                index + 1,
                coordinate.token
            );
            json_string(
                trace_.stream,
                token_piece(vocab_, coordinate.token)
            );
            std::fprintf(
                trace_.stream,
                ",\"logit\":%.9g}",
                coordinate.value
            );
        }
        std::fputs("],\"outcome_profile\":[", trace_.stream);
        const product::BoundPath & path = *alternative.complete_path;
        const product::ObservationTuple & tuple =
            *alternative.observation_tuple;
        if (path.positions.size() != tuple.positions.size()) {
            fail("trace path and covector tuple have different lengths");
        }
        for (size_t index = 0; index < path.positions.size(); ++index) {
            if (index != 0) std::fputc(',', trace_.stream);
            const product::BoundValue & position = path.positions[index];
            const product::PositionCovector & position_covector =
                tuple.positions[index];
            if (position.position != position_covector.position) {
                fail("trace path and covector tuple are position-misaligned");
            }
            std::fprintf(
                trace_.stream,
                "{\"position\":%zu,\"token\":%d,\"piece\":",
                position.position,
                position.token
            );
            json_string(trace_.stream, token_piece(vocab_, position.token));
            std::fprintf(
                trace_.stream,
                ",\"proposal_local_rank\":%d"
                ",\"covector_rank\":%zu"
                ",\"boundary_count\":%zu"
                ",\"scan_ev_log_probability\":%.17g}",
                position.local_rank,
                coordinate_rank(position_covector, position.token),
                position_covector.boundary_count,
                scan_evaluation(position_covector, position.token)
            );
        }
        std::fputs("]}\n", trace_.stream);
        trace_.flush();
    }

    void continuation_selected(
        product::SelectionId selection_id,
        const product::ObservedAlternative & alternative
    ) override {
        if (alternative.binding == nullptr ||
            alternative.complete_path == nullptr ||
            alternative.current_covector == nullptr) {
            fail("trace received an incomplete selected continuation");
        }
        if (trace_.stream == nullptr) return;
        const product::BoundValue & value = *alternative.binding;
        const product::PositionCovector & covector =
            *alternative.current_covector;
        std::fprintf(
            trace_.stream,
            "{\"event\":\"select\",\"frame\":%llu"
            ",\"depth\":%d,\"remaining\":%d,\"token\":%d"
            ",\"local_rank\":%d,\"observer_frame\":%llu"
            ",\"attains_on_common_support\":%s"
            ",\"scan_ev_log_probability\":%.17g"
            ",\"selection_rule\":\"max_boundary_scan_ev\""
            ",\"propagated\":\"complete_covector_family\""
            ",\"text\":",
            static_cast<unsigned long long>(selection_id - 1),
            static_cast<int>(value.position),
            options_.length - static_cast<int>(value.position),
            value.token,
            value.local_rank,
            static_cast<unsigned long long>(covector.frame.frame_id - 1),
            alternative.attains ? "true" : "false",
            alternative.scan_ev_log_probability
        );
        json_string(trace_.stream, path_text(*alternative.complete_path));
        std::fputs("}\n", trace_.stream);
        trace_.flush();
    }

    llama_model * model_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    const Options & options_;
    std::vector<llama_token> prompt_tokens_;
    Trace & trace_;
    Counters & counters_;
    int vocab_size_ = 0;
    int embedding_out_ = 0;
    BatchOwner token_batch_;
    std::unique_ptr<llama_context, ContextDeleter> context_;
    llama_memory_t memory_ = nullptr;
    std::unique_ptr<SequencePool> sequence_pool_;
    std::unordered_map<uint64_t, NodeState *> states_;
    std::unordered_map<product::NodeId, ActiveFrontier> active_frontiers_;
    bool cleanup_failed_ = false;
    uint64_t next_state_id_ = 1;
    product::NodeId next_node_id_ = 1;
    product::BindingId next_binding_id_ = 1;
    uint64_t next_observer_frame_ = 1;
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
                ",\"mode\":\"exact_select_product_boundary_scan\""
                ",\"observer\":\"recursive_boundary_geometric_barycenter\""
                ",\"selection_rule\":\"max_boundary_scan_ev\"}\n",
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
        product::ProductResult selected = term.run();
        auto stopped = Clock::now();
        double seconds = std::chrono::duration<double>(
            stopped - started
        ).count();

        std::vector<llama_token> selected_tokens;
        selected_tokens.reserve(selected.path.positions.size());
        for (const product::BoundValue & value : selected.path.positions) {
            selected_tokens.push_back(value.token);
        }
        std::string completion = decode_tokens(vocab, selected_tokens);
        std::string displayed = options.use_chat_template
            ? completion
            : user_prompt + completion;

        std::puts("completion:");
        std::puts(displayed.c_str());
        std::printf(
            "score_kind=full_vocabulary_boundary_scan_ev\n"
            "selection_carrier=token_path_with_covector_family\n"
            "selection_observer=recursive_boundary_geometric_barycenter\n"
            "selection_rule=max_boundary_scan_ev\n"
            "observer_attention=affine_scan_of_causal_boundary_covectors\n"
            "observer_composition=geometric_mean_native_boundary_covectors\n"
            "observer_lowering=one_completed_path_lane_per_candidate\n"
            "aggregate_path_score=none\n"
            "root_terminalizations=1\n"
            "strength_nodes=%llu\n"
            "candidate_observations=%llu\n"
            "continuation_demands=%llu\n"
            "model_decode_calls=%llu\n"
            "model_decoded_terms=%llu\n"
            "observer_decode_calls=%llu\n"
            "sequence_copies=%llu\n"
            "attaining_alternatives=%llu\n"
            "ambiguous_selection_nodes=%llu\n"
            "zero_attaining_selection_nodes=%llu\n"
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
            static_cast<unsigned long long>(
                counters.attaining_alternatives
            ),
            static_cast<unsigned long long>(
                counters.ambiguous_selection_nodes
            ),
            static_cast<unsigned long long>(
                counters.zero_attaining_selection_nodes
            ),
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
                ",\"candidate_observations\":%llu"
                ",\"attaining_alternatives\":%llu"
                ",\"ambiguous_selection_nodes\":%llu"
                ",\"zero_attaining_selection_nodes\":%llu}\n",
                static_cast<unsigned long long>(counters.strength_nodes),
                static_cast<unsigned long long>(
                    counters.candidate_observations
                ),
                static_cast<unsigned long long>(
                    counters.attaining_alternatives
                ),
                static_cast<unsigned long long>(
                    counters.ambiguous_selection_nodes
                ),
                static_cast<unsigned long long>(
                    counters.zero_attaining_selection_nodes
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
