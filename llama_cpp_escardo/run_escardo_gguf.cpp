#include "llama.h"
#include "llama-context.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "chat.h"

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

enum class FeedbackBoundary {
    Identity,
    AffineTokenBarycenter,
};

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

struct Options {
    std::string model_path;
    std::string prompt;
    std::string trace_path;
    int length = -1;
    int sample_ms = -1;
    int top_k = 0;
    int batch_size = 16;
    int gpu_layers = 999;
    int threads = 8;
    uint64_t seed = 42;
    FeedbackBoundary feedback = FeedbackBoundary::Identity;
    bool allow_eog = false;
    bool use_chat_template = false;
    bool enable_thinking = true;
};

long parse_long(const char * text, const char * name) {
    char * end = nullptr;
    errno = 0;
    long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        fail(std::string(name) + " must be an integer");
    }
    return value;
}

uint64_t parse_u64(const char * text, const char * name) {
    char * end = nullptr;
    errno = 0;
    unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        fail(std::string(name) + " must be an unsigned integer");
    }
    return static_cast<uint64_t>(value);
}

[[noreturn]] void usage(const char * program) {
    std::fprintf(
        stderr,
        "usage: %s MODEL.gguf --prompt TEXT --length N --sample-ms MS "
        "[--top-k K] [--batch-size N] [--seed N] [--trace FILE] "
        "[--gpu-layers N] [--threads N] [--allow-eog] "
        "[--feedback affine|identity] [--chat --reasoning on|off]\n",
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
        } else if (std::strcmp(flag, "--sample-ms") == 0) {
            options.sample_ms = static_cast<int>(parse_long(value, "sample-ms"));
        } else if (std::strcmp(flag, "--top-k") == 0) {
            options.top_k = static_cast<int>(parse_long(value, "top-k"));
        } else if (std::strcmp(flag, "--batch-size") == 0) {
            options.batch_size = static_cast<int>(parse_long(value, "batch-size"));
        } else if (std::strcmp(flag, "--seed") == 0) {
            options.seed = parse_u64(value, "seed");
        } else if (std::strcmp(flag, "--trace") == 0) {
            options.trace_path = value;
        } else if (std::strcmp(flag, "--gpu-layers") == 0) {
            options.gpu_layers = static_cast<int>(parse_long(value, "gpu-layers"));
        } else if (std::strcmp(flag, "--threads") == 0) {
            options.threads = static_cast<int>(parse_long(value, "threads"));
        } else if (std::strcmp(flag, "--feedback") == 0) {
            if (std::strcmp(value, "affine") == 0) {
                options.feedback = FeedbackBoundary::AffineTokenBarycenter;
            } else if (std::strcmp(value, "identity") == 0) {
                options.feedback = FeedbackBoundary::Identity;
            } else {
                fail("feedback must be 'affine' or 'identity'");
            }
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
        options.sample_ms <= 0 || options.top_k < 0 ||
        options.batch_size <= 0 || options.gpu_layers < 0 ||
        options.threads <= 0) {
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
        templates.get(),
        inputs
    );
    if (rendered.prompt.empty()) fail("chat template rendered an empty prompt");
    return rendered.prompt;
}

uint64_t random_mix(uint64_t value) {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31;
    return value;
}

uint64_t random_next(uint64_t & state) {
    uint64_t value = state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    state = value;
    return value * UINT64_C(2685821657736338717);
}

double random_unit(uint64_t & state) {
    return static_cast<double>(random_next(state) >> 11) *
        (1.0 / 9007199254740992.0);
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
        vocab, token, result.data(), static_cast<int32_t>(result.size()), 0, true
    );
    if (count < 0) fail("could not decode token piece");
    result.resize(static_cast<size_t>(count));
    return result;
}

std::string decode_tokens(
    const llama_vocab * vocab,
    const std::vector<llama_token> & tokens
) {
    std::string text;
    for (llama_token token : tokens) text += token_piece(vocab, token);
    return text;
}

double log_partition(const float * logits, size_t count) {
    float maximum = -std::numeric_limits<float>::max();
    for (size_t index = 0; index < count; ++index) {
        maximum = std::max(maximum, logits[index]);
    }
    double total = 0.0;
    for (size_t index = 0; index < count; ++index) {
        total += std::exp(static_cast<double>(logits[index]) - maximum);
    }
    if (!(total > 0.0) || !std::isfinite(total)) fail("invalid logit partition");
    return static_cast<double>(maximum) + std::log(total);
}

double log_partition(const std::vector<float> & logits) {
    return log_partition(logits.data(), logits.size());
}

struct Trace {
    FILE * stream = nullptr;
    ~Trace() { if (stream != nullptr) std::fclose(stream); }
    void flush() { if (stream != nullptr) std::fflush(stream); }
};

struct Counters {
    uint64_t retained_hidden_states = 0;
    uint64_t feedback_decode_calls = 0;
    uint64_t feedback_decoded_tokens = 0;
    uint64_t feedback_distribution_observations = 0;
    uint64_t feedback_reembedding_calls = 0;
    uint64_t deferred_projection_calls = 0;
    uint64_t deferred_projection_rows = 0;
    uint64_t sampled_paths = 0;
    uint64_t sampled_candidate_demands = 0;
    uint64_t term_nodes = 1;
    uint64_t whole_path_decode_calls = 0;
    uint64_t whole_path_decoded_tokens = 0;
    uint64_t observer_hidden_states = 0;
    uint64_t observer_projection_calls = 0;
    uint64_t observer_projection_rows = 0;
    uint64_t company_observations = 0;
    uint64_t strength_nodes = 0;
    uint64_t strength_candidate_ratings = 0;
};

struct Value {
    llama_token token = LLAMA_TOKEN_NULL;
    int local_rank = 0;
    float logit = 0.0f;
    double log_probability = 0.0;
};

struct ProjectionFrame {
    int position = 0;
    std::vector<float> logits;
    double partition = 0.0;
    std::vector<llama_token> support;
    std::vector<double> cumulative_mass;
    bool ranked_support = false;

    Value draw(uint64_t & random_state) const {
        if (support.empty()) fail("empty deferred projection carrier");
        double target = random_unit(random_state) * cumulative_mass.back();
        auto found = std::lower_bound(
            cumulative_mass.begin(), cumulative_mass.end(), target
        );
        size_t index = static_cast<size_t>(found - cumulative_mass.begin());
        if (index >= support.size()) index = support.size() - 1;
        llama_token token = support[index];
        return Value{
            token,
            ranked_support ? static_cast<int>(index + 1) : 0,
            logits[static_cast<size_t>(token)],
            static_cast<double>(logits[static_cast<size_t>(token)]) - partition,
        };
    }
};

ProjectionFrame make_projection_frame(
    int position,
    std::vector<float> logits,
    const llama_vocab * vocab,
    int top_k,
    bool allow_eog
) {
    ProjectionFrame frame;
    frame.position = position;
    frame.partition = log_partition(logits);
    frame.logits = std::move(logits);
    frame.support.reserve(frame.logits.size());
    for (size_t token = 0; token < frame.logits.size(); ++token) {
        llama_token value = static_cast<llama_token>(token);
        if (!allow_eog && llama_vocab_is_eog(vocab, value)) continue;
        frame.support.push_back(value);
    }
    if (frame.support.empty()) fail("no selectable deferred projections");
    if (top_k > 0 && static_cast<size_t>(top_k) < frame.support.size()) {
        std::partial_sort(
            frame.support.begin(), frame.support.begin() + top_k,
            frame.support.end(),
            [&](llama_token left, llama_token right) {
                float a = frame.logits[static_cast<size_t>(left)];
                float b = frame.logits[static_cast<size_t>(right)];
                return a != b ? a > b : left < right;
            }
        );
        frame.support.resize(static_cast<size_t>(top_k));
        frame.ranked_support = true;
    }
    float maximum = -std::numeric_limits<float>::max();
    for (llama_token token : frame.support) {
        maximum = std::max(maximum, frame.logits[static_cast<size_t>(token)]);
    }
    double cumulative = 0.0;
    frame.cumulative_mass.reserve(frame.support.size());
    for (llama_token token : frame.support) {
        cumulative += std::exp(
            static_cast<double>(frame.logits[static_cast<size_t>(token)]) - maximum
        );
        frame.cumulative_mass.push_back(cumulative);
    }
    if (!(cumulative > 0.0) || !std::isfinite(cumulative)) {
        fail("invalid deferred projection sampling mass");
    }
    return frame;
}

void clear_batch(llama_batch & batch) { batch.n_tokens = 0; }

void add_token(
    llama_batch & batch,
    llama_token token,
    llama_pos position,
    llama_seq_id sequence,
    bool output
) {
    int32_t index = batch.n_tokens++;
    batch.token[index] = token;
    batch.pos[index] = position;
    batch.n_seq_id[index] = 1;
    batch.seq_id[index][0] = sequence;
    batch.logits[index] = output ? 1 : 0;
}

void add_embedding(
    llama_batch & batch,
    const float * embedding,
    int embedding_size,
    llama_pos position
) {
    int32_t index = batch.n_tokens++;
    std::memcpy(
        batch.embd + static_cast<size_t>(index) * embedding_size,
        embedding,
        static_cast<size_t>(embedding_size) * sizeof(float)
    );
    batch.pos[index] = position;
    batch.n_seq_id[index] = 1;
    batch.seq_id[index][0] = 0;
    batch.logits[index] = 1;
}

const char * feedback_boundary_name(FeedbackBoundary boundary) {
    switch (boundary) {
        case FeedbackBoundary::Identity:
            return "hidden_state_identity";
        case FeedbackBoundary::AffineTokenBarycenter:
            return "full_distribution_token_barycenter";
    }
    fail("invalid feedback boundary");
}

class AffineFeedbackBridge {
public:
    AffineFeedbackBridge(
        llama_model * model,
        llama_context * context,
        int embedding_size,
        int vocab_size
    ) : context_(context), embedding_size_(embedding_size) {
        const ggml_tensor * output = model->get_tensor("output.weight");
        if (output == nullptr) output = model->get_tensor("token_embd.weight");
        const ggml_tensor * token_embedding =
            model->get_tensor("token_embd.weight");
        if (output == nullptr || token_embedding == nullptr) {
            fail("affine feedback requires output and token embedding tensors");
        }
        if (output->ne[0] != embedding_size || output->ne[1] != vocab_size ||
            token_embedding->ne[0] != embedding_size ||
            token_embedding->ne[1] != vocab_size) {
            fail("affine feedback tensor shapes do not match model dimensions");
        }

        metadata_.resize(16U * 1024U * 1024U);
        ggml_init_params params = {
            metadata_.size(), metadata_.data(), true
        };
        graph_context_ = ggml_init(params);
        if (graph_context_ == nullptr) {
            fail("could not create affine feedback graph context");
        }
        input_ = ggml_new_tensor_1d(
            graph_context_, GGML_TYPE_F32, embedding_size
        );
        ggml_set_name(input_, "feedback_output_hidden");
        ggml_set_input(input_);
        ggml_tensor * logits = ggml_mul_mat(
            graph_context_, const_cast<ggml_tensor *>(output), input_
        );
        ggml_set_name(logits, "feedback_token_observations");
        ggml_tensor * probabilities = ggml_soft_max(graph_context_, logits);
        ggml_set_name(probabilities, "feedback_token_distribution");
        ggml_tensor * distribution = ggml_reshape_2d(
            graph_context_, probabilities, 1, vocab_size
        );
        output_ = ggml_out_prod(
            graph_context_,
            const_cast<ggml_tensor *>(token_embedding),
            distribution
        );
        ggml_set_name(output_, "feedback_input_embedding_barycenter");
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(graph_context_, 32, false);
        ggml_build_forward_expand(graph_, output_);

        ggml_backend_sched_t model_scheduler = context_->get_sched();
        int backend_count =
            ggml_backend_sched_get_n_backends(model_scheduler);
        if (backend_count <= 0) fail("model exposes no GGML backends");
        std::vector<ggml_backend_t> backends(
            static_cast<size_t>(backend_count)
        );
        std::vector<ggml_backend_buffer_type_t> buffer_types(
            static_cast<size_t>(backend_count)
        );
        for (int index = 0; index < backend_count; ++index) {
            backends[static_cast<size_t>(index)] =
                ggml_backend_sched_get_backend(model_scheduler, index);
            buffer_types[static_cast<size_t>(index)] =
                ggml_backend_sched_get_buffer_type(
                    model_scheduler,
                    backends[static_cast<size_t>(index)]
                );
        }
        scheduler_ = ggml_backend_sched_new(
            backends.data(), buffer_types.data(), backend_count,
            64, false, true
        );
        if (scheduler_ == nullptr) {
            fail("could not create isolated affine feedback scheduler");
        }
    }

    ~AffineFeedbackBridge() {
        if (scheduler_ != nullptr) ggml_backend_sched_free(scheduler_);
        if (graph_context_ != nullptr) ggml_free(graph_context_);
    }

    AffineFeedbackBridge(const AffineFeedbackBridge &) = delete;
    AffineFeedbackBridge & operator=(const AffineFeedbackBridge &) = delete;

    std::vector<float> apply(const std::vector<float> & hidden) {
        if (hidden.size() != static_cast<size_t>(embedding_size_)) {
            fail("affine feedback received a hidden state of the wrong width");
        }
        context_->synchronize();
        ggml_backend_sched_reset(scheduler_);
        if (!ggml_backend_sched_alloc_graph(scheduler_, graph_)) {
            fail("could not allocate affine feedback graph");
        }
        ggml_backend_tensor_set(
            input_, hidden.data(), 0, hidden.size() * sizeof(float)
        );
        if (ggml_backend_sched_graph_compute(scheduler_, graph_) !=
                GGML_STATUS_SUCCESS) {
            fail("affine feedback graph failed");
        }
        ggml_backend_sched_synchronize(scheduler_);
        std::vector<float> embedding(static_cast<size_t>(embedding_size_));
        ggml_backend_tensor_get(
            output_, embedding.data(), 0,
            embedding.size() * sizeof(float)
        );
        return embedding;
    }

private:
    llama_context * context_ = nullptr;
    int embedding_size_ = 0;
    std::vector<unsigned char> metadata_;
    ggml_context * graph_context_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_backend_sched_t scheduler_ = nullptr;
};

std::vector<float> project_hidden_family(
    llama_model * model,
    llama_context * context,
    const std::vector<float> & hidden,
    int row_count,
    int embedding_size,
    int vocab_size
) {
    const ggml_tensor * output = model->get_tensor("output.weight");
    if (output == nullptr) output = model->get_tensor("token_embd.weight");
    if (output == nullptr) {
        fail("model exposes neither output.weight nor tied token_embd.weight");
    }
    if (output->ne[0] != embedding_size || output->ne[1] != vocab_size) {
        fail("output tensor shape does not match retained hidden states");
    }

    std::vector<unsigned char> metadata(16U * 1024U * 1024U);
    ggml_init_params params = { metadata.size(), metadata.data(), true };
    ggml_context * graph_context = ggml_init(params);
    if (graph_context == nullptr) {
        fail("could not create deferred projection graph context");
    }
    ggml_tensor * input = ggml_new_tensor_2d(
        graph_context, GGML_TYPE_F32, embedding_size, row_count
    );
    ggml_set_name(input, "deferred_hidden_family");
    ggml_set_input(input);
    ggml_tensor * projected = ggml_mul_mat(
        graph_context, const_cast<ggml_tensor *>(output), input
    );
    ggml_set_name(projected, "deferred_token_observations");
    ggml_set_output(projected);
    ggml_cgraph * graph = ggml_new_graph_custom(graph_context, 16, false);
    ggml_build_forward_expand(graph, projected);

    context->synchronize();
    ggml_backend_sched_t scheduler = context->get_sched();
    ggml_backend_sched_reset(scheduler);
    if (!ggml_backend_sched_alloc_graph(scheduler, graph)) {
        ggml_free(graph_context);
        fail("could not allocate deferred output-head graph");
    }
    ggml_backend_tensor_set(
        input, hidden.data(), 0, hidden.size() * sizeof(float)
    );
    if (context->graph_compute(graph, true) != GGML_STATUS_SUCCESS) {
        ggml_free(graph_context);
        fail("deferred output-head graph failed");
    }
    context->synchronize();
    std::vector<float> logits(
        static_cast<size_t>(row_count) * static_cast<size_t>(vocab_size)
    );
    ggml_backend_tensor_get(
        projected, logits.data(), 0, logits.size() * sizeof(float)
    );
    ggml_free(graph_context);
    return logits;
}

struct HiddenTape {
    std::vector<ProjectionFrame> frames;
    std::vector<double> prompt_log_probabilities;
};

HiddenTape build_hidden_tape(
    llama_model * model,
    const llama_vocab * vocab,
    const std::vector<llama_token> & prompt_tokens,
    const Options & options,
    Counters & counters
) {
    const int embedding_in = llama_model_n_embd_inp(model);
    const int embedding_out = llama_model_n_embd_out(model);
    if (embedding_in != embedding_out) {
        fail("hidden feedback requires equal model input/output embedding widths");
    }
    const int vocab_size = llama_vocab_n_tokens(vocab);
    uint64_t needed = prompt_tokens.size() + static_cast<uint64_t>(options.length);
    uint64_t context_size = (needed + 255U) & ~UINT64_C(255);
    int32_t batch_capacity = std::max<int32_t>(
        static_cast<int32_t>(prompt_tokens.size()), 1
    );

    llama_context_params params = llama_context_default_params();
    params.n_ctx = static_cast<uint32_t>(context_size);
    params.n_batch = static_cast<uint32_t>(batch_capacity);
    params.n_ubatch = static_cast<uint32_t>(batch_capacity);
    params.n_seq_max = 1;
    params.n_outputs_max = static_cast<uint32_t>(batch_capacity);
    params.n_threads = options.threads;
    params.n_threads_batch = options.threads;
    params.embeddings = true;
    params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    params.no_perf = false;
    llama_context * raw_context = llama_init_from_model(model, params);
    if (raw_context == nullptr) fail("could not create hidden-feedback context");
    std::unique_ptr<llama_context, decltype(&llama_free)> context(
        raw_context, llama_free
    );

    llama_batch prompt = llama_batch_init(
        static_cast<int32_t>(prompt_tokens.size()), 0, 1
    );
    for (size_t index = 0; index < prompt_tokens.size(); ++index) {
        add_token(
            prompt, prompt_tokens[index], static_cast<llama_pos>(index), 0, true
        );
    }
    if (llama_decode(context.get(), prompt) != 0) {
        llama_batch_free(prompt);
        fail("hidden-feedback prompt decode failed");
    }
    llama_batch_free(prompt);
    counters.feedback_decode_calls++;
    counters.feedback_decoded_tokens += prompt_tokens.size();

    std::vector<double> prompt_log_probabilities(prompt_tokens.size(), 0.0);
    for (size_t index = 1; index < prompt_tokens.size(); ++index) {
        float * preceding = llama_get_logits_ith(
            context.get(), static_cast<int32_t>(index - 1)
        );
        if (preceding == nullptr) {
            fail("hidden-feedback prompt returned no token observation");
        }
        std::vector<float> row(preceding, preceding + vocab_size);
        prompt_log_probabilities[index] = static_cast<double>(
            row[static_cast<size_t>(prompt_tokens[index])]
        ) - log_partition(row);
    }

    float * last = llama_get_embeddings_ith(context.get(), -1);
    if (last == nullptr) fail("model did not expose its final prompt hidden state");
    std::vector<float> current(last, last + embedding_out);
    std::vector<float> hidden(
        static_cast<size_t>(options.length) * static_cast<size_t>(embedding_out)
    );
    std::memcpy(
        hidden.data(), current.data(),
        static_cast<size_t>(embedding_out) * sizeof(float)
    );

    std::unique_ptr<AffineFeedbackBridge> affine_bridge;
    if (options.feedback == FeedbackBoundary::AffineTokenBarycenter) {
        affine_bridge = std::make_unique<AffineFeedbackBridge>(
            model, context.get(), embedding_in, vocab_size
        );
    }
    llama_batch feedback = llama_batch_init(1, embedding_in, 1);
    for (int position = 1; position < options.length; ++position) {
        std::vector<float> feedback_input;
        const float * input = current.data();
        if (affine_bridge != nullptr) {
            feedback_input = affine_bridge->apply(current);
            input = feedback_input.data();
            counters.feedback_distribution_observations++;
            counters.feedback_reembedding_calls++;
        }
        clear_batch(feedback);
        add_embedding(
            feedback,
            input,
            embedding_in,
            static_cast<llama_pos>(
                prompt_tokens.size() + static_cast<size_t>(position - 1)
            )
        );
        if (llama_decode(context.get(), feedback) != 0) {
            llama_batch_free(feedback);
            fail("hidden-state feedback decode failed");
        }
        counters.feedback_decode_calls++;
        counters.feedback_decoded_tokens++;
        last = llama_get_embeddings_ith(context.get(), -1);
        if (last == nullptr) {
            llama_batch_free(feedback);
            fail("hidden-state feedback returned no hidden state");
        }
        current.assign(last, last + embedding_out);
        std::memcpy(
            hidden.data() + static_cast<size_t>(position) * embedding_out,
            current.data(),
            static_cast<size_t>(embedding_out) * sizeof(float)
        );
    }
    llama_batch_free(feedback);
    counters.retained_hidden_states = static_cast<uint64_t>(options.length);

    std::vector<float> all_logits = project_hidden_family(
        model, context.get(), hidden, options.length, embedding_out, vocab_size
    );
    counters.deferred_projection_calls++;
    counters.deferred_projection_rows += static_cast<uint64_t>(options.length);

    HiddenTape tape;
    tape.prompt_log_probabilities = std::move(prompt_log_probabilities);
    tape.frames.reserve(static_cast<size_t>(options.length));
    for (int position = 0; position < options.length; ++position) {
        auto begin = all_logits.begin() +
            static_cast<std::ptrdiff_t>(position) * vocab_size;
        std::vector<float> row(begin, begin + vocab_size);
        tape.frames.push_back(make_projection_frame(
            position, std::move(row), vocab, options.top_k, options.allow_eog
        ));
    }
    return tape;
}

struct TermNode {
    uint64_t id = 0;
    TermNode * parent = nullptr;
    int absolute_position = -1;
    int completion_position = -1;
    bool unit = false;
    Value value;
    uint64_t multiplicity = 0;
    bool hidden_set = false;
    std::vector<float> hidden;
    size_t projection_row = std::numeric_limits<size_t>::max();
    bool company_score_set = false;
    double incoming_company_log_probability = 0.0;
    std::vector<std::unique_ptr<TermNode>> children;
    std::unordered_map<llama_token, TermNode *> child_by_token;
    bool forcing = false;
    bool forced = false;
    TermNode * selected = nullptr;
    TermNode * selected_leaf = nullptr;
    double backed_rating = std::numeric_limits<double>::quiet_NaN();
};

struct SampledPath {
    uint64_t id = 0;
    std::vector<Value> values;
    std::vector<TermNode *> nodes;
};

class SelectionTerm {
public:
    SelectionTerm(
        const std::vector<llama_token> & prompt_tokens,
        const llama_vocab * vocab,
        Trace & trace,
        Counters & counters
    ) : vocab_(vocab), trace_(trace), counters_(counters) {
        root_.id = 0;
        prompt_tail_ = &root_;
        for (size_t position = 0; position < prompt_tokens.size(); ++position) {
            Value unit;
            unit.token = prompt_tokens[position];
            unit.local_rank = 1;
            prompt_tail_ = child_for(
                prompt_tail_, unit, static_cast<int>(position), -1, true
            );
            prompt_tail_->multiplicity = 1;
            prompt_nodes_.push_back(prompt_tail_);
        }
    }

    void observe_prompt(const std::vector<double> & incoming_scores) {
        if (incoming_scores.size() != prompt_nodes_.size()) {
            fail("prefill observation count does not match unit selections");
        }
        for (size_t index = 0; index < prompt_nodes_.size(); ++index) {
            observe(*prompt_nodes_[index], incoming_scores[index]);
        }
    }

    TermNode & prompt_node(size_t index) {
        if (index >= prompt_nodes_.size()) fail("invalid prompt term position");
        return *prompt_nodes_[index];
    }

    void retain_hidden(TermNode & node, const float * hidden, int width) {
        if (node.hidden_set) return;
        node.hidden.assign(hidden, hidden + width);
        node.hidden_set = true;
        node.projection_row = hidden_nodes_.size();
        hidden_nodes_.push_back(&node);
        counters_.observer_hidden_states++;
    }

    void finalize_observations(
        llama_model * model,
        llama_context * context
    ) {
        if (hidden_nodes_.empty()) fail("whole-path observer retained no hidden states");
        const int embedding_size = llama_model_n_embd_out(model);
        const int vocab_size = llama_vocab_n_tokens(vocab_);
        std::vector<float> family;
        family.reserve(hidden_nodes_.size() * static_cast<size_t>(embedding_size));
        for (TermNode * node : hidden_nodes_) {
            if (!node->hidden_set ||
                node->hidden.size() != static_cast<size_t>(embedding_size)) {
                fail("whole-path observer retained an invalid hidden state");
            }
            family.insert(family.end(), node->hidden.begin(), node->hidden.end());
        }
        observer_logits_ = project_hidden_family(
            model,
            context,
            family,
            static_cast<int>(hidden_nodes_.size()),
            embedding_size,
            vocab_size
        );
        counters_.observer_projection_calls++;
        counters_.observer_projection_rows += hidden_nodes_.size();

        observer_vocab_size_ = vocab_size;
        observer_partitions_.resize(hidden_nodes_.size());
        for (size_t row = 0; row < hidden_nodes_.size(); ++row) {
            observer_partitions_[row] = log_partition(
                observer_logits_.data() +
                    row * static_cast<size_t>(vocab_size),
                static_cast<size_t>(vocab_size)
            );
        }
        for (TermNode * node : hidden_nodes_) {
            if (node->parent == &root_) {
                observe(*node, 0.0);
                continue;
            }
            TermNode * context_node = node->parent;
            if (!context_node->hidden_set ||
                context_node->projection_row >= hidden_nodes_.size()) {
                fail("whole-path observation lost its preceding context");
            }
            size_t row = context_node->projection_row;
            size_t token = static_cast<size_t>(node->value.token);
            if (token >= static_cast<size_t>(vocab_size)) {
                fail("whole-path observation contains an invalid token");
            }
            double score = static_cast<double>(
                observer_logits_[
                    row * static_cast<size_t>(vocab_size) + token
                ]
            ) - observer_partitions_[row];
            observe(*node, score);
        }
    }

    SampledPath insert(uint64_t id, const std::vector<Value> & values) {
        SampledPath path;
        path.id = id;
        path.values = values;
        path.nodes.reserve(values.size());
        TermNode * parent = prompt_tail_;
        for (size_t position = 0; position < values.size(); ++position) {
            parent = child_for(
                parent,
                values[position],
                prompt_tail_->absolute_position + 1 +
                    static_cast<int>(position),
                static_cast<int>(position),
                false
            );
            parent->multiplicity++;
            path.nodes.push_back(parent);
            counters_.sampled_candidate_demands++;
            trace_demand(path, *parent);
        }
        return path;
    }

    void observe(TermNode & node, double score) {
        if (node.company_score_set) {
            if (std::fabs(node.incoming_company_log_probability - score) >
                    1.0e-5) {
                fail("memoized company continuation changed score");
            }
            return;
        }
        node.company_score_set = true;
        node.incoming_company_log_probability = score;
        counters_.company_observations++;
    }

    void trace_observed(const SampledPath & path) {
        if (trace_.stream == nullptr) return;
        std::fprintf(
            trace_.stream,
            "{\"event\":\"path_observed\",\"sample\":%llu,\"tokens\":[",
            static_cast<unsigned long long>(path.id)
        );
        for (size_t index = 0; index < path.values.size(); ++index) {
            if (index != 0) std::fputc(',', trace_.stream);
            std::fprintf(trace_.stream, "%d", path.values[index].token);
        }
        std::fputs("],\"proposal_log_probabilities\":[", trace_.stream);
        for (size_t index = 0; index < path.values.size(); ++index) {
            if (index != 0) std::fputc(',', trace_.stream);
            std::fprintf(
                trace_.stream, "%.17g", path.values[index].log_probability
            );
        }
        std::fputs("],\"company_log_probabilities\":[", trace_.stream);
        for (size_t index = 0; index < path.nodes.size(); ++index) {
            if (index != 0) std::fputc(',', trace_.stream);
            std::fprintf(
                trace_.stream,
                "%.17g",
                path.nodes[index]->incoming_company_log_probability
            );
        }
        std::fputs("],\"aggregate_score\":null,\"text\":", trace_.stream);
        json_string(
            trace_.stream,
            decode_tokens(
                vocab_,
                completion_tokens(
                    path.nodes.empty() ? prompt_tail_ : path.nodes.back()
                )
            )
        );
        std::fputs("}\n", trace_.stream);
        trace_.flush();
    }

    TermNode * select() {
        force(root_);
        if (root_.selected_leaf == nullptr) fail("selection term has no leaf");
        return root_.selected_leaf;
    }

    std::vector<llama_token> completion_tokens(TermNode * leaf) const {
        std::vector<llama_token> reversed;
        for (TermNode * node = leaf; node != nullptr && node != &root_;
             node = node->parent) {
            if (node->completion_position >= 0) {
                reversed.push_back(node->value.token);
            }
        }
        std::reverse(reversed.begin(), reversed.end());
        return reversed;
    }

    double first_completion_rating() const {
        if (prompt_tail_->selected == nullptr) {
            fail("prefill unit did not retain a completion selection");
        }
        return prompt_tail_->backed_rating;
    }

private:
    TermNode * child_for(
        TermNode * parent,
        const Value & value,
        int absolute_position,
        int completion_position,
        bool unit
    ) {
        auto found = parent->child_by_token.find(value.token);
        if (found != parent->child_by_token.end()) return found->second;
        auto child = std::make_unique<TermNode>();
        child->id = next_id_++;
        child->parent = parent;
        child->absolute_position = absolute_position;
        child->completion_position = completion_position;
        child->unit = unit;
        child->value = value;
        TermNode * result = child.get();
        parent->children.push_back(std::move(child));
        parent->child_by_token.emplace(value.token, result);
        counters_.term_nodes++;
        return result;
    }

    void trace_demand(const SampledPath & path, const TermNode & node) {
        if (trace_.stream == nullptr) return;
        std::fprintf(
            trace_.stream,
            "{\"event\":\"continuation_demand\",\"sample\":%llu"
            ",\"node\":%llu,\"position\":%d,\"token\":%d"
            ",\"local_rank\":%d,\"proposal_logit\":%.9g"
            ",\"proposal_log_probability\":%.17g"
            ",\"multiplicity\":%llu,\"piece\":",
            static_cast<unsigned long long>(path.id),
            static_cast<unsigned long long>(node.id),
            node.completion_position,
            node.value.token,
            node.value.local_rank,
            node.value.logit,
            node.value.log_probability,
            static_cast<unsigned long long>(node.multiplicity)
        );
        json_string(trace_.stream, token_piece(vocab_, node.value.token));
        std::fputs("}\n", trace_.stream);
        trace_.flush();
    }

    static bool improves(
        const TermNode & candidate,
        double rating,
        const TermNode * selected,
        double selected_rating
    ) {
        if (selected == nullptr || rating > selected_rating) return true;
        if (rating < selected_rating) return false;
        if (candidate.value.local_rank > 0 && selected->value.local_rank > 0 &&
            candidate.value.local_rank != selected->value.local_rank) {
            return candidate.value.local_rank < selected->value.local_rank;
        }
        return candidate.value.token < selected->value.token;
    }

    struct ObserverCoordinate {
        double log_probability = -std::numeric_limits<double>::infinity();
        const TermNode * context = nullptr;
        const TermNode * company = nullptr;
        bool incoming_boundary = false;
    };

    /*
     * A candidate is rated by the company retained by its recursively forced
     * continuation.  For an internal occurrence, its hidden state observes
     * the selected immediate successor.  At the finite right boundary, where
     * no successor was invented, the preceding hidden state observes the
     * candidate itself.  In particular, an earlier token is never read from
     * the final causal state after the whole completion.
     */
    ObserverCoordinate observer_coordinate(const TermNode & candidate) const {
        ObserverCoordinate observation;
        if (candidate.selected != nullptr) {
            if (candidate.selected->parent != &candidate) {
                fail("candidate retained a non-successor company");
            }
            observation.context = &candidate;
            observation.company = candidate.selected;
        } else {
            if (!candidate.children.empty()) {
                fail("candidate lost its recursively selected company");
            }
            observation.context = candidate.parent;
            observation.company = &candidate;
            observation.incoming_boundary = true;
        }
        if (observation.context == nullptr ||
            observation.context == &root_ ||
            !observation.context->hidden_set ||
            observation.context->projection_row >= observer_partitions_.size() ||
            observer_vocab_size_ <= 0) {
            fail("candidate company has no projected contextual hidden state");
        }
        size_t token = static_cast<size_t>(observation.company->value.token);
        if (token >= static_cast<size_t>(observer_vocab_size_)) {
            fail("candidate company contains an invalid token coordinate");
        }
        size_t row = observation.context->projection_row;
        observation.log_probability = static_cast<double>(
            observer_logits_[
                row * static_cast<size_t>(observer_vocab_size_) + token
            ]
        ) - observer_partitions_[row];
        return observation;
    }

    void trace_selection_candidate(
        const TermNode & frame,
        const TermNode & candidate,
        const ObserverCoordinate & observation
    ) {
        if (trace_.stream == nullptr) return;
        std::fprintf(
            trace_.stream,
            "{\"event\":\"selection_candidate\",\"frame\":%llu"
            ",\"candidate_node\":%llu,\"position\":%d,\"token\":%d"
            ",\"observer_company_log_probability\":%.17g"
            ",\"observer_context_node\":%llu"
            ",\"observer_company_node\":%llu"
            ",\"observer_company_token\":%d"
            ",\"observer_direction\":\"%s\""
            ",\"aggregate_score\":null,\"text\":",
            static_cast<unsigned long long>(frame.id),
            static_cast<unsigned long long>(candidate.id),
            candidate.completion_position,
            candidate.value.token,
            observation.log_probability,
            static_cast<unsigned long long>(observation.context->id),
            static_cast<unsigned long long>(observation.company->id),
            observation.company->value.token,
            observation.incoming_boundary ? "incoming_boundary" : "outgoing"
        );
        json_string(
            trace_.stream,
            decode_tokens(vocab_, completion_tokens(candidate.selected_leaf))
        );
        std::fputs("}\n", trace_.stream);
        trace_.flush();
    }

    void trace_select(const TermNode & frame) {
        if (trace_.stream == nullptr || frame.selected == nullptr) return;
        ObserverCoordinate observation = observer_coordinate(*frame.selected);
        std::fprintf(
            trace_.stream,
            "{\"event\":\"select\",\"frame\":%llu,\"selected_node\":%llu"
            ",\"position\":%d,\"token\":%d,\"alternatives\":%zu"
            ",\"observer_company_log_probability\":%.17g"
            ",\"observer_context_node\":%llu"
            ",\"observer_company_node\":%llu"
            ",\"observer_company_token\":%d"
            ",\"observer_direction\":\"%s\",\"text\":",
            static_cast<unsigned long long>(frame.id),
            static_cast<unsigned long long>(frame.selected->id),
            frame.selected->completion_position,
            frame.selected->value.token,
            frame.children.size(),
            observation.log_probability,
            static_cast<unsigned long long>(observation.context->id),
            static_cast<unsigned long long>(observation.company->id),
            observation.company->value.token,
            observation.incoming_boundary ? "incoming_boundary" : "outgoing"
        );
        json_string(
            trace_.stream,
            decode_tokens(vocab_, completion_tokens(frame.selected_leaf))
        );
        std::fputs("}\n", trace_.stream);
        trace_.flush();
    }

    /* Mechanical function-tree form of Escardo's product:
     *   b(x) = force the subtree retained below x
     *   a    = epsilon(x -> p(x : b(x)))
     *   result = a : b(a)
     * Sampling limits which x nodes exist; it never chooses a result path. */
    void force(TermNode & frame) {
        if (frame.forced) return;
        if (frame.forcing) fail("selection term contains a cycle");
        frame.forcing = true;
        if (frame.children.empty()) {
            frame.selected_leaf = &frame;
            frame.forced = true;
            frame.forcing = false;
            return;
        }
        counters_.strength_nodes++;
        TermNode * best = nullptr;
        double best_rating = -std::numeric_limits<double>::infinity();
        for (const std::unique_ptr<TermNode> & owned : frame.children) {
            TermNode & candidate = *owned;
            force(candidate);
            ObserverCoordinate observation = observer_coordinate(candidate);
            double rating = observation.log_probability;
            counters_.strength_candidate_ratings++;
            trace_selection_candidate(frame, candidate, observation);
            if (improves(candidate, rating, best, best_rating)) {
                best = &candidate;
                best_rating = rating;
            }
        }
        if (best == nullptr || best->selected_leaf == nullptr) {
            fail("selection frame retained no continuation");
        }
        frame.selected = best;
        frame.selected_leaf = best->selected_leaf;
        frame.backed_rating = best_rating;
        frame.forced = true;
        frame.forcing = false;
        trace_select(frame);
    }

    const llama_vocab * vocab_;
    Trace & trace_;
    Counters & counters_;
    TermNode root_;
    TermNode * prompt_tail_ = nullptr;
    std::vector<TermNode *> prompt_nodes_;
    std::vector<TermNode *> hidden_nodes_;
    std::vector<float> observer_logits_;
    std::vector<double> observer_partitions_;
    int observer_vocab_size_ = 0;
    uint64_t next_id_ = 1;
};

class WholePathObserver {
public:
    WholePathObserver(
        llama_model * model,
        std::vector<llama_token> prompt_tokens,
        const Options & options,
        SelectionTerm & term,
        Counters & counters
    ) :
        model_(model),
        prompt_tokens_(std::move(prompt_tokens)),
        options_(options),
        term_(term),
        counters_(counters) {
        uint64_t needed = prompt_tokens_.size() +
            static_cast<uint64_t>(options_.length);
        uint64_t per_sequence = (needed + 255U) & ~UINT64_C(255);
        uint64_t sequence_count = static_cast<uint64_t>(options_.batch_size);
        uint64_t total_context = per_sequence * sequence_count;
        if (total_context > std::numeric_limits<uint32_t>::max()) {
            fail("whole-path observer context is too large");
        }
        uint64_t batch_capacity = needed * sequence_count;
        if (batch_capacity > static_cast<uint64_t>(
                std::numeric_limits<int32_t>::max())) {
            fail("whole-path observer batch is too large");
        }
        batch_capacity_ = static_cast<int32_t>(batch_capacity);
        embedding_size_ = llama_model_n_embd_out(model_);
        llama_context_params params = llama_context_default_params();
        params.n_ctx = static_cast<uint32_t>(total_context);
        params.n_batch = static_cast<uint32_t>(batch_capacity_);
        params.n_ubatch = static_cast<uint32_t>(batch_capacity_);
        params.n_seq_max = static_cast<uint32_t>(sequence_count);
        params.n_outputs_max = static_cast<uint32_t>(batch_capacity_);
        params.n_threads = options_.threads;
        params.n_threads_batch = options_.threads;
        params.embeddings = true;
        params.pooling_type = LLAMA_POOLING_TYPE_NONE;
        params.no_perf = false;
        context_.reset(llama_init_from_model(model, params));
        if (!context_) fail("could not create whole-path observer context");
    }

    void observe(std::vector<SampledPath> & paths) {
        if (paths.empty()) return;
        if (paths.size() > static_cast<size_t>(options_.batch_size)) {
            fail("whole-path observer batch exceeds configured size");
        }
        llama_memory_t memory = llama_get_memory(context_.get());
        llama_memory_clear(memory, true);

        llama_batch batch = llama_batch_init(batch_capacity_, 0, 1);
        std::vector<TermNode *> output_nodes;
        output_nodes.reserve(static_cast<size_t>(batch_capacity_));
        for (size_t sequence_index = 0; sequence_index < paths.size();
             ++sequence_index) {
            llama_seq_id sequence = static_cast<llama_seq_id>(sequence_index);
            for (size_t position = 0; position < prompt_tokens_.size();
                 ++position) {
                add_token(
                    batch,
                    prompt_tokens_[position],
                    static_cast<llama_pos>(position),
                    sequence,
                    true
                );
                output_nodes.push_back(&term_.prompt_node(position));
            }
            SampledPath & path = paths[sequence_index];
            for (size_t depth = 0; depth < path.values.size(); ++depth) {
                add_token(
                    batch,
                    path.values[depth].token,
                    static_cast<llama_pos>(prompt_tokens_.size() + depth),
                    sequence,
                    true
                );
                output_nodes.push_back(path.nodes[depth]);
            }
        }
        if (batch.n_tokens != static_cast<int32_t>(output_nodes.size())) {
            llama_batch_free(batch);
            fail("whole-path observer lost its output row mapping");
        }
        int32_t result = llama_decode(context_.get(), batch);
        if (result != 0) {
            llama_batch_free(batch);
            fail("whole-path observer decode failed with code " +
                std::to_string(result));
        }
        counters_.whole_path_decode_calls++;
        counters_.whole_path_decoded_tokens +=
            static_cast<uint64_t>(batch.n_tokens);
        for (size_t row = 0; row < output_nodes.size(); ++row) {
            float * hidden = llama_get_embeddings_ith(
                context_.get(), static_cast<int32_t>(row)
            );
            if (hidden == nullptr) {
                llama_batch_free(batch);
                fail("whole-path observer returned no hidden-state row");
            }
            term_.retain_hidden(*output_nodes[row], hidden, embedding_size_);
        }
        llama_batch_free(batch);
    }

    void finalize() {
        term_.finalize_observations(model_, context_.get());
    }

private:
    struct ContextDeleter {
        void operator()(llama_context * context) const {
            if (context != nullptr) llama_free(context);
        }
    };

    llama_model * model_;
    std::vector<llama_token> prompt_tokens_;
    const Options & options_;
    SelectionTerm & term_;
    Counters & counters_;
    std::unique_ptr<llama_context, ContextDeleter> context_;
    int32_t batch_capacity_ = 0;
    int embedding_size_ = 0;
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

        std::string user_prompt = options.prompt;
        if (options.use_chat_template) {
            options.prompt = format_chat_prompt(
                model.get(), user_prompt, options.enable_thinking
            );
        }
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        std::vector<llama_token> prompt_tokens = tokenize(vocab, options.prompt);
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
            json_string(trace.stream, options.prompt);
            std::fprintf(
                trace.stream,
                ",\"chat_template\":%s,\"enable_thinking\":%s"
                ",\"length\":%d,\"proposal_top_k\":",
                options.use_chat_template ? "true" : "false",
                options.enable_thinking ? "true" : "false",
                options.length
            );
            if (options.top_k == 0) std::fputs("null", trace.stream);
            else std::fprintf(trace.stream, "%d", options.top_k);
            std::fprintf(
                trace.stream,
                ",\"sample_ms\":%d,\"seed\":%llu,\"batch_size\":%d"
                ",\"feedback_boundary\":\"%s\""
                ",\"projection\":\"deferred_batched_output_head\""
                ",\"observer\":\"whole_path_hidden_firthian_context_coordinate\""
                ",\"strength\":\"memoized_escardo_function_tree\"}\n",
                options.sample_ms,
                static_cast<unsigned long long>(options.seed),
                options.batch_size,
                feedback_boundary_name(options.feedback)
            );
            trace.flush();
        }

        Counters counters;
        auto started = Clock::now();
        HiddenTape tape = build_hidden_tape(
            model.get(), vocab, prompt_tokens, options, counters
        );
        SelectionTerm term(prompt_tokens, vocab, trace, counters);
        WholePathObserver observer(
            model.get(), prompt_tokens, options, term, counters
        );
        uint64_t random_state = random_mix(
            options.seed ^ UINT64_C(0x6a09e667f3bcc909)
        );
        if (random_state == 0) random_state = UINT64_C(0x4d595df4d0f33173);
        auto deadline = Clock::now() + std::chrono::milliseconds(options.sample_ms);
        std::vector<SampledPath> observed_paths;
        do {
            std::vector<SampledPath> paths;
            paths.reserve(static_cast<size_t>(options.batch_size));
            for (int sample = 0; sample < options.batch_size; ++sample) {
                std::vector<Value> values;
                values.reserve(tape.frames.size());
                for (const ProjectionFrame & frame : tape.frames) {
                    Value value = frame.draw(random_state);
                    values.push_back(value);
                    if (options.allow_eog &&
                        llama_vocab_is_eog(vocab, value.token)) {
                        break;
                    }
                }
                uint64_t sample_id = counters.sampled_paths++;
                paths.push_back(term.insert(sample_id, values));
            }
            observer.observe(paths);
            for (SampledPath & path : paths) {
                observed_paths.push_back(std::move(path));
            }
        } while (Clock::now() < deadline);
        observer.finalize();
        for (const SampledPath & path : observed_paths) {
            term.trace_observed(path);
        }

        TermNode * selected_leaf = term.select();
        std::vector<llama_token> selected = term.completion_tokens(selected_leaf);
        double selected_rating = term.first_completion_rating();
        auto stopped = Clock::now();
        double seconds = std::chrono::duration<double>(stopped - started).count();
        std::string completion = decode_tokens(vocab, selected);

        std::puts("completion:");
        std::puts(completion.c_str());
        std::printf(
            "selected_score=%.17g\n"
            "score_kind=first_completion_firthian_context_coordinate\n"
            "aggregate_path_score=none\n"
            "selection_carrier=DeferredGGUFLogit\n"
            "feedback_boundary=%s\n"
            "projection_decisions_during_recurrence=0\n"
            "root_terminalizations=1\n"
            "retained_hidden_states=%llu\n"
            "feedback_decode_calls=%llu\n"
            "feedback_decoded_tokens=%llu\n"
            "feedback_distribution_observations=%llu\n"
            "feedback_reembedding_calls=%llu\n"
            "deferred_projection_calls=%llu\n"
            "deferred_projection_rows=%llu\n"
            "sampled_paths=%llu\n"
            "sampled_candidate_demands=%llu\n"
            "term_nodes=%llu\n"
            "whole_path_decode_calls=%llu\n"
            "whole_path_decoded_tokens=%llu\n"
            "observer_hidden_states=%llu\n"
            "observer_projection_calls=%llu\n"
            "observer_projection_rows=%llu\n"
            "company_observations=%llu\n"
            "strength_nodes=%llu\n"
            "strength_candidate_ratings=%llu\n"
            "model_decode_calls_during_strength=0\n"
            "numerical_backend=llama.cpp/ggml-metal\n"
            "chat_template=%s\n"
            "enable_thinking=%s\n"
            "search_batch_size=%d\n"
            "search_seconds=%.6f\n",
            selected_rating,
            feedback_boundary_name(options.feedback),
            static_cast<unsigned long long>(counters.retained_hidden_states),
            static_cast<unsigned long long>(counters.feedback_decode_calls),
            static_cast<unsigned long long>(counters.feedback_decoded_tokens),
            static_cast<unsigned long long>(
                counters.feedback_distribution_observations
            ),
            static_cast<unsigned long long>(counters.feedback_reembedding_calls),
            static_cast<unsigned long long>(counters.deferred_projection_calls),
            static_cast<unsigned long long>(counters.deferred_projection_rows),
            static_cast<unsigned long long>(counters.sampled_paths),
            static_cast<unsigned long long>(counters.sampled_candidate_demands),
            static_cast<unsigned long long>(counters.term_nodes),
            static_cast<unsigned long long>(counters.whole_path_decode_calls),
            static_cast<unsigned long long>(counters.whole_path_decoded_tokens),
            static_cast<unsigned long long>(counters.observer_hidden_states),
            static_cast<unsigned long long>(counters.observer_projection_calls),
            static_cast<unsigned long long>(counters.observer_projection_rows),
            static_cast<unsigned long long>(counters.company_observations),
            static_cast<unsigned long long>(counters.strength_nodes),
            static_cast<unsigned long long>(counters.strength_candidate_ratings),
            options.use_chat_template ? "embedded_jinja" : "raw",
            options.enable_thinking ? "true" : "false",
            options.batch_size,
            seconds
        );

        if (trace.stream != nullptr) {
            std::fputs(
                "{\"event\":\"root_terminalization\",\"score\":",
                trace.stream
            );
            std::fprintf(trace.stream, "%.17g,\"text\":", selected_rating);
            json_string(trace.stream, completion);
            std::fprintf(
                trace.stream,
                ",\"sampled_paths\":%llu,\"term_nodes\":%llu"
                ",\"strength_candidate_ratings\":%llu}\n",
                static_cast<unsigned long long>(counters.sampled_paths),
                static_cast<unsigned long long>(counters.term_nodes),
                static_cast<unsigned long long>(counters.strength_candidate_ratings)
            );
            trace.flush();
        }
        return EXIT_SUCCESS;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "escardo-gguf: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
