#include <metal_stdlib>

using namespace metal;

struct EmbeddingParameters {
    uint count;
    uint vocab_size;
    uint dim;
};

struct RmsParameters {
    uint count;
    uint width;
};

kernel void embedding_family(
    device const int *tokens [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *outputs [[buffer(2)]],
    constant EmbeddingParameters &parameters [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    uint element_count = parameters.count * parameters.dim;
    if (index >= element_count) return;
    uint occurrence = index / parameters.dim;
    uint lane = index - occurrence * parameters.dim;
    int token = tokens[occurrence];
    if (token < 0 || uint(token) >= parameters.vocab_size) return;
    outputs[index] = weights[uint(token) * parameters.dim + lane];
}

kernel void rms_family(
    device const float *inputs [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *outputs [[buffer(2)]],
    constant RmsParameters &parameters [[buffer(3)]],
    threadgroup float *partial [[threadgroup(0)]],
    uint occurrence [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lane_count [[threads_per_threadgroup]]
) {
    if (occurrence >= parameters.count) return;
    uint base = occurrence * parameters.width;
    float sum = 0.0f;
    for (uint index = lane; index < parameters.width; index += lane_count) {
        float value = inputs[base + index];
        sum += value * value;
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = lane_count / 2; stride > 0; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = rsqrt(partial[0] / float(parameters.width) + 1.0e-5f);
    for (uint index = lane; index < parameters.width; index += lane_count) {
        outputs[base + index] = weights[index] * (scale * inputs[base + index]);
    }
}
