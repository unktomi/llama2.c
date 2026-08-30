#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "metal_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

@interface AtkeyMetalContext : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) id<MTLComputePipelineState> embeddingPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> rmsPipeline;
@property(nonatomic, strong) NSMutableDictionary<NSString *, id<MTLBuffer>> *weights;
@property(nonatomic, strong) id<MTLBuffer> inputScratch;
@property(nonatomic, strong) id<MTLBuffer> outputScratch;
@property(nonatomic, strong) id<MTLBuffer> tokenScratch;
@end

@implementation AtkeyMetalContext
@end

struct AtkeyMetalBackend {
    void *retained_context;
    char *device_name;
    uint64_t dispatch_count;
    uint64_t weight_upload_count;
    uint64_t weight_upload_bytes;
};

typedef struct {
    uint32_t count;
    uint32_t vocab_size;
    uint32_t dim;
} EmbeddingParameters;

typedef struct {
    uint32_t count;
    uint32_t width;
} RmsParameters;

static AtkeyMetalContext *context_for(AtkeyMetalBackend *backend) {
    if (backend == NULL || backend->retained_context == NULL) return nil;
    return (__bridge AtkeyMetalContext *)backend->retained_context;
}

static id<MTLComputePipelineState> make_pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString *name
) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) {
        fprintf(stderr, "metal: function %s is missing\n", name.UTF8String);
        return nil;
    }
    NSError *error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
        fprintf(
            stderr,
            "metal: could not create %s pipeline: %s\n",
            name.UTF8String,
            error.localizedDescription.UTF8String
        );
    }
    return pipeline;
}

AtkeyMetalBackend *atkey_metal_backend_new(const char *library_path) {
    if (library_path == NULL) return NULL;
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            fprintf(stderr, "metal: no system Metal device\n");
            return NULL;
        }
        NSError *error = nil;
        NSURL *url = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:library_path]];
        id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
        if (library == nil) {
            fprintf(
                stderr,
                "metal: could not load %s: %s\n",
                library_path,
                error.localizedDescription.UTF8String
            );
            return NULL;
        }

        AtkeyMetalContext *context = [AtkeyMetalContext new];
        context.device = device;
        context.queue = [device newCommandQueue];
        context.library = library;
        context.embeddingPipeline = make_pipeline(
            device,
            library,
            @"embedding_family"
        );
        context.rmsPipeline = make_pipeline(device, library, @"rms_family");
        context.weights = [NSMutableDictionary dictionary];
        if (context.queue == nil || context.embeddingPipeline == nil ||
            context.rmsPipeline == nil) {
            return NULL;
        }

        AtkeyMetalBackend *backend = (AtkeyMetalBackend *)calloc(
            1,
            sizeof(*backend)
        );
        if (backend == NULL) return NULL;
        backend->retained_context = (__bridge_retained void *)context;
        backend->device_name = strdup(device.name.UTF8String);
        if (backend->device_name == NULL) {
            atkey_metal_backend_free(backend);
            return NULL;
        }
        return backend;
    }
}

void atkey_metal_backend_free(AtkeyMetalBackend *backend) {
    if (backend == NULL) return;
    if (backend->retained_context != NULL) {
        AtkeyMetalContext *context = (__bridge_transfer AtkeyMetalContext *)
            backend->retained_context;
        (void)context;
    }
    free(backend->device_name);
    free(backend);
}

const char *atkey_metal_backend_device_name(AtkeyMetalBackend *backend) {
    return backend == NULL ? NULL : backend->device_name;
}

uint64_t atkey_metal_backend_dispatch_count(AtkeyMetalBackend *backend) {
    return backend == NULL ? 0 : backend->dispatch_count;
}

uint64_t atkey_metal_backend_weight_upload_count(AtkeyMetalBackend *backend) {
    return backend == NULL ? 0 : backend->weight_upload_count;
}

uint64_t atkey_metal_backend_weight_upload_bytes(AtkeyMetalBackend *backend) {
    return backend == NULL ? 0 : backend->weight_upload_bytes;
}

static id<MTLBuffer> ensure_scratch(
    id<MTLDevice> device,
    id<MTLBuffer> current,
    NSUInteger length
) {
    if (current != nil && current.length >= length) return current;
    return [device newBufferWithLength:length
        options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> cached_weights(
    AtkeyMetalBackend *backend,
    const float *weights,
    int rows,
    int columns,
    NSUInteger row_bytes
) {
    AtkeyMetalContext *context = context_for(backend);
    NSString *key = [NSString stringWithFormat:
        @"%p:%d:%d:%lu",
        weights,
        rows,
        columns,
        (unsigned long)row_bytes];
    id<MTLBuffer> buffer = context.weights[key];
    if (buffer != nil) return buffer;

    NSUInteger length = (NSUInteger)rows * row_bytes;
    buffer = [context.device newBufferWithLength:length
        options:MTLResourceStorageModeShared];
    if (buffer == nil) return nil;
    unsigned char *destination = (unsigned char *)buffer.contents;
    size_t packed_row_bytes = (size_t)columns * sizeof(float);
    for (int row = 0; row < rows; row++) {
        memcpy(
            destination + (NSUInteger)row * row_bytes,
            weights + (size_t)row * (size_t)columns,
            packed_row_bytes
        );
        if (row_bytes > packed_row_bytes) {
            memset(
                destination + (NSUInteger)row * row_bytes + packed_row_bytes,
                0,
                row_bytes - packed_row_bytes
            );
        }
    }
    context.weights[key] = buffer;
    backend->weight_upload_count++;
    backend->weight_upload_bytes += (uint64_t)length;
    return buffer;
}

static bool finish_command(
    AtkeyMetalBackend *backend,
    id<MTLCommandBuffer> command,
    const char *operation
) {
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        fprintf(
            stderr,
            "metal: %s failed: %s\n",
            operation,
            command.error.localizedDescription.UTF8String
        );
        return false;
    }
    backend->dispatch_count++;
    return true;
}

bool atkey_metal_embedding_family(
    AtkeyMetalBackend *backend,
    float *outputs,
    const int *tokens,
    int count,
    const float *weights,
    int vocab_size,
    int dim
) {
    if (backend == NULL || outputs == NULL || tokens == NULL || count <= 0 ||
        weights == NULL || vocab_size <= 0 || dim <= 0) return false;
    @autoreleasepool {
        AtkeyMetalContext *context = context_for(backend);
        NSUInteger token_bytes = (NSUInteger)count * sizeof(*tokens);
        NSUInteger output_bytes =
            (NSUInteger)count * (NSUInteger)dim * sizeof(*outputs);
        context.tokenScratch = ensure_scratch(
            context.device,
            context.tokenScratch,
            token_bytes
        );
        context.outputScratch = ensure_scratch(
            context.device,
            context.outputScratch,
            output_bytes
        );
        id<MTLBuffer> weight_buffer = cached_weights(
            backend,
            weights,
            vocab_size,
            dim,
            (NSUInteger)dim * sizeof(float)
        );
        if (context.tokenScratch == nil || context.outputScratch == nil ||
            weight_buffer == nil) return false;
        memcpy(context.tokenScratch.contents, tokens, token_bytes);

        EmbeddingParameters parameters = {
            (uint32_t)count,
            (uint32_t)vocab_size,
            (uint32_t)dim,
        };
        id<MTLCommandBuffer> command = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:context.embeddingPipeline];
        [encoder setBuffer:context.tokenScratch offset:0 atIndex:0];
        [encoder setBuffer:weight_buffer offset:0 atIndex:1];
        [encoder setBuffer:context.outputScratch offset:0 atIndex:2];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
        NSUInteger thread_count = (NSUInteger)count * (NSUInteger)dim;
        NSUInteger group_width = MIN(
            context.embeddingPipeline.maxTotalThreadsPerThreadgroup,
            (NSUInteger)256
        );
        [encoder dispatchThreads:MTLSizeMake(thread_count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(group_width, 1, 1)];
        [encoder endEncoding];
        if (!finish_command(backend, command, "embedding family")) return false;
        memcpy(outputs, context.outputScratch.contents, output_bytes);
        return true;
    }
}

bool atkey_metal_rms_family(
    AtkeyMetalBackend *backend,
    float *outputs,
    const float *inputs,
    int count,
    const float *weights,
    int width
) {
    if (backend == NULL || outputs == NULL || inputs == NULL || count <= 0 ||
        weights == NULL || width <= 0) return false;
    @autoreleasepool {
        AtkeyMetalContext *context = context_for(backend);
        NSUInteger bytes =
            (NSUInteger)count * (NSUInteger)width * sizeof(*inputs);
        context.inputScratch = ensure_scratch(
            context.device,
            context.inputScratch,
            bytes
        );
        context.outputScratch = ensure_scratch(
            context.device,
            context.outputScratch,
            bytes
        );
        id<MTLBuffer> weight_buffer = cached_weights(
            backend,
            weights,
            1,
            width,
            (NSUInteger)width * sizeof(float)
        );
        if (context.inputScratch == nil || context.outputScratch == nil ||
            weight_buffer == nil) return false;
        memcpy(context.inputScratch.contents, inputs, bytes);

        RmsParameters parameters = {(uint32_t)count, (uint32_t)width};
        id<MTLCommandBuffer> command = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:context.rmsPipeline];
        [encoder setBuffer:context.inputScratch offset:0 atIndex:0];
        [encoder setBuffer:weight_buffer offset:0 atIndex:1];
        [encoder setBuffer:context.outputScratch offset:0 atIndex:2];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
        NSUInteger group_width = MIN(
            context.rmsPipeline.maxTotalThreadsPerThreadgroup,
            (NSUInteger)256
        );
        [encoder setThreadgroupMemoryLength:group_width * sizeof(float)
            atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(group_width, 1, 1)];
        [encoder endEncoding];
        if (!finish_command(backend, command, "RMS family")) return false;
        memcpy(outputs, context.outputScratch.contents, bytes);
        return true;
    }
}

bool atkey_metal_matmul_family(
    AtkeyMetalBackend *backend,
    float *outputs,
    const float *inputs,
    int count,
    const float *weights,
    int input_width,
    int output_width
) {
    if (backend == NULL || outputs == NULL || inputs == NULL || count <= 0 ||
        weights == NULL || input_width <= 0 || output_width <= 0) return false;
    @autoreleasepool {
        AtkeyMetalContext *context = context_for(backend);
        NSUInteger input_row_bytes = [MPSMatrixDescriptor
            rowBytesFromColumns:(NSUInteger)input_width
            dataType:MPSDataTypeFloat32];
        NSUInteger output_row_bytes = [MPSMatrixDescriptor
            rowBytesFromColumns:(NSUInteger)output_width
            dataType:MPSDataTypeFloat32];
        NSUInteger input_bytes = (NSUInteger)count * input_row_bytes;
        NSUInteger output_bytes = (NSUInteger)count * output_row_bytes;
        context.inputScratch = ensure_scratch(
            context.device,
            context.inputScratch,
            input_bytes
        );
        context.outputScratch = ensure_scratch(
            context.device,
            context.outputScratch,
            output_bytes
        );
        id<MTLBuffer> weight_buffer = cached_weights(
            backend,
            weights,
            output_width,
            input_width,
            input_row_bytes
        );
        if (context.inputScratch == nil || context.outputScratch == nil ||
            weight_buffer == nil) return false;

        unsigned char *input_destination =
            (unsigned char *)context.inputScratch.contents;
        size_t packed_input_bytes = (size_t)input_width * sizeof(float);
        for (int row = 0; row < count; row++) {
            memcpy(
                input_destination + (NSUInteger)row * input_row_bytes,
                inputs + (size_t)row * (size_t)input_width,
                packed_input_bytes
            );
        }

        MPSMatrixDescriptor *input_descriptor = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(NSUInteger)count
            columns:(NSUInteger)input_width
            rowBytes:input_row_bytes
            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *weight_descriptor = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(NSUInteger)output_width
            columns:(NSUInteger)input_width
            rowBytes:input_row_bytes
            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *output_descriptor = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(NSUInteger)count
            columns:(NSUInteger)output_width
            rowBytes:output_row_bytes
            dataType:MPSDataTypeFloat32];
        MPSMatrix *input_matrix = [[MPSMatrix alloc]
            initWithBuffer:context.inputScratch
            descriptor:input_descriptor];
        MPSMatrix *weight_matrix = [[MPSMatrix alloc]
            initWithBuffer:weight_buffer
            descriptor:weight_descriptor];
        MPSMatrix *output_matrix = [[MPSMatrix alloc]
            initWithBuffer:context.outputScratch
            descriptor:output_descriptor];
        MPSMatrixMultiplication *multiplication = [[MPSMatrixMultiplication alloc]
            initWithDevice:context.device
            transposeLeft:NO
            transposeRight:YES
            resultRows:(NSUInteger)count
            resultColumns:(NSUInteger)output_width
            interiorColumns:(NSUInteger)input_width
            alpha:1.0
            beta:0.0];
        id<MTLCommandBuffer> command = [context.queue commandBuffer];
        [multiplication encodeToCommandBuffer:command
            leftMatrix:input_matrix
            rightMatrix:weight_matrix
            resultMatrix:output_matrix];
        if (!finish_command(backend, command, "matrix family")) return false;

        const unsigned char *output_source =
            (const unsigned char *)context.outputScratch.contents;
        size_t packed_output_bytes = (size_t)output_width * sizeof(float);
        for (int row = 0; row < count; row++) {
            memcpy(
                outputs + (size_t)row * (size_t)output_width,
                output_source + (NSUInteger)row * output_row_bytes,
                packed_output_bytes
            );
        }
        return true;
    }
}
