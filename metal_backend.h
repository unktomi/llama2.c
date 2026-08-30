#ifndef METAL_BACKEND_H
#define METAL_BACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AtkeyMetalBackend AtkeyMetalBackend;

AtkeyMetalBackend *atkey_metal_backend_new(const char *library_path);
void atkey_metal_backend_free(AtkeyMetalBackend *backend);

const char *atkey_metal_backend_device_name(AtkeyMetalBackend *backend);
uint64_t atkey_metal_backend_dispatch_count(AtkeyMetalBackend *backend);
uint64_t atkey_metal_backend_weight_upload_count(AtkeyMetalBackend *backend);
uint64_t atkey_metal_backend_weight_upload_bytes(AtkeyMetalBackend *backend);

bool atkey_metal_embedding_family(
    AtkeyMetalBackend *backend,
    float *outputs,
    const int *tokens,
    int count,
    const float *weights,
    int vocab_size,
    int dim
);

bool atkey_metal_rms_family(
    AtkeyMetalBackend *backend,
    float *outputs,
    const float *inputs,
    int count,
    const float *weights,
    int width
);

bool atkey_metal_matmul_family(
    AtkeyMetalBackend *backend,
    float *outputs,
    const float *inputs,
    int count,
    const float *weights,
    int input_width,
    int output_width
);

#ifdef __cplusplus
}
#endif

#endif
