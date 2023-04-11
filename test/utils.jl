const GROUP = get(ENV, "GROUP", "CUDA")
const backend = if GROUP == "CUDA"
    using CUDA
    CUDA.CUDABackend()
elseif GROUP == "AMDGPU"
    using AMDGPU, ROCKernels
    ROCDevice()
elseif GROUP == "oneAPI"
    using oneAPI, oneAPIKernels
    oneAPIDevice()
elseif GROUP == "Metal"
    using Metal, MetalKernels
    MetalDevice()
end

import GPUArraysCore
GPUArraysCore.allowscalar(false)

@info "Testing on " backend
