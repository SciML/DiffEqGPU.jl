const GROUP = get(ENV, "GROUP", "CUDA")
const backend = if GROUP == "CUDA"
    using CUDA
    CUDA.CUDABackend()
elseif GROUP == "AMDGPU"
    using AMDGPU
    AMDGPU.ROCBackend()
elseif GROUP == "oneAPI"
    using oneAPI
    oneAPI.oneAPIBackend()
elseif GROUP == "Metal"
    using Metal
    Metal.MetalBackend()
elseif GROUP == "CPU"
    using KernelAbstractions
    KernelAbstractions.CPU()
end

import GPUArraysCore
GPUArraysCore.allowscalar(false)

@info "Testing on " backend
