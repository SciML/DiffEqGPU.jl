const GROUP = get(ENV, "GROUP", "CUDA")
const device = if GROUP == "CUDA"
    using CUDA, CUDAKernels
    CUDA.allowscalar(false)
    CUDADevice()
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

@info "Testing on " device