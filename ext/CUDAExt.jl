module CUDAExt
isdefined(Base, :get_extension) ? (using CUDA) : (using ..CUDA)
import DiffEqGPU

# import via parent
# import ..CUDAKernels: CUDA, KernelAbstractions
using .CUDA
import .CUDA: CUDABackend

function DiffEqGPU.EnsembleGPUArray(cpu_offload::Float64)
    DiffEqGPU.EnsembleGPUArray(CUDABackend(), cpu_offload)
end
DiffEqGPU.maxthreads(::CUDABackend) = 256
DiffEqGPU.maybe_prefer_blocks(::CUDABackend) = CUDABackend(; prefer_blocks = true)

# TODO move to KA
# Adapt.adapt_storage(::KernelAbstractions.CPU, a::CuArray) = adapt(Array, a)
# Adapt.adapt_storage(::CUDADevice, a::CuArray) = a
# Adapt.adapt_storage(::CUDADevice, a::Array) = adapt(CuArray, a)

# DiffEqGPU.allocate(::CUDADevice, ::Type{T}, init, dims) where {T} = CuArray{T}(init, dims)
# DiffEqGPU.supports(::CUDADevice, ::Type{Float64}) = true

function DiffEqGPU.lufact!(::CUDABackend, W)
    CUDA.CUBLAS.getrf_strided_batched!(W, false)
    return nothing
end

end
