module CUDAExt
isdefined(Base, :get_extension) ? (using CUDAKernels) : (using ..CUDAKernels)
import DiffEqGPU

# import via parent
import ..CUDAKernels: CUDA, KernelAbstractions
import .KernelAbstractions: Adapt
using .CUDA, .Adapt

function DiffEqGPU.EnsembleGPUArray(cpu_offload::Float64)
    DiffEqGPU.EnsembleGPUArray(CUDADevice(), cpu_offload)
end
DiffEqGPU.maxthreads(::CUDADevice) = 256
DiffEqGPU.maybe_prefer_blocks(::CUDADevice) = CUDADevice(; prefer_blocks = true)

# TODO move to KA
Adapt.adapt_storage(::KernelAbstractions.CPU, a::CuArray) = adapt(Array, a)
Adapt.adapt_storage(::CUDADevice, a::CuArray) = a
Adapt.adapt_storage(::CUDADevice, a::Array) = adapt(CuArray, a)

DiffEqGPU.allocate(::CUDADevice, ::Type{T}, init, dims) where {T} = CuArray{T}(init, dims)

function DiffEqGPU.lufact!(::CUDADevice, W)
    CUDA.CUBLAS.getrf_strided_batched!(W, false)
    return nothing
end

end
