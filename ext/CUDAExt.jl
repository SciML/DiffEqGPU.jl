module CUDAExt
using CUDA
import DiffEqGPU

using .CUDA
import .CUDA: CUDABackend

function DiffEqGPU.EnsembleGPUArray(cpu_offload::Float64)
    DiffEqGPU.EnsembleGPUArray(CUDABackend(), cpu_offload)
end
DiffEqGPU.maxthreads(::CUDABackend) = 256
DiffEqGPU.maybe_prefer_blocks(::CUDABackend) = CUDABackend(; prefer_blocks = true)

function DiffEqGPU.lufact!(::CUDABackend, W)
    CUDA.CUBLAS.getrf_strided_batched!(W, false)
    return nothing
end

end
