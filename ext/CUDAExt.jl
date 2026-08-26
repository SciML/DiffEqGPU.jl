module CUDAExt
using CUDA: CUDA, CuArray, CUDABackend
import DiffEqGPU
import DiffEqBase: findall_events!

function DiffEqGPU.EnsembleGPUArray(cpu_offload::Float64)
    return DiffEqGPU.EnsembleGPUArray(CUDABackend(), cpu_offload)
end
DiffEqGPU.maxthreads(::CUDABackend) = 256
DiffEqGPU.maybe_prefer_blocks(::CUDABackend) = CUDABackend(; prefer_blocks = true)

function DiffEqGPU.lufact!(::CUDABackend, W)
    CUDA.CUBLAS.getrf_strided_batched!(W, false)
    return nothing
end

# DiffEqBase only defines `findall_events!` for CPU arrays, but
# `EnsembleGPUArray` keeps callback caches on the GPU.
function findall_events!(next_sign::CuArray, prev_sign::CuArray)
    next_cpu = Array(next_sign)
    prev_cpu = Array(prev_sign)
    event_occurred = findall_events!(next_cpu, prev_cpu)
    copyto!(next_sign, next_cpu)
    return event_occurred
end

end
