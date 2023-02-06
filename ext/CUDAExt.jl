module CUDAExt
using KernelAbstractions
using CUDA, CUDAKernels, Adapt
import DiffEqGPU

using CUDA: CuPtr, CU_NULL, Mem, CuDefaultStream
using CUDA: CUBLAS

function DiffEqGPU.EnsembleGPUArray(cpu_offload::Float64)
    DiffEqGPU.EnsembleGPUArray(CUDADevice(), cpu_offload)
end
DiffEqGPU.maxthreads(::CUDADevice) = 256

# TODO move to KA
Adapt.adapt_storage(::CPU, a::CuArray) = adapt(Array, a)
Adapt.adapt_storage(::CUDADevice, a::CuArray) = a
Adapt.adapt_storage(::CUDADevice, a::Array) = adapt(CuArray, a)

DiffEqGPU.allocate(::CUDADevice, ::Type{T}, init, dims) where {T} = CuArray{T}(init, dims)

function DiffEqGPU.lufact!(::CUDADevice, W)
    CUBLAS.getrf_strided_batched!(W, false)
    return nothing
end

function DiffEqGPU.__printjac(A, ii)
    @cuprintf "[%d, %d]\n" ii.offset1 ii.stride2
    @cuprintf "%d %d %d\n%d %d %d\n%d %d %d\n" ii[1, 1] ii[1, 2] ii[1, 3] ii[2, 1] ii[2, 2] ii[2,
                                                                                               3] ii[3,
                                                                                                     1] ii[3,
                                                                                                           2] ii[3,
                                                                                                                 3]
    @cuprintf "%2.2f %2.2f %2.2f\n%2.2f %2.2f %2.2f\n%2.2f %2.2f %2.2f" A[ii[1, 1]] A[ii[1,
                                                                                         2]] A[ii[1,
                                                                                                  3]] A[ii[2,
                                                                                                           1]] A[ii[2,
                                                                                                                    2]] A[ii[2,
                                                                                                                             3]] A[ii[3,
                                                                                                                                      1]] A[ii[3,
                                                                                                                                               2]] A[ii[3,
                                                                                                                                                        3]]
end
end
