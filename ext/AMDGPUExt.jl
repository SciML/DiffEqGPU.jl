module AMDGPUExt
using KernelAbstractions
using AMDGPU, ROCKernels, Adapt
import DiffEqGPU

function DiffEqGPU.EnsembleGPUArray(cpu_offload::Float64)
    DiffEqGPU.EnsembleGPUArray(ROCDevice(), cpu_offload)
end

DiffEqGPU.maxthreads(::ROCDevice) = 256
DiffEqGPU.maybe_prefer_blocks(::ROCDevice) = ROCDevice()

# TODO move to KA
Adapt.adapt_storage(::CPU, a::ROCArray) = adapt(Array, a)
Adapt.adapt_storage(::ROCDevice, a::ROCArray) = a
Adapt.adapt_storage(::ROCDevice, a::Array) = adapt(ROCArray, a)

DiffEqGPU.allocate(::ROCDevice, ::Type{T}, init, dims) where {T} = ROCArray{T}(init, dims)

end
