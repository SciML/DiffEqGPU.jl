module oneAPIExt
using KernelAbstractions
using oneAPI, oneAPIKernels, Adapt
import DiffEqGPU

DiffEqGPU.maxthreads(::oneAPIDevice) = 256
DiffEqGPU.maybe_prefer_blocks(::oneAPIDevice) = oneAPIDevice()

# TODO move to KA
Adapt.adapt_storage(::CPU, a::oneArray) = adapt(Array, a)
Adapt.adapt_storage(::oneAPIDevice, a::oneArray) = a
Adapt.adapt_storage(::oneAPIDevice, a::Array) = adapt(oneArray, a)

function DiffEqGPU.allocate(::oneAPIDevice, ::Type{T}, init, dims) where {T}
    oneArray{T}(init, dims)
end

end
