module oneAPIExt
isdefined(Base, :get_extension) ? (using oneAPIKernels) : (using ..oneAPIKernels)
import DiffEqGPU

# import via parent
import ..oneAPIKernels: oneAPI, KernelAbstractions
import .KernelAbstractions: Adapt
using .oneAPI


DiffEqGPU.maxthreads(::oneAPIDevice) = 256
DiffEqGPU.maybe_prefer_blocks(::oneAPIDevice) = oneAPIDevice()

# TODO move to KA
Adapt.adapt_storage(::KernelAbstractions.CPU, a::oneArray) = adapt(Array, a)
Adapt.adapt_storage(::oneAPIDevice, a::oneArray) = a
Adapt.adapt_storage(::oneAPIDevice, a::Array) = adapt(oneArray, a)

function DiffEqGPU.allocate(::oneAPIDevice, ::Type{T}, init, dims) where {T}
    oneArray{T}(init, dims)
end

end
