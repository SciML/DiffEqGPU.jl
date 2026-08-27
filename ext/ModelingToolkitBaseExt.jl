module ModelingToolkitBaseExt

using ModelingToolkitBase: MTKParameters
using StaticArraysCore: SArray, StaticArray
import DiffEqGPU

function static_parameter_storage(x::StaticArray)
    values = map(static_parameter_storage, x)
    return SArray{Tuple{size(x)...}}(values)
end
function static_parameter_storage(x::Array)
    values = map(static_parameter_storage, x)
    return SArray{Tuple{size(x)...}}(values)
end
static_parameter_storage(x::Tuple) = map(static_parameter_storage, x)
static_parameter_storage(x::NamedTuple) = map(static_parameter_storage, x)
static_parameter_storage(x) = x

function DiffEqGPU.make_parameter_compatible(p::MTKParameters)
    return MTKParameters(
        static_parameter_storage(p.tunable),
        static_parameter_storage(p.initials),
        static_parameter_storage(p.discrete),
        static_parameter_storage(p.constant),
        static_parameter_storage(p.nonnumeric),
        static_parameter_storage(p.caches)
    )
end

end
