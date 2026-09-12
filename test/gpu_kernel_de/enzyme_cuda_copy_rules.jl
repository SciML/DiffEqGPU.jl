# EnzymeCUDAExt registers unsafe_copyto! augmented_primal for all Ptr/CuPtr eltypes
# but only implements reverse for AbstractFloat. Ensemble problem data uses SVector
# and KernelODEProblem, so reverse must handle those isbits aggregates (via `.+=`).
using CUDA
using DiffEqGPU
using DiffEqGPU: KernelODEProblem
using Enzyme
using Enzyme: EnzymeRules
using StaticArrays: StaticArray

# Float64 ensemble reverse hits Active GPU fill! on the time matrix; treat init as inactive.
EnzymeRules.inactive(::typeof(DiffEqGPU._init_time_matrix!), ::Any, ::Any) = nothing

function _agg_zero!(ptr::Ptr{T}, off::Integer, n::Integer) where {T}
    Base.Libc.memset(ptr + off * sizeof(T), 0, n * sizeof(T))
    return nothing
end
function _agg_zero!(ptr::CuPtr{T}, off::Integer, n::Integer) where {T}
    bytes = reinterpret(CuPtr{UInt8}, ptr + off * sizeof(T))
    CUDA.memset(bytes, UInt8(0), n * sizeof(T))
    return nothing
end

const _AggStridedSubArray{
    T, N, I <: Tuple{
        Vararg{
            Union{
                Base.RangeIndex, Base.ReshapedUnitRange,
                Base.AbstractCartesianIndex,
            },
        },
    },
} = SubArray{T, N, <:Array, I}
const _AggStridedArray{T, N} = Union{Array{T, N}, _AggStridedSubArray{T, N}}

_agg_accumulate!(dst::_AggStridedArray, src::_AggStridedArray) = (dst .+= src; nothing)
_agg_accumulate!(dst::StridedCuArray, src::StridedCuArray) = (dst .+= src; nothing)
_agg_accumulate!(dst::StridedCuArray, src::_AggStridedArray) = _agg_accumulate!(dst, CuArray(src))
_agg_accumulate!(dst::_AggStridedArray, src::StridedCuArray) = _agg_accumulate!(dst, Array(src))

function _agg_accumulate!(
        acc::Union{Ptr, CuPtr}, aoff::Integer,
        val::Union{Ptr, CuPtr}, voff::Integer, n::Integer
    )
    dst = acc isa CuPtr ? unsafe_wrap(CuArray, acc + aoff, n; own = false) :
        unsafe_wrap(Array, acc + aoff, n; own = false)
    src = val isa CuPtr ? unsafe_wrap(CuArray, val + voff, n; own = false) :
        unsafe_wrap(Array, val + voff, n; own = false)
    _agg_accumulate!(dst, src)
    return nothing
end

@inline function _agg_shadow(x, config, batch)
    return EnzymeRules.width(config) == 1 ? x.dval : x.dval[batch]
end

const _AGG_PTR_COPY_DIRECTIONS = (
    (Ptr, CuPtr),
    (CuPtr, Ptr),
    (CuPtr, CuPtr),
)

const _AggCopyEltype = Union{StaticArray, KernelODEProblem}

# EnzymeCUDAExt only registers the array-level unsafe_copyto! rules for
# AbstractFloat/Complex eltypes, so host↔device copies of aggregate records need
# their own augmented_primal + reverse (the Ptr-level rules above are only hit
# once Enzyme successfully descends through the array methods).
const _AGG_ARRAY_COPY_DIRECTIONS = (
    (StridedCuArray, StridedCuArray),
    (StridedCuArray, Array),
    (Array, StridedCuArray),
)

for (DstArr, SrcArr) in _AGG_ARRAY_COPY_DIRECTIONS
    @eval begin
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfig,
                func::Const{typeof(Base.unsafe_copyto!)},
                ::Type{RT},
                dest::Annotation{<:$DstArr{T}},
                src::Annotation{<:$SrcArr{T}},
                n::Const;
                kwargs...,
            ) where {RT, T <: _AggCopyEltype}
            func.val(dest.val, src.val, n.val; kwargs...)
            primal = EnzymeRules.needs_primal(config) ? dest.val : nothing
            shadow = if !(RT <: Const) && EnzymeRules.needs_shadow(config) &&
                    !(dest isa Const)
                dest.dval
            else
                nothing
            end
            return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
        end

        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfig,
                func::Const{typeof(Base.unsafe_copyto!)},
                ::Type{RT},
                tape,
                dest::Annotation{<:$DstArr{T}},
                src::Annotation{<:$SrcArr{T}},
                n::Const;
                kwargs...,
            ) where {RT, T <: _AggCopyEltype}
            if !(dest isa Const)
                for batch in 1:EnzymeRules.width(config)
                    ddest = _agg_shadow(dest, config, batch)
                    if !(src isa Const)
                        dsrc = _agg_shadow(src, config, batch)
                        _agg_accumulate!(
                            @view(dsrc[1:n.val]), @view(ddest[1:n.val])
                        )
                    end
                    _agg_zero!(pointer(ddest), 0, n.val)
                end
            end
            return (nothing, nothing, nothing)
        end

        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfig,
                func::Const{typeof(Base.unsafe_copyto!)},
                ::Type{RT},
                dest::Annotation{<:$DstArr{T}},
                doffs::Const,
                src::Annotation{<:$SrcArr{T}},
                soffs::Const,
                n::Const;
                kwargs...,
            ) where {RT, T <: _AggCopyEltype}
            func.val(dest.val, doffs.val, src.val, soffs.val, n.val; kwargs...)
            primal = EnzymeRules.needs_primal(config) ? dest.val : nothing
            shadow = if !(RT <: Const) && EnzymeRules.needs_shadow(config) &&
                    !(dest isa Const)
                dest.dval
            else
                nothing
            end
            return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
        end

        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfig,
                func::Const{typeof(Base.unsafe_copyto!)},
                ::Type{RT},
                tape,
                dest::Annotation{<:$DstArr{T}},
                doffs::Const,
                src::Annotation{<:$SrcArr{T}},
                soffs::Const,
                n::Const;
                kwargs...,
            ) where {RT, T <: _AggCopyEltype}
            if !(dest isa Const)
                for batch in 1:EnzymeRules.width(config)
                    ddest = _agg_shadow(dest, config, batch)
                    if !(src isa Const)
                        dsrc = _agg_shadow(src, config, batch)
                        _agg_accumulate!(
                            @view(dsrc[soffs.val:(soffs.val + n.val - 1)]),
                            @view(ddest[doffs.val:(doffs.val + n.val - 1)])
                        )
                    end
                    _agg_zero!(pointer(ddest, doffs.val), 0, n.val)
                end
            end
            return (nothing, nothing, nothing, nothing, nothing)
        end
    end
end

for (DstPtr, SrcPtr) in _AGG_PTR_COPY_DIRECTIONS
    @eval begin
        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfig,
                func::Const{typeof(Base.unsafe_copyto!)},
                ::Type{RT},
                tape,
                dest::Annotation{<:$DstPtr{T}},
                src::Annotation{<:$SrcPtr{T}},
                n::Const;
                kwargs...,
            ) where {RT, T <: _AggCopyEltype}
            if !(dest isa Const)
                for batch in 1:EnzymeRules.width(config)
                    ddest = _agg_shadow(dest, config, batch)
                    if !(src isa Const)
                        dsrc = _agg_shadow(src, config, batch)
                        _agg_accumulate!(dsrc, 0, ddest, 0, n.val)
                    end
                    _agg_zero!(ddest, 0, n.val)
                end
            end
            return (nothing, nothing, nothing)
        end
    end
end
