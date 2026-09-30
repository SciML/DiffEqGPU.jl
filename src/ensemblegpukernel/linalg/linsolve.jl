# Credits: StaticArrays.jl
# https://github.com/JuliaArrays/StaticArrays.jl/blob/master/src/solve.jl

@inline function linear_solve(A::StaticMatrix, b::StaticVecOrMat)
    return _linear_solve(Size(A), Size(b), A, b)
end

@inline function _linear_solve(
        ::Size{(1, 1)},
        ::Size{(1,)},
        a::StaticMatrix{<:Any, <:Any, Ta},
        b::StaticVector{<:Any, Tb}
    ) where {Ta, Tb}
    @inbounds return similar_type(b, typeof(a[1] \ b[1]))(a[1] \ b[1])
end

@inline function _linear_solve(
        ::Size{(2, 2)},
        ::Size{(2,)},
        a::StaticMatrix{<:Any, <:Any, Ta},
        b::StaticVector{<:Any, Tb}
    ) where {Ta, Tb}
    d = det(a)
    T = typeof((one(Ta) * zero(Tb) + one(Ta) * zero(Tb)) / d)
    @inbounds return similar_type(b, T)(
        (a[2, 2] * b[1] - a[1, 2] * b[2]) / d,
        (a[1, 1] * b[2] - a[2, 1] * b[1]) / d
    )
end

@inline function _linear_solve(
        ::Size{(3, 3)},
        ::Size{(3,)},
        a::StaticMatrix{<:Any, <:Any, Ta},
        b::StaticVector{<:Any, Tb}
    ) where {Ta, Tb}
    d = det(a)
    T = typeof((one(Ta) * zero(Tb) + one(Ta) * zero(Tb)) / d)
    @inbounds return similar_type(b, T)(
        (
            (a[2, 2] * a[3, 3] - a[2, 3] * a[3, 2]) * b[1] +
                (a[1, 3] * a[3, 2] - a[1, 2] * a[3, 3]) * b[2] +
                (a[1, 2] * a[2, 3] - a[1, 3] * a[2, 2]) * b[3]
        ) /
            d,
        (
            (a[2, 3] * a[3, 1] - a[2, 1] * a[3, 3]) * b[1] +
                (a[1, 1] * a[3, 3] - a[1, 3] * a[3, 1]) * b[2] +
                (a[1, 3] * a[2, 1] - a[1, 1] * a[2, 3]) * b[3]
        ) / d,
        (
            (a[2, 1] * a[3, 2] - a[2, 2] * a[3, 1]) * b[1] +
                (a[1, 2] * a[3, 1] - a[1, 1] * a[3, 2]) * b[2] +
                (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1]) * b[3]
        ) / d
    )
end

for Sa in [(2, 2), (3, 3)]  # not needed for Sa = (1, 1);
    @eval begin
        @inline function _linear_solve(
                ::Size{$Sa},
                ::Size{Sb},
                a::StaticMatrix{<:Any, <:Any, Ta},
                b::StaticMatrix{<:Any, <:Any, Tb}
            ) where {Sb, Ta, Tb}
            d = det(a)
            T = typeof((one(Ta) * zero(Tb) + one(Ta) * zero(Tb)) / d)
            if isbitstype(T)
                # This if block can be removed when https://github.com/JuliaArrays/StaticArrays.jl/pull/749 is merged.
                c = similar(b, T)
                for col in 1:Sb[2]
                    @inbounds c[:, col] = _linear_solve(
                        Size($Sa),
                        Size($Sa[1]),
                        a,
                        b[:, col]
                    )
                end
                return similar_type(b, T)(c)
            else
                return _linear_solve_general($(Size(Sa)), Size(Sb), a, b)
            end
        end
    end # @eval
end

@inline function _linear_solve(sa::Size, sb::Size, a::StaticMatrix, b::StaticVecOrMat)
    return _linear_solve_general(sa, sb, a, b)
end

@generated function _linear_solve_general(
        ::Size{Sa},
        ::Size{Sb},
        a::StaticMatrix{<:Any, <:Any, Ta},
        b::StaticVecOrMat{Tb}
    ) where {Sa, Sb, Ta, Tb}
    if Sa[1] != Sb[1]
        return quote
            throw(DimensionMismatch("Left and right hand side first dimensions do not match in backdivide (got sizes $Sa and $Sb)"))
        end
    end
    return quote
        @_inline_meta
        LUp = static_lu(a)
        LUp.U \ (LUp.L \ $(length(Sb) > 1 ? :(b[LUp.p, :]) : :(b[LUp.p])))
    end
end

const _KVAERNO_SINGULAR_MASS_MSG = "GPUKvaerno3 and GPUKvaerno5 do not support DAEs (singular mass matrices). Use GPURosenbrock23, GPURodas4, or GPURodas5P instead."

@inline function _check_kvaerno_mass_matrix(alg, mass_matrix)
    if alg isa Union{GPUKvaerno3, GPUKvaerno5} && _is_singular_mass_matrix(mass_matrix)
        throw(ArgumentError(_KVAERNO_SINGULAR_MASS_MSG))
    end
    return nothing
end

@inline _is_singular_mass_matrix(::LinearAlgebra.UniformScaling{Bool}) = false
@inline _is_singular_mass_matrix(M::LinearAlgebra.UniformScaling) = iszero(M.λ)
@inline _is_singular_mass_matrix(M) = iszero(det(M))

# Explicit first-stage / FSAL derivative for ESDIRK: solve M * du = f(u,p,t).
@inline function _kvaerno_explicit_du(f, u, p, t)
    return _mass_matrix_solve_du(f.mass_matrix, f(u, p, t))
end

@inline _mass_matrix_solve_du(::LinearAlgebra.UniformScaling{Bool}, rhs) = rhs
@inline function _mass_matrix_solve_du(M::LinearAlgebra.UniformScaling, rhs)
    iszero(M.λ) && throw(ArgumentError(_KVAERNO_SINGULAR_MASS_MSG))
    return rhs / M.λ
end
@inline function _mass_matrix_solve_du(M::StaticMatrix, rhs)
    iszero(det(M)) && throw(ArgumentError(_KVAERNO_SINGULAR_MASS_MSG))
    return linear_solve(M, rhs)
end
