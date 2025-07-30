
#Credits: StaticArrays.jl
# https://github.com/JuliaArrays/StaticArrays.jl/blob/master/src/lu.jl

# LU decomposition
pivot_options = if isdefined(LinearAlgebra, :PivotingStrategy) # introduced in Julia v1.7
    (:(Val{true}), :(Val{false}), :NoPivot, :RowMaximum)
else
    (:(Val{true}), :(Val{false}))
end
for pv in pivot_options
    # ... define each `pivot::Val{true/false}` method individually to avoid ambiguties
    @eval function static_lu(A::StaticLUMatrix, pivot::$pv; check = true)
        L, U, p = _lu(A, pivot, check)
        LU(L, U, p)
    end

    # For the square version, return explicit lower and upper triangular matrices.
    # We would do this for the rectangular case too, but Base doesn't support that.
    @eval function static_lu(A::StaticLUMatrix{N, N}, pivot::$pv; check = true) where {N}
        L, U, p = _lu(A, pivot, check)
        LU(LowerTriangular(L), UpperTriangular(U), p)
    end
end
static_lu(A::StaticLUMatrix; check = true) = static_lu(A, Val(true); check = check)

# location of the first zero on the diagonal, 0 when not found
function _first_zero_on_diagonal(A::StaticLUMatrix{M, N, T}) where {M, N, T}
    if @generated
        quote
            $(map(i -> :(A[$i, $i] == zero(T) && return $i), 1:min(M, N))...)
            0
        end
    else
        for i in 1:min(M, N)
            A[i, i] == 0 && return i
        end
        0
    end
end

@generated function _lu(A::StaticLUMatrix{M, N, T}, pivot, check) where {M, N, T}
    _pivot = if isdefined(LinearAlgebra, :PivotingStrategy) # v1.7 feature
        pivot === RowMaximum ? Val(true) : pivot === NoPivot ? Val(false) : pivot()
    else
        pivot()
    end
    quote
        L, U, P = __lu(A, $(_pivot))
        if check
            i = _first_zero_on_diagonal(U)
            i == 0 || throw(SingularException(i))
        end
        L, U, P
    end
end

function __lu(A::StaticMatrix{0, 0, T}, ::Val{Pivot}) where {T, Pivot}
    (SMatrix{0, 0, typeof(one(T))}(), A, SVector{0, Int}())
end

function __lu(A::StaticMatrix{0, 1, T}, ::Val{Pivot}) where {T, Pivot}
    (SMatrix{0, 0, typeof(one(T))}(), A, SVector{0, Int}())
end

function __lu(A::StaticMatrix{0, N, T}, ::Val{Pivot}) where {T, N, Pivot}
    (SMatrix{0, 0, typeof(one(T))}(), A, SVector{0, Int}())
end

function __lu(A::StaticMatrix{1, 0, T}, ::Val{Pivot}) where {T, Pivot}
    (SMatrix{1, 0, typeof(one(T))}(), SMatrix{0, 0, T}(), SVector{1, Int}(1))
end

function __lu(A::StaticMatrix{M, 0, T}, ::Val{Pivot}) where {T, M, Pivot}
    (SMatrix{M, 0, typeof(one(T))}(), SMatrix{0, 0, T}(), SVector{M, Int}(1:M))
end

function __lu(A::StaticMatrix{1, 1, T}, ::Val{Pivot}) where {T, Pivot}
    (SMatrix{1, 1}(one(T)), A, SVector(1))
end

function __lu(A::LinearAlgebra.HermOrSym{T, <:StaticMatrix{1, 1, T}},
        ::Val{Pivot}) where {T, Pivot}
    (SMatrix{1, 1}(one(T)), A.data, SVector(1))
end

function __lu(A::StaticMatrix{1, N, T}, ::Val{Pivot}) where {N, T, Pivot}
    (SMatrix{1, 1, T}(one(T)), A, SVector{1, Int}(1))
end

function __lu(A::StaticMatrix{M, 1}, ::Val{Pivot}) where {M, Pivot}
    @inbounds begin
        kp = 1
        if Pivot
            amax = abs(A[1, 1])
            for i in 2:M
                absi = abs(A[i, 1])
                if absi > amax
                    kp = i
                    amax = absi
                end
            end
        end
        ps = tailindices(Val{M})
        if kp != 1
            # Swap elements: put 1 at position kp-1, and kp at the first position  
            ps_array = [ps...]
            ps_array[kp - 1] = 1
            ps = SVector{length(ps)}(ps_array)
        end
        U = SMatrix{1, 1}(A[kp, 1])
        # Scale first column
        Akkinv = inv(A[kp, 1])
        Ls = A[ps, 1] * Akkinv
        if !isfinite(Akkinv)
            Ls = zeros(typeof(Ls))
        end
        L = [SVector{1}(one(eltype(Ls))); Ls]
        p = [SVector{1, Int}(kp); ps]
    end
    return (SMatrix{M, 1}(L), U, p)
end

function __lu(A::StaticLUMatrix{M, N, T}, ::Val{Pivot}) where {M, N, T, Pivot}
    @inbounds begin
        kp = 1
        if Pivot
            amax = abs(A[1, 1])
            for i in 2:M
                absi = abs(A[i, 1])
                if absi > amax
                    kp = i
                    amax = absi
                end
            end
        end
        ps = tailindices(Val{M})
        if kp != 1
            # Swap elements: put 1 at position kp-1, and kp at the first position
            ps_array = [ps...]
            ps_array[kp - 1] = 1  
            ps = SVector{length(ps)}(ps_array)
        end
        Ufirst = SMatrix{1, N}(A[kp, :])
        # Scale first column
        Akkinv = inv(A[kp, 1])
        Ls = A[ps, 1] * Akkinv
        if !isfinite(Akkinv)
            Ls = zeros(typeof(Ls))
        end

        # Update the rest
        Arest = A[ps, tailindices(Val{N})] - Ls * Ufirst[:, tailindices(Val{N})]
        Lrest, Urest, prest = __lu(Arest, Val(Pivot))
        p = [SVector{1, Int}(kp); ps[prest]]
        L = [[SVector{1}(one(eltype(Ls))); Ls[prest]] [zeros(typeof(SMatrix{1}(Lrest[1,
                                                           :])));
                                                       Lrest]]
        U = [Ufirst; [zeros(typeof(Urest[:, 1])) Urest]]
    end
    return (L, U, p)
end

@generated function tailindices(::Type{Val{M}}) where {M}
    :(SVector{$(M - 1), Int}($(tuple(2:M...))))
end
