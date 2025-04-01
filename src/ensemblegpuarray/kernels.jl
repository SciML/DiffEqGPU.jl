"""
Wrapper for modifying parameters to contain additional data. Useful for simulating
trajectories with different time span values.
"""
struct ParamWrapper{P, T}
    params::P
    data::T
end

function Adapt.adapt_structure(to, ps::ParamWrapper{P, T}) where {P, T}
    ParamWrapper(adapt(to, ps.params),
        adapt(to, ps.data))
end

# The reparameterization is adapted from:https://github.com/rtqichen/torchdiffeq/issues/122#issuecomment-738978844
@kernel function gpu_kernel(f, du, @Const(u),
        @Const(params::AbstractArray{ParamWrapper{P, T}}),
        @Const(t)) where {P, T}
    i = @index(Global, Linear)
    @inbounds p = params[i].params
    @inbounds tspan = params[i].data
    # reparameterization t->(t_0, t_f) from t->(0, 1).
    t = (tspan[2] - tspan[1]) * t + tspan[1]
    @views @inbounds f(du[:, i], u[:, i], p, t)
    @inbounds for j in 1:size(du, 1)
        du[j, i] = du[j, i] * (tspan[2] - tspan[1])
    end
end

@kernel function gpu_kernel_oop(f, du, @Const(u),
        @Const(params::AbstractArray{ParamWrapper{P, T}}),
        @Const(t)) where {P, T}
    i = @index(Global, Linear)
    @inbounds p = params[i].params
    @inbounds tspan = params[i].data
    # reparameterization
    t = (tspan[2] - tspan[1]) * t + tspan[1]
    @views @inbounds x = f(u[:, i], p, t)
    @inbounds for j in 1:size(du, 1)
        du[j, i] = x[j] * (tspan[2] - tspan[1])
    end
end

@kernel function gpu_kernel(f, du, @Const(u), @Const(p), @Const(t))
    i = @index(Global, Linear)
    if eltype(p) <: Number
        @views @inbounds f(du[:, i], u[:, i], p[:, i], t)
    else
        @views @inbounds f(du[:, i], u[:, i], p[i], t)
    end
end

@kernel function gpu_kernel_oop(f, du, @Const(u), @Const(p), @Const(t))
    i = @index(Global, Linear)
    if eltype(p) <: Number
        @views @inbounds x = f(u[:, i], p[:, i], t)
    else
        @views @inbounds x = f(u[:, i], p[i], t)
    end
    @inbounds for j in 1:size(du, 1)
        du[j, i] = x[j]
    end
end

@kernel function jac_kernel(f, J, @Const(u),
        @Const(params::AbstractArray{ParamWrapper{P, T}}),
        @Const(t)) where {P, T}
    i = @index(Global, Linear) - 1
    section = (1 + (i * size(u, 1))):((i + 1) * size(u, 1))
    @inbounds p = params[i + 1].params
    @inbounds tspan = params[i + 1].data

    # reparameterization
    t = (tspan[2] - tspan[1]) * t + tspan[1]

    @views @inbounds f(J[section, section], u[:, i + 1], p, t)
    @inbounds for j in section, k in section
        J[k, j] = J[k, j] * (tspan[2] - tspan[1])
    end
end

@kernel function jac_kernel_oop(f, J, @Const(u),
        @Const(params::AbstractArray{ParamWrapper{P, T}}),
        @Const(t)) where {P, T}
    i = @index(Global, Linear) - 1
    section = (1 + (i * size(u, 1))):((i + 1) * size(u, 1))

    @inbounds p = params[i + 1].params
    @inbounds tspan = params[i + 1].data

    # reparameterization
    t = (tspan[2] - tspan[1]) * t + tspan[1]

    @views @inbounds x = f(u[:, i + 1], p, t)

    @inbounds for j in section, k in section
        J[k, j] = x[k, j] * (tspan[2] - tspan[1])
    end
end

@kernel function jac_kernel(f, J, @Const(u), @Const(p), @Const(t))
    i = @index(Global, Linear) - 1
    section = (1 + (i * size(u, 1))):((i + 1) * size(u, 1))
    if eltype(p) <: Number
        @views @inbounds f(J[section, section], u[:, i + 1], p[:, i + 1], t)
    else
        @views @inbounds f(J[section, section], u[:, i + 1], p[i + 1], t)
    end
end

@kernel function jac_kernel_oop(f, J, @Const(u), @Const(p), @Const(t))
    i = @index(Global, Linear) - 1
    section = (1 + (i * size(u, 1))):((i + 1) * size(u, 1))
    if eltype(p) <: Number
        @views @inbounds x = f(u[:, i + 1], p[:, i + 1], t)
    else
        @views @inbounds x = f(u[:, i + 1], p[i + 1], t)
    end
    @inbounds for j in section, k in section
        J[k, j] = x[k, j]
    end
end

@kernel function discrete_condition_kernel(condition, cur, @Const(u), @Const(t), @Const(p))
    i = @index(Global, Linear)
    @views @inbounds cur[i] = condition(u[:, i], t, FakeIntegrator(u[:, i], t, p[:, i]))
end

@kernel function discrete_affect!_kernel(affect!, cur, u, t, p)
    i = @index(Global, Linear)
    @views @inbounds cur[i] && affect!(FakeIntegrator(u[:, i], t, p[:, i]))
end

@kernel function continuous_condition_kernel(condition, out, @Const(u), @Const(t),
        @Const(p))
    i = @index(Global, Linear)
    @views @inbounds out[i] = condition(u[:, i], t, FakeIntegrator(u[:, i], t, p[:, i]))
end

@kernel function continuous_affect!_kernel(affect!, event_idx, u, t, p)
    for i in event_idx
        @views @inbounds affect!(FakeIntegrator(u[:, i], t, p[:, i]))
    end
end

maxthreads(::CPU) = 1024
maybe_prefer_blocks(::CPU) = CPU()

function workgroupsize(backend, n)
    min(maxthreads(backend), n)
end

@kernel function W_kernel(jac, W, @Const(u),
        @Const(params::AbstractArray{ParamWrapper{P, T}}), @Const(gamma),
        @Const(t)) where {P, T}
    i = @index(Global, Linear)
    len = size(u, 1)
    _W = @inbounds @view(W[:, :, i])

    @inbounds p = params[i].params
    @inbounds tspan = params[i].data

    # reparameterization
    t = (tspan[2] - tspan[1]) * t + tspan[1]

    @views @inbounds jac(_W, u[:, i], p, t)

    @inbounds for i in eachindex(_W)
        _W[i] = gamma * _W[i] * (tspan[2] - tspan[1])
    end
    _one = one(eltype(_W))
    @inbounds for i in 1:len
        _W[i, i] = _W[i, i] - _one
    end
end

@kernel function W_kernel(jac, W, @Const(u), @Const(p), @Const(gamma), @Const(t))
    i = @index(Global, Linear)
    len = size(u, 1)
    _W = @inbounds @view(W[:, :, i])
    @views @inbounds jac(_W, u[:, i], p[:, i], t)
    @inbounds for i in eachindex(_W)
        _W[i] = gamma * _W[i]
    end
    _one = one(eltype(_W))
    @inbounds for i in 1:len
        _W[i, i] = _W[i, i] - _one
    end
end

@kernel function W_kernel_oop(jac, W, @Const(u),
        @Const(params::AbstractArray{ParamWrapper{P, T}}),
        @Const(gamma),
        @Const(t)) where {P, T}
    i = @index(Global, Linear)
    len = size(u, 1)

    @inbounds p = params[i].params
    @inbounds tspan = params[i].data

    _W = @inbounds @view(W[:, :, i])

    # reparameterization
    t = (tspan[2] - tspan[1]) * t + tspan[1]

    @views @inbounds x = jac(u[:, i], p, t)
    @inbounds for j in 1:length(_W)
        _W[j] = x[j] * (tspan[2] - tspan[1])
    end
    @inbounds for i in eachindex(_W)
        _W[i] = gamma * _W[i]
    end
    _one = one(eltype(_W))
    @inbounds for i in 1:len
        _W[i, i] = _W[i, i] - _one
    end
end

@kernel function W_kernel_oop(jac, W, @Const(u), @Const(p), @Const(gamma), @Const(t))
    i = @index(Global, Linear)
    len = size(u, 1)
    _W = @inbounds @view(W[:, :, i])
    @views @inbounds x = jac(u[:, i], p[:, i], t)
    @inbounds for j in 1:length(_W)
        _W[j] = x[j]
    end
    @inbounds for i in eachindex(_W)
        _W[i] = gamma * _W[i]
    end
    _one = one(eltype(_W))
    @inbounds for i in 1:len
        _W[i, i] = _W[i, i] - _one
    end
end

@kernel function Wt_kernel(
        jac, W, @Const(u), @Const(params::AbstractArray{ParamWrapper{P, T}}),
        @Const(gamma), @Const(t)) where {P, T}
    i = @index(Global, Linear)
    len = size(u, 1)
    @inbounds p = params[i].params
    @inbounds tspan = params[i].data

    # reparameterization
    t = (tspan[2] - tspan[1]) * t + tspan[1]

    _W = @inbounds @view(W[:, :, i])
    @views @inbounds jac(_W, u[:, i], p, t)
    @inbounds for i in 1:len
        _W[i, i] = -inv(gamma) + _W[i, i] * (tspan[2] - tspan[1])
    end
end

@kernel function Wt_kernel_oop(
        jac, W, @Const(u), @Const(params::AbstractArray{ParamWrapper{P, T}}),
        @Const(gamma), @Const(t)) where {P, T}
    i = @index(Global, Linear)
    len = size(u, 1)

    @inbounds p = params[i].params
    @inbounds tspan = params[i].data

    # reparameterization
    t = (tspan[2] - tspan[1]) * t + tspan[1]

    _W = @inbounds @view(W[:, :, i])
    @views @inbounds x = jac(u[:, i], p, t)
    @inbounds for j in 1:length(_W)
        _W[j] = x[j] * (tspan[2] - tspan[1])
    end
    @inbounds for i in 1:len
        _W[i, i] = -inv(gamma) + _W[i, i]
    end
end

@kernel function Wt_kernel(jac, W, @Const(u), @Const(p), @Const(gamma), @Const(t))
    i = @index(Global, Linear)
    len = size(u, 1)
    _W = @inbounds @view(W[:, :, i])
    @views @inbounds jac(_W, u[:, i], p[:, i], t)
    @inbounds for i in 1:len
        _W[i, i] = -inv(gamma) + _W[i, i]
    end
end

@kernel function Wt_kernel_oop(jac, W, @Const(u), @Const(p), @Const(gamma), @Const(t))
    i = @index(Global, Linear)
    len = size(u, 1)
    _W = @inbounds @view(W[:, :, i])
    @views @inbounds x = jac(u[:, i], p[:, i], t)
    @inbounds for j in 1:length(_W)
        _W[j] = x[j]
    end
    @inbounds for i in 1:len
        _W[i, i] = -inv(gamma) + _W[i, i]
    end
end

# @kernel function gpu_kernel_tgrad(f::AbstractArray{T}, du, @Const(u), @Const(p),
#         @Const(t)) where {T}
#     i = @index(Global, Linear)
#     @inbounds f = f[i].tgrad
#     if eltype(p) <: Number
#         @views @inbounds f(du[:, i], u[:, i], p[:, i], t)
#     else
#         @views @inbounds f(du[:, i], u[:, i], p[i], t)
#     end
# end

# @kernel function gpu_kernel_oop_tgrad(f::AbstractArray{T}, du, @Const(u), @Const(p),
#         @Const(t)) where {T}
#     i = @index(Global, Linear)
#     @inbounds f = f[i].tgrad
#     if eltype(p) <: Number
#         @views @inbounds x = f(u[:, i], p[:, i], t)
#     else
#         @views @inbounds x = f(u[:, i], p[i], t)
#     end
#     @inbounds for j in 1:size(du, 1)
#         du[j, i] = x[j]
#     end
# end

function lufact!(::CPU, W)
    len = size(W, 1)
    for i in 1:size(W, 3)
        _W = @inbounds @view(W[:, :, i])
        generic_lufact!(_W, len)
    end
    return nothing
end

struct FakeIntegrator{uType, tType, P}
    u::uType
    t::tType
    p::P
end

### GPU Factorization
"""
A parameter-parallel `SciMLLinearSolveAlgorithm`.
"""
struct LinSolveGPUSplitFactorize <: LinearSolve.SciMLLinearSolveAlgorithm
    len::Int
    nfacts::Int
end
LinSolveGPUSplitFactorize() = LinSolveGPUSplitFactorize(0, 0)

LinearSolve.needs_concrete_A(::LinSolveGPUSplitFactorize) = true

function LinearSolve.init_cacheval(linsol::LinSolveGPUSplitFactorize, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::LinearSolve.OperatorAssumptions)
    LinSolveGPUSplitFactorize(linsol.len, length(u) ÷ linsol.len)
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::LinSolveGPUSplitFactorize,
        args...; kwargs...)
    p = cache.cacheval
    A = cache.A
    b = cache.b
    x = cache.u
    version = get_backend(b)
    copyto!(x, b)
    wgs = workgroupsize(version, p.nfacts)
    # Note that the matrix is already factorized, only ldiv is needed.
    ldiv!_kernel(version)(A, x, p.len, p.nfacts;
        ndrange = p.nfacts,
        workgroupsize = wgs)
    SciMLBase.build_linear_solution(alg, x, nothing, cache)
end

# Old stuff
function (p::LinSolveGPUSplitFactorize)(x, A, b, update_matrix = false; kwargs...)
    version = get_backend(b)
    copyto!(x, b)
    wgs = workgroupsize(version, p.nfacts)
    ldiv!_kernel(version)(A, x, p.len, p.nfacts;
        ndrange = p.nfacts,
        workgroupsize = wgs)
    return nothing
end

function (p::LinSolveGPUSplitFactorize)(::Type{Val{:init}}, f, u0_prototype)
    LinSolveGPUSplitFactorize(size(u0_prototype)...)
end

@kernel function ldiv!_kernel(W, u, @Const(len), @Const(nfacts))
    i = @index(Global, Linear)
    section = (1 + ((i - 1) * len)):(i * len)
    _W = @inbounds @view(W[:, :, i])
    _u = @inbounds @view u[section]
    naivesolve!(_W, _u, len)
end

function generic_lufact!(A::AbstractMatrix{T}, minmn) where {T}
    m = n = minmn
    #@cuprintf "\n\nbefore lufact!\n"
    #__printjac(A, ii)
    #@cuprintf "\n"
    @inbounds for k in 1:minmn
        #@cuprintf "inner factorization loop\n"
        # Scale first column
        Akkinv = inv(A[k, k])
        for i in (k + 1):m
            #@cuprintf "L\n"
            A[i, k] *= Akkinv
        end
        # Update the rest
        for j in (k + 1):n, i in (k + 1):m
            #@cuprintf "U\n"
            A[i, j] -= A[i, k] * A[k, j]
        end
    end
    #@cuprintf "after lufact!"
    #__printjac(A, ii)
    #@cuprintf "\n\n\n"
    return nothing
end

struct MyL{T} # UnitLowerTriangular
    data::T
end
struct MyU{T} # UpperTriangular
    data::T
end

function naivesub!(A::MyU, b::AbstractVector, n)
    x = b
    @inbounds for j in n:-1:1
        xj = x[j] = A.data[j, j] \ b[j]
        for i in (j - 1):-1:1 # counterintuitively 1:j-1 performs slightly better
            b[i] -= A.data[i, j] * xj
        end
    end
    return nothing
end
function naivesub!(A::MyL, b::AbstractVector, n)
    x = b
    @inbounds for j in 1:n
        xj = x[j]
        for i in (j + 1):n
            b[i] -= A.data[i, j] * xj
        end
    end
    return nothing
end

function naivesolve!(A::AbstractMatrix, x::AbstractVector, n)
    naivesub!(MyL(A), x, n)
    naivesub!(MyU(A), x, n)
    return nothing
end
