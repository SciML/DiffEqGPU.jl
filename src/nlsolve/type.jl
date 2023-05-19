abstract type AbstractNLSolver end
abstract type AbstractNLSolverCache end

struct NLSolver{uType, gamType, tmpType, tType, JType, WType, pType} <: AbstractNLSolver
    z::uType
    tmp::uType # DIRK and multistep methods only use tmp
    tmp2::tmpType # for GLM if neccssary
    ztmp::uType
    γ::gamType
    c::tType
    α::tType
    κ::tType
    J::JType
    W::WType
    dt::tType
    t::tType
    p::pType
    iter::Int
    maxiters::Int
end

function NLSolver{tType}(z, tmp, ztmp, γ, c, α, κ, J, W, dt, t, p,
                         iter, maxiters, tmp2 = nothing) where {tType}
    NLSolver{typeof(z), typeof(γ), typeof(tmp2), tType, typeof(J), typeof(W), typeof(p)}(z,
                                                                                         tmp,
                                                                                         tmp2,
                                                                                         ztmp,
                                                                                         γ,
                                                                                         convert(tType,
                                                                                                 c),
                                                                                         convert(tType,
                                                                                                 α),
                                                                                         convert(tType,
                                                                                                 κ),
                                                                                         J,
                                                                                         W,
                                                                                         dt,
                                                                                         t,
                                                                                         p,
                                                                                         iter,
                                                                                         maxiters)
end

@inline function build_J_W(f, γ, dt)
    J(u, p, t) =
        if f.jac !== nothing
            ForwardDiff.jacobian(u -> f(u, p, t), u)
        else
            f.jac(u, p, t)
        end
    W(u, p, t) = -LinearAlgebra.I + γ * dt * J(u, p, t)
    J, W
end

@inline function build_nlsolver(u, p,
                                t, dt,
                                f,
                                γ, c)
    build_nlsolver(u, p, t, dt, f, γ, c, 1)
end

@inline function build_nlsolver(u, p,
                                t, dt,
                                f,
                                γ, c, α)
    # define fields of non-linear solver
    z = u
    tmp = u
    ztmp = u
    J, W = build_J_W(f, γ, dt)
    # build non-linear solver
    tt = ForwardDiff.jacobian(u -> f(u, p, t), SVector{3}(1.0f0, 1.0f0, 1.0f0))
    # KernelAbstractions.@print(typeof(tt))
    max_iter = 30
    κ = 1 / 100
    NLSolver{typeof(dt)}(z, tmp, ztmp, γ, c, α, κ,
                         J, W, dt, t, p, 0, max_iter)
end
