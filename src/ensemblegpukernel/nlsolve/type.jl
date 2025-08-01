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

@inline function build_J_W(alg, f, γ, dt)
    J(u, p, t) =
        if DiffEqBase.has_jac(f)
            f.jac(u, p, t)
        elseif alg_autodiff(alg)
            ForwardDiff.jacobian(u -> f(u, p, t), u)
        else
            finite_diff_jac(u -> f(u, p, t), f.jac_prototype, u)
        end
    W(u, p, t) = -f.mass_matrix + γ * dt * J(u, p, t)
    J, W
end

@inline function build_tgrad(alg, f)
    function tgrad(u, p, t)
        if DiffEqBase.has_tgrad(f)
            f.tgrad(u, p, t)
        elseif alg_autodiff(alg)
            ForwardDiff.derivative(t -> f(u, p, t), t)
        else
            # derivative using finite difference
            begin
                dt = sqrt(eps(eltype(t)))
                (f(u, p, t + dt) - f(u, p, t)) / dt
            end
        end
    end
    tgrad
end

@inline function build_nlsolver(alg, u, p,
        t, dt,
        f,
        γ, c)
    build_nlsolver(alg, u, p, t, dt, f, γ, c, 1)
end

@inline function build_nlsolver(alg, u, p,
        t, dt,
        f,
        γ, c, α)
    # define fields of non-linear solver
    z = u
    tmp = u
    ztmp = u
    J, W = build_J_W(alg, f, γ, dt)
    max_iter = 30
    κ = 1 / 100
    NLSolver{typeof(dt)}(z, tmp, ztmp, γ, c, α, κ,
        J, W, dt, t, p, 0, max_iter)
end
