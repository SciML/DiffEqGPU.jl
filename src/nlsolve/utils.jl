@inline function nlsolve(nlsolver::NLType, integrator::IntegType) where {NLType, IntegType}
    maxiters = nlsolver.maxiters
    dt = nlsolver.dt
    p = integrator.p
    t = nlsolver.t
    γ = nlsolver.γ
    tmp = nlsolver.tmp
    z_i = nlsolver.z
    c = nlsolver.c

    abstol = 100eps(eltype(z_i))

    for i in 1:maxiters
        W_eval = nlsolver.W(tmp + γ * z_i, p, t + c * dt)
        f_eval = integrator.f(tmp + γ * z_i, p, t + c * dt)
        f_rhs = dt * f_eval - z_i
        Δz = linear_solve(W_eval, f_rhs)
        z_i = z_i - Δz

        if norm(dt * integrator.f(tmp + γ * z_i, p, t + c * dt) - z_i) < abstol
            break
        end
    end
    @set! nlsolver.z = z_i
end
