@kernel function em_kernel(
        @Const(probs), _us, _ts, dt, saveat, ::Val{save_everystep}) where {save_everystep}
    i = @index(Global, Linear)
    prob = @inbounds probs[i]
    Random.seed!(prob.seed)

    ts = @inbounds view(_ts, :, i)
    us = @inbounds view(_us, :, i)

    f = prob.f
    g = prob.g
    u0 = prob.u0
    t0, t1 = prob.tspan
    p = prob.p
    Tt = typeof(t0)

    # init
    if saveat === nothing
        @inbounds ts[1] = t0
        @inbounds us[1] = u0
    end

    # keep everything in the element types; no Float64
    sqdt = sqrt(Tt(dt))
    u = u0                 # avoid copy()
    t = t0

    if saveat === nothing && save_everystep
        # use the preallocated length; no float→Int
        n = size(us, 1)
        @inbounds for j in 2:n
            uprev = u
            u = uprev + f(uprev, p, t) * dt + sqdt * g(uprev, p, t) .* randn(typeof(u0))
            t += dt
            us[j] = u
            ts[j] = t
        end

    else
        # no need for n; just advance until t reaches t1
        # tolerance avoids off-by-one from fp error
        tol = eps(Tt) * abs(t1) + Tt(1e-7) * dt
        while t + dt <= t1 + tol
            uprev = u
            u = uprev + f(uprev, p, t) * dt + sqdt * g(uprev, p, t) .* randn(typeof(u0))
            t += dt
        end
        if saveat === nothing && !save_everystep
            @inbounds us[2] = u
            @inbounds ts[2] = t
        end
    end
end
