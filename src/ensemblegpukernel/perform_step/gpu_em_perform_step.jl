@kernel function em_kernel(@Const(probs), _us, _ts, dt,
        saveat, ::Val{save_everystep}) where {save_everystep}
    i = @index(Global, Linear)

    # get the actual problem for this thread
    prob = @inbounds probs[i]

    Random.seed!(prob.seed)

    # get the input/output arrays for this thread
    ts = @inbounds view(_ts, :, i)
    us = @inbounds view(_us, :, i)

    _saveat = get(prob.kwargs, :saveat, nothing)

    saveat = _saveat === nothing ? saveat : _saveat

    f = prob.f
    g = prob.g
    u0 = prob.u0
    tspan = prob.tspan
    p = prob.p

    is_diagonal_noise = SciMLBase.is_diagonal_noise(prob)

    cur_t = 0
    if saveat !== nothing
        cur_t = 1
        if tspan[1] == saveat[1]
            cur_t += 1
            @inbounds us[1] = u0
        end
    else
        @inbounds ts[1] = tspan[1]
        @inbounds us[1] = u0
    end

    sqdt = sqrt(dt)
    u = copy(u0)
    t = copy(tspan[1])
    t1 = copy(tspan[2])

    j = 2
    tol = dt / 2
    while t < t1 - tol
        uprev = u

        if is_diagonal_noise
            u = uprev + f(uprev, p, t) * dt +
                sqdt * g(uprev, p, t) .* randn(typeof(u0))
        else
            u = uprev + f(uprev, p, t) * dt +
                sqdt * g(uprev, p, t) * randn(typeof(prob.noise_rate_prototype[1, :]))
        end

        t += dt

        if saveat === nothing && save_everystep
            @inbounds us[j] = u
            @inbounds ts[j] = t
        elseif saveat !== nothing
            while cur_t <= length(saveat) && saveat[cur_t] <= t
                savet = saveat[cur_t]
                Θ = (savet - (t - dt)) / dt
                # Linear Interpolation
                @inbounds us[cur_t] = uprev + (u - uprev) * Θ
                @inbounds ts[cur_t] = savet
                cur_t += 1
            end
        end
        j += 1
    end

    if saveat === nothing && !save_everystep
        @inbounds us[2] = u
        @inbounds ts[2] = t
    end
end