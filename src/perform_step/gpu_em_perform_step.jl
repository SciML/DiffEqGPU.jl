function em_kernel(probs, _us, _ts, dt,
                   saveat, ::Val{save_everystep}) where {save_everystep}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(probs) && return

    # get the actual problem for this thread
    prob = @inbounds probs[i]

    # get the input/output arrays for this thread
    ts = @inbounds view(_ts, :, i)
    us = @inbounds view(_us, :, i)

    f = prob.f
    g = prob.g
    u0 = prob.u0
    tspan = prob.tspan
    p = prob.p

    is_diagonal_noise = SciMLBase.is_diagonal_noise(prob)

    @inbounds ts[1] = prob.tspan[1]
    @inbounds us[1] = prob.u0

    sqdt = sqrt(dt)
    u = copy(u0)
    t = copy(tspan[1])
    n = length(ts)

    for j in 2:n
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
            error("GPUEM does not support saveat yet")
        end
    end

    if saveat === nothing && !save_everystep
        @inbounds us[2] = u
        @inbounds ts[2] = t
    end

    return nothing
end
