struct SIESMEConstantCache{T, T2}
    α1::T
    α2::T

    γ1::T

    λ1::T
    λ2::T
    λ3::T

    µ1::T
    µ2::T
    µ3::T

    µ0::T2
    µbar0::T2

    λ0::T
    λbar0::T

    ν1::T
    ν2::T

    β2::T
    β3::T

    δ2::T
    δ3::T
end

function SIEAConstantCache(::Type{T}, ::Type{T2}) where {T, T2}
    α1 = convert(T, 1 / 2)
    α2 = convert(T, 1 / 2)

    γ1 = convert(T, 1 / 2)

    λ1 = convert(T, 1 / 4)
    λ2 = convert(T, -1 / 4)
    λ3 = convert(T, 1 / 4)

    µ1 = convert(T, 1 / 4)
    µ2 = convert(T, 1 / 4)
    µ3 = convert(T, -1 / 4)

    µ0 = convert(T2, 1 / 1)
    µbar0 = convert(T2, 1 / 1)

    λ0 = convert(T, 1 / 1)
    λbar0 = convert(T, 1 / 1)

    ν1 = convert(T, 1 / 1)
    ν2 = convert(T, 0)

    β2 = convert(T, 1 / 1)
    β3 = convert(T, 0)

    δ2 = convert(T, -1 / 1)
    δ3 = convert(T, 0)

    SIESMEConstantCache(α1, α2, γ1, λ1, λ2, λ3, µ1, µ2, µ3, µ0, µbar0, λ0, λbar0, ν1, ν2,
        β2, β3, δ2, δ3)
end

@kernel function siea_kernel(
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
    dtT = Tt(dt)                 # keep all time math in Tt
    sqdt = sqrt(dtT)

    # init write only when not saveat (matches EM fix semantics)
    if saveat === nothing
        @inbounds ts[1] = t0
        @inbounds us[1] = u0
    end

    # make constants in the right element types
    cache = SIEAConstantCache(eltype(u0), typeof(t0))
    α1 = cache.α1
    α2 = cache.α2
    γ1 = cache.γ1
    λ1 = cache.λ1
    λ2 = cache.λ2
    λ3 = cache.λ3
    µ1 = cache.µ1
    µ2 = cache.µ2
    µ3 = cache.µ3
    µ0 = cache.µ0
    µbar0 = cache.µbar0
    λ0 = cache.λ0
    λbar0 = cache.λbar0
    ν1 = cache.ν1
    ν2 = cache.ν2
    β2 = cache.β2
    β3 = cache.β3
    δ2 = cache.δ2
    δ3 = cache.δ3

    u = u0           # avoid copy(); keeps SVector type
    t = t0

    if saveat === nothing && save_everystep
        # drive by preallocated length – no float range, no Int casts
        n = size(us, 1)
        @inbounds for j in 2:n
            uprev = u
            # stage values
            k0 = f(uprev, p, t)
            g0 = g(uprev, p, t)
            # diagonal noise branch (matches your code path)
            dW = sqdt * randn(typeof(u0))
            W2 = (dW .* dW) / sqdt
            W3 = ν2 * (dW .* dW .* dW) / dtT
            k1 = f(uprev + λ0 * k0 * dtT + ν1 * g0 .* dW + g0 .* W3, p, t + µ0 * dtT)
            g1 = g(uprev + λbar0 * k0 * dtT + β2 * g0 * sqdt + β3 * g0 .* W2,
                p, t + µbar0 * dtT)
            g2 = g(uprev + λbar0 * k0 * dtT + δ2 * g0 * sqdt + δ3 * g0 .* W2,
                p, t + µbar0 * dtT)
            u = uprev + (α1 * k0 + α2 * k1) * dtT
            u += γ1 * g0 .* dW +
                 (λ1 .* dW .+ λ2 * sqdt .+ λ3 .* W2) .* g1 +
                 (µ1 .* dW .+ µ2 * sqdt .+ µ3 .* W2) .* g2
            t += dtT
            us[j] = u
            ts[j] = t
        end
    else
        # time-driven loop with tolerance; no ranges, no float→Int
        tol = eps(Tt) * abs(t1) + Tt(1e-7) * dtT
        cur_t = 0
        if saveat !== nothing
            cur_t = 1
            if t0 == saveat[1]
                cur_t += 1
                @inbounds us[1] = u0
            end
        end
        while t + dtT <= t1 + tol
            uprev = u
            k0 = f(uprev, p, t)
            g0 = g(uprev, p, t)
            dW = sqdt * randn(typeof(u0))
            W2 = (dW .* dW) / sqdt
            W3 = ν2 * (dW .* dW .* dW) / dtT
            k1 = f(uprev + λ0 * k0 * dtT + ν1 * g0 .* dW + g0 .* W3, p, t + µ0 * dtT)
            g1 = g(uprev + λbar0 * k0 * dtT + β2 * g0 * sqdt + β3 * g0 .* W2,
                p, t + µbar0 * dtT)
            g2 = g(uprev + λbar0 * k0 * dtT + δ2 * g0 * sqdt + δ3 * g0 .* W2,
                p, t + µbar0 * dtT)
            u = uprev + (α1 * k0 + α2 * k1) * dtT
            u += γ1 * g0 .* dW +
                 (λ1 .* dW .+ λ2 * sqdt .+ λ3 .* W2) .* g1 +
                 (µ1 .* dW .+ µ2 * sqdt .+ µ3 .* W2) .* g2
            t += dtT

            if saveat !== nothing
                # interpolate any save points we passed
                while cur_t <= length(saveat) && saveat[cur_t] <= t + tol
                    savet = saveat[cur_t]
                    Θ = (savet - (t - dtT)) / dtT
                    @inbounds us[cur_t] = uprev + (u - uprev) * Θ
                    @inbounds ts[cur_t] = savet
                    cur_t += 1
                end
            end
        end

        if saveat === nothing && !save_everystep
            @inbounds us[2] = u
            @inbounds ts[2] = t
        end
    end
end