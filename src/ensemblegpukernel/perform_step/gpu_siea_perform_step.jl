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

@kernel function siea_kernel(@Const(probs), _us, _ts, dt,
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
    
    # FIX for Issue #379: Get time type from tspan
    Tt = typeof(tspan[1])
    
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
    
    # FIX: Use Tt for sqrt to ensure proper type
    sqdt = sqrt(Tt(dt))
    u = copy(u0)
    t = copy(tspan[1])
    
    # FIX: Ensure n calculation uses proper types
    t0, tf = tspan[1], tspan[2]
    n = floor(Int, abs(tf - t0) / abs(Tt(dt))) + 1
    
    cache = SIEAConstantCache(eltype(u0), Tt)
    @unpack α1, α2, γ1, λ1, λ2, λ3, µ1, µ2, µ3, µ0, µbar0, λ0, λbar0, ν1, ν2, β2, β3, δ2,
    δ3 = cache
    
    for j in 2:n
        uprev = u
        # compute stage values
        k0 = f(uprev, p, t)
        g0 = g(uprev, p, t)
        if is_diagonal_noise
            dW = sqdt * randn(typeof(u0))
            W2 = (dW) .^ 2 / sqdt
            W3 = ν2 * (dW) .^ 3 / Tt(dt)
            k1 = f(uprev + λ0 * k0 * Tt(dt) + ν1 * g0 .* dW + g0 .* W3, p, t + µ0 * Tt(dt))
            g1 = g(uprev + λbar0 * k0 * Tt(dt) + β2 * g0 * sqdt + β3 * g0 .* W2, p,
                t + µbar0 * Tt(dt))
            g2 = g(uprev + λbar0 * k0 * Tt(dt) + δ2 * g0 * sqdt + δ3 * g0 .* W2, p,
                t + µbar0 * Tt(dt))
            u = uprev + (α1 * k0 + α2 * k1) * Tt(dt)
            u += γ1 * g0 .* dW + (λ1 .* dW .+ λ2 * sqdt + λ3 .* W2) .* g1 +
                 (µ1 .* dW .+ µ2 * sqdt + µ3 .* W2) .* g2
        end
        t += Tt(dt)
        if saveat === nothing && save_everystep
            @inbounds us[j] = u
            @inbounds ts[j] = t
        elseif saveat !== nothing
            while cur_t <= length(saveat) && saveat[cur_t] <= t
                savet = saveat[cur_t]
                Θ = (savet - (t - Tt(dt))) / Tt(dt)
                # Linear Interpolation
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