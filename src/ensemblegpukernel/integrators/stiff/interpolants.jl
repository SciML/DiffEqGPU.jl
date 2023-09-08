@inline @muladd function _ode_interpolant(Θ, dt, y₀,
    integ::T) where {
    T <:
    Union{GPURodas4I, GPUARodas4I}}
    Θ1 = 1 - Θ
    y₁ = integ.u
    return Θ1 * y₀ + Θ * (y₁ + Θ1 * (integ.k1 + Θ * integ.k2))
end

@inline @muladd function _ode_interpolant(Θ, dt, y₀,
    integ::T) where {
    T <:
    Union{GPURodas5PI, GPUARodas5PI}}
    Θ1 = 1 - Θ
    y₁ = integ.u
    return Θ1 * y₀ + Θ * (y₁ + Θ1 * (integ.k1 + Θ * (integ.k2 + Θ * integ.k3)))
end
