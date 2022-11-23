
@inline function bΘs(integ::T, Θ) where {T <: Union{GPUVern7I, GPUAVern7I}}
    @unpack r011, r012, r013, r014, r015, r016, r017, r042, r043, r044, r045, r046, r047, r052, r053, r054, r055, r056, r057, r062, r063, r064, r065, r066, r067, r072, r073, r074, r075, r076, r077, r082, r083, r084, r085, r086, r087, r092, r093, r094, r095, r096, r097, r112, r113, r114, r115, r116, r117, r122, r123, r124, r125, r126, r127, r132, r133, r134, r135, r136, r137, r142, r143, r144, r145, r146, r147, r152, r153, r154, r155, r156, r157, r162, r163, r164, r165, r166, r167 = integ.tab.interp

    b1Θ = @evalpoly(Θ, 0, r011, r012, r013, r014, r015, r016, r017)
    b4Θ = @evalpoly(Θ, 0, 0, r042, r043, r044, r045, r046, r047)
    b5Θ = @evalpoly(Θ, 0, 0, r052, r053, r054, r055, r056, r057)
    b6Θ = @evalpoly(Θ, 0, 0, r062, r063, r064, r065, r066, r067)
    b7Θ = @evalpoly(Θ, 0, 0, r072, r073, r074, r075, r076, r077)
    b8Θ = @evalpoly(Θ, 0, 0, r082, r083, r084, r085, r086, r087)
    b9Θ = @evalpoly(Θ, 0, 0, r092, r093, r094, r095, r096, r097)
    b11Θ = @evalpoly(Θ, 0, 0, r112, r113, r114, r115, r116, r117)
    b12Θ = @evalpoly(Θ, 0, 0, r122, r123, r124, r125, r126, r127)
    b13Θ = @evalpoly(Θ, 0, 0, r132, r133, r134, r135, r136, r137)
    b14Θ = @evalpoly(Θ, 0, 0, r142, r143, r144, r145, r146, r147)
    b15Θ = @evalpoly(Θ, 0, 0, r152, r153, r154, r155, r156, r157)
    b16Θ = @evalpoly(Θ, 0, 0, r162, r163, r164, r165, r166, r167)

    return b1Θ, b4Θ, b5Θ, b6Θ, b7Θ, b8Θ, b9Θ, b11Θ, b12Θ, b13Θ, b14Θ, b15Θ, b16Θ
end

@inline @muladd function _ode_interpolant(Θ, dt, y₀,
                                          integ::T) where {T <:
                                                           Union{GPUVern7I, GPUAVern7I}}
    b1Θ, b4Θ, b5Θ, b6Θ, b7Θ, b8Θ, b9Θ, b11Θ, b12Θ, b13Θ, b14Θ, b15Θ, b16Θ = bΘs(integ, Θ)
    return y₀ +
           dt * (integ.k1 * b1Θ
            + integ.k4 * b4Θ + integ.k5 * b5Θ + integ.k6 * b6Θ + integ.k7 * b7Θ +
            integ.k8 * b8Θ + integ.k9 * b9Θ
            + integ.k11 * b11Θ + integ.k12 * b12Θ + integ.k13 * b13Θ +
            integ.k14 * b14Θ + integ.k15 * b15Θ + integ.k16 * b16Θ)
end
