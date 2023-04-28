
@inline function bΘs(integ::T, Θ) where {T <: Union{GPUV7I, GPUAV7I}}
    @unpack r011, r012, r013, r014, r015, r016, r017, r042, r043, r044, r045, r046, r047,
    r052, r053, r054, r055, r056, r057, r062, r063, r064, r065, r066, r067, r072, r073,
    r074, r075, r076, r077, r082, r083, r084, r085, r086, r087, r092, r093, r094, r095,
    r096, r097, r112, r113, r114, r115, r116, r117, r122, r123, r124, r125, r126, r127,
    r132, r133, r134, r135, r136, r137, r142, r143, r144, r145, r146, r147, r152, r153,
    r154, r155, r156, r157, r162, r163, r164, r165, r166, r167 = integ.tab.interp

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
                                                           Union{GPUV7I, GPUAV7I}}
    b1Θ, b4Θ, b5Θ, b6Θ, b7Θ, b8Θ, b9Θ, b11Θ, b12Θ, b13Θ, b14Θ, b15Θ, b16Θ = bΘs(integ, Θ)

    @unpack c11, a1101, a1104, a1105, a1106, a1107, a1108, a1109, c12, a1201, a1204,
    a1205, a1206, a1207, a1208, a1209, a1211, c13, a1301, a1304, a1305, a1306, a1307,
    a1308, a1309, a1311, a1312, c14, a1401, a1404, a1405, a1406, a1407, a1408, a1409,
    a1411, a1412, a1413, c15, a1501, a1504, a1505, a1506, a1507, a1508, a1509, a1511,
    a1512, a1513, c16, a1601, a1604, a1605, a1606, a1607, a1608, a1609,
    a1611, a1612, a1613 = integ.tab.extra

    @unpack k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, uprev, f, t, p = integ

    k11 = f(uprev +
            dt * (a1101 * k1 + a1104 * k4 + a1105 * k5 + a1106 * k6 +
             a1107 * k7 + a1108 * k8 + a1109 * k9), p, t + c11 * dt)
    k12 = f(uprev +
            dt * (a1201 * k1 + a1204 * k4 + a1205 * k5 + a1206 * k6 +
             a1207 * k7 + a1208 * k8 + a1209 * k9 + a1211 * k11), p,
            t + c12 * dt)
    k13 = f(uprev +
            dt * (a1301 * k1 + a1304 * k4 + a1305 * k5 + a1306 * k6 +
             a1307 * k7 + a1308 * k8 + a1309 * k9 + a1311 * k11 +
             a1312 * k12), p, t + c13 * dt)
    k14 = f(uprev +
            dt * (a1401 * k1 + a1404 * k4 + a1405 * k5 + a1406 * k6 +
             a1407 * k7 + a1408 * k8 + a1409 * k9 + a1411 * k11 +
             a1412 * k12 + a1413 * k13), p, t + c14 * dt)
    k15 = f(uprev +
            dt * (a1501 * k1 + a1504 * k4 + a1505 * k5 + a1506 * k6 +
             a1507 * k7 + a1508 * k8 + a1509 * k9 + a1511 * k11 +
             a1512 * k12 + a1513 * k13), p, t + c15 * dt)
    k16 = f(uprev +
            dt * (a1601 * k1 + a1604 * k4 + a1605 * k5 + a1606 * k6 +
             a1607 * k7 + a1608 * k8 + a1609 * k9 + a1611 * k11 +
             a1612 * k12 + a1613 * k13), p, t + c16 * dt)

    return y₀ +
           dt * (integ.k1 * b1Θ
            + integ.k4 * b4Θ + integ.k5 * b5Θ + integ.k6 * b6Θ + integ.k7 * b7Θ +
            integ.k8 * b8Θ + integ.k9 * b9Θ
            + k11 * b11Θ + k12 * b12Θ + k13 * b13Θ +
            k14 * b14Θ + k15 * b15Θ + k16 * b16Θ)
end

@inline function bΘs(integ::T, Θ) where {T <: Union{GPUV9I, GPUAV9I}}
    @unpack r011, r012, r013, r014, r015, r016, r017, r018, r019, r082, r083, r084, r085,
    r086, r087, r088, r089, r092, r093, r094, r095, r096, r097, r098, r099, r102, r103,
    r104, r105, r106, r107, r108, r109, r112, r113, r114, r115, r116, r117, r118, r119,
    r122, r123, r124, r125, r126, r127, r128, r129, r132, r133, r134, r135, r136, r137,
    r138, r139, r142, r143, r144, r145, r146, r147, r148, r149, r152, r153, r154, r155,
    r156, r157, r158, r159, r172, r173, r174, r175, r176, r177, r178, r179, r182, r183,
    r184, r185, r186, r187, r188, r189, r192, r193, r194, r195, r196, r197, r198, r199,
    r202, r203, r204, r205, r206, r207, r208, r209, r212, r213, r214, r215, r216, r217,
    r218, r219, r222, r223, r224, r225, r226, r227, r228, r229, r232, r233, r234, r235,
    r236, r237, r238, r239, r242, r243, r244, r245, r246, r247, r248, r249, r252, r253,
    r254, r255, r256, r257, r258, r259, r262, r263, r264, r265, r266,
    r267, r268, r269 = integ.tab.interp

    b1Θ = @evalpoly(Θ, 0, r011, r012, r013, r014, r015, r016, r017, r018, r019)
    b8Θ = @evalpoly(Θ, 0, 0, r082, r083, r084, r085, r086, r087, r088, r089)
    b9Θ = @evalpoly(Θ, 0, 0, r092, r093, r094, r095, r096, r097, r098, r099)
    b10Θ = @evalpoly(Θ, 0, 0, r102, r103, r104, r105, r106, r107, r108, r109)
    b11Θ = @evalpoly(Θ, 0, 0, r112, r113, r114, r115, r116, r117, r118, r119)
    b12Θ = @evalpoly(Θ, 0, 0, r122, r123, r124, r125, r126, r127, r128, r129)
    b13Θ = @evalpoly(Θ, 0, 0, r132, r133, r134, r135, r136, r137, r138, r139)
    b14Θ = @evalpoly(Θ, 0, 0, r142, r143, r144, r145, r146, r147, r148, r149)
    b15Θ = @evalpoly(Θ, 0, 0, r152, r153, r154, r155, r156, r157, r158, r159)
    b17Θ = @evalpoly(Θ, 0, 0, r172, r173, r174, r175, r176, r177, r178, r179)
    b18Θ = @evalpoly(Θ, 0, 0, r182, r183, r184, r185, r186, r187, r188, r189)
    b19Θ = @evalpoly(Θ, 0, 0, r192, r193, r194, r195, r196, r197, r198, r199)
    b20Θ = @evalpoly(Θ, 0, 0, r202, r203, r204, r205, r206, r207, r208, r209)
    b21Θ = @evalpoly(Θ, 0, 0, r212, r213, r214, r215, r216, r217, r218, r219)
    b22Θ = @evalpoly(Θ, 0, 0, r222, r223, r224, r225, r226, r227, r228, r229)
    b23Θ = @evalpoly(Θ, 0, 0, r232, r233, r234, r235, r236, r237, r238, r239)
    b24Θ = @evalpoly(Θ, 0, 0, r242, r243, r244, r245, r246, r247, r248, r249)
    b25Θ = @evalpoly(Θ, 0, 0, r252, r253, r254, r255, r256, r257, r258, r259)
    b26Θ = @evalpoly(Θ, 0, 0, r262, r263, r264, r265, r266, r267, r268, r269)

    return b1Θ, b8Θ, b9Θ, b10Θ, b11Θ, b12Θ, b13Θ, b14Θ, b15Θ, b17Θ, b18Θ, b19Θ, b20Θ,
           b21Θ, b22Θ, b23Θ, b24Θ, b25Θ, b26Θ
end

@inline @muladd function _ode_interpolant(Θ, dt, y₀,
                                          integ::T) where {T <:
                                                           Union{GPUV9I, GPUAV9I}}
    b1Θ, b8Θ, b9Θ, b10Θ, b11Θ, b12Θ, b13Θ, b14Θ, b15Θ, b17Θ, b18Θ, b19Θ, b20Θ,
    b21Θ, b22Θ, b23Θ, b24Θ, b25Θ, b26Θ = bΘs(integ, Θ)

    @unpack c17, a1701, a1708, a1709, a1710, a1711, a1712, a1713, a1714, a1715, c18, a1801,
    a1808, a1809, a1810, a1811, a1812, a1813, a1814, a1815, a1817, c19, a1901, a1908, a1909,
    a1910, a1911, a1912, a1913, a1914, a1915, a1917, a1918, c20, a2001, a2008, a2009, a2010,
    a2011, a2012, a2013, a2014, a2015, a2017, a2018, a2019, c21, a2101, a2108, a2109, a2110,
    a2111, a2112, a2113, a2114, a2115, a2117, a2118, a2119, a2120, c22, a2201, a2208, a2209,
    a2210, a2211, a2212, a2213, a2214, a2215, a2217, a2218, a2219, a2220, a2221, c23, a2301,
    a2308, a2309, a2310, a2311, a2312, a2313, a2314, a2315, a2317, a2318, a2319, a2320,
    a2321, c24, a2401, a2408, a2409, a2410, a2411, a2412, a2413, a2414, a2415, a2417,
    a2418, a2419, a2420, a2421, c25, a2501, a2508, a2509, a2510, a2511, a2512, a2513,
    a2514, a2515, a2517, a2518, a2519, a2520, a2521, c26, a2601, a2608, a2609, a2610,
    a2611, a2612, a2613, a2614, a2615, a2617, a2618, a2619, a2620, a2621 = integ.tab.extra

    @unpack k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, uprev, f, t, p = integ

    k11 = f(uprev +
            dt * (a1701 * k1 + a1708 * k2 + a1709 * k3 + a1710 * k4 +
             a1711 * k5 + a1712 * k6 + a1713 * k7 + a1714 * k8 + a1715 * k9),
            p, t + c17 * dt)
    k12 = f(uprev +
            dt * (a1801 * k1 + a1808 * k2 + a1809 * k3 + a1810 * k4 +
             a1811 * k5 + a1812 * k6 + a1813 * k7 + a1814 * k8 +
             a1815 * k9 + a1817 * k11), p, t + c18 * dt)
    k13 = f(uprev +
            dt * (a1901 * k1 + a1908 * k2 + a1909 * k3 + a1910 * k4 +
             a1911 * k5 + a1912 * k6 + a1913 * k7 + a1914 * k8 +
             a1915 * k9 + a1917 * k11 + a1918 * k12), p, t + c19 * dt)
    k14 = f(uprev +
            dt * (a2001 * k1 + a2008 * k2 + a2009 * k3 + a2010 * k4 +
             a2011 * k5 + a2012 * k6 + a2013 * k7 + a2014 * k8 +
             a2015 * k9 + a2017 * k11 + a2018 * k12 + a2019 * k13), p,
            t + c20 * dt)
    k15 = f(uprev +
            dt * (a2101 * k1 + a2108 * k2 + a2109 * k3 + a2110 * k4 +
             a2111 * k5 + a2112 * k6 + a2113 * k7 + a2114 * k8 +
             a2115 * k9 + a2117 * k11 + a2118 * k12 + a2119 * k13 +
             a2120 * k14), p, t + c21 * dt)
    k16 = f(uprev +
            dt * (a2201 * k1 + a2208 * k2 + a2209 * k3 + a2210 * k4 +
             a2211 * k5 + a2212 * k6 + a2213 * k7 + a2214 * k8 +
             a2215 * k9 + a2217 * k11 + a2218 * k12 + a2219 * k13 +
             a2220 * k14 + a2221 * k15), p, t + c22 * dt)
    k17 = f(uprev +
            dt * (a2301 * k1 + a2308 * k2 + a2309 * k3 + a2310 * k4 +
             a2311 * k5 + a2312 * k6 + a2313 * k7 + a2314 * k8 +
             a2315 * k9 + a2317 * k11 + a2318 * k12 + a2319 * k13 +
             a2320 * k14 + a2321 * k15), p, t + c23 * dt)
    k18 = f(uprev +
            dt * (a2401 * k1 + a2408 * k2 + a2409 * k3 + a2410 * k4 +
             a2411 * k5 + a2412 * k6 + a2413 * k7 + a2414 * k8 +
             a2415 * k9 + a2417 * k11 + a2418 * k12 + a2419 * k13 +
             a2420 * k14 + a2421 * k15), p, t + c24 * dt)
    k19 = f(uprev +
            dt * (a2501 * k1 + a2508 * k2 + a2509 * k3 + a2510 * k4 +
             a2511 * k5 + a2512 * k6 + a2513 * k7 + a2514 * k8 +
             a2515 * k9 + a2517 * k11 + a2518 * k12 + a2519 * k13 +
             a2520 * k14 + a2521 * k15), p, t + c25 * dt)
    k20 = f(uprev +
            dt * (a2601 * k1 + a2608 * k2 + a2609 * k3 + a2610 * k4 +
             a2611 * k5 + a2612 * k6 + a2613 * k7 + a2614 * k8 +
             a2615 * k9 + a2617 * k11 + a2618 * k12 + a2619 * k13 +
             a2620 * k14 + a2621 * k15), p, t + c26 * dt)

    return y₀ +
           dt *
           (integ.k1 * b1Θ + integ.k2 * b8Θ + integ.k3 * b9Θ + integ.k4 * b10Θ +
            integ.k5 * b11Θ +
            integ.k6 * b12Θ + integ.k7 * b13Θ + integ.k8 * b14Θ + integ.k9 * b15Θ +
            k11 * b17Θ +
            k12 * b18Θ + k13 * b19Θ + k14 * b20Θ + k15 * b21Θ +
            k16 * b22Θ +
            k17 * b23Θ + k18 * b24Θ + k19 * b25Θ + k20 * b26Θ)
end

@inline @muladd function _ode_interpolant(Θ, dt, y₀,
                                          integ::T) where {T <:
                                                           Union{GPUT5I, GPUAT5I}}
    b1θ, b2θ, b3θ, b4θ, b5θ, b6θ, b7θ = SimpleDiffEq.bθs(integ.rs, Θ)
    return y₀ +
           dt *
           (b1θ * integ.k1 + b2θ * integ.k2 + b3θ * integ.k3 +
            b4θ * integ.k4 + b5θ * integ.k5 + b6θ * integ.k6 +
            b7θ * integ.k7)
end

@inline @muladd function _ode_interpolant(Θ, dt, y₀,
                                          integ::T) where {T <:
                                                           Union{GPURB23I, GPUARB23I}}
    c1 = Θ * (1 - Θ) / (1 - 2 * integ.d)
    c2 = Θ * (Θ - 2 * integ.d) / (1 - 2 * integ.d)
    return y₀ + dt * (c1 * integ.k1 + c2 * integ.k2)
end

@inline @muladd function _ode_interpolant(Θ, dt, y₀,
                                          integ::T) where {T <:
                                                           Union{GPURodas4I, GPUARodas4I}}
    Θ1 = 1 - Θ
    y₁ = integ.u
    return Θ1 * y₀ + Θ * (y₁ + Θ1 * (integ.k1 + Θ * integ.k2))
end


@inline @muladd function _ode_interpolant(Θ, dt, y₀,
                                          integ::T) where {T <:
                                                           Union{GPURodas5PI,GPUARodas5PI}}
    Θ1 = 1 - Θ
    y₁ = integ.u
    return Θ1 * y₀ + Θ * (y₁ + Θ1 * (integ.k1 + Θ * (integ.k2 + Θ * integ.k3)))
end