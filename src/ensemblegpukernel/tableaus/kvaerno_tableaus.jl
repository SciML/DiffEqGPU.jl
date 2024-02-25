struct Kvaerno3Tableau{T, T2}
    γ::T2
    a31::T
    a32::T
    a41::T
    a42::T
    a43::T
    btilde1::T
    btilde2::T
    btilde3::T
    btilde4::T
    c3::T2
    α31::T2
    α32::T2
    α41::T2
    α42::T2
end

function Kvaerno3Tableau(T, T2)
    γ = convert(T2, 0.4358665215)
    a31 = convert(T, 0.490563388419108)
    a32 = convert(T, 0.073570090080892)
    a41 = convert(T, 0.308809969973036)
    a42 = convert(T, 1.490563388254106)
    a43 = -convert(T, 1.235239879727145)
    # bhat1 = convert(T,0.490563388419108)
    # bhat2 = convert(T,0.073570090080892)
    # bhat3 = convert(T,0.4358665215)
    # bhat4 = convert(T,0.0)
    btilde1 = convert(T, 0.181753418446072) # bhat1-a41
    btilde2 = convert(T, -1.416993298173214) # bhat2-a42
    btilde3 = convert(T, 1.671106401227145) # bhat3-a43
    btilde4 = convert(T, -γ) # bhat4-γ
    c3 = convert(T2, 1)
    c2 = 2γ
    θ = c3 / c2
    α31 = ((1 + (-4θ + 3θ^2)) + (6θ * (1 - θ) / c2) * γ)
    α32 = ((-2θ + 3θ^2) + (6θ * (1 - θ) / c2) * γ)
    α41 = convert(T2, 0.0)
    α42 = convert(T2, 0.0)
    Kvaerno3Tableau(
        γ, a31, a32, a41, a42, a43, btilde1, btilde2, btilde3, btilde4, c3, α31,
        α32, α41, α42)
end

struct Kvaerno5Tableau{T, T2}
    γ::T2
    a31::T
    a32::T
    a41::T
    a42::T
    a43::T
    a51::T
    a52::T
    a53::T
    a54::T
    a61::T
    a63::T
    a64::T
    a65::T
    a71::T
    a73::T
    a74::T
    a75::T
    a76::T
    btilde1::T
    btilde3::T
    btilde4::T
    btilde5::T
    btilde6::T
    btilde7::T
    c3::T2
    c4::T2
    c5::T2
    c6::T2
    α31::T2
    α32::T2
    α41::T2
    α42::T2
    α43::T2
    α51::T2
    α52::T2
    α53::T2
    α61::T2
    α62::T2
    α63::T2
end

#=
# Kvaerno5
# Predict z3 from Hermite z2 and z1

c2 = 2γ
θ = c3/c2
dt = c2
((1 + (-4θ + 3θ^2)) + (6θ*(1-θ)/c2)*γ)
((-2θ + 3θ^2) + (6θ*(1-θ)/c2)*γ)

# Predict others from z1 and z3 since it covers [0,1.23]

dt = c3 since interval is [c1,c3] and c1 = 0
θ =  c4/c3, c5/c3, c6/c3, c7/c3
z = dt*k

z₁ + Θ*(-4dt*z₁ - 2dt*z₃ - 6y₀ + Θ*(3dt*z₁ + 3z₃ + 6y₀ - 6y₁ ) + 6y₁)/dt

(1 + (-4θ + 3θ^2))*z₁ + (-2θ + 3θ^2)*z₃ + (6θ*(1-θ)/dt)*(y₁-y₀)

y₀ = uprev
y₁ = uprev + a31*z₁ + a32*z₂ + γ*z₃
y₁-y₀ = a31*z₁ + a32*z₂ + γ*z₃

(1 + (-4θ + 3θ^2) + a31*(6θ*(1-θ)/dt))*z₁ +
(-2θ + 3θ^2 + γ*(6θ*(1-θ)/dt))*z₃ + (6θ*(1-θ)/dt)*a32*z₂

dt = c3
θ = c4/c3
(1 + (-4θ + 3θ^2) + a31*(6θ*(1-θ)/dt))
(6θ*(1-θ)/dt)*a32
(-2θ + 3θ^2 + γ*(6θ*(1-θ)/dt))

θ = c5/c3
(1 + (-4θ + 3θ^2) + a31*(6θ*(1-θ)/dt))
(6θ*(1-θ)/dt)*a32
(-2θ + 3θ^2 + γ*(6θ*(1-θ)/dt))

θ = c6/c3
(1 + (-4θ + 3θ^2) + a31*(6θ*(1-θ)/dt))
(6θ*(1-θ)/dt)*a32
(-2θ + 3θ^2 + γ*(6θ*(1-θ)/dt))
=#

function Kvaerno5Tableau(T, T2)
    γ = convert(T2, 0.26)
    a31 = convert(T, 0.13)
    a32 = convert(T, 0.84033320996790809)
    a41 = convert(T, 0.22371961478320505)
    a42 = convert(T, 0.47675532319799699)
    a43 = -convert(T, 0.06470895363112615)
    a51 = convert(T, 0.16648564323248321)
    a52 = convert(T, 0.10450018841591720)
    a53 = convert(T, 0.03631482272098715)
    a54 = -convert(T, 0.13090704451073998)
    a61 = convert(T, 0.13855640231268224)
    a63 = -convert(T, 0.04245337201752043)
    a64 = convert(T, 0.02446657898003141)
    a65 = convert(T, 0.61943039072480676)
    a71 = convert(T, 0.13659751177640291)
    a73 = -convert(T, 0.05496908796538376)
    a74 = -convert(T, 0.04118626728321046)
    a75 = convert(T, 0.62993304899016403)
    a76 = convert(T, 0.06962479448202728)
    btilde1 = convert(T, 0.00195889053627933) # a61-a71
    btilde3 = convert(T, 0.01251571594786333) # a63-a73
    btilde4 = convert(T, 0.06565284626324187) # a64-a74
    btilde5 = -convert(T, 0.01050265826535727) # a65-a75
    btilde6 = convert(T, 0.19037520551797272) # γ-a76
    btilde7 = -γ
    α21 = convert(T, 2) # c2/γ
    α31 = convert(T2, -1.366025403784441)
    α32 = convert(T2, 2.3660254037844357)
    α41 = convert(T2, -0.19650552613122207)
    α42 = convert(T2, 0.8113579546496623)
    α43 = convert(T2, 0.38514757148155954)
    α51 = convert(T2, 0.10375304369958693)
    α52 = convert(T2, 0.937994698066431)
    α53 = convert(T2, -0.04174774176601781)
    α61 = convert(T2, -0.17281112873898072)
    α62 = convert(T2, 0.6235784481025847)
    α63 = convert(T2, 0.5492326806363959)
    c3 = convert(T, 1.230333209967908)
    c4 = convert(T, 0.895765984350076)
    c5 = convert(T, 0.436393609858648)
    c6 = convert(T, 1)
    Kvaerno5Tableau(γ, a31, a32, a41, a42, a43, a51, a52, a53, a54,
        a61, a63, a64, a65, a71, a73, a74, a75, a76,
        btilde1, btilde3, btilde4, btilde5, btilde6, btilde7,
        c3, c4, c5, c6, α31, α32, α41, α42, α43, α51, α52, α53,
        α61, α62, α63)
end
