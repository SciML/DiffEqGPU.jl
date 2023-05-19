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
    Kvaerno3Tableau(γ, a31, a32, a41, a42, a43, btilde1, btilde2, btilde3, btilde4, c3, α31,
                    α32, α41, α42)
end
