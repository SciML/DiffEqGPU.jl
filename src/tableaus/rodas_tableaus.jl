struct Rodas4Tableau{T, T2}
    a21::T
    a31::T
    a32::T
    a41::T
    a42::T
    a43::T
    a51::T
    a52::T
    a53::T
    a54::T
    C21::T
    C31::T
    C32::T
    C41::T
    C42::T
    C43::T
    C51::T
    C52::T
    C53::T
    C54::T
    C61::T
    C62::T
    C63::T
    C64::T
    C65::T
    gamma::T
    c2::T2
    c3::T2
    c4::T2
    d1::T
    d2::T
    d3::T
    d4::T
    h21::T
    h22::T
    h23::T
    h24::T
    h25::T
    h31::T
    h32::T
    h33::T
    h34::T
    h35::T
end

function Rodas4Tableau(T::Type{T1}, T2::Type{T1}) where {T1}
    gamma = convert(T, 0.25)
    #BET2P=0.0317D0
    #BET3P=0.0635D0
    #BET4P=0.3438D0
    a21 = convert(T, 1.544000000000000)
    a31 = convert(T, 0.9466785280815826)
    a32 = convert(T, 0.2557011698983284)
    a41 = convert(T, 3.314825187068521)
    a42 = convert(T, 2.896124015972201)
    a43 = convert(T, 0.9986419139977817)
    a51 = convert(T, 1.221224509226641)
    a52 = convert(T, 6.019134481288629)
    a53 = convert(T, 12.53708332932087)
    a54 = -convert(T, 0.6878860361058950)
    C21 = -convert(T, 5.668800000000000)
    C31 = -convert(T, 2.430093356833875)
    C32 = -convert(T, 0.2063599157091915)
    C41 = -convert(T, 0.1073529058151375)
    C42 = -convert(T, 9.594562251023355)
    C43 = -convert(T, 20.47028614809616)
    C51 = convert(T, 7.496443313967647)
    C52 = -convert(T, 10.24680431464352)
    C53 = -convert(T, 33.99990352819905)
    C54 = convert(T, 11.70890893206160)
    C61 = convert(T, 8.083246795921522)
    C62 = -convert(T, 7.981132988064893)
    C63 = -convert(T, 31.52159432874371)
    C64 = convert(T, 16.31930543123136)
    C65 = -convert(T, 6.058818238834054)

    c2 = convert(T2, 0.386)
    c3 = convert(T2, 0.21)
    c4 = convert(T2, 0.63)

    d1 = convert(T, 0.2500000000000000)
    d2 = -convert(T, 0.1043000000000000)
    d3 = convert(T, 0.1035000000000000)
    d4 = -convert(T, 0.03620000000000023)

    h21 = convert(T, 10.12623508344586)
    h22 = -convert(T, 7.487995877610167)
    h23 = -convert(T, 34.80091861555747)
    h24 = -convert(T, 7.992771707568823)
    h25 = convert(T, 1.025137723295662)
    h31 = -convert(T, 0.6762803392801253)
    h32 = convert(T, 6.087714651680015)
    h33 = convert(T, 16.43084320892478)
    h34 = convert(T, 24.76722511418386)
    h35 = -convert(T, 6.594389125716872)

    Rodas4Tableau(a21, a31, a32, a41, a42, a43, a51, a52, a53, a54,
                  C21, C31, C32, C41, C42, C43, C51, C52, C53, C54, C61, C62, C63, C64, C65,
                  gamma, c2, c3, c4, d1, d2, d3, d4,
                  h21, h22, h23, h24, h25, h31, h32, h33, h34, h35)
end
