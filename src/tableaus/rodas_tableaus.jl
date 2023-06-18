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
    γ::T
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
    γ = convert(T, 0.25)
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
        γ, c2, c3, c4, d1, d2, d3, d4,
        h21, h22, h23, h24, h25, h31, h32, h33, h34, h35)
end

struct Rodas5PTableau{T, T2}
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
    a61::T
    a62::T
    a63::T
    a64::T
    a65::T
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
    C71::T
    C72::T
    C73::T
    C74::T
    C75::T
    C76::T
    C81::T
    C82::T
    C83::T
    C84::T
    C85::T
    C86::T
    C87::T
    γ::T2
    d1::T
    d2::T
    d3::T
    d4::T
    d5::T
    c2::T2
    c3::T2
    c4::T2
    c5::T2
    h21::T
    h22::T
    h23::T
    h24::T
    h25::T
    h26::T
    h27::T
    h28::T
    h31::T
    h32::T
    h33::T
    h34::T
    h35::T
    h36::T
    h37::T
    h38::T
    h41::T
    h42::T
    h43::T
    h44::T
    h45::T
    h46::T
    h47::T
    h48::T
end

function Rodas5PTableau(T, T2)
    γ = convert(T2, 0.21193756319429014)

    a21 = convert(T, 3.0)
    a31 = convert(T, 2.849394379747939)
    a32 = convert(T, 0.45842242204463923)
    a41 = convert(T, -6.954028509809101)
    a42 = convert(T, 2.489845061869568)
    a43 = convert(T, -10.358996098473584)
    a51 = convert(T, 2.8029986275628964)
    a52 = convert(T, 0.5072464736228206)
    a53 = convert(T, -0.3988312541770524)
    a54 = convert(T, -0.04721187230404641)
    a61 = convert(T, -7.502846399306121)
    a62 = convert(T, 2.561846144803919)
    a63 = convert(T, -11.627539656261098)
    a64 = convert(T, -0.18268767659942256)
    a65 = convert(T, 0.030198172008377946)

    C21 = convert(T, -14.155112264123755)
    C31 = convert(T, -17.97296035885952)
    C32 = convert(T, -2.859693295451294)
    C41 = convert(T, 147.12150275711716)
    C42 = convert(T, -1.41221402718213)
    C43 = convert(T, 71.68940251302358)
    C51 = convert(T, 165.43517024871676)
    C52 = convert(T, -0.4592823456491126)
    C53 = convert(T, 42.90938336958603)
    C54 = convert(T, -5.961986721573306)
    C61 = convert(T, 24.854864614690072)
    C62 = convert(T, -3.0009227002832186)
    C63 = convert(T, 47.4931110020768)
    C64 = convert(T, 5.5814197821558125)
    C65 = convert(T, -0.6610691825249471)
    C71 = convert(T, 30.91273214028599)
    C72 = convert(T, -3.1208243349937974)
    C73 = convert(T, 77.79954646070892)
    C74 = convert(T, 34.28646028294783)
    C75 = convert(T, -19.097331116725623)
    C76 = convert(T, -28.087943162872662)
    C81 = convert(T, 37.80277123390563)
    C82 = convert(T, -3.2571969029072276)
    C83 = convert(T, 112.26918849496327)
    C84 = convert(T, 66.9347231244047)
    C85 = convert(T, -40.06618937091002)
    C86 = convert(T, -54.66780262877968)
    C87 = convert(T, -9.48861652309627)

    c2 = convert(T2, 0.6358126895828704)
    c3 = convert(T2, 0.4095798393397535)
    c4 = convert(T2, 0.9769306725060716)
    c5 = convert(T2, 0.4288403609558664)

    d1 = convert(T, 0.21193756319429014)
    d2 = convert(T, -0.42387512638858027)
    d3 = convert(T, -0.3384627126235924)
    d4 = convert(T, 1.8046452872882734)
    d5 = convert(T, 2.325825639765069)

    h21 = convert(T, 25.948786856663858)
    h22 = convert(T, -2.5579724845846235)
    h23 = convert(T, 10.433815404888879)
    h24 = convert(T, -2.3679251022685204)
    h25 = convert(T, 0.524948541321073)
    h26 = convert(T, 1.1241088310450404)
    h27 = convert(T, 0.4272876194431874)
    h28 = convert(T, -0.17202221070155493)

    h31 = convert(T, -9.91568850695171)
    h32 = convert(T, -0.9689944594115154)
    h33 = convert(T, 3.0438037242978453)
    h34 = convert(T, -24.495224566215796)
    h35 = convert(T, 20.176138334709044)
    h36 = convert(T, 15.98066361424651)
    h37 = convert(T, -6.789040303419874)
    h38 = convert(T, -6.710236069923372)

    h41 = convert(T, 11.419903575922262)
    h42 = convert(T, 2.8879645146136994)
    h43 = convert(T, 72.92137995996029)
    h44 = convert(T, 80.12511834622643)
    h45 = convert(T, -52.072871366152654)
    h46 = convert(T, -59.78993625266729)
    h47 = convert(T, -0.15582684282751913)
    h48 = convert(T, 4.883087185713722)

    Rodas5PTableau(a21, a31, a32, a41, a42, a43, a51, a52, a53, a54,
        a61, a62, a63, a64, a65,
        C21, C31, C32, C41, C42, C43, C51, C52, C53, C54,
        C61, C62, C63, C64, C65, C71, C72, C73, C74, C75, C76,
        C81, C82, C83, C84, C85, C86, C87,
        γ, d1, d2, d3, d4, d5, c2, c3, c4, c5,
        h21, h22, h23, h24, h25, h26, h27, h28, h31, h32, h33, h34, h35, h36,
        h37,
        h38, h41, h42, h43, h44, h45, h46, h47, h48)
end
