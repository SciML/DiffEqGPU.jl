## Vern7
struct Vern7ExtraStages{T, T2}
    c11::T2
    a1101::T
    a1104::T
    a1105::T
    a1106::T
    a1107::T
    a1108::T
    a1109::T
    c12::T2
    a1201::T
    a1204::T
    a1205::T
    a1206::T
    a1207::T
    a1208::T
    a1209::T
    a1211::T
    c13::T2
    a1301::T
    a1304::T
    a1305::T
    a1306::T
    a1307::T
    a1308::T
    a1309::T
    a1311::T
    a1312::T
    c14::T2
    a1401::T
    a1404::T
    a1405::T
    a1406::T
    a1407::T
    a1408::T
    a1409::T
    a1411::T
    a1412::T
    a1413::T
    c15::T2
    a1501::T
    a1504::T
    a1505::T
    a1506::T
    a1507::T
    a1508::T
    a1509::T
    a1511::T
    a1512::T
    a1513::T
    c16::T2
    a1601::T
    a1604::T
    a1605::T
    a1606::T
    a1607::T
    a1608::T
    a1609::T
    a1611::T
    a1612::T
    a1613::T
end

function Vern7ExtraStages(T::Type{T1}, T2::Type{T1}) where {T1}
    c11 = convert(T2, 1)
    a1101 = convert(T, 0.04715561848627222)
    a1104 = convert(T, 0.25750564298434153)
    a1105 = convert(T, 0.2621665397741262)
    a1106 = convert(T, 0.15216092656738558)
    a1107 = convert(T, 0.49399691700324844)
    a1108 = convert(T, -0.29430311714032503)
    a1109 = convert(T, 0.0813174723249511)
    c12 = convert(T2, 0.29)
    a1201 = convert(T, 0.0523222769159969)
    a1204 = convert(T, 0.22495861826705715)
    a1205 = convert(T, 0.017443709248776376)
    a1206 = convert(T, -0.007669379876829393)
    a1207 = convert(T, 0.03435896044073285)
    a1208 = convert(T, -0.0410209723009395)
    a1209 = convert(T, 0.025651133005205617)
    a1211 = convert(T, -0.0160443457)
    c13 = convert(T2, 0.125)
    a1301 = convert(T, 0.053053341257859085)
    a1304 = convert(T, 0.12195301011401886)
    a1305 = convert(T, 0.017746840737602496)
    a1306 = convert(T, -0.0005928372667681495)
    a1307 = convert(T, 0.008381833970853752)
    a1308 = convert(T, -0.01293369259698612)
    a1309 = convert(T, 0.009412056815253861)
    a1311 = convert(T, -0.005353253107275676)
    a1312 = convert(T, -0.06666729992455811)
    c14 = convert(T2, 0.25)
    a1401 = convert(T, 0.03887903257436304)
    a1404 = convert(T, -0.0024403203308301317)
    a1405 = convert(T, -0.0013928917214672623)
    a1406 = convert(T, -0.00047446291558680135)
    a1407 = convert(T, 0.00039207932413159514)
    a1408 = convert(T, -0.00040554733285128004)
    a1409 = convert(T, 0.00019897093147716726)
    a1411 = convert(T, -0.00010278198793179169)
    a1412 = convert(T, 0.03385661513870267)
    a1413 = convert(T, 0.1814893063199928)
    c15 = convert(T2, 0.53)
    a1501 = convert(T, 0.05723681204690013)
    a1504 = convert(T, 0.22265948066761182)
    a1505 = convert(T, 0.12344864200186899)
    a1506 = convert(T, 0.04006332526666491)
    a1507 = convert(T, -0.05269894848581452)
    a1508 = convert(T, 0.04765971214244523)
    a1509 = convert(T, -0.02138895885042213)
    a1511 = convert(T, 0.015193891064036402)
    a1512 = convert(T, 0.12060546716289655)
    a1513 = convert(T, -0.022779423016187374)
    c16 = convert(T2, 0.79)
    a1601 = convert(T, 0.051372038802756814)
    a1604 = convert(T, 0.5414214473439406)
    a1605 = convert(T, 0.350399806692184)
    a1606 = convert(T, 0.14193112269692182)
    a1607 = convert(T, 0.10527377478429423)
    a1608 = convert(T, -0.031081847805874016)
    a1609 = convert(T, -0.007401883149519145)
    a1611 = convert(T, -0.006377932504865363)
    a1612 = convert(T, -0.17325495908361865)
    a1613 = convert(T, -0.18228156777622026)

    Vern7ExtraStages(c11, a1101, a1104, a1105, a1106, a1107, a1108, a1109, c12, a1201,
                     a1204, a1205, a1206, a1207, a1208, a1209, a1211, c13, a1301, a1304,
                     a1305, a1306, a1307, a1308, a1309, a1311, a1312, c14, a1401, a1404,
                     a1405, a1406, a1407, a1408, a1409, a1411, a1412, a1413, c15, a1501,
                     a1504, a1505, a1506, a1507, a1508, a1509, a1511, a1512, a1513, c16,
                     a1601, a1604, a1605, a1606, a1607, a1608, a1609, a1611, a1612, a1613)
end

struct Vern7InterpolationCoefficients{T}
    r011::T
    r012::T
    r013::T
    r014::T
    r015::T
    r016::T
    r017::T
    r042::T
    r043::T
    r044::T
    r045::T
    r046::T
    r047::T
    r052::T
    r053::T
    r054::T
    r055::T
    r056::T
    r057::T
    r062::T
    r063::T
    r064::T
    r065::T
    r066::T
    r067::T
    r072::T
    r073::T
    r074::T
    r075::T
    r076::T
    r077::T
    r082::T
    r083::T
    r084::T
    r085::T
    r086::T
    r087::T
    r092::T
    r093::T
    r094::T
    r095::T
    r096::T
    r097::T
    r112::T
    r113::T
    r114::T
    r115::T
    r116::T
    r117::T
    r122::T
    r123::T
    r124::T
    r125::T
    r126::T
    r127::T
    r132::T
    r133::T
    r134::T
    r135::T
    r136::T
    r137::T
    r142::T
    r143::T
    r144::T
    r145::T
    r146::T
    r147::T
    r152::T
    r153::T
    r154::T
    r155::T
    r156::T
    r157::T
    r162::T
    r163::T
    r164::T
    r165::T
    r166::T
    r167::T
end

function Vern7InterpolationCoefficients(::Type{T}) where {T}
    r011 = convert(T, 1)
    r012 = convert(T, -8.413387198332767)
    r013 = convert(T, 33.675508884490895)
    r014 = convert(T, -70.80159089484886)
    r015 = convert(T, 80.64695108301298)
    r016 = convert(T, -47.19413969837522)
    r017 = convert(T, 11.133813442539243)
    r042 = convert(T, 8.754921980674396)
    r043 = convert(T, -88.4596828699771)
    r044 = convert(T, 346.9017638429916)
    r045 = convert(T, -629.2580030059837)
    r046 = convert(T, 529.6773755604193)
    r047 = convert(T, -167.35886986514018)
    r052 = convert(T, 8.913387586637922)
    r053 = convert(T, -90.06081846893218)
    r054 = convert(T, 353.1807459217058)
    r055 = convert(T, -640.6476819744374)
    r056 = convert(T, 539.2646279047156)
    r057 = convert(T, -170.38809442991547)
    r062 = convert(T, 5.1733120298478)
    r063 = convert(T, -52.271115900055385)
    r064 = convert(T, 204.9853867374073)
    r065 = convert(T, -371.8306118563603)
    r066 = convert(T, 312.9880934374529)
    r067 = convert(T, -98.89290352172495)
    r072 = convert(T, 16.79537744079696)
    r073 = convert(T, -169.70040000059728)
    r074 = convert(T, 665.4937727009246)
    r075 = convert(T, -1207.1638892336007)
    r076 = convert(T, 1016.1291515818546)
    r077 = convert(T, -321.06001557237494)
    r082 = convert(T, -10.005997536098665)
    r083 = convert(T, 101.1005433052275)
    r084 = convert(T, -396.47391512378437)
    r085 = convert(T, 719.1787707014183)
    r086 = convert(T, -605.3681033918824)
    r087 = convert(T, 191.27439892797935)
    r092 = convert(T, 2.764708833638599)
    r093 = convert(T, -27.934602637390462)
    r094 = convert(T, 109.54779186137893)
    r095 = convert(T, -198.7128113064482)
    r096 = convert(T, 167.26633571640318)
    r097 = convert(T, -52.85010499525706)
    r112 = convert(T, -2.1696320280163506)
    r113 = convert(T, 22.016696037569876)
    r114 = convert(T, -86.90152427798948)
    r115 = convert(T, 159.22388973861476)
    r116 = convert(T, -135.9618306534588)
    r117 = convert(T, 43.792401183280006)
    r122 = convert(T, -4.890070188793804)
    r123 = convert(T, 22.75407737425176)
    r124 = convert(T, -30.78034218537731)
    r125 = convert(T, -2.797194317207249)
    r126 = convert(T, 31.369456637508403)
    r127 = convert(T, -15.655927320381801)
    r132 = convert(T, 10.862170929551967)
    r133 = convert(T, -50.542971417827104)
    r134 = convert(T, 68.37148040407511)
    r135 = convert(T, 6.213326521632409)
    r136 = convert(T, -69.68006323194157)
    r137 = convert(T, 34.776056794509195)
    r142 = convert(T, -11.37286691922923)
    r143 = convert(T, 130.79058078246717)
    r144 = convert(T, -488.65113677785604)
    r145 = convert(T, 832.2148793276441)
    r146 = convert(T, -664.7743368554426)
    r147 = convert(T, 201.79288044241662)
    r152 = convert(T, -5.919778732715007)
    r153 = convert(T, 63.27679965889219)
    r154 = convert(T, -265.432682088738)
    r155 = convert(T, 520.1009254140611)
    r156 = convert(T, -467.412109533902)
    r157 = convert(T, 155.3868452824017)
    r162 = convert(T, -10.492146197961823)
    r163 = convert(T, 105.35538525188011)
    r164 = convert(T, -409.43975011988937)
    r165 = convert(T, 732.831448907654)
    r166 = convert(T, -606.3044574733512)
    r167 = convert(T, 188.0495196316683)

    Vern7InterpolationCoefficients(r011, r012, r013, r014, r015, r016, r017, r042, r043,
                                   r044, r045, r046, r047, r052, r053, r054, r055, r056,
                                   r057, r062, r063, r064, r065, r066, r067, r072, r073,
                                   r074, r075, r076, r077, r082, r083, r084, r085, r086,
                                   r087, r092, r093, r094, r095, r096, r097, r112, r113,
                                   r114, r115, r116, r117, r122, r123, r124, r125, r126,
                                   r127, r132, r133, r134, r135, r136, r137, r142, r143,
                                   r144, r145, r146, r147, r152, r153, r154, r155, r156,
                                   r157, r162, r163, r164, r165, r166, r167)
end

struct Vern7Tableau{T, T2}
    c2::T2
    c3::T2
    c4::T2
    c5::T2
    c6::T2
    c7::T2
    c8::T2
    a021::T
    a031::T
    a032::T
    a041::T
    a043::T
    a051::T
    a053::T
    a054::T
    a061::T
    a063::T
    a064::T
    a065::T
    a071::T
    a073::T
    a074::T
    a075::T
    a076::T
    a081::T
    a083::T
    a084::T
    a085::T
    a086::T
    a087::T
    a091::T
    a093::T
    a094::T
    a095::T
    a096::T
    a097::T
    a098::T
    a101::T
    a103::T
    a104::T
    a105::T
    a106::T
    a107::T
    b1::T
    b4::T
    b5::T
    b6::T
    b7::T
    b8::T
    b9::T
    btilde1::T
    btilde4::T
    btilde5::T
    btilde6::T
    btilde7::T
    btilde8::T
    btilde9::T
    btilde10::T
    extra::Vern7ExtraStages{T, T2}
    interp::Vern7InterpolationCoefficients{T}
end

function Vern7Tableau(T::Type{T1}, T2::Type{T1}) where {T1}
    c2 = convert(T2, 0.005)
    c3 = convert(T2, 0.10888888888888888)
    c4 = convert(T2, 0.16333333333333333)
    c5 = convert(T2, 0.4555)
    c6 = convert(T2, 0.6095094489978381)
    c7 = convert(T2, 0.884)
    c8 = convert(T2, 0.925)
    a021 = convert(T, 0.005)
    a031 = convert(T, -1.07679012345679)
    a032 = convert(T, 1.185679012345679)
    a041 = convert(T, 0.04083333333333333)
    a043 = convert(T, 0.1225)
    a051 = convert(T, 0.6389139236255726)
    a053 = convert(T, -2.455672638223657)
    a054 = convert(T, 2.272258714598084)
    a061 = convert(T, -2.6615773750187572)
    a063 = convert(T, 10.804513886456137)
    a064 = convert(T, -8.3539146573962)
    a065 = convert(T, 0.820487594956657)
    a071 = convert(T, 6.067741434696772)
    a073 = convert(T, -24.711273635911088)
    a074 = convert(T, 20.427517930788895)
    a075 = convert(T, -1.9061579788166472)
    a076 = convert(T, 1.006172249242068)
    a081 = convert(T, 12.054670076253203)
    a083 = convert(T, -49.75478495046899)
    a084 = convert(T, 41.142888638604674)
    a085 = convert(T, -4.461760149974004)
    a086 = convert(T, 2.042334822239175)
    a087 = convert(T, -0.09834843665406107)
    a091 = convert(T, 10.138146522881808)
    a093 = convert(T, -42.6411360317175)
    a094 = convert(T, 35.76384003992257)
    a095 = convert(T, -4.3480228403929075)
    a096 = convert(T, 2.0098622683770357)
    a097 = convert(T, 0.3487490460338272)
    a098 = convert(T, -0.27143900510483127)
    a101 = convert(T, -45.030072034298676)
    a103 = convert(T, 187.3272437654589)
    a104 = convert(T, -154.02882369350186)
    a105 = convert(T, 18.56465306347536)
    a106 = convert(T, -7.141809679295079)
    a107 = convert(T, 1.3088085781613787)
    b1 = convert(T, 0.04715561848627222)
    b4 = convert(T, 0.25750564298434153)
    b5 = convert(T, 0.26216653977412624)
    b6 = convert(T, 0.15216092656738558)
    b7 = convert(T, 0.4939969170032485)
    b8 = convert(T, -0.29430311714032503)
    b9 = convert(T, 0.08131747232495111)
    # bhat1     =  convert(T,0.044608606606341174)
    # bhat4     =  convert(T,0.26716403785713727)
    # bhat5     =  convert(T,0.22010183001772932)
    # bhat6     =  convert(T,0.2188431703143157)
    # bhat7     =  convert(T,0.22898717054112028)
    # bhat10    =  convert(T,0.02029518466335628)
    btilde1 = convert(T, 0.002547011879931045)
    btilde4 = convert(T, -0.00965839487279575)
    btilde5 = convert(T, 0.04206470975639691)
    btilde6 = convert(T, -0.0666822437469301)
    btilde7 = convert(T, 0.2650097464621281)
    btilde8 = convert(T, -0.29430311714032503)
    btilde9 = convert(T, 0.08131747232495111)
    btilde10 = convert(T, -0.02029518466335628)

    extra = Vern7ExtraStages(T, T2)
    interp = Vern7InterpolationCoefficients(T)

    Vern7Tableau(c2, c3, c4, c5, c6, c7, c8, a021, a031, a032, a041, a043, a051, a053, a054,
                 a061, a063, a064, a065, a071, a073, a074, a075, a076, a081, a083, a084,
                 a085, a086, a087, a091, a093, a094, a095, a096, a097, a098, a101, a103,
                 a104, a105, a106, a107, b1, b4, b5, b6, b7, b8, b9, btilde1, btilde4,
                 btilde5, btilde6, btilde7, btilde8, btilde9, btilde10, extra, interp)
end
