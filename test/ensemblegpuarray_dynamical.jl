using DiffEqGPU
using OrdinaryDiffEq: Tsit5
using SciMLBase: DynamicalODEProblem, SecondOrderODEProblem, EnsembleProblem,
    EnsembleSerial, remake, solve
using Test

# harmonic oscillators: dv/dt = -k x, dx/dt = v, with a different k per trajectory
dv(v, x, p, t) = -p[1] .* x
dx(v, x, p, t) = v
function dv!(a, v, x, p, t)
    for j in eachindex(a)
        a[j] = -p[1] * x[j]
    end
    return nothing
end
function dx!(a, v, x, p, t)
    for j in eachindex(a)
        a[j] = v[j]
    end
    return nothing
end

v0 = Float32[0.5, 0]
x0 = Float32[1, -2]
tspan = (0.0f0, 2.0f0)
ks = Float32[1, 2, 3, 4]
prob_func = (prob, ctx) -> remake(prob, p = Float32[ks[ctx.sim_id]])

function exact(k, t)
    w = sqrt(k)
    return vcat(-w .* x0 .* sin(w * t) .+ v0 .* cos(w * t), x0 .* cos(w * t) .+ v0 ./ w .* sin(w * t))
end

probs = [
    "DynamicalODEProblem oop" => DynamicalODEProblem{false}(dv, dx, v0, x0, tspan, Float32[1]),
    "DynamicalODEProblem iip" => DynamicalODEProblem{true}(dv!, dx!, v0, x0, tspan, Float32[1]),
    "SecondOrderODEProblem" => SecondOrderODEProblem{false}(dv, v0, x0, tspan, Float32[1]),
]

@testset "$name" for (name, prob) in probs
    ens = EnsembleProblem(prob; prob_func, safetycopy = false)
    kw = (; trajectories = length(ks), abstol = 1.0f-7, reltol = 1.0f-7, save_everystep = false)
    sol = solve(ens, Tsit5(), EnsembleCPUArray(); kw...)
    ref = solve(ens, Tsit5(), EnsembleSerial(); kw...)
    for (i, k) in enumerate(ks)
        @test collect(sol.u[i].u[end]) ≈ exact(k, tspan[2]) atol = 1.0f-3
        @test collect(sol.u[i].u[end]) ≈ collect(ref.u[i].u[end]) atol = 1.0f-3
    end
end
