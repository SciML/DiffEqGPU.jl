using DiffEqGPU, StaticArrays, OrdinaryDiffEq, LinearAlgebra

include("../../utils.jl")

function rober(u, p, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    return @SVector [
        -k₁ * y₁ + k₃ * y₂ * y₃,
        k₁ * y₁ - k₂ * y₂^2 - k₃ * y₂ * y₃,
        y₁ + y₂ + y₃ - 1,
    ]
end
function rober_jac(u, p, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    return @SMatrix[
        (k₁ * -1) (y₃ * k₃) (k₃ * y₂)
        k₁ (y₂ * k₂ * -2 + y₃ * k₃ * -1) (k₃ * y₂ * -1)
        0 (y₂ * 2 * k₂) (0)
    ]
end
M = @SMatrix [
    1.0f0 0.0f0 0.0f0
    0.0f0 1.0f0 0.0f0
    0.0f0 0.0f0 0.0f0
]
ff = ODEFunction(rober, mass_matrix = M)
prob = ODEProblem(
    ff, @SVector([1.0f0, 0.0f0, 0.0f0]), (0.0f0, 1.0f5),
    (0.04f0, 3.0f7, 1.0f4)
)

monteprob = EnsembleProblem(prob, safetycopy = false)

alg = GPURosenbrock23()

# dt must match the Float32 eltype of u0/p. A Float64 dt makes calc_W build a
# Float64 StaticWOperator that OrdinaryDiffEqRosenbrock's JacReuseState (typed
# from the Float32 state) cannot store, erroring in the cached_W assignment.
bench_sol = solve(
    prob, Rosenbrock23(), dt = 0.1f0, abstol = 1.0f-5, reltol = 1.0f-5,
    initializealg = BrownFullBasicInit()
)

sol = solve(
    monteprob, alg, EnsembleGPUKernel(backend),
    trajectories = 2,
    dt = 0.1f0,
    adaptive = true, abstol = 1.0f-5, reltol = 1.0f-5
)

@test norm(bench_sol.u[1] - sol.u[1].u[1]) < 8.0e-4
@test norm(bench_sol.u[end] - sol.u[1].u[end]) < 8.0e-4

@testset "UniformScaling mass matrix" for mass in (I, 2.0f0I, 0.0f0I)
    initial = iszero(mass.λ) ? 0.0f0 : 1.0f0
    f = ODEFunction{false}((u, p, t) -> -u; mass_matrix = mass)
    prob = ODEProblem(f, SVector(initial), (0.0f0, 1.0f0))
    compatible = @inferred DiffEqGPU.make_prob_compatible(prob)
    @test compatible.f.mass_matrix === mass
    sol = solve(
        EnsembleProblem(prob), GPURosenbrock23(), EnsembleGPUKernel(backend, 0.0);
        trajectories = 2, dt = 0.001f0, adaptive = true,
        abstol = 1.0f-8, reltol = 1.0f-8, save_everystep = false
    )
    expected = iszero(initial) ? initial : exp(-1.0f0 / mass.λ)
    @test all(s -> isapprox(s.u[end][1], expected; atol = 2.0f-6), sol.u)
end

# Analytic check for M * u' = -u with nonsingular M = 2I (issue #530).
# Expected endpoint is u0 * exp(-t/2); first component is exp(-1/2).
@testset "Kvaerno nonsingular mass matrix" begin
    f(u, p, t) = -u
    jac(u, p, t) = -one(SMatrix{2, 2, Float64})
    expected = exp(-0.5) * SVector(1.0, 2.0)
    for alg in (GPUKvaerno3(), GPUKvaerno5()), adaptive in (false, true)
        prob = ODEProblem(
            ODEFunction{false}(f; jac, mass_matrix = 2.0I),
            SVector(1.0, 2.0), (0.0, 1.0)
        )
        sol = solve(
            EnsembleProblem(prob; safetycopy = false), alg,
            EnsembleGPUKernel(backend, 0.0); trajectories = 2, adaptive,
            dt = 0.01, abstol = 1.0e-8, reltol = 1.0e-8, save_everystep = false
        )
        @test sol.u[1].u[end] ≈ expected rtol = 1.0e-5 atol = 1.0e-8
    end
end

@testset "Kvaerno rejects singular mass matrix" begin
    f = ODEFunction{false}((u, p, t) -> -u; mass_matrix = 0.0I)
    prob = ODEProblem(f, SVector(0.0), (0.0, 1.0))
    for alg in (GPUKvaerno3(), GPUKvaerno5())
        @test_throws ArgumentError solve(
            EnsembleProblem(prob; safetycopy = false), alg,
            EnsembleGPUKernel(backend, 0.0); trajectories = 2, adaptive = false,
            dt = 0.01, save_everystep = false
        )
    end
end
