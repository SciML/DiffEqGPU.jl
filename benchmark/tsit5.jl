using DiffEqGPU, SciMLBase, StaticArrays, KernelAbstractions, Adapt, BenchmarkTools

const backend = if get(ENV, "GROUP", "CPU") == "CUDA"
    using CUDA
    CUDA.CUDABackend()
else
    CPU()
end

function lorenz(u::SVector{3, T}, p, t) where {T}
    return SVector(
        T(10) * (u[2] - u[1]),
        p[1] * u[1] - u[2] - u[1] * u[3],
        u[1] * u[2] - T(8 / 3) * u[3]
    )
end

function benchmark_tsit5(::Type{T}, n, adaptive) where {T}
    prob = ODEProblem{false}(
        lorenz, SVector{3, T}(1, 0, 0), (zero(T), one(T)), SVector{1, T}(21)
    )
    probs = [remake(prob; p = SVector{1, T}(rho)) for rho in range(0, 21; length = n)]
    gpu_probs = adapt(backend, adapt.((backend,), probs))
    solver = adaptive ? DiffEqGPU.vectorized_asolve : DiffEqGPU.vectorized_solve
    function solve_once()
        result = solver(
            gpu_probs, prob, GPUTsit5(); dt = T(1.0e-3),
            reltol = T(1.0e-6), abstol = T(1.0e-9), save_everystep = false
        )
        KernelAbstractions.synchronize(backend)
        return result
    end
    ts, us = solve_once()
    trial = @benchmark $solve_once() samples = 20 evals = 1
    return (; T, n, adaptive, minimum_ms = minimum(trial).time / 1.0e6, final = Array(us)[end, end])
end

for T in (Float32, Float64), adaptive in (false, true)
    @show benchmark_tsit5(T, parse(Int, get(ENV, "TRAJECTORIES", "4096")), adaptive)
end
