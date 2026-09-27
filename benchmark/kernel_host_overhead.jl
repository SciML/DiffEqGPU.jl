using DiffEqGPU, SciMLBase, KernelAbstractions, StaticArrays
const backend = CPU()
const f = (u, p, t) -> SVector(p[1] * u[1])
const prob = ODEProblem{false}(f, SVector(1.0f0), (0.0f0, 1.0f0), SVector(0.2f0))
const ens = EnsembleProblem(prob; prob_func = (prob, ctx) -> remake(prob; p = SVector(0.2f0 + Float32(ctx.sim_id % 4) / 10)), safetycopy = false)
const alg = GPUTsit5()
const ealg = EnsembleGPUKernel(backend, 0.0)
function time_prep(n)
    GC.gc()
    t = @timed DiffEqGPU._prepare_kernel_problems(ens, backend, 1:n, nothing, SciMLBase.default_rng_func, nothing)
    return (t.time, t.bytes, t.gctime)
end
function time_kernel(n)
    probs, adapted = DiffEqGPU._prepare_kernel_problems(ens, backend, 1:n, nothing, SciMLBase.default_rng_func, nothing)
    GC.gc()
    t = @timed DiffEqGPU.vectorized_asolve(adapted, prob, alg; dt = 0.2f0, save_everystep = false)
    return (t.time, t.bytes, t.gctime)
end
function run_one(n)
    prep = time_prep(n)
    kernel = time_kernel(n)
    GC.gc()
    full = @timed solve(ens, alg, ealg; trajectories = n, adaptive = true, dt = 0.2f0, save_everystep = false)
    sol = full.value
    vals = (sol.u[1].u[end][1], sol.u[end].u[end][1])
    checksum = sum(s -> Float64(s.u[end][1]), sol.u)
    println(
        "N=$n prep_s=$(round(prep[1], digits = 4))",
        " prep_mb=$(round(prep[2] / 2^20, digits = 1))",
        " prep_gc=$(round(prep[3], digits = 4))",
        " kernel_s=$(round(kernel[1], digits = 4))",
        " total_s=$(round(full.time, digits = 4))",
        " total_mb=$(round(full.bytes / 2^20, digits = 1))",
        " total_gc=$(round(full.gctime, digits = 4))",
        " endpoints=$vals checksum=$checksum"
    )
    flush(stdout)
    return nothing
end
run_one(16)
for p in 14:20
    run_one(2^p)
end
