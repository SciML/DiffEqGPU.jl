using DiffEqGPU, SciMLBase, KernelAbstractions, StaticArrays, Statistics
using SciMLBase.EnsembleAnalysis

const backend = CPU()
function lorenz(u, p, t)
    σ, ρ, β = p
    return SVector(σ * (u[2] - u[1]), u[1] * (ρ - u[3]) - u[2], u[1] * u[2] - β * u[3])
end
const u0 = @SVector [1.0f0, 0.0f0, 0.0f0]
const p0 = @SVector [10.0f0, 28.0f0, 8 / 3.0f0]
const prob = ODEProblem{false}(lorenz, u0, (0.0f0, 1.0f0), p0)
const prob_func = (prob, ctx) -> remake(prob; p = p0 .* (1 + Float32(ctx.sim_id % 97) / 1000))
const ens = EnsembleProblem(prob; prob_func, safetycopy = false)
const alg = GPUTsit5()
const ealg = EnsembleGPUKernel(backend, 0.0)
const saveat = 0.0f0:0.01f0:1.0f0
const REPS = 5

function solve_endpoints(n)
    return solve(
        ens, alg, ealg;
        trajectories = n, adaptive = true, dt = 0.1f0, save_everystep = false
    )
end

function solve_saveat(n)
    return solve(
        ens, alg, ealg;
        trajectories = n, adaptive = true, dt = 0.1f0, saveat
    )
end

function endpoint_pass(sol)
    s = 0.0
    @inbounds for x in sol.u
        s += Float64(x.u[end][1])
    end
    return s
end

function median_elapsed(f, reps = REPS)
    ts = Vector{Float64}(undef, reps)
    for i in 1:reps
        GC.gc()
        ts[i] = @elapsed f()
    end
    return median(ts)
end

# Warmup
solve_endpoints(16)
endpoint_pass(solve_endpoints(16))
timeseries_steps_meanvar(solve_saveat(16))

const TREE = isempty(ARGS) ? "branch" : ARGS[1]
println("tree=$TREE backend=CPU threads=$(Threads.nthreads()) reps=$REPS")
println("metric=median wall seconds; checksum=sum of endpoint u[1]; analysis=timeseries_steps_meanvar")
flush(stdout)

for p in 14:20
    n = 2^p
    local_sol = Ref{Any}()
    solve_t = median_elapsed() do
        local_sol[] = solve_endpoints(n)
    end
    sol = local_sol[]
    @assert sol.u isa Vector
    loop_t = median_elapsed(() -> endpoint_pass(sol))
    checksum = endpoint_pass(sol)
    analysis_sol = Ref{Any}()
    saveat_solve_t = median_elapsed() do
        analysis_sol[] = solve_saveat(n)
    end
    analysis_t = median_elapsed(() -> timeseries_steps_meanvar(analysis_sol[]))
    println(
        "N=$n",
        " solve_s=$(round(solve_t; digits = 4))",
        " endpoint_loop_s=$(round(loop_t; digits = 4))",
        " saveat_solve_s=$(round(saveat_solve_t; digits = 4))",
        " timeseries_steps_meanvar_s=$(round(analysis_t; digits = 4))",
        " checksum=$checksum",
        " utype=$(typeof(sol.u))"
    )
    flush(stdout)
    local_sol[] = nothing
    analysis_sol[] = nothing
    sol = nothing
end
