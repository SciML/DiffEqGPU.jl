using DiffEqGPU, BenchmarkTools
using OrdinaryDiffEqTsit5, StableRNGs

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

function lorenz(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    return du[3] = u[1] * u[2] - p[3] * u[3]
end

u0 = Float32[1.0; 0.0; 0.0]
tspan = (0.0f0, 50.0f0)
p = (10.0f0, 28.0f0, 8 / 3.0f0)
prob = ODEProblem(lorenz, u0, tspan, p)
pre_p = [rand(rng, Float32, 3) for i in 1:32]
prob_func = (prob, ctx) -> remake(prob, p = pre_p[ctx.sim_id] .* p)
monteprob = EnsembleProblem(prob; prob_func)

# =============================================================================
# Ensemble solving on the CPU-array backend
# =============================================================================

SUITE["ensemble"] = BenchmarkGroup()

SUITE["ensemble"]["cpu_array_16"] = @benchmarkable solve(
    $monteprob, Tsit5(), EnsembleCPUArray(); trajectories = 16, saveat = 1.0f0
)
SUITE["ensemble"]["cpu_array_32"] = @benchmarkable solve(
    $monteprob, Tsit5(), EnsembleCPUArray(); trajectories = 32, saveat = 1.0f0
)
SUITE["ensemble"]["threads_16"] = @benchmarkable solve(
    $monteprob, Tsit5(), EnsembleThreads(); trajectories = 16, saveat = 1.0f0
)
SUITE["ensemble"]["cpu_array_batch"] = @benchmarkable solve(
    $monteprob, Tsit5(), EnsembleCPUArray(); trajectories = 32,
    batch_size = 8, saveat = 1.0f0
)
