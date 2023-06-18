using DiffEqGPU, StaticArrays, CUDA, Adapt, OrdinaryDiffEq

include("utils.jl")

@info "Testing lower level API for EnsembleGPUKernel"

trajectories = 10_000
function lorenz(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    du1 = σ * (u[2] - u[1])
    du2 = u[1] * (ρ - u[3]) - u[2]
    du3 = u[1] * u[2] - β * u[3]
    return SVector{3}(du1, du2, du3)
end

function lorenz_jac(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    x = u[1]
    y = u[2]
    z = u[3]
    J11 = -σ
    J21 = ρ - z
    J31 = y
    J12 = σ
    J22 = -1
    J32 = x
    J13 = 0
    J23 = -x
    J33 = -β
    return SMatrix{3, 3}(J11, J21, J31, J12, J22, J32, J13, J23, J33)
end

function lorenz_tgrad(u, p, t)
    return SVector{3, eltype(u)}(0.0, 0.0, 0.0)
end

u0 = @SVector [1.0f0; 0.0f0; 0.0f0]
tspan = (0.0f0, 10.0f0)
p = @SVector [10.0f0, 28.0f0, 8 / 3.0f0]

func = ODEFunction(lorenz, jac = lorenz_jac, tgrad = lorenz_tgrad)
prob = ODEProblem{false}(func, u0, tspan, p)

## Building different problems for different parameters
batch = 1:trajectories
probs = map(batch) do i
    remake(prob, p = (@SVector rand(Float32, 3)) .* p)
end

## Move the arrays to the GPU
gpu_probs = adapt(backend, probs)

## Finally use the lower API for faster solves! (Fixed time-stepping)

algs = (GPUTsit5(), GPUVern7(), GPUVern9(), GPURosenbrock23(), GPURodas4())

for alg in algs
    @info alg

    DiffEqGPU.vectorized_solve(gpu_probs, prob, alg;
        save_everystep = false, dt = 0.1f0)

    DiffEqGPU.vectorized_asolve(gpu_probs, prob, alg;
        save_everystep = false, dt = 0.1f0)
end

@info "Testing lower level API for EnsembleGPUArray"

@time sol = DiffEqGPU.vectorized_map_solve(probs, Tsit5(), EnsembleGPUArray(backend, 0.0),
    batch, false, dt = 0.001f0,
    save_everystep = false, dense = false)

## Adaptive time-stepping (Notice the boolean argument)
@time sol = DiffEqGPU.vectorized_map_solve(probs, Tsit5(), EnsembleGPUArray(backend, 0.0),
    batch, true, dt = 0.001f0,
    save_everystep = false, dense = false)
