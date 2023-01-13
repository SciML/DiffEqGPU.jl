# [Within-Method GPU Parallelism of Ordinary Differential Equation Solves](@id withingpu)

Within-Method GPU Parallelism for ODE solvers is a method for accelerating large ODE
solves with regularity, i.e., only using array-based “vectorized” operations like
linear algebra, maps, and broadcast statements. In these cases, the solve can be GPU
accelerated simply by placing the initial condition array on the GPU. As a quick example:

```julia
using OrdinaryDiffEq, CUDA, LinearAlgebra
function f(du, u, p, t)
    mul!(du, A, u)
end

A = cu(-rand(3, 3))
u0 = cu([1.0; 0.0; 0.0])
tspan = (0.0f0, 100.0f0)

prob = ODEProblem(ff, u0, tspan)
sol = solve(prob, Tsit5())
sol = solve(prob, Rosenbrock23())
```

Notice that both stiff and non-stiff ODE solvers were used here.

!!! note
    Time span was changed to `Float32` types, as GPUs generally have very slow `Float64`
    operations, usually around 1/32 of the speed of `Float32`. `cu(x)` on an array
    automatically changes an `Array{Float64}` to a `CuArray{Float32}`. If this is not
    intended, use the `CuArray` constructor directly. For more information on GPU
    `Float64` performance issues, search around Google for
    [discussions like this](https://www.techpowerup.com/forums/threads/nerfed-fp64-performance-in-consumer-gpu-cards.272732/).

!!! warn
    `Float32` precision is sometimes not enough precision to accurately solve a
    stiff ODE. Make sure that the precision is necessary by investigating the condition
    number of the Jacobian. If this value is well-above `1e8`, use `Float32` with caution!

## Restrictions of CuArrays

Note that all the rules of [CUDA.jl](https://cuda.juliagpu.org/stable/) apply when
`CuArrays` are being used in the solver. While for most of the `AbstractArray` interface
they act similarly to `Array`s, such as having valid broadcasting operations (`x .* y`)
defined, they will work on GPUs. For more information on the rules and restrictions of
`CuArrays`, see
[this page from the CUDA.jl documentation](https://cuda.juliagpu.org/stable/usage/array/).
