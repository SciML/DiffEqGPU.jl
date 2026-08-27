# Symbolic-Numeric GPU Acceleration with ModelingToolkit

[ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/) is a symbolic-numeric
computing system which allows for using symbolic transformations of equations before
code generation. The goal is to improve numerical simulations by first turning them into
the simplest set of equations to solve and exploiting things that normally cannot be done
by hand. Those exact features are also potentially useful for GPU computing, and thus this
tutorial showcases how to effectively use MTK with DiffEqGPU.jl.

!!! note

    `EnsembleGPUKernel` supports mass-matrix DAEs whose ModelingToolkit initialization
    problem is square and unbounded. See [DAE initialization](@ref dae_initialization) for
    an example and the current restrictions. Other DAE formulations may still require
    `EnsembleGPUArray`.

The core aspect to doing this right is two things. First of all, MTK respects the types
chosen by the user, and thus in order for GPU kernel generation to work the user needs
to ensure that the problem that is built uses static structures. For example this means
that the `u0` and `p` specifications should use static arrays. This looks as follows:

```@example mtk
using OrdinaryDiffEq, ModelingToolkit, StaticArrays
using ModelingToolkit: t_nounits as t, D_nounits as D

@parameters σ ρ β
@variables x(t) y(t) z(t)

eqs = [D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z]

@named lorenz = System(eqs, t)
sys = mtkcompile(lorenz; split = false)

op = @SVector [D(x) => 2.0f0,
    x => 1.0f0,
    y => 0.0f0,
    z => 0.0f0,
    σ => 28.0f0,
    ρ => 10.0f0,
    β => 8.0f0 / 3.0f0]

tspan = (0.0f0, 100.0f0)
prob = ODEProblem{false}(sys, op, tspan)
sol = solve(prob, Tsit5())
```

There are two things to notice here. The first is the `split = false` argument to
`mtkcompile`. By default MTK builds an `MTKParameters` object, which stores the parameters
in separate buffers grouped by how they are used. That object holds `Vector`s and is
therefore not `isbits`, so it cannot be placed into a GPU kernel. `split = false` instead
puts every parameter into a single flat buffer.

The second is that the operating point `op` is given as a single
[StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) vector of pairs, using
`Float32` values. MTK builds `u0` and `p` in the same container type it was handed, so a
static vector in gives static vectors out:

```@example mtk
typeof(prob.u0), typeof(prob.p)
```

Both are `isbits` and thus usable from a GPU kernel.

Now one of the difficulties for building an ensemble problem is that we must make a kernel
for how to construct the problems, but the use of symbolics is inherently dynamic. As such,
we need to make sure that any changes to `u0` and `p` are done on the CPU and that we
compile an optimized function to run on the GPU. This can be done using the
[SymbolicIndexingInterface.jl](https://docs.sciml.ai/SymbolicIndexingInterface/stable/).
For example, let's define a problem which randomizes the choice of `(σ, ρ, β)`. We do this
by first constructing the function that will change a `prob.p` object into the updated
form by changing those 3 values by using the `setsym_oop` as follows:

```@example mtk
using SymbolicIndexingInterface
sym_setter = setsym_oop(sys, [σ, ρ, β])
nothing # hide
```

The return `sym_setter` is our optimized function, let's see it in action:

```@example mtk
u0, p = sym_setter(prob, SVector{3}(rand(Float32, 3)))
```

Notice it takes in the vector of values for `[σ, ρ, β]` and spits out the new `u0, p`. So
we can build and solve an MTK generated ODE on the GPU using the following:

!!! warning

    The generic SymbolicIndexingInterface ensemble transformation below does not
    currently run inside `EnsembleGPUKernel`: a host-side symbolic setter is not
    automatically lowered to a device-compatible function. Making `u0` and `p` static
    is necessary but not sufficient for this workflow. See
    [DiffEqGPU.jl#375](https://github.com/SciML/DiffEqGPU.jl/issues/375). The DAE path
    described below is separate: it recognizes ModelingToolkit initialization maps and
    replaces them with static gather recipes before launching the kernel.

```julia
using DiffEqGPU, CUDA
function prob_func2(prob, ctx)
    u0, p = sym_setter(prob, SVector{3}(rand(Float32, 3)))
    remake(prob, u0 = u0, p = p)
end

monteprob = EnsembleProblem(prob, prob_func = prob_func2, safetycopy = false)
sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend()),
    trajectories = 10_000)
```

We can then using symbolic indexing on the result to inspect it:

```julia
[sol.u[i][y] for i in 1:length(sol.u)]
```

## [DAE initialization](@id dae_initialization)

ModelingToolkit can generate a nonlinear initialization problem for a mass-matrix DAE.
`EnsembleGPUKernel` converts that problem to static storage on the host, then solves one
copy per trajectory inside the GPU kernel with `SimpleTrustRegion` from
SimpleNonlinearSolve.jl. The resulting consistent state and parameters are used to start
the ODE solve.

For example, the Cartesian pendulum can be initialized and solved as follows:

```julia
using CUDA, DiffEqGPU, ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D

@parameters g = 9.81 L = 1.0
@variables px(t) py(t) [state_priority = 10] pλ(t)

eqs = [
    D(D(px)) ~ pλ * px / L
    D(D(py)) ~ pλ * py / L - g
    px^2 + py^2 ~ L^2
]

@mtkcompile pendulum = ODESystem(eqs, t, [px, py, pλ], [g, L])

prob = ODEProblem(
    pendulum,
    [py => 0.99, D(px) => 0.0],
    (0.0, 1.0);
    guesses = [pλ => 0.0, px => 0.1, D(py) => 0.0],
    use_scc = false,
)

ensemble_prob = EnsembleProblem(prob; safetycopy = false)
sol = solve(
    ensemble_prob,
    GPURodas5P(),
    EnsembleGPUKernel(CUDA.CUDABackend());
    trajectories = 10_000,
    dt = 0.01,
    adaptive = false,
)
```

The current initialization path has the following restrictions:

  - The nonlinear initialization problem must be square. Both underdetermined and
    overdetermined problems throw an error before the GPU kernel is launched. Supply
    enough initial conditions and guesses for ModelingToolkit to produce a square system.
  - Bounds on the nonlinear initialization problem are not supported.
  - ModelingToolkit's state and parameter initialization maps must copy or restructure
    numeric values from the ODE and initialization problems. DiffEqGPU traces those maps
    on the host and stores only static gather recipes in the kernel.

Structured `MTKParameters` storage is converted recursively to static storage, so this
path does not require `split = false`.
