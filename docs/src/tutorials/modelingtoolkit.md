# Symbolic-Numeric GPU Acceleration with ModelingToolkit

[ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/) is a symbolic-numeric
computing system which allows for using symbolic transformations of equations before
code generation. The goal is to improve numerical simulations by first turning them into
the simplest set of equations to solve and exploiting things that normally cannot be done
by hand. Those exact features are also potentially useful for GPU computing, and thus this
tutorial showcases how to effectively use MTK with DiffEqGPU.jl.

!!! note

    `EnsembleGPUKernel` supports mass-matrix DAEs whose ModelingToolkit initialization
    problem can be converted to a static nonlinear problem. See
    [DAE initialization](@ref dae_initialization) for an example and the current
    restrictions. Other DAE formulations may still require `EnsembleGPUArray`.

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

Symbolic problem transformations are inherently dynamic, so changes to `u0` and `p`
should be made on the CPU before the per-trajectory problems are sent to the GPU. This can
be done using the
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

!!! note

    `EnsembleProblem.prob_func` is evaluated on the host for every trajectory before
    `EnsembleGPUKernel` launches. The symbolic setter therefore does not need to compile
    for the device. The problem returned by `prob_func` is then converted to static,
    device-compatible storage by `make_prob_compatible`.

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
copy per trajectory inside the GPU kernel. Square systems use `SimpleTrustRegion`, while
rectangular nonlinear least-squares systems use `SimpleGaussNewton`, both from
SimpleNonlinearSolve.jl. The resulting consistent state and parameters are used to start
the ODE solve. ModelingToolkit's default `SCCNonlinearProblem` representation is lowered
to an immutable nonlinear problem plus statically typed SCC block metadata. The blocks
are solved sequentially in the kernel: nonlinear blocks use `SimpleTrustRegion`, and
linear blocks use DiffEqGPU's device-compatible static square solve (closed-form for
blocks of size three or smaller and pivoted LU otherwise). The mutable SCC caches and
linear-problem update wrappers are not placed in the kernel.

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

  - Square, underdetermined, and overdetermined nonlinear least-squares initialization
    problems are supported. Square systems use `SimpleTrustRegion`; rectangular systems
    use the equivalent static normal-equation Gauss-Newton step because rectangular
    `StaticArray` factorizations are not device-compatible. Their Jacobian must have the
    rank required by that step.
  - Lower and upper bounds are supported through a smooth transformation to unconstrained
    variables. A solution exactly on a finite bound is represented by a limiting
    unconstrained value and can therefore converge less robustly than an interior solution.
  - ModelingToolkit's state and parameter initialization maps may directly select,
    reorder, and repack numeric values from the ODE and initialization problems. DiffEqGPU
    traces those operations on the host and stores only static gather recipes in the
    kernel. Fallback getters that evaluate derived symbolic expressions are not yet
    lowered.
  - Ordinary nonlinear and linear SCC initialization blocks are supported. SCC
    initialization containing Modelica homotopy blocks is not supported.

Structured `MTKParameters` storage is converted recursively to static storage, so this
path does not require `split = false`.
