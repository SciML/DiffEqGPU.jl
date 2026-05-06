# Symbolic-Numeric GPU Acceleration with ModelingToolkit

[ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/) is a symbolic-numeric
computing system which allows for using symbolic transformations of equations before
code generation. The goal is to improve numerical simulations by first turning them into
the simplest set of equations to solve and exploiting things that normally cannot be done
by hand. Those exact features are also potentially useful for GPU computing, and thus this
tutorial showcases how to effectively use MTK with DiffEqGPU.jl.

!!! warn
    
    This tutorial currently only works for ODEs defined by ModelingToolkit. More work
    will be required to support DAEs in full. This is work that is ongoing and expected
    to be completed by the summer of 2025.

The core aspect to doing this right is two things. First of all, MTK respects the types
chosen by the user, and thus in order for GPU kernel generation to work the user needs
to ensure that the problem that is built uses static structures. For example this means
that the `u0` and `p` specifications should use static arrays. This looks as follows:

```@example mtk
using OrdinaryDiffEq, ModelingToolkit, StaticArrays, SciMLBase
using ModelingToolkit: t_nounits as t, D_nounits as D

@parameters σ ρ β
@variables x(t) y(t) z(t)

eqs = [D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z]

# Two MTK options matter for `EnsembleGPUKernel` below
# (see SciML/DiffEqGPU.jl#375):
#   * `split = false` keeps all parameters in a single SVector instead of
#     splitting them into multiple `Vector`-backed `MTKParameters` fields,
#     which CuArray cannot store inline.
#   * `build_initializeprob = false` (passed to ODEProblem) skips the
#     `OverrideInitData` initialization-problem metadata; otherwise the
#     `MTKChainRulesCoreExt` path errors with
#     `type Nothing has no field oop_reconstruct_u0_p` during the GPU
#     `remake` in the ensemble below.
@mtkcompile sys = System(eqs, t) split=false

u0 = @SVector [D(x) => 2.0f0,
    x => 1.0f0,
    y => 0.0f0,
    z => 0.0f0]

p = @SVector [σ => 28.0f0,
    ρ => 10.0f0,
    β => 8.0f0 / 3.0f0]

tspan = (0.0f0, 100.0f0)
prob = ODEProblem{false, SciMLBase.FullSpecialize}(
    sys, [u0; p], tspan; build_initializeprob = false
)
sol = solve(prob, Tsit5())
```

with the core aspect to notice are the `SA`
[StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) annotations on the parts
for the problem construction, along with the use of `Float32`.

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

```@example mtk
using DiffEqGPU, CUDA
function prob_func2(prob, ctx)
    u0, p = sym_setter(prob, SVector{3}(rand(Float32, 3)))
    remake(prob, u0 = u0, p = p)
end

monteprob = EnsembleProblem(prob, prob_func = prob_func2, safetycopy = false)
sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend()),
    trajectories = 10_000)
```

We can then use symbolic indexing on the result to inspect it. The
per-trajectory solutions returned by `EnsembleGPUKernel` are minimal
`ImmutableODESolution`s (their state vectors are plain `SVector`s and don't
carry the symbolic metadata themselves), so we use a `getu` accessor built
from `sys` to pick out the `y` component of each trajectory's final state:

```@example mtk
y_at_end = SymbolicIndexingInterface.getu(sys, y)
[y_at_end(sol.u[i].u[end]) for i in 1:length(sol.u)]
```
