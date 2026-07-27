# Symbolic-Numeric GPU Acceleration with ModelingToolkit

[ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/) is a symbolic-numeric
computing system which allows for using symbolic transformations of equations before
code generation. The goal is to improve numerical simulations by first turning them into
the simplest set of equations to solve and exploiting things that normally cannot be done
by hand. Those exact features are also potentially useful for GPU computing, and thus this
tutorial showcases how to effectively use MTK with DiffEqGPU.jl.

!!! warn
    
    This tutorial currently only works for ODEs defined by ModelingToolkit. More work
    will be required to support DAEs in full. This is work that is ongoing.

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
    
    The two blocks below do not currently run. `EnsembleGPUKernel` moves the vector of
    problems to the device, which requires the whole `ImmutableODEProblem` to be
    `isbitstype`. An MTK-generated `ODEFunction` never is: `f.sys`, `f.observed`,
    `f.initialization_data` and `f.nlstep_data` all hold non-inline data, so the
    conversion fails with `CuArray only supports element types that are allocated
    inline`. Making `u0` and `p` static, as above, is necessary but not sufficient.
    See [DiffEqGPU.jl#375](https://github.com/SciML/DiffEqGPU.jl/issues/375). These
    blocks are left unevaluated until the device conversion strips those fields.

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
