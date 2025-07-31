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
using OrdinaryDiffEqTsit5, ModelingToolkit, StaticArrays
using ModelingToolkit: t_nounits as t, D_nounits as D

@parameters σ ρ β
@variables x(t) y(t) z(t)

eqs = [D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z]

@mtkbuild sys = ODESystem(eqs, t)
u0 = SA[D(x) => 2f0,
    x => 1f0,
    y => 0f0,
    z => 0f0]

p = SA[σ => 28f0,
    ρ => 10f0,
    β => 8f0 / 3f0]

tspan = (0f0, 100f0)
prob = ODEProblem{false}(sys, u0, tspan, p)
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
```

The return `sym_setter` is our optimized function, let's see it in action:

```@example mtk
u0, p = sym_setter(prob,@SVector(rand(Float32,3)))
```

Notice it takes in the vector of values for `[σ, ρ, β]` and spits out the new `u0, p`. So
we can build and solve an MTK generated ODE on the GPU using the following:

```@example mtk
using DiffEqGPU, CUDA
function prob_func2(prob, i, repeat)
    u0, p = sym_setter(prob,@SVector(rand(Float32,3)))
    remake(prob, u0 = u0, p = p)
end

monteprob = EnsembleProblem(prob, prob_func = prob_func2, safetycopy = false)
sol = solve(monteprob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend()),
    trajectories = 10_000)
```

We can then using symbolic indexing on the result to inspect it:

```@example mtk
[sol.u[i][y] for i in 1:length(sol.u)]
```
