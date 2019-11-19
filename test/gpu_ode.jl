using GPUifyLoops, CuArrays, SimpleDiffEq
using StaticArrays

function loop(u, p, t)
    @inbounds begin
        σ = p[1]; ρ = p[2]; β = p[3]
        du1 = σ*(u[2]-u[1])
        du2 = u[1]*(ρ-u[3]) - u[2]
        du3 = u[1]*u[2] - β*u[3]
        return SVector{3}(du1, du2, du3)
    end
end
const func = ODEFunction(loop)
u0 = 10ones(Float32,3)
const su0= SVector{3}(u0)
const dt = 1f-1
const tspan = (0.0f0, 10.0f0)

const odeoop = ODEProblem{false}(loop, SVector{3}(u0), (0.0f0, 10.0f0),  Float32[10, 28, 8/3])
sol2 = solve(odeoop,GPUSimpleTsit5(),dt=dt)
cps = Array([@SVector [10f0,28f0,8/3f0] for i in 1:32])
ps = CuArray([@SVector [10f0,28f0,8/3f0] for i in 1:32])
CuArrays.allowscalar(false)

function f(p)
    solve(odeoop,GPUSimpleTsit5(),dt=dt)
end

map(f,cps)

#@code_warntype map(f,ps)
_f = GPUifyLoops.contextualize(f)
map(_f,ps);

function f2(p)
    prob = ODEProblem{false}(loop, su0, tspan,  p)
    solve(odeoop,GPUSimpleTsit5(),dt=dt)
end

_f2 = GPUifyLoops.contextualize(f2)
map(_f2,ps)


using GPUifyLoops, SimpleDiffEq, StaticArrays
using CuArrays, Cthulhu

function loop(u, p, t)
    @inbounds begin
        σ = p[1]; ρ = p[2]; β = p[3]
        du1 = σ*(u[2]-u[1])
        du2 = u[1]*(ρ-u[3]) - u[2]
        du3 = u[1]*u[2] - β*u[3]
        return SVector{3}(du1, du2, du3)
    end
end

u0 = 10ones(Float32,3)
su0= SVector{3}(u0)
dt = 1f-1

odeoop = ODEProblem{false}(loop, SVector{3}(u0), (0.0f0, 10.0f0),  Float32[10, 28, 8/3])
sol2 = solve(odeoop,GPUSimpleATsit5(),save_everystep=false).u[end]
ps = CuArray([@SVector [10f0,28f0,8/3f0] for i in 1:10])
CuArrays.allowscalar(false)

function f(p)
    prob = ODEProblem{false}(loop, su0, (0.0f0, 10.0f0),  p)
    solve(prob,GPUSimpleATsit5(),save_everystep=false).u[end]
end

_f = GPUifyLoops.contextualize(f)
map(_f,ps)

function f(p)
    prob = ODEProblem{false}(loop, su0, (0.0f0, 10.0f0),  p)
    solve(prob,GPUSimpleATsit5(),saveat=0.0f0:dt:10.0f0).u
end

_f = GPUifyLoops.contextualize(f)
map(_f,ps)

function f2(p)
    prob = ODEProblem{false}(loop, su0, (0.0f0, 10.0f0),  p)
    solve(prob,GPUSimpleATsit5(),saveat=0.0f0:dt:10.0f0)
end

_f2 = GPUifyLoops.contextualize(f2)
map(_f2,ps)

using GPUifyLoops, CuArrays, SimpleDiffEq
using StaticArrays, Cthulhu

function loop(u, p, t)
    @inbounds begin
        σ = p[1]; ρ = p[2]; β = p[3]
        du1 = σ*(u[2]-u[1])
        du2 = u[1]*(ρ-u[3]) - u[2]
        du3 = u[1]*u[2] - β*u[3]
        return SVector{3}(du1, du2, du3)
    end
end
function liip(du, u, p, t)
    σ = p[1]; ρ = p[2]; β = p[3]
    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
    return nothing
end

u0 = 10ones(Float32,3)
su0= SVector{3}(u0)
dt = 1f-1

odeoop = ODEProblem{false}(loop, SVector{3}(u0), (0.0f0, 10.0f0),  Float32[10, 28, 8/3])
sol2 = solve(odeoop,GPUSimpleTsit5(),dt=dt)
ps = CuArray([@SVector [10f0,28f0,8/3f0] for i in 1:600])
_ps = [@SVector [10f0,28f0,8/3f0] for i in 1:600]
CuArrays.allowscalar(false)

function f(p)
    prob = ODEProblem{false}(loop, su0, (0.0f0, 10.0f0),  p)
    solve(prob,GPUSimpleTsit5(),dt=dt).u
end

_f = GPUifyLoops.contextualize(f)
@time map(_f,ps)
@time map(_f,ps)

map(f,_ps)
