using KernelAbstractions, CUDAKernels, CUDA, SimpleDiffEq
using StaticArrays

function loop(u, p, t)
    σ = p[1]; ρ = p[2]; β = p[3]
    du1 = σ*(u[2]-u[1])
    du2 = u[1]*(ρ-u[3]) - u[2]
    du3 = u[1]*u[2] - β*u[3]
    return SVector{3}(du1, du2, du3)
end
const func = ODEFunction(loop)
u0 = 10ones(Float32,3)
const su0= SVector{3}(u0)
const dt = 1f-1
const tspan = (0.0f0, 10.0f0)

const odeoop = ODEProblem{false}(loop, SVector{3}(u0), (0.0f0, 10.0f0),  Float32.(SA[10, 28, 8/3]))
sol2 = solve(odeoop,GPUSimpleTsit5(),dt=dt)
cps = Array([SA[10f0,28f0,8/3f0] for i in 1:32])
ps = CuArray([SA[10f0,28f0,8/3f0] for i in 1:32])
CUDA.allowscalar(false)

function f(p)
    solve(odeoop,GPUSimpleTsit5(),dt=dt)[end]
end

map(f,cps)

@kernel function f_kernel(ps)
    solve(odeoop,GPUSimpleTsit5(),dt=dt)[end]
end

version = CUDADevice()
wait(version, f_kernel(version)(ps;ndrange=length(ps)))

function DiffEqBase.solve(prob::ODEProblem,
                          alg::GPUSimpleTsit5;
                          dt = 0.1f0)
  u0 = prob.u0
  tspan = prob.tspan
  f = prob.f
  p = prob.p
  t = tspan[1]
  tf = prob.tspan[2]
  #ts = tspan[1]:dt:tspan[2]
  us = MVector{101,typeof(u0)}(undef)
  us[1] = u0
  u = u0
  k7 = f(u, p, t)

  cs, as, btildes, rs = SimpleDiffEq._build_atsit5_caches(eltype(u0))
  c1, c2, c3, c4, c5, c6 = cs
  a21, a31, a32, a41, a42, a43, a51, a52, a53, a54,
  a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76 = as
  btilde1, btilde2, btilde3, btilde4, btilde5, btilde6, btilde7 = btildes

  # FSAL
  for i in 2:101
      uprev = u; k1 = k7; t = tspan[1] + dt * i
      tmp = uprev+dt*a21*k1
      k2 = f(tmp, p, t+c1*dt)
      tmp = uprev+dt*(a31*k1+a32*k2)
      k3 = f(tmp, p, t+c2*dt)
      tmp = uprev+dt*(a41*k1+a42*k2+a43*k3)
      k4 = f(tmp, p, t+c3*dt)
      tmp = uprev+dt*(a51*k1+a52*k2+a53*k3+a54*k4)
      k5 = f(tmp, p, t+c4*dt)
      tmp = uprev+dt*(a61*k1+a62*k2+a63*k3+a64*k4+a65*k5)
      k6 = f(tmp, p, t+dt)
      u = uprev+dt*(a71*k1+a72*k2+a73*k3+a74*k4+a75*k5+a76*k6)
      k7 = f(u, p, t+dt)
      us[i] = u
  end

  SArray(us)
end

# Works now!
function f(p)
    solve(odeoop,GPUSimpleTsit5(),dt=dt)[end]
end

map(f,cps)

@kernel function f_kernel(ps)
    solve(odeoop,GPUSimpleTsit5(),dt=dt)[end]
end

version = CUDADevice()
wait(version, f_kernel(version)(ps;ndrange=length(ps)))

# Fails to change parameters
function f(p)
    _prob = remake(odeoop,p=p)
    solve(_prob,GPUSimpleTsit5(),dt=dt)[end]
end

map(f,cps)

@kernel function f_kernel(ps)
    i = @index(Global)
    _prob = remake(odeoop,p=ps[i])
    solve(_prob,GPUSimpleTsit5(),dt=dt)[end]
end

version = CUDADevice()
wait(version, f_kernel(version)(ps;ndrange=length(ps)))
