# GPU-Acceleration of a Stiff Nonlinear Partial Differential Equation

The following is a demonstration of a GPU-accelerated implicit solve of a stiff
nonlinear partial differential equation (the Brusselator model). The user
function below uses scalar indexing into the `CuArray` state, so each step of
the implicit solver triggers many small GPU kernel launches; we keep the grid
modest (`N = 8`) to keep the doc build runtime reasonable. For large-grid
GPU PDEs you would write a true broadcast/kernel form; this example is for
demonstrating the wiring with `OrdinaryDiffEq` and `CuArray`.

```@example bruss
using OrdinaryDiffEq, CUDA, LinearAlgebra

const N = 8
const xyd_brusselator = range(0, stop = 1, length = N)
brusselator_f(x, y, t) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * (t >= 1.1) * 5.0
limit(a, N) = a == N + 1 ? 1 : a == 0 ? N : a
kernel_u! = let N = N, xyd = xyd_brusselator, dx = step(xyd_brusselator)
    @inline function (du, u, A, B, α, II, I, t)
        i, j = Tuple(I)
        x = xyd[I[1]]
        y = xyd[I[2]]
        ip1 = limit(i + 1, N)
        im1 = limit(i - 1, N)
        jp1 = limit(j + 1, N)
        jm1 = limit(j - 1, N)
        du[II[i, j, 1]] = α * (u[II[im1, j, 1]] + u[II[ip1, j, 1]] + u[II[i, jp1, 1]] +
                           u[II[i, jm1, 1]] - 4u[II[i, j, 1]]) +
                          B + u[II[i, j, 1]]^2 * u[II[i, j, 2]] - (A + 1) * u[II[i, j, 1]] +
                          brusselator_f(x, y, t)
    end
end
kernel_v! = let N = N, xyd = xyd_brusselator, dx = step(xyd_brusselator)
    @inline function (du, u, A, B, α, II, I, t)
        i, j = Tuple(I)
        ip1 = limit(i + 1, N)
        im1 = limit(i - 1, N)
        jp1 = limit(j + 1, N)
        jm1 = limit(j - 1, N)
        du[II[i, j, 2]] = α * (u[II[im1, j, 2]] + u[II[ip1, j, 2]] + u[II[i, jp1, 2]] +
                           u[II[i, jm1, 2]] - 4u[II[i, j, 2]]) +
                          A * u[II[i, j, 1]] - u[II[i, j, 1]]^2 * u[II[i, j, 2]]
    end
end
brusselator_2d = let N = N, xyd = xyd_brusselator, dx = step(xyd_brusselator)
    function (du, u, p, t)
        @inbounds begin
            ii1 = N^2
            ii2 = ii1 + N^2
            ii3 = ii2 + 2(N^2)
            A = p[1]
            B = p[2]
            α = p[3] / dx^2
            II = LinearIndices((N, N, 2))
            kernel_u!.(Ref(du), Ref(u), A, B, α, Ref(II), CartesianIndices((N, N)), t)
            kernel_v!.(Ref(du), Ref(u), A, B, α, Ref(II), CartesianIndices((N, N)), t)
            return nothing
        end
    end
end
p = (3.4, 1.0, 10.0, step(xyd_brusselator))

function init_brusselator_2d(xyd)
    N = length(xyd)
    u = zeros(N, N, 2)
    for I in CartesianIndices((N, N))
        x = xyd[I[1]]
        y = xyd[I[2]]
        u[I, 1] = 22 * (y * (1 - y))^(3 / 2)
        u[I, 2] = 27 * (x * (1 - x))^(3 / 2)
    end
    u
end
u0 = init_brusselator_2d(xyd_brusselator)

# Sanity check the RHS on the CPU before moving to GPU.
du = similar(u0)
brusselator_2d(du, u0, p, 0.0)
du[1, 1, 1], du[end, end, 2]

# The brusselator RHS only depends on `t` through the source term in
# `brusselator_f`, which is a step function activating at t=1.1 (handled
# explicitly via `tstops`). Away from that discontinuity, `df/dt = 0`. We
# supply this tgrad analytically so OrdinaryDiffEq does not try to AD it,
# which fails on a CuArray-backed user function.
brusselator_tgrad(du, u, p, t) = (du .= 0)
brusselator_f_cuda = ODEFunction(brusselator_2d, tgrad = brusselator_tgrad)
prob_ode_brusselator_2d_cuda = ODEProblem(
    brusselator_f_cuda, CuArray(u0), (0.0f0, 11.5f0), p, tstops = [1.1f0]
)
# The user function indexes into a CPU `Vector` (`xyd`) and writes scalar
# entries into the CuArray, so allow scalar GPU ops for the duration.
CUDA.allowscalar(true)
sol = solve(prob_ode_brusselator_2d_cuda, Rosenbrock23(), save_everystep = false)
CUDA.allowscalar(false)
sol.retcode
```
