# GPU-Acceleration of a Stiff Nonlinear Partial Differential Equation

The following is a demonstration of a GPU-accelerated implicit solve of a stiff
nonlinear partial differential equation (the Brusselator model):

```@example bruss
using OrdinaryDiffEq, CUDA, LinearAlgebra

const N = 32
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
        du[II[i,
            j,
            1]] = α * (u[II[im1, j, 1]] + u[II[ip1, j, 1]] + u[II[i, jp1, 1]] +
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
        du[II[i,
            j,
            2]] = α * (u[II[im1, j, 2]] + u[II[ip1, j, 2]] + u[II[i, jp1, 2]] +
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
prob_ode_brusselator_2d = ODEProblem(brusselator_2d, u0, (0.0, 11.5), p)

du = similar(u0)
brusselator_2d(du, u0, p, 0.0)
du[34] # 802.9807693762164
du[1058] # 985.3120721709204
du[2000] # -403.5817880634729
du[end] # 1431.1460373522068
du[521] # -323.1677459142322

du2 = similar(u0)
brusselator_2d(du2, u0, p, 1.3)
du2[34] # 802.9807693762164
du2[1058] # 985.3120721709204
du2[2000] # -403.5817880634729
du2[end] # 1431.1460373522068
du2[521] # -318.1677459142322

prob_ode_brusselator_2d_cuda = ODEProblem(brusselator_2d, CuArray(u0), (0.0f0, 11.5f0), p,
    tstops = [1.1f0])
solve(prob_ode_brusselator_2d_cuda, Rosenbrock23(), save_everystep = false);
```
