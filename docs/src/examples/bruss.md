# GPU-Acceleration of a Stiff Nonlinear Partial Differential Equation

The following is a demonstration of a GPU-accelerated implicit solve of a
stiff nonlinear partial differential equation (the Brusselator model). The
RHS is written as a pure broadcast over the state arrays — the periodic
5-point Laplacian uses `circshift`, the reaction terms are element-wise,
and the spatially-varying source term `brusselator_f` is evaluated against
precomputed `x`/`y` grids — so no scalar indexing into the GPU array is
needed. We assert that explicitly with `CUDA.allowscalar(false)`: any
accidental scalar GPU read or write during the solve will raise.

```@example bruss
using OrdinaryDiffEq, CUDA, LinearAlgebra

const N = 32
const xyd_brusselator = range(0.0f0, stop = 1.0f0, length = N)

# Source term — element-wise pure function so it broadcasts over the
# (x, y) grids without any scalar indexing.
brusselator_f(x, y, t) =
    (((x - 0.3f0)^2 + (y - 0.6f0)^2) <= 0.01f0) * (t >= 1.1f0) * 5.0f0

function brusselator_2d!(du, u, p, t)
    A, B, α_raw, dx_val, x_grid, y_grid = p
    α = α_raw / dx_val^2

    u1 = @view u[:, :, 1]
    u2 = @view u[:, :, 2]
    du1 = @view du[:, :, 1]
    du2 = @view du[:, :, 2]

    # Periodic boundary 5-point Laplacian via circshift — entirely
    # broadcast on the GPU array, no scalar indexing.
    u1_im1 = circshift(u1, (1, 0));  u1_ip1 = circshift(u1, (-1, 0))
    u1_jm1 = circshift(u1, (0, 1));  u1_jp1 = circshift(u1, (0, -1))
    u2_im1 = circshift(u2, (1, 0));  u2_ip1 = circshift(u2, (-1, 0))
    u2_jm1 = circshift(u2, (0, 1));  u2_jp1 = circshift(u2, (0, -1))

    @. du1 = α * (u1_im1 + u1_ip1 + u1_jm1 + u1_jp1 - 4 * u1) +
        B + u1^2 * u2 - (A + 1) * u1 + brusselator_f(x_grid, y_grid, t)
    @. du2 = α * (u2_im1 + u2_ip1 + u2_jm1 + u2_jp1 - 4 * u2) +
        A * u1 - u1^2 * u2

    return nothing
end

function init_brusselator_2d(xyd)
    N = length(xyd)
    u = zeros(Float32, N, N, 2)
    for i in 1:N, j in 1:N
        x = xyd[i]; y = xyd[j]
        u[i, j, 1] = 22 * (y * (1 - y))^(3 / 2)
        u[i, j, 2] = 27 * (x * (1 - x))^(3 / 2)
    end
    u
end

u0_cpu = init_brusselator_2d(xyd_brusselator)

# Pre-compute x/y grids shaped for broadcasting with `u[:, :, k]`.
xg = CuArray(reshape(collect(xyd_brusselator), N, 1))
yg = CuArray(reshape(collect(xyd_brusselator), 1, N))
u0 = CuArray(u0_cpu)
p = (3.4f0, 1.0f0, 10.0f0, step(xyd_brusselator), xg, yg)

# Sanity check the RHS on the GPU under allowscalar(false).
CUDA.allowscalar(false)
du = similar(u0)
brusselator_2d!(du, u0, p, 0.0f0)
Array(du[1:1, 1:1, 1])[1], Array(du[end:end, end:end, 2])[1]

# The brusselator RHS only depends on `t` through `brusselator_f`, which is a
# step source activating at t=1.1 (handled explicitly via `tstops`). Away
# from that discontinuity, df/dt = 0; supply that analytically so the
# implicit solver doesn't AD a non-AD-friendly time-gradient through the
# closure.
brusselator_tgrad(du, u, p, t) = (du .= 0)
brusselator_f_cuda = ODEFunction(brusselator_2d!, tgrad = brusselator_tgrad)
prob_ode_brusselator_2d_cuda = ODEProblem(
    brusselator_f_cuda, u0, (0.0f0, 11.5f0), p, tstops = [1.1f0]
)
sol = solve(prob_ode_brusselator_2d_cuda, Rosenbrock23(), save_everystep = false)
sol.retcode
```
