# Adaptive controller conventions

Preserve the existing `GPUTsit5` PI-controller behavior. Alternative controllers must be explicit opt-in variants, with documentation explaining their step-size stability and accuracy tradeoffs. Do not change the default based on equal-tolerance timing alone: use work–precision comparisons at matched achieved error, include rejected steps, and test beyond simple nonstiff problems such as Lorenz.

# Differentiation validation

Gradient tests must use the package and its loaded extensions without adding differentiation rules in test code. Keep CPU Enzyme coverage alongside CUDA coverage. Exercise all active problem fields when validating transfer rules, including times and captured function values; parameter-only losses cannot detect discarded sensitivities. Keep initialization regressions enabled across backends and fix their owning dependencies rather than gating the tests.

# Kernel stepper inlining

Keep `@inline` on the `step!` methods in `src/ensemblegpukernel/perform_step/`. Without it the mutable integrator escapes into a call and is heap-allocated in device code: on OpenCL the resulting `gpu_malloc` fails and the kernel exits without writing output, and on Julia 1.12 Enzyme returns an incorrect adaptive Float64 gradient (the primal is unchanged). Scope `@muladd` to arithmetic blocks rather than removing the inline annotation.
