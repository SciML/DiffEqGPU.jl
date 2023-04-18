# Pkg.test runs with --check_bounds=1, forcing all bounds checks.
# This is incompatible with GPUifyLoops.
# TODO: Is this needed any longer?
if Base.JLOptions().check_bounds == 1
    cmd = Cmd(filter(arg -> !startswith(arg, "--check-bounds"), Base.julia_cmd().exec))
    code = """
      $(Base.load_path_setup_code(false))
      cd($(repr(@__DIR__)))
      include($(repr(@__FILE__)))
      """
    run(`$cmd --eval $code`)
    exit()
end
@assert Base.JLOptions().check_bounds == 0

const SUPPORTS_LUFACT = Set(["CUDA", "AMDGPU"])
const SUPPORTS_DOUBLE_PRECISION = Set(["CUDA", "AMDGPU"])
const GROUP = get(ENV, "GROUP", "CUDA")

using SafeTestsets, Test

@time @testset "GPU Kernelized Non Stiff ODE Regression" begin include("gpu_kernel_de/gpu_ode_regression.jl") end
@time @testset "GPU Kernelized Stiff ODE Regression" begin include("gpu_kernel_de/stiff_ode/gpu_ode_regression.jl") end
@time @safetestset "GPU Kernelized ODE DiscreteCallback" begin include("gpu_kernel_de/gpu_ode_discrete_callbacks.jl") end

if GROUP in SUPPORTS_LUFACT
    @time @safetestset "EnsembleGPUArray" begin include("ensemblegpuarray.jl") end
    @time @safetestset "EnsembleGPUArray OOP" begin include("ensemblegpuarray_oop.jl") end
end

# EnsembleGPUArray kernels has Int64 arguments, causing them to fail with Metal and oneAPI
if GROUP in SUPPORTS_DOUBLE_PRECISION
    @time @safetestset "EnsembleGPUArray SDE" begin include("ensemblegpuarray_sde.jl") end
    @time @safetestset "EnsembleGPUArray Input Types" begin include("ensemblegpuarray_inputtypes.jl") end
    @time @safetestset "Reduction" begin include("reduction.jl") end
    @time @safetestset "Reverse Mode AD" begin include("reverse_ad_tests.jl") end
    # Not safe because distributed doesn't play nicely with modules.
    @time @testset "Distributed Multi-GPU" begin include("distributed_multi_gpu.jl") end
end

if GROUP == "CUDA"
    # Causes dynamic function invocation
    @time @testset "GPU Kernelized ODE ContinuousCallback" begin include("gpu_kernel_de/gpu_ode_continuous_callbacks.jl") end
    # device Random not implemented yet
    @time @testset "GPU Kernelized SDE Regression" begin include("gpu_kernel_de/gpu_sde_regression.jl") end
    @time @testset "GPU Kernelized SDE Convergence" begin include("gpu_kernel_de/gpu_sde_convergence.jl") end
end
