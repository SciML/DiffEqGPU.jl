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

using SafeTestsets, Test

@time @safetestset "EnsembleGPUArray" begin
    include("ensemblegpuarray.jl")
end
@time @safetestset "EnsembleGPUArray OOP" begin
    include("ensemblegpuarray_oop.jl")
end
@time @safetestset "EnsembleGPUArray SDE" begin
    include("ensemblegpuarray_sde.jl")
end
@time @safetestset "EnsembleGPUArray Input Types" begin
    include("ensemblegpuarray_inputtypes.jl")
end

# Fails, but not locally?
@time @safetestset "Reduction" begin
    include("reduction.jl")
end

@time @safetestset "Reverse Mode AD" begin
    include("reverse_ad_tests.jl")
end
# Not safe because distributed doesn't play nicely with modules.
@time @testset "Distributed Multi-GPU" begin
    include("distributed_multi_gpu.jl")
end
@time @testset "GPU Tsit5" begin
    include("gpu_tsit5_tests.jl")
end
