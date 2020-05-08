using SafeTestsets

@safetestset "Binning" begin include("binning/generic.jl") end
@safetestset "Uniform binning" begin include("binning/uniform.jl") end
@safetestset "Median variance binning" begin include("binning/medianvariance.jl") end
@safetestset "ECE" begin include("ece.jl") end

@safetestset "Kernels" begin include("kernels.jl") end
@safetestset "Biased SKCE" begin include("skce/biased.jl") end
@safetestset "Unbiased SKCE" begin include("skce/unbiased.jl") end

@safetestset "SUMCE" begin include("sumce.jl") end
