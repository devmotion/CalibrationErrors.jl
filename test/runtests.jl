using SafeTestsets

using Random
Random.seed!(1234)

@safetestset "Bures metric" begin
    include("distances/bures.jl")
end
@safetestset "Wasserstein distance" begin
    include("distances/wasserstein.jl")
end
@safetestset "Kernels" begin
    include("kernels.jl")
end
@safetestset "Normal" begin
    include("normal.jl")
end
@safetestset "MvNormal" begin
    include("mvnormal.jl")
end
