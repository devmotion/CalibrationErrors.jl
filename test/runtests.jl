using CalibrationErrorsDistributions
using Distances
using LinearAlgebra
using PDMats
using Random
using Test

using CalibrationErrorsDistributions:
    Wasserstein, SqWasserstein, MixtureWasserstein, SqMixtureWasserstein
using Tulip: Tulip

Random.seed!(1234)

@testset "CalibrationErrorsDistributions" begin
    @testset "distances" begin
        @testset "Bures metric" begin
            include("distances/bures.jl")
        end
        @testset "Wasserstein distance" begin
            include("distances/wasserstein.jl")
        end
    end

    @testset "Kernels" begin
        include("kernels.jl")
    end

    @testset "Normal" begin
        include("normal.jl")
    end
    @testset "MvNormal" begin
        include("mvnormal.jl")
    end
end
