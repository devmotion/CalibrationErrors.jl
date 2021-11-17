using CalibrationErrorsDistributions
using Distances
using LinearAlgebra
using PDMats
using Random
using Test

using CalibrationErrorsDistributions:
    Wasserstein, SqWasserstein, MixtureWasserstein, SqMixtureWasserstein

Random.seed!(1234)

@testset "CalibrationErrorsDistributions" begin
    @testset "distances" begin
        @testset "Wasserstein distance" begin
            include("distances/wasserstein.jl")
        end
    end

    @testset "Normal" begin
        include("normal.jl")
    end
    @testset "MvNormal" begin
        include("mvnormal.jl")
    end

    @testset "deprecations" begin
        include("deprecated.jl")
    end
end
