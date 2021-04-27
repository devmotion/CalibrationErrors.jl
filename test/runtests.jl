using CalibrationErrors
using Distances
using Distributions
using StatsBase

using LinearAlgebra
using Random
using Statistics
using Test

Random.seed!(1234)

@testset "CalibrationErrors" begin
    @testset "binning" begin
        @testset "generic" begin
            include("binning/generic.jl")
        end
        @testset "uniform" begin
            include("binning/uniform.jl")
        end
        @testset "median variance" begin
            include("binning/medianvariance.jl")
        end
    end

    @testset "ECE" begin
        include("ece.jl")
    end

    @testset "Kernels" begin
        include("kernels.jl")
    end

    @testset "SKCE" begin
        @testset "biased" begin
            include("skce/biased.jl")
        end
        @testset "unbiased" begin
            include("skce/unbiased.jl")
        end
    end

    @testset "UCME" begin
        include("ucme.jl")
    end

    @testset "deprecations" begin
        include("deprecated.jl")
    end
end
