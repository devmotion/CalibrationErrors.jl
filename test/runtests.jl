using CalibrationErrors
using Aqua
using Distances
using Distributions
using PDMats
using StatsBase

using LinearAlgebra
using Random
using Statistics
using Test

using CalibrationErrors: unsafe_skce_eval, unsafe_ucme_eval

Random.seed!(1234)

@testset "CalibrationErrors" begin
    @testset "General" begin
        include("aqua.jl")
    end

    @testset "binning" begin
        include("binning/generic.jl")
        include("binning/uniform.jl")
        include("binning/medianvariance.jl")
    end

    @testset "distances" begin
        include("distances/wasserstein.jl")
    end

    @testset "ECE" begin
        include("ece.jl")
    end

    @testset "SKCE" begin
        include("skce.jl")
    end

    @testset "UCME" begin
        include("ucme.jl")
    end

    @testset "distributions" begin
        include("distributions/normal.jl")
        include("distributions/mvnormal.jl")
    end
end
