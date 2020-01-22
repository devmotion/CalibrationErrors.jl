using CalibrationErrors
using Distributions

using Random
using Test

Random.seed!(1234)

@testset "Trivial tests" begin
    ece = ECE(UniformBinning(10))

    @test @inferred(calibrationerror(ece, ([0 1; 1 0], [2, 1]))) == 0
    @test @inferred(calibrationerror(ece, ([0 0.5 0.5 1; 1 0.5 0.5 0], [2, 2, 1, 1]))) == 0
end

@testset "Uniform binning: Basic properties" begin
    ece = ECE(UniformBinning(10))
    estimates = Vector{Float64}(undef, 1_000)

    for nclasses in (2, 10, 100)
        dist = Dirichlet(nclasses, 1)
        
        for i in 1:length(estimates)
            predictions = [rand(dist) for _ in 1:20]
            targets = [rand(Categorical(p)) for p in predictions]

            estimates[i] = calibrationerror(ece, predictions, targets)
        end

        @test all(x -> zero(x) < x < one(x), estimates)
    end
end

@testset "Median variance binning: Basic properties" begin
    ece = ECE(MedianVarianceBinning(10))
    estimates = Vector{Float64}(undef, 1_000)

    for nclasses in (2, 10, 100)
        dist = Dirichlet(nclasses, 1)
        
        for i in 1:length(estimates)
            predictions = [rand(dist) for _ in 1:20]
            targets = [rand(Categorical(p)) for p in predictions]

            estimates[i] = calibrationerror(ece, predictions, targets)
        end

        @test all(x -> zero(x) < x < one(x), estimates)
    end
end