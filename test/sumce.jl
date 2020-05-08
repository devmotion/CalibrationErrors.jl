using CalibrationErrors
using Distributions

using Random
using Statistics
using Test

Random.seed!(1234)

@testset "SUMCE: Two-dimensional example" begin
    # three test locations
    sumce = SUMCE(transform(SqExponentialKernel(), 2), WhiteKernel(),
                  [[1.0, 0], [0.5, 0.5], [0.0, 1]], [1, 1, 2])

    # two predictions
    @test @inferred(calibrationerror(sumce, ([1 0; 0 1], [1, 2]))) ≈ 0
    @test @inferred(calibrationerror(sumce, [1 0; 0 1], [1, 1])) ≈ (exp(-16) + exp(-4) + 1) / 12
    @test @inferred(calibrationerror(sumce, [[1, 0], [0, 1]], [2, 1])) ≈ (1 - exp(-8))^2 / 6
    @test @inferred(calibrationerror(sumce, ([1 0; 0 1], [2, 2]))) ≈ (exp(-16) + exp(-4) + 1) / 12
end

@testset "SUMCE: Basic properties" begin
    estimates = Vector{Float64}(undef, 1_000)

    for ntest in (1, 5, 10), nclasses in (2, 10, 100)
        dist = Dirichlet(nclasses, 1)

        testpredictions = [rand(dist) for _ in 1:ntest]
        testtargets = rand(1:nclasses, ntest)
        sumce = SUMCE(transform(ExponentialKernel(), 0.1), WhiteKernel(), testpredictions, testtargets)

        for i in 1:length(estimates)
            predictions = [rand(dist) for _ in 1:20]
            targets = [rand(Categorical(p)) for p in predictions]

            estimates[i] = calibrationerror(sumce, predictions, targets)
        end

        @test all(x > zero(x) for x in estimates)
    end
end

