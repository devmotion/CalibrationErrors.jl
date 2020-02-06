using CalibrationErrors
using Distributions

using Random
using Statistics
using Test

Random.seed!(1234)

@testset "Quadratic: Two-dimensional example" begin
    skce = QuadraticUnbiasedSKCE(SqExponentialKernel(2), WhiteKernel())

    # only two predictions, i.e., one term in the estimator
    @test @inferred(calibrationerror(skce, ([1 0; 0 1], [1, 2]))) ≈ 0
    @test @inferred(calibrationerror(skce, [1 0; 0 1], [1, 1])) ≈ 0
    @test @inferred(calibrationerror(skce, [[1, 0], [0, 1]], [2, 1])) ≈ -2 * exp(-8)
    @test @inferred(calibrationerror(skce, ([1 0; 0 1], [2, 2]))) ≈ 0
end

@testset "Quadratic: Basic properties" begin
    skce = QuadraticUnbiasedSKCE(ExponentialKernel(0.1), WhiteKernel())
    estimates = Vector{Float64}(undef, 1_000)

    for nclasses in (2, 10, 100)
        dist = Dirichlet(nclasses, 1)
        
        for i in 1:length(estimates)
            predictions = [rand(dist) for _ in 1:20]
            targets = [rand(Categorical(p)) for p in predictions]

            estimates[i] = calibrationerror(skce, predictions, targets)
        end

        @test any(x -> x > zero(x), estimates)
        @test any(x -> x < zero(x), estimates)
        @test mean(estimates) ≈ 0 atol=1e-3
    end
end

@testset "Linear: Two-dimensional example" begin
    skce = LinearUnbiasedSKCE(SqExponentialKernel(2), WhiteKernel())

    # only two predictions, i.e., one term in the estimator
    @test @inferred(calibrationerror(skce, ([1 0; 0 1], [1, 2]))) ≈ 0
    @test @inferred(calibrationerror(skce, [1 0; 0 1], [1, 1])) ≈ 0
    @test @inferred(calibrationerror(skce, [[1, 0], [0, 1]], [2, 1])) ≈ -2 * exp(-8)
    @test @inferred(calibrationerror(skce, ([1 0; 0 1], [2, 2]))) ≈ 0
end

@testset "Linear: Basic properties" begin
    skce = LinearUnbiasedSKCE(ExponentialKernel(0.1), WhiteKernel())
    estimates = Vector{Float64}(undef, 1_000)

    for nclasses in (2, 10, 100)
        dist = Dirichlet(nclasses, 1)
        
        for i in 1:length(estimates)
            predictions = [rand(dist) for _ in 1:20]
            targets = [rand(Categorical(p)) for p in predictions]

            estimates[i] = calibrationerror(skce, predictions, targets)
        end

        @test any(x -> x > zero(x), estimates)
        @test any(x -> x < zero(x), estimates)
        @test mean(estimates) ≈ 0 atol=5e-3
    end
end