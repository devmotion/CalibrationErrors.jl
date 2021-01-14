using CalibrationErrors
using Distributions

using Random
using Statistics
using Test

Random.seed!(1234)

@testset "Unbiased: Two-dimensional example" begin
    skce = UnbiasedSKCE(SqExponentialKernel(), WhiteKernel())

    # only two predictions, i.e., one term in the estimator
    @test @inferred(calibrationerror(skce, ([1 0; 0 1], [1, 2]))) ≈ 0
    @test @inferred(calibrationerror(skce, [1 0; 0 1], [1, 1])) ≈ 0
    @test @inferred(calibrationerror(skce, [[1, 0], [0, 1]], [2, 1])) ≈ -2 * exp(-1)
    @test @inferred(calibrationerror(skce, ([1 0; 0 1], [2, 2]))) ≈ 0
end

@testset "Unbiased: Basic properties" begin
    skce = UnbiasedSKCE(transform(ExponentialKernel(), 0.1), WhiteKernel())
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

@testset "Block: Two-dimensional example" begin
    # Blocks of two samples
    skce = BlockUnbiasedSKCE(SqExponentialKernel(), WhiteKernel())

    # only two predictions, i.e., one term in the estimator
    @test @inferred(calibrationerror(skce, ([1 0; 0 1], [1, 2]))) ≈ 0
    @test @inferred(calibrationerror(skce, [1 0; 0 1], [1, 1])) ≈ 0
    @test @inferred(calibrationerror(skce, [[1, 0], [0, 1]], [2, 1])) ≈ -2 * exp(-1)
    @test @inferred(calibrationerror(skce, ([1 0; 0 1], [2, 2]))) ≈ 0

    # two predictions, ten times replicated
    @test @inferred(calibrationerror(skce, (repeat([1 0; 0 1], 1, 10),
                                            repeat([1, 2], 10)))) ≈ 0
    @test @inferred(calibrationerror(skce, repeat([1 0; 0 1], 1, 10),
                                     repeat([1, 1], 10))) ≈ 0
    @test @inferred(calibrationerror(skce, (repeat([[1, 0], [0, 1]], 10),
                                            repeat([2, 1], 10)))) ≈ -2 * exp(-1)
    @test @inferred(calibrationerror(skce, (repeat([1 0; 0 1], 1, 10),
                                            repeat([2, 2], 10)))) ≈ 0
end

@testset "Block: Basic properties" begin
    nsamples = 20
    skce = UnbiasedSKCE(transform(ExponentialKernel(), 0.1), WhiteKernel())
    blockskce = BlockUnbiasedSKCE(skce.kernel)
    blockskce_all = BlockUnbiasedSKCE(skce.kernel, nsamples)
    estimates = Vector{Float64}(undef, 1_000)

    for nclasses in (2, 10, 100)
        dist = Dirichlet(nclasses, 1)

        for i in 1:length(estimates)
            predictions = [rand(dist) for _ in 1:nsamples]
            targets = [rand(Categorical(p)) for p in predictions]

            estimates[i] = calibrationerror(blockskce, predictions, targets)

            # consistency checks
            @test estimates[i] ≈ mean(
                calibrationerror(
                    skce,
                    predictions[(2 * i - 1):(2 * i)],
                    targets[(2 * i - 1):(2 * i)],
                ) for i in 1:(nsamples ÷ 2)
            )
            @test calibrationerror(skce, predictions, targets) ==
                calibrationerror(blockskce_all, predictions, targets)
        end

        @test any(x -> x > zero(x), estimates)
        @test any(x -> x < zero(x), estimates)
        @test mean(estimates) ≈ 0 atol=5e-3
    end
end
