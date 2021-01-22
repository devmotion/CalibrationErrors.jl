using CalibrationErrors
using Distributions

using Random
using Statistics
using Test

Random.seed!(1234)

@testset "Unbiased: Two-dimensional example" begin
    skce = UnbiasedSKCE(SqExponentialKernel() ⊗ WhiteKernel())

    # Deprecation
    skce2 = @test_deprecated UnbiasedSKCE(SqExponentialKernel(), WhiteKernel())
    @test typeof(skce2) === typeof(skce)
    @test skce2.kernel == skce.kernel

    # only two predictions, i.e., one term in the estimator
    @test @inferred(calibrationerror(skce, ([1 0; 0 1], [1, 2]))) ≈ 0
    @test @inferred(calibrationerror(skce, [1 0; 0 1], [1, 1])) ≈ 0
    @test @inferred(calibrationerror(skce, [[1, 0], [0, 1]], [2, 1])) ≈ -2 * exp(-1)
    @test @inferred(calibrationerror(skce, ([1 0; 0 1], [2, 2]))) ≈ 0
end

@testset "Unbiased: Basic properties" begin
    skce = UnbiasedSKCE(transform(ExponentialKernel(), 0.1) ⊗ WhiteKernel())
    estimates = Vector{Float64}(undef, 1_000)

    for nclasses in (2, 10, 100)
        dist = Dirichlet(nclasses, 1.0)

        predictions = [Vector{Float64}(undef, nclasses) for _ in 1:20]
        targets = Vector{Int}(undef, 20)

        for i in 1:length(estimates)
            rand!.(Ref(dist), predictions)
            targets .= rand.(Categorical.(predictions))

            estimates[i] = calibrationerror(skce, predictions, targets)
        end

        @test any(x -> x > zero(x), estimates)
        @test any(x -> x < zero(x), estimates)
        @test mean(estimates) ≈ 0 atol = 1e-3
    end
end

@testset "Block: Two-dimensional example" begin
    # Blocks of two samples
    skce = BlockUnbiasedSKCE(SqExponentialKernel() ⊗ WhiteKernel())

    # Deprecation
    skce2 = @test_deprecated BlockUnbiasedSKCE(SqExponentialKernel(), WhiteKernel())
    @test typeof(skce2) === typeof(skce)
    @test skce2.kernel == skce.kernel
    @test skce2.blocksize == skce.blocksize
    skce3 = @test_deprecated BlockUnbiasedSKCE(SqExponentialKernel(), WhiteKernel(), 2)
    @test typeof(skce3) === typeof(skce)
    @test skce3.kernel == skce.kernel
    @test skce3.blocksize == skce.blocksize

    # only two predictions, i.e., one term in the estimator
    @test @inferred(calibrationerror(skce, ([1 0; 0 1], [1, 2]))) ≈ 0
    @test @inferred(calibrationerror(skce, [1 0; 0 1], [1, 1])) ≈ 0
    @test @inferred(calibrationerror(skce, [[1, 0], [0, 1]], [2, 1])) ≈ -2 * exp(-1)
    @test @inferred(calibrationerror(skce, ([1 0; 0 1], [2, 2]))) ≈ 0

    # two predictions, ten times replicated
    @test @inferred(
        calibrationerror(skce, (repeat([1 0; 0 1], 1, 10), repeat([1, 2], 10)))
    ) ≈ 0
    @test @inferred(calibrationerror(skce, repeat([1 0; 0 1], 1, 10), repeat([1, 1], 10))) ≈
          0
    @test @inferred(
        calibrationerror(skce, (repeat([[1, 0], [0, 1]], 10), repeat([2, 1], 10)))
    ) ≈ -2 * exp(-1)
    @test @inferred(
        calibrationerror(skce, (repeat([1 0; 0 1], 1, 10), repeat([2, 2], 10)))
    ) ≈ 0
end

@testset "Block: Basic properties" begin
    nsamples = 20
    kernel = transform(ExponentialKernel(), 0.1) ⊗ WhiteKernel()
    skce = UnbiasedSKCE(kernel)
    blockskce = BlockUnbiasedSKCE(kernel)
    blockskce_all = BlockUnbiasedSKCE(kernel, nsamples)
    estimates = Vector{Float64}(undef, 1_000)

    for nclasses in (2, 10, 100)
        dist = Dirichlet(nclasses, 1.0)

        predictions = [Vector{Float64}(undef, nclasses) for _ in 1:nsamples]
        targets = Vector{Int}(undef, nsamples)

        for i in 1:length(estimates)
            rand!.(Ref(dist), predictions)
            targets .= rand.(Categorical.(predictions))

            estimates[i] = calibrationerror(blockskce, predictions, targets)

            # consistency checks
            @test estimates[i] ≈ mean(
                calibrationerror(
                    skce, predictions[(2 * i - 1):(2 * i)], targets[(2 * i - 1):(2 * i)]
                ) for i in 1:(nsamples ÷ 2)
            )
            @test calibrationerror(skce, predictions, targets) ==
                  calibrationerror(blockskce_all, predictions, targets)
        end

        @test any(x -> x > zero(x), estimates)
        @test any(x -> x < zero(x), estimates)
        @test mean(estimates) ≈ 0 atol = 5e-3
    end
end
