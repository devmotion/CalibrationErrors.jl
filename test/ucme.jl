using CalibrationErrors
using Distributions

using Random
using Statistics
using Test

Random.seed!(1234)

@testset "UCME: Two-dimensional example" begin
    # three test locations
    ucme = UCME(SqExponentialKernel() ⊗ WhiteKernel(),
                [[1.0, 0], [0.5, 0.5], [0.0, 1]], [1, 1, 2])

    # Deprecations
    ucme2 = @test_deprecated UCME(SqExponentialKernel(), WhiteKernel(),
                 [[1.0, 0], [0.5, 0.5], [0.0, 1]], [1, 1, 2])
    @test typeof(ucme2) === typeof(ucme)
    @test ucme2.kernel == ucme.kernel
    @test ucme2.testpredictions == ucme.testpredictions
    @test ucme2.testtargets == ucme.testtargets

    # two predictions
    @test @inferred(calibrationerror(ucme, ([1 0; 0 1], [1, 2]))) ≈ 0
    @test @inferred(calibrationerror(ucme, [1 0; 0 1], [1, 1])) ≈
        (exp(-2) + exp(-0.5) + 1) / 12
    @test @inferred(calibrationerror(ucme, [[1, 0], [0, 1]], [2, 1])) ≈ (1 - exp(-1))^2 / 6
    @test @inferred(calibrationerror(ucme, ([1 0; 0 1], [2, 2]))) ≈
        (exp(-2) + exp(-0.5) + 1) / 12
end

@testset "UCME: Basic properties" begin
    estimates = Vector{Float64}(undef, 1_000)

    for ntest in (1, 5, 10), nclasses in (2, 10, 100)
        dist = Dirichlet(nclasses, 1.0)

        testpredictions = [rand(dist) for _ in 1:ntest]
        testtargets = rand(1:nclasses, ntest)
        ucme = UCME(transform(ExponentialKernel(), 0.1) ⊗ WhiteKernel(),
                    testpredictions, testtargets)

        predictions = [Vector{Float64}(undef, nclasses) for _ in 1:20]
        targets = Vector{Int}(undef, 20)

        for i in 1:length(estimates)
            rand!.(Ref(dist), predictions)
            targets .= rand.(Categorical.(predictions))

            estimates[i] = calibrationerror(ucme, predictions, targets)
        end

        @test all(x > zero(x) for x in estimates)
    end
end
