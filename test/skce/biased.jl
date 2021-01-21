using CalibrationErrors
using Distributions

using Random
using Test

Random.seed!(1234)

@testset "Two-dimensional example" begin
    skce = BiasedSKCE(SqExponentialKernel() ⊗ WhiteKernel())

    # Deprecation
    skce2 = @test_deprecated BiasedSKCE(SqExponentialKernel(), WhiteKernel())
    @test typeof(skce2) === typeof(skce)
    @test skce2.kernel == skce.kernel

    # only two predictions, i.e., three unique terms in the estimator
    @test @inferred(calibrationerror(skce, ([1 0; 0 1], [1, 2]))) ≈ 0
    @test @inferred(calibrationerror(skce, [1 0; 0 1], [1, 1])) ≈ 0.5
    @test @inferred(calibrationerror(skce, [[1, 0], [0, 1]], [2, 1])) ≈ 1 - exp(-1)
    @test @inferred(calibrationerror(skce, ([1 0; 0 1], [2, 2]))) ≈ 0.5
end

@testset "Basic properties" begin
    skce = BiasedSKCE(transform(ExponentialKernel(), 0.1) ⊗ WhiteKernel())
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

        @test all(x -> x > zero(x), estimates)
    end
end
