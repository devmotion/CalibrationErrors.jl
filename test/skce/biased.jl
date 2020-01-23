using CalibrationErrors
using Distributions

using Random
using Test

Random.seed!(1234)

@testset "Two-dimensional example" begin
    skce = BiasedSKCE(UniformScalingKernel(4, SquaredExponentialKernel(2)))

    # only two predictions, i.e., one term in the estimator
    @test @inferred(calibrationerror(skce, ([1 0; 0 1], [1, 2]))) ≈ 0
    @test @inferred(calibrationerror(skce, [1 0; 0 1], [1, 1])) ≈ 2
    @test @inferred(calibrationerror(skce, [[1, 0], [0, 1]], [2, 1])) ≈ 4 - 4 * exp(-4)
    @test @inferred(calibrationerror(skce, ([1 0; 0 1], [2, 2]))) ≈ 2
end

@testset "Basic properties" begin
    skce = BiasedSKCE(UniformScalingKernel(ExponentialKernel(0.1)))
    estimates = Vector{Float64}(undef, 1_000)

    for nclasses in (2, 10, 100)
        dist = Dirichlet(nclasses, 1)
        
        for i in 1:length(estimates)
            predictions = [rand(dist) for _ in 1:20]
            targets = [rand(Categorical(p)) for p in predictions]

            estimates[i] = calibrationerror(skce, predictions, targets)
        end

        @test all(x -> x > zero(x), estimates)
    end
end