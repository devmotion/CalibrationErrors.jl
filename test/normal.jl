using CalibrationErrorsDistributions
using Test

@testset "normal.jl" begin
    @testset "SKCE: basic example" begin
        skce = UnbiasedSKCE(WassersteinExponentialKernel(), SqExponentialKernel())

        # only two predictions, i.e., one term in the estimator
        normal1 = Normal(0, 1)
        normal2 = Normal(1, 2)
        @test @inferred(calibrationerror(skce, ([normal1, normal1], [0, 0]))) ≈
            1 - sqrt(2) + 1 / sqrt(3)
        @test @inferred(calibrationerror(skce, ([normal1, normal2], [1, 0]))) ≈
            exp(-sqrt(2)) * (exp(-1/2) - 1 / sqrt(2) - 1 / sqrt(5) + exp(-1/12) / sqrt(6))
        @test @inferred(calibrationerror(skce, ([normal1, normal2], [0, 1]))) ≈
            exp(-sqrt(2)) * (exp(-1/2) - exp(-1/4) / sqrt(2) - exp(-1/10) / sqrt(5) + exp(-1/12) / sqrt(6))
    end

    @testset "SKCE: basic example (transformed)" begin
        skce = UnbiasedSKCE(WassersteinExponentialKernel(), transform(SqExponentialKernel(), 0.5))

        # only two predictions, i.e., one term in the estimator
        normal1 = Normal(0, 1)
        normal2 = Normal(1, 2)
        @test @inferred(calibrationerror(skce, ([normal1, normal1], [0, 0]))) ≈
            1 - 2 / sqrt(1.25) + 1 / sqrt(1.5)
        @test @inferred(calibrationerror(skce, ([normal1, normal2], [1, 0]))) ≈
            exp(-sqrt(2)) * (exp(-1/8) - 1 / sqrt(1.25) - 1 / sqrt(2) + exp(-1/18) / sqrt(2.25))
        @test @inferred(calibrationerror(skce, ([normal1, normal2], [0, 1]))) ≈
            exp(-sqrt(2)) * (exp(-1/8) - exp(-1/10) / sqrt(1.25) - exp(-1/16) / sqrt(2) + exp(-1/18) / sqrt(2.25))
    end

    @testset "SKCE: basic properties" begin
        skce = UnbiasedSKCE(WassersteinExponentialKernel(), SqExponentialKernel())

        estimates = Vector{Float64}(undef, 1_000)
        for i in 1:length(estimates)
            predictions = [Normal(randn(), rand()) for _ in 1:20]
            targets = [rand(p) for p in predictions]

            estimates[i] = calibrationerror(skce, predictions, targets)
        end
        
        @test any(x -> x > zero(x), estimates)
        @test any(x -> x < zero(x), estimates)
        @test mean(estimates) ≈ 0 atol=1e-3
    end
end
