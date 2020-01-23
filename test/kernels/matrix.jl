using CalibrationErrors

using LinearAlgebra
using Test

const x1, y1 = 1, 3
const x2, y2 = [1, -1], [2, 1]

@testset "Uniform scaling kernel" begin
    @test_throws ArgumentError UniformScalingKernel(-1, SquaredExponentialKernel())

    @test UniformScalingKernel(SquaredExponentialKernel()) ===
        UniformScalingKernel(1, SquaredExponentialKernel())

    for γ in (1, 2), λ in (1, 5, 10)
        kernel = UniformScalingKernel(λ, SquaredExponentialKernel(γ))
        @test kernel(x1, y1) ≈ λ * exp(- 4 * γ) * I
        @test kernel(x2, y2) ≈ λ * exp(- 5 * γ) * I
    end
end

@testset "Diagonal kernel" begin
    @test_throws ArgumentError DiagonalKernel([-1], SquaredExponentialKernel())

    for γ in (1, 2), v in (ones(3), 1:4)
        kernel = DiagonalKernel(v, SquaredExponentialKernel(γ))
        @test kernel(x1, y1) ≈ exp(- 4 * γ) * Diagonal(v)
        @test kernel(x2, y2) ≈ exp(- 5 * γ) * Diagonal(v)
    end
end
