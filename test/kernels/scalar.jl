using CalibrationErrors, Distances

const x1, y1 = 1, 3
const x2, y2 = [1, -1], [2, 1]

@testset "Squared exponential kernel" begin
    @test_throws ArgumentError SquaredExponentialKernel(-1)

    @test SquaredExponentialKernel() === SquaredExponentialKernel(1)
    @test SquaredExponentialKernel() === SquaredExponentialKernel(Euclidean())
    @test SquaredExponentialKernel() === SquaredExponentialKernel(1, Euclidean())

    for γ in (1, 2)
        kernel = SquaredExponentialKernel(γ, TotalVariation())
        @test kernel(x1, y1) ≈ exp(- γ)
        @test kernel(x2, y2) ≈ exp(- 2.25 * γ)

        kernel2 = SquaredExponentialKernel(γ, Cityblock())
        @test kernel2(x1, y1) ≈ exp(- 4 * γ)
        @test kernel2(x2, y2) ≈ exp(- 9 * γ)

        kernel3 = SquaredExponentialKernel(γ, Euclidean())
        @test kernel3(x1, y1) ≈ exp(- 4 * γ)
        @test kernel3(x2, y2) ≈ exp(- 5 * γ)
    end
end

@testset "Exponential kernel" begin
    @test_throws ArgumentError ExponentialKernel(-1)

    @test ExponentialKernel() === ExponentialKernel(1)
    @test ExponentialKernel() === ExponentialKernel(Euclidean())
    @test ExponentialKernel() === ExponentialKernel(1, Euclidean())

    for γ in (1, 2)
        kernel = ExponentialKernel(γ, TotalVariation())
        @test kernel(x1, y1) ≈ exp(- γ)
        @test kernel(x2, y2) ≈ exp(- 1.5 * γ)

        kernel2 = ExponentialKernel(γ, Cityblock())
        @test kernel2(x1, y1) ≈ exp(- 2 * γ)
        @test kernel2(x2, y2) ≈ exp(- 3 * γ)

        kernel3 = ExponentialKernel(γ, Euclidean())
        @test kernel3(x1, y1) ≈ exp(- 2 * γ)
        @test kernel3(x2, y2) ≈ exp(- sqrt(5) * γ)
    end
end
