using CalibrationErrorsDistributions
using CalibrationErrorsDistributions: wasserstein, Wasserstein,
    mixturewasserstein, MixtureWasserstein

using Test

@testset "kernels.jl" begin
    @testset "WassersteinExponentialKernel" begin
        kernel = WassersteinExponentialKernel()

        # traits
        @test KernelFunctions.metric(kernel) === Wasserstein()

        # simple evaluation
        x, y = Normal(randn(), rand()), Normal(randn(), rand())
        @test kernel(x, y) == exp(-wasserstein(x, y))

        # transformations
        @test transform(kernel, 0.1)(x, y) == exp(- 0.1 * wasserstein(x, y))
        @test transform(kernel, ScaleTransform(0.1))(x, y) ==
            exp(- 0.1 * wasserstein(x, y))
    end

    @testset "MixtureWassersteinExponentialKernel" begin
        kernel = MixtureWassersteinExponentialKernel()

        # traits
        @test KernelFunctions.metric(kernel) === MixtureWasserstein()

        # simple evaluation
        x = MixtureModel(Normal, [(randn(), rand())])
        y = MixtureModel(Normal, [(randn(), rand())])
        @test kernel(x, y) == exp(-mixturewasserstein(x, y))
        @test kernel(x, y) ==
            WassersteinExponentialKernel()(first(components(x)), first(components(y)))

        # transformations
        @test transform(kernel, 0.1)(x, y) == exp(- 0.1 * mixturewasserstein(x, y))
        @test transform(kernel, ScaleTransform(0.1))(x, y) ==
            exp(- 0.1 * mixturewasserstein(x, y))
    end
end