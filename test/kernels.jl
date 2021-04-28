@testset "kernels.jl" begin
    @testset "WassersteinExponentialKernel" begin
        kernel = WassersteinExponentialKernel()

        # traits
        @test KernelFunctions.metric(kernel) === Wasserstein()

        # simple evaluation
        x, y = Normal(randn(), rand()), Normal(randn(), rand())
        @test kernel(x, y) == exp(-Wasserstein()(x, y))

        # transformations
        @test (kernel ∘ ScaleTransform(0.1))(x, y) == exp(-0.1 * Wasserstein()(x, y))
    end

    @testset "MixtureWassersteinExponentialKernel" begin
        kernel = MixtureWassersteinExponentialKernel()

        # traits
        @test KernelFunctions.metric(kernel) isa MixtureWasserstein{<:Tulip.Optimizer}

        # simple evaluation
        x = MixtureModel(Normal, [(randn(), rand())])
        y = MixtureModel(Normal, [(randn(), rand())])
        @test kernel(x, y) == exp(-MixtureWasserstein()(x, y))
        @test kernel(x, y) ==
              WassersteinExponentialKernel()(first(components(x)), first(components(y)))

        # transformations
        @test (kernel ∘ ScaleTransform(0.1))(x, y) == exp(-0.1 * MixtureWasserstein()(x, y))
    end
end
