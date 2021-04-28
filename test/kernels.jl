@testset "kernels.jl" begin
    kernel = TVExponentialKernel()

    # traits
    @test KernelFunctions.metric(kernel) === TotalVariation()

    # simple evaluation
    x, y = rand(10), rand(10)
    @test kernel(x, y) == exp(-totalvariation(x, y))

    # transformations
    @test (kernel ∘ ScaleTransform(0.1))(x, y) == exp(-0.1 * totalvariation(x, y))
    ard = rand(10)
    @test (kernel ∘ ARDTransform(ard))(x, y) == exp(-totalvariation(ard .* x, ard .* y))
end
