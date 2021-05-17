@testset "deprecated.jl" begin
    @testset "WassersteinExponentialKernel" begin
        kernel = @test_deprecated(WassersteinExponentialKernel())
        @test kernel isa ExponentialKernel{Wasserstein}
    end

    @testset "MixtureWassersteinExponentialKernel" begin
        kernel = @test_deprecated(MixtureWassersteinExponentialKernel())
        @test kernel isa ExponentialKernel{<:MixtureWasserstein}
    end
end
