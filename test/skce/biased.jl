using CalibrationErrors

@testset "Two-dimensional example" begin
    skce = BiasedSKCE(UniformScalingKernel(4, SquaredExponentialKernel(2)))

    # only two predictions, i.e., one term in the estimator
    @test calibrationerror(skce, ([1 0; 0 1], [1, 2])) ≈ 0
    @test calibrationerror(skce, ([1 0; 0 1], [1, 1])) ≈ 2
    @test calibrationerror(skce, ([1 0; 0 1], [2, 1])) ≈ 4 - 4 * exp(-4)
    @test calibrationerror(skce, ([1 0; 0 1], [2, 2])) ≈ 2
end
