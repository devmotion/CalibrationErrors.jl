using CalibrationErrors

@testset "Quadratic: Two-dimensional example" begin
    skce = QuadraticUnbiasedSKCE(UniformScalingKernel(4, SquaredExponentialKernel(2)))

    # only two predictions, i.e., one term in the estimator
    @test calibrationerror(skce, ([1 0; 0 1], [1, 2])) ≈ 0
    @test calibrationerror(skce, ([1 0; 0 1], [1, 1])) ≈ 0
    @test calibrationerror(skce, ([1 0; 0 1], [2, 1])) ≈ -8 * exp(-4)
    @test calibrationerror(skce, ([1 0; 0 1], [2, 2])) ≈ 0
end

@testset "Linear: Two-dimensional example" begin
    skce = LinearUnbiasedSKCE(UniformScalingKernel(4, SquaredExponentialKernel(2)))

    # only two predictions, i.e., one term in the estimator
    @test calibrationerror(skce, ([1 0; 0 1], [1, 2])) ≈ 0
    @test calibrationerror(skce, ([1 0; 0 1], [1, 1])) ≈ 0
    @test calibrationerror(skce, ([1 0; 0 1], [2, 1])) ≈ -8 * exp(-4)
    @test calibrationerror(skce, ([1 0; 0 1], [2, 2])) ≈ 0
end
