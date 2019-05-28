using CalibrationErrors
using Random

Random.seed!(1234)

@testset "Trivial tests" begin
    ece = ECE(UniformBinning(10))

    @test calibrationerror(ece, ([0 1; 1 0], [2, 1])) == 0
    @test calibrationerror(ece, ([0 0.5 0.5 1; 1 0.5 0.5 0], [2, 2, 1, 1])) == 0
end
