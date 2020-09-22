using CalibrationErrorsDistributions
using CalibrationErrorsDistributions: sqwasserstein, SqWasserstein
using Distances

using Test

@testset "pairwise.jl" begin
    for (m, n) in ((1, 10), (10, 1), (10, 10))
        dists1 = [Normal(randn(), rand()) for _ in 1:m]
        dists2 = [Normal(randn(), rand()) for _ in 1:n]

        # compute distance matrix
        distmat = [sqwasserstein(x, y) for x in dists1, y in dists2]

        # out-of-place
        @test pairwise(SqWasserstein(), dists1, dists2) ≈ distmat

        # in-place
        z = similar(distmat)
        pairwise!(z, SqWasserstein(), dists1, dists2)
        @test z ≈ distmat
    end
end