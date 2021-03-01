using CalibrationErrorsDistributions
using Distances

using Test

using CalibrationErrorsDistributions:
    sqwasserstein,
    sqmixturewasserstein,
    wasserstein,
    mixturewasserstein,
    Wasserstein,
    SqWasserstein

@testset "wasserstein.jl" begin
    @testset "SqWasserstein" begin
        μ1, μ2 = randn(2)
        σ1, σ2 = rand(2)
        normal1 = Normal(μ1, σ1)
        normal2 = Normal(μ2, σ2)

        @test iszero(sqwasserstein(normal1, normal1))
        @test iszero(sqwasserstein(normal2, normal2))
        @test sqwasserstein(normal1, normal2) == (μ1 - μ2)^2 + (σ1 - σ2)^2

        for (d1, d2) in Iterators.product((normal1, normal2), (normal1, normal2))
            mvnormal1 = MvNormal([mean(d1)], [std(d1)])
            mvnormal2 = MvNormal([mean(d2)], [std(d2)])
            @test sqwasserstein(mvnormal1, mvnormal2) == sqwasserstein(d1, d2)

            mvnormal_fill1 = MvNormal(fill(mean(d1), 10), fill(std(d1), 10))
            mvnormal_fill2 = MvNormal(fill(mean(d2), 10), fill(std(d2), 10))
            @test sqwasserstein(mvnormal_fill1, mvnormal_fill2) ≈ 10 * sqwasserstein(d1, d2)
        end

        laplace1 = Laplace(μ1, σ1)
        laplace2 = Laplace(μ2, σ2)
        @test iszero(sqwasserstein(laplace1, laplace1))
        @test iszero(sqwasserstein(laplace2, laplace2))
        @test sqwasserstein(laplace1, laplace2) == (μ1 - μ2)^2 + 2 * (σ1 - σ2)^2

        # pairwise computations
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

    @testset "Wasserstein" begin
        μ1, μ2 = randn(2)
        σ1, σ2 = rand(2)
        normal1 = Normal(μ1, σ1)
        normal2 = Normal(μ2, σ2)

        @test iszero(wasserstein(normal1, normal1))
        @test iszero(wasserstein(normal2, normal2))
        @test wasserstein(normal1, normal2) == sqrt(sqwasserstein(normal1, normal2))

        for (d1, d2) in Iterators.product((normal1, normal2), (normal1, normal2))
            mvnormal1 = MvNormal([mean(d1)], [std(d1)])
            mvnormal2 = MvNormal([mean(d2)], [std(d2)])
            @test wasserstein(mvnormal1, mvnormal2) == wasserstein(d1, d2)
            @test wasserstein(mvnormal1, mvnormal2) == sqrt(sqwasserstein(d1, d2))

            mvnormal_fill1 = MvNormal(fill(mean(d1), 10), fill(std(d1), 10))
            mvnormal_fill2 = MvNormal(fill(mean(d2), 10), fill(std(d2), 10))
            @test wasserstein(mvnormal_fill1, mvnormal_fill2) ≈
                  sqrt(10) * wasserstein(d1, d2)
            @test wasserstein(mvnormal_fill1, mvnormal_fill2) ==
                  sqrt(sqwasserstein(mvnormal_fill1, mvnormal_fill2))
        end

        laplace1 = Laplace(μ1, σ1)
        laplace2 = Laplace(μ2, σ2)
        @test iszero(wasserstein(laplace1, laplace1))
        @test iszero(wasserstein(laplace2, laplace2))
        @test wasserstein(laplace1, laplace2) == sqrt(sqwasserstein(laplace1, laplace2))

        # pairwise computations
        for (m, n) in ((1, 10), (10, 1), (10, 10))
            dists1 = [Normal(randn(), rand()) for _ in 1:m]
            dists2 = [Normal(randn(), rand()) for _ in 1:n]

            # compute distance matrix
            distmat = [wasserstein(x, y) for x in dists1, y in dists2]

            # out-of-place
            @test pairwise(Wasserstein(), dists1, dists2) ≈ distmat

            # in-place
            z = similar(distmat)
            pairwise!(z, Wasserstein(), dists1, dists2)
            @test z ≈ distmat
        end
    end

    @testset "SqMixtureWasserstein" begin
        for T in (Normal, Laplace)
            mixture1 = MixtureModel(T, [(randn(), rand())], [1.0])
            mixture2 = MixtureModel(T, [(randn(), rand())], [1.0])
            @test sqmixturewasserstein(mixture1, mixture2) ≈
                  sqwasserstein(first(components(mixture1)), first(components(mixture2)))

            mixture1 = MixtureModel(T, [(randn(), rand()), (randn(), rand())], [1.0, 0.0])
            mixture2 = MixtureModel(T, [(randn(), rand()), (randn(), rand())], [0.0, 1.0])
            @test sqmixturewasserstein(mixture1, mixture2) ≈
                  sqwasserstein(first(components(mixture1)), last(components(mixture2)))

            mixture1 = MixtureModel(T, fill((randn(), rand()), 10))
            mixture2 = MixtureModel(T, fill((randn(), rand()), 10))
            @test sqmixturewasserstein(mixture1, mixture2) ≈
                  sqwasserstein(first(components(mixture1)), first(components(mixture2)))

            mixture1 = MixtureModel(T, fill((randn(), rand()), 10))
            mixture2 = MixtureModel(T, [(randn(), rand())])
            @test sqmixturewasserstein(mixture1, mixture2) ≈
                  sqwasserstein(first(components(mixture1)), first(components(mixture2)))
        end
    end

    @testset "MixtureWasserstein" begin
        for T in (Normal, Laplace)
            mixture1 = MixtureModel(T, [(randn(), rand())], [1.0])
            mixture2 = MixtureModel(T, [(randn(), rand())], [1.0])
            @test mixturewasserstein(mixture1, mixture2) ≈
                  wasserstein(first(components(mixture1)), first(components(mixture2)))
            @test mixturewasserstein(mixture1, mixture2) ≈
                  sqrt(sqmixturewasserstein(mixture1, mixture2))

            mixture1 = MixtureModel(T, [(randn(), rand()), (randn(), rand())], [1.0, 0.0])
            mixture2 = MixtureModel(T, [(randn(), rand()), (randn(), rand())], [0.0, 1.0])
            @test mixturewasserstein(mixture1, mixture2) ≈
                  wasserstein(first(components(mixture1)), last(components(mixture2)))
            @test mixturewasserstein(mixture1, mixture2) ≈
                  sqrt(sqmixturewasserstein(mixture1, mixture2))

            mixture1 = MixtureModel(T, fill((randn(), rand()), 10))
            mixture2 = MixtureModel(T, fill((randn(), rand()), 10))
            @test mixturewasserstein(mixture1, mixture2) ≈
                  wasserstein(first(components(mixture1)), first(components(mixture2)))
            @test mixturewasserstein(mixture1, mixture2) ≈
                  sqrt(sqmixturewasserstein(mixture1, mixture2))

            mixture1 = MixtureModel(T, fill((randn(), rand()), 10))
            mixture2 = MixtureModel(T, [(randn(), rand())])
            @test mixturewasserstein(mixture1, mixture2) ≈
                  wasserstein(first(components(mixture1)), first(components(mixture2)))
            @test mixturewasserstein(mixture1, mixture2) ≈
                  sqrt(sqmixturewasserstein(mixture1, mixture2))
        end
    end
end
