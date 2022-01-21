@testset "normal.jl" begin
    @testset "SKCE: basic example" begin
        skce = UnbiasedSKCE(
            ExponentialKernel(; metric=Wasserstein()) ⊗ SqExponentialKernel()
        )

        # only two predictions, i.e., one term in the estimator
        normal1 = Normal(0, 1)
        normal2 = Normal(1, 2)
        @test @inferred(skce([normal1, normal1], [0, 0])) ≈ 1 - sqrt(2) + 1 / sqrt(3)
        @test @inferred(skce([normal1, normal2], [1, 0])) ≈
            exp(-sqrt(2)) *
              (exp(-1 / 2) - 1 / sqrt(2) - 1 / sqrt(5) + exp(-1 / 12) / sqrt(6))
        @test @inferred(skce([normal1, normal2], [0, 1])) ≈
            exp(-sqrt(2)) * (
            exp(-1 / 2) - exp(-1 / 4) / sqrt(2) - exp(-1 / 10) / sqrt(5) +
            exp(-1 / 12) / sqrt(6)
        )
    end

    @testset "SKCE: basic example (transformed)" begin
        skce = UnbiasedSKCE(
            ExponentialKernel(; metric=Wasserstein()) ⊗
            (SqExponentialKernel() ∘ ScaleTransform(0.5)),
        )

        # only two predictions, i.e., one term in the estimator
        normal1 = Normal(0, 1)
        normal2 = Normal(1, 2)
        @test @inferred(skce([normal1, normal1], [0, 0])) ≈
            1 - 2 / sqrt(1.25) + 1 / sqrt(1.5)
        @test @inferred(skce([normal1, normal2], [1, 0])) ≈
            exp(-sqrt(2)) *
              (exp(-1 / 8) - 1 / sqrt(1.25) - 1 / sqrt(2) + exp(-1 / 18) / sqrt(2.25))
        @test @inferred(skce([normal1, normal2], [0, 1])) ≈
            exp(-sqrt(2)) * (
            exp(-1 / 8) - exp(-1 / 10) / sqrt(1.25) - exp(-1 / 16) / sqrt(2) +
            exp(-1 / 18) / sqrt(2.25)
        )
    end

    @testset "SKCE: basic properties" begin
        skce = UnbiasedSKCE(
            ExponentialKernel(; metric=Wasserstein()) ⊗ SqExponentialKernel()
        )

        estimates = map(1:10_000) do _
            predictions = map(Normal, randn(20), rand(20))
            targets = map(rand, predictions)
            return skce(predictions, targets)
        end

        @test any(x -> x > zero(x), estimates)
        @test any(x -> x < zero(x), estimates)
        @test mean(estimates) ≈ 0 atol = 1e-4
    end

    @testset "UCME: basic example" begin
        # one test location
        ucme = UCME(
            ExponentialKernel(; metric=Wasserstein()) ⊗ SqExponentialKernel(),
            [Normal(0.5, 0.5)],
            [1],
        )

        # two predictions
        normal1 = Normal(0, 1)
        normal2 = Normal(1, 2)
        @test @inferred(ucme([normal1, normal2], [0, 0.5])) ≈
            (
            exp(-1 / sqrt(2)) * (exp(-1 / 2) - exp(-1 / 4) / sqrt(2)) +
            exp(-sqrt(5 / 2)) * (exp(-1 / 8) - 1 / sqrt(5))
        )^2 / 4

        # two test locations
        ucme = UCME(
            ExponentialKernel(; metric=Wasserstein()) ⊗ SqExponentialKernel(),
            [Normal(0.5, 0.5), Normal(-1, 1.5)],
            [1, -0.5],
        )
        @test @inferred(ucme([normal1, normal2], [0, 0.5])) ≈
            (
            (
                exp(-1 / sqrt(2)) * (exp(-1 / 2) - exp(-1 / 4) / sqrt(2)) +
                exp(-sqrt(5 / 2)) * (exp(-1 / 8) - 1 / sqrt(5))
            )^2 +
            (
                exp(-sqrt(5) / 2) * (exp(-1 / 8) - exp(-1 / 16) / sqrt(2)) +
                exp(-sqrt(17) / 2) * (exp(-1 / 2) - exp(-9 / 40) / sqrt(5))
            )^2
        ) / 8
    end

    @testset "kernels with input transformations" begin
        nsamples = 100
        ntestsamples = 5

        # create predictions and targets
        predictions = map(Normal, randn(nsamples), rand(nsamples))
        targets = randn(nsamples)

        # create random test locations
        testpredictions = map(Normal, randn(ntestsamples), rand(ntestsamples))
        testtargets = randn(ntestsamples)

        for γ in (1.0, rand())
            kernel1 =
                ExponentialKernel(; metric=Wasserstein()) ⊗
                (SqExponentialKernel() ∘ ScaleTransform(γ))
            kernel2 =
                ExponentialKernel(; metric=Wasserstein()) ⊗
                (SqExponentialKernel() ∘ ARDTransform([γ]))

            # check evaluation of the first two observations
            p1 = predictions[1]
            p2 = predictions[2]
            t1 = targets[1]
            t2 = targets[2]
            for f in (
                CalibrationErrors.unsafe_skce_eval_targets,
                CalibrationErrors.unsafe_ucme_eval_targets,
            )
                out1 = f(kernel1.kernels[2], p1, t1, p2, t2)
                out2 = f(kernel2.kernels[2], p1, t1, p2, t2)
                @test out2 ≈ out1
                if isone(γ)
                    @test f(SqExponentialKernel(), p1, t1, p2, t2) ≈ out1
                end
            end

            # check estimates
            for estimator in (UnbiasedSKCE, x -> UCME(x, testpredictions, testtargets))
                estimate1 = estimator(kernel1)(predictions, targets)
                estimate2 = estimator(kernel2)(predictions, targets)
                @test estimate2 ≈ estimate1
                if isone(γ)
                    @test estimator(
                        ExponentialKernel(; metric=Wasserstein()) ⊗ SqExponentialKernel()
                    )(
                        predictions, targets
                    ) ≈ estimate1
                end
            end
        end
    end
end
