@testset "mvnormal.jl" begin
    @testset "consistency with Normal" begin
        nsamples = 1_000
        ntestsamples = 5

        # create predictions
        predictions_μ = randn(nsamples)
        predictions_σ = rand(nsamples)
        predictions_normal = map(Normal, predictions_μ, predictions_σ)
        predictions_mvnormal = map(predictions_μ, predictions_σ) do μ, σ
            MvNormal([μ], σ)
        end

        # create targets
        targets_normal = randn(nsamples)
        targets_mvnormal = map(vcat, targets_normal)

        # create test locations
        testpredictions_μ = randn(ntestsamples)
        testpredictions_σ = rand(ntestsamples)
        testpredictions_normal = map(Normal, testpredictions_μ, testpredictions_σ)
        testpredictions_mvnormal = map(testpredictions_μ, testpredictions_σ) do μ, σ
            MvNormal([μ], σ)
        end
        testtargets_normal = randn(ntestsamples)
        testtargets_mvnormal = map(vcat, testtargets_normal)

        for kernel in (
            WassersteinExponentialKernel() ⊗ SqExponentialKernel(),
            WassersteinExponentialKernel() ⊗
            (SqExponentialKernel() ∘ ScaleTransform(rand())),
            WassersteinExponentialKernel() ⊗
            (SqExponentialKernel() ∘ ARDTransform([rand()])),
        )
            for estimator in
                (BiasedSKCE(kernel), UnbiasedSKCE(kernel), BlockUnbiasedSKCE(kernel, 5))
                skce_mvnormal = estimator(predictions_mvnormal, targets_mvnormal)
                skce_normal = estimator(predictions_normal, targets_normal)
                @test skce_mvnormal ≈ skce_normal
            end

            ucme_mvnormal = UCME(kernel, testpredictions_mvnormal, testtargets_mvnormal)(
                predictions_mvnormal, targets_mvnormal
            )
            ucme_normal = UCME(kernel, testpredictions_normal, testtargets_normal)(
                predictions_normal, targets_normal
            )
            @test ucme_mvnormal ≈ ucme_normal
        end
    end

    @testset "kernels with input transformations" begin
        nsamples = 100
        ntestsamples = 5

        for dim in (1, 10)
            # create predictions and targets
            predictions = [MvNormal(randn(dim), rand()) for _ in 1:nsamples]
            targets = [randn(dim) for _ in 1:nsamples]

            # create random test locations
            testpredictions = [MvNormal(randn(dim), rand()) for _ in 1:ntestsamples]
            testtargets = [randn(dim) for _ in 1:ntestsamples]

            for γ in (1.0, rand())
                kernel1 =
                    WassersteinExponentialKernel() ⊗
                    (SqExponentialKernel() ∘ ScaleTransform(γ))
                kernel2 =
                    WassersteinExponentialKernel() ⊗
                    (SqExponentialKernel() ∘ ARDTransform(fill(γ, dim)))
                kernel3 =
                    WassersteinExponentialKernel() ⊗
                    (SqExponentialKernel() ∘ LinearTransform(diagm(fill(γ, dim))))

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
                    out3 = f(kernel3.kernels[2], p1, t1, p2, t2)
                    @test out2 ≈ out1
                    @test out3 ≈ out1
                    if isone(γ)
                        @test f(SqExponentialKernel(), p1, t1, p2, t2) ≈ out1
                    end
                end

                # check estimates
                for estimator in (UnbiasedSKCE, x -> UCME(x, testpredictions, testtargets))
                    estimate1 = estimator(kernel1)(predictions, targets)
                    estimate2 = estimator(kernel2)(predictions, targets)
                    estimate3 = estimator(kernel3)(predictions, targets)
                    @test estimate2 ≈ estimate1
                    @test estimate3 ≈ estimate1
                    if isone(γ)
                        @test estimator(
                            WassersteinExponentialKernel() ⊗ SqExponentialKernel()
                        )(
                            predictions, targets
                        ) ≈ estimate1
                    end
                end
            end
        end
    end

    @testset "apply" begin
        dim = 10
        μ = randn(dim)
        A = randn(dim, dim)

        for d in (
            MvNormal(μ, rand()), MvNormal(μ, rand(dim)), MvNormal(μ, Symmetric(I + A' * A))
        )
            # unscaled transformation
            for t in (
                ScaleTransform(1.0),
                ARDTransform(ones(dim)),
                LinearTransform(Diagonal(ones(dim))),
            )
                out = CalibrationErrorsDistributions.apply(t, d)
                @test mean(out) ≈ mean(d)
                @test cov(out) ≈ cov(d)
            end

            # scaling
            scale = rand()
            d_scaled = diagm(fill(scale, dim)) * d
            for t in (
                ScaleTransform(scale),
                ARDTransform(fill(scale, dim)),
                LinearTransform(Diagonal(fill(scale, dim))),
            )
                out = CalibrationErrorsDistributions.apply(t, d)
                @test mean(out) ≈ mean(d_scaled)
                @test cov(out) ≈ cov(d_scaled)
            end
        end
    end

    @testset "scale_cov" begin
        dim = 10
        v = rand(dim)
        γ = rand()
        X = diagm(γ .* v .^ 2)

        for A in (ScalMat(dim, γ), PDiagMat(fill(γ, dim)), PDMat(diagm(fill(γ, dim))))
            Y = CalibrationErrorsDistributions.scale_cov(A, v)
            @test Matrix(Y) ≈ X
        end
    end

    @testset "invquad_diff" begin
        dim = 10
        x = rand(dim)
        y = rand(dim)
        γ = rand()
        u = sum(abs2, x - y) / γ

        for A in (ScalMat(dim, γ), PDiagMat(fill(γ, dim)), PDMat(diagm(fill(γ, dim))))
            v = CalibrationErrorsDistributions.invquad_diff(A, x, y)
            @test v ≈ u
        end
    end
end
