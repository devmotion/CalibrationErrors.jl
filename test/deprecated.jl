@testset "deprecated.jl" begin
    @testset "calibrationerror" begin
        ece = ECE(UniformBinning(10))
        skce1 = UnbiasedSKCE(SqExponentialKernel() ⊗ WhiteKernel())
        skce2 = BlockUnbiasedSKCE(SqExponentialKernel() ⊗ WhiteKernel())
        skce3 = BiasedSKCE(SqExponentialKernel() ⊗ WhiteKernel())
        ucme = UCME(
            SqExponentialKernel() ⊗ WhiteKernel(),
            [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]],
            [2, 1],
        )

        predictions = map(1:10) do _
            x = rand(5)
            x ./= sum(x)
            return x
        end
        targets = rand(1:5, 10)

        for estimator in (ece, skce1, skce2, skce3, ucme)
            estimate = estimator(predictions, targets)
            @test @test_deprecated(calibrationerror(estimator, predictions, targets)) ==
                estimate
            @test @test_deprecated(calibrationerror(estimator, (predictions, targets))) ==
                estimate
            @test @test_deprecated(
                calibrationerror(estimator, reduce(hcat, predictions), targets)
            ) == estimate
            @test @test_deprecated(
                calibrationerror(estimator, map(tuple, predictions, targets))
            ) == estimate
        end
    end

    @testset "BiasedSKCE" begin
        skce = BiasedSKCE(SqExponentialKernel() ⊗ WhiteKernel())

        skce2 = @test_deprecated BiasedSKCE(SqExponentialKernel(), WhiteKernel())
        @test typeof(skce2) === typeof(skce)
        @test skce2.kernel == skce.kernel
    end

    @testset "UnbiasedSKCE" begin
        skce = UnbiasedSKCE(SqExponentialKernel() ⊗ WhiteKernel())

        skce2 = @test_deprecated UnbiasedSKCE(SqExponentialKernel(), WhiteKernel())
        @test typeof(skce2) === typeof(skce)
        @test skce2.kernel == skce.kernel
    end

    @testset "BlockUnbiasedSKCE" begin
        skce = BlockUnbiasedSKCE(SqExponentialKernel() ⊗ WhiteKernel())

        skce2 = @test_deprecated BlockUnbiasedSKCE(SqExponentialKernel(), WhiteKernel())
        @test typeof(skce2) === typeof(skce)
        @test skce2.kernel == skce.kernel
        @test skce2.blocksize == skce.blocksize
        skce3 = @test_deprecated BlockUnbiasedSKCE(SqExponentialKernel(), WhiteKernel(), 2)
        @test typeof(skce3) === typeof(skce)
        @test skce3.kernel == skce.kernel
        @test skce3.blocksize == skce.blocksize
    end

    @testset "UCME" begin
        # three test locations
        ucme = UCME(
            SqExponentialKernel() ⊗ WhiteKernel(),
            [[1.0, 0], [0.5, 0.5], [0.0, 1]],
            [1, 1, 2],
        )

        # Deprecations
        ucme2 = @test_deprecated UCME(
            SqExponentialKernel(),
            WhiteKernel(),
            [[1.0, 0], [0.5, 0.5], [0.0, 1]],
            [1, 1, 2],
        )
        @test typeof(ucme2) === typeof(ucme)
        @test ucme2.kernel == ucme.kernel
        @test ucme2.testpredictions == ucme.testpredictions
        @test ucme2.testtargets == ucme.testtargets

        ucme3 = @test_deprecated UCME(
            SqExponentialKernel() ⊗ WhiteKernel(), [1.0 0.5 0.0; 0 0.5 1], [1, 1, 2]
        )
        @test typeof(ucme3) === typeof(ucme)
        @test ucme3.kernel == ucme.kernel
        @test ucme3.testpredictions == ucme.testpredictions
        @test ucme3.testtargets == ucme.testtargets

        ucme4 = @test_deprecated UCME(
            SqExponentialKernel() ⊗ WhiteKernel(),
            ([[1.0, 0], [0.5, 0.5], [0.0, 1]], [1, 1, 2]),
        )
        @test typeof(ucme4) === typeof(ucme)
        @test ucme4.kernel == ucme.kernel
        @test ucme4.testpredictions == ucme.testpredictions
        @test ucme4.testtargets == ucme.testtargets

        ucme5 = @test_deprecated UCME(
            SqExponentialKernel() ⊗ WhiteKernel(),
            [([1.0, 0], 1), ([0.5, 0.5], 1), ([0.0, 1], 2)],
        )
        @test typeof(ucme5) === typeof(ucme)
        @test ucme5.kernel == ucme.kernel
        @test ucme5.testpredictions == ucme.testpredictions
        @test ucme5.testtargets == ucme.testtargets
    end

    @testset "predictions_targets" begin
        predictions = map(1:10) do _
            x = rand(5)
            x ./= sum(x)
            return x
        end
        targets = rand(1:5, 10)
        data = (predictions, targets)

        @test @test_deprecated(
            CalibrationErrors.predictions_targets(predictions, targets)
        ) == data
        @test @test_deprecated(
            CalibrationErrors.predictions_targets((predictions, targets))
        ) == data
        @test @test_deprecated(
            CalibrationErrors.predictions_targets(reduce(hcat, predictions), targets)
        ) == data
        @test @test_deprecated(
            CalibrationErrors.predictions_targets(map(tuple, predictions, targets))
        ) == data
    end

    @testset "unbiasedskce" begin
        kernel = SqExponentialKernel() ⊗ WhiteKernel()
        predictions = map(1:10) do _
            x = rand(5)
            x ./= sum(x)
            return x
        end
        targets = rand(1:5, 10)
        @test @test_deprecated(
            CalibrationErrors.unbiasedskce(kernel, predictions, targets)
        ) == UnbiasedSKCE(kernel)(predictions, targets)
    end

    @testset "TVExponentialKernel" begin
        kernel = @test_deprecated(TVExponentialKernel())
        @test kernel isa ExponentialKernel{TotalVariation}
    end

    @testset "WassersteinExponentialKernel" begin
        kernel = @test_deprecated(WassersteinExponentialKernel())
        @test kernel isa ExponentialKernel{Wasserstein}
    end

    @testset "MixtureWassersteinExponentialKernel" begin
        kernel = @test_deprecated(MixtureWassersteinExponentialKernel())
        @test kernel isa ExponentialKernel{<:MixtureWasserstein}
    end
end
