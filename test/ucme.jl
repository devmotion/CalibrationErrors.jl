@testset "ucme.jl" begin
    @testset "UCME: Binary examples" begin
        # categorical distributions
        ucme = UCME(
            SqExponentialKernel() ⊗ WhiteKernel(),
            [[1.0, 0], [0.5, 0.5], [0.0, 1]],
            [1, 1, 2],
        )
        @test iszero(@inferred(ucme([[1, 0], [0, 1]], [1, 2])))
        @test @inferred(ucme([[1, 0], [0, 1]], [1, 1])) ≈ (exp(-2) + exp(-0.5) + 1) / 12
        @test @inferred(ucme([[1, 0], [0, 1]], [2, 1])) ≈ (1 - exp(-1))^2 / 6
        @test @inferred(ucme([[1, 0], [0, 1]], [2, 2])) ≈ (exp(-2) + exp(-0.5) + 1) / 12

        # probabilities
        ucme = UCME(
            (SqExponentialKernel() ∘ ScaleTransform(sqrt(2))) ⊗ WhiteKernel(),
            [1.0, 0.5, 0.0],
            [true, true, false],
        )
        @test iszero(@inferred(ucme([1, 0], [true, false])))
        @test @inferred(ucme([1, 0], [true, true])) ≈ (exp(-2) + exp(-0.5) + 1) / 12
        @test @inferred(ucme([1, 0], [false, true])) ≈ (1 - exp(-1))^2 / 6
        @test @inferred(ucme([1, 0], [false, false])) ≈ (exp(-2) + exp(-0.5) + 1) / 12
    end

    @testset "UCME: Basic properties" begin
        estimates = Vector{Float64}(undef, 1_000)

        for ntest in (1, 5, 10)
            # categorical distributions
            for nclasses in (2, 10, 100)
                dist = Dirichlet(nclasses, 1.0)

                testpredictions = [rand(dist) for _ in 1:ntest]
                testtargets = rand(1:nclasses, ntest)
                ucme = UCME(
                    (ExponentialKernel() ∘ ScaleTransform(0.1)) ⊗ WhiteKernel(),
                    testpredictions,
                    testtargets,
                )

                predictions = [Vector{Float64}(undef, nclasses) for _ in 1:20]
                targets = Vector{Int}(undef, 20)

                for i in 1:length(estimates)
                    rand!.(Ref(dist), predictions)
                    targets .= rand.(Categorical.(predictions))

                    estimates[i] = ucme(predictions, targets)
                end

                @test all(x > zero(x) for x in estimates)
            end

            # probabilities
            testpredictions = rand(ntest)
            testtargets = rand(Bool, ntest)
            ucme = UCME(
                (ExponentialKernel() ∘ ScaleTransform(0.1)) ⊗ WhiteKernel(),
                testpredictions,
                testtargets,
            )

            predictions = Vector{Float64}(undef, 20)
            targets = Vector{Bool}(undef, 20)

            for i in 1:length(estimates)
                rand!(predictions)
                map!(targets, predictions) do p
                    return rand() < p
                end

                estimates[i] = ucme(predictions, targets)
            end

            @test all(x > zero(x) for x in estimates)
        end
    end

    # alternative implementation of white kernel
    struct WhiteKernel2 <: Kernel end
    (::WhiteKernel2)(x, y) = x == y

    # alternative implementation TensorProductKernel
    struct TensorProduct2{K1<:Kernel,K2<:Kernel} <: Kernel
        kernel1::K1
        kernel2::K2
    end
    function (kernel::TensorProduct2)((x1, x2), (y1, y2))
        return kernel.kernel1(x1, y1) * kernel.kernel2(x2, y2)
    end

    @testset "binary classification" begin
        # probabilities and boolean targets
        p, testp = rand(2)
        y, testy = rand(Bool, 2)
        scale = rand()
        kernel = SqExponentialKernel() ∘ ScaleTransform(scale)
        val = unsafe_ucme_eval(kernel ⊗ WhiteKernel(), p, y, testp, testy)
        @test unsafe_ucme_eval(kernel ⊗ WhiteKernel2(), p, y, testp, testy) ≈ val
        @test unsafe_ucme_eval(TensorProduct2(kernel, WhiteKernel()), p, y, testp, testy) ≈
              val
        @test unsafe_ucme_eval(TensorProduct2(kernel, WhiteKernel2()), p, y, testp, testy) ≈
              val

        # corresponding values and kernel for full categorical distribution
        pfull = [p, 1 - p]
        yint = 2 - y
        testpfull = [testp, 1 - testp]
        testyint = 2 - testy
        kernelfull = SqExponentialKernel() ∘ ScaleTransform(scale / sqrt(2))

        @test unsafe_ucme_eval(
            kernelfull ⊗ WhiteKernel(), pfull, yint, testpfull, testyint
        ) ≈ val
        @test unsafe_ucme_eval(
            kernelfull ⊗ WhiteKernel2(), pfull, yint, testpfull, testyint
        ) ≈ val
        @test unsafe_ucme_eval(
            TensorProduct2(kernelfull, WhiteKernel()), pfull, yint, testpfull, testyint
        ) ≈ val
        @test unsafe_ucme_eval(
            TensorProduct2(kernelfull, WhiteKernel2()), pfull, yint, testpfull, testyint
        ) ≈ val
    end

    @testset "multi-class classification" begin
        n = 10
        p = rand(n)
        p ./= sum(p)
        y = rand(1:n)
        testp = rand(n)
        testp ./= sum(testp)
        testy = rand(1:n)

        kernel = SqExponentialKernel() ∘ ScaleTransform(rand())
        val = unsafe_ucme_eval(kernel ⊗ WhiteKernel(), p, y, testp, testy)

        @test unsafe_ucme_eval(kernel ⊗ WhiteKernel2(), p, y, testp, testy) ≈ val
        @test unsafe_ucme_eval(TensorProduct2(kernel, WhiteKernel()), p, y, testp, testy) ≈
              val
        @test unsafe_ucme_eval(TensorProduct2(kernel, WhiteKernel2()), p, y, testp, testy) ≈
              val
    end
end
