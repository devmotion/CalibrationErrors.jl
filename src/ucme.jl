struct UCME{K<:Kernel,TP,TT} <: CalibrationErrorEstimator
    """Kernel."""
    kernel::K
    """Test predictions."""
    testpredictions::TP
    """Test targets."""
    testtargets::TT

    function UCME{K,TP,TT}(kernel::K, testpredictions::TP, testtargets::TT) where {K,TP,TT}
        # obtain number of test locations
        ntest = length(testpredictions)
        ntest ≥ 1 || error("there must be at least one test location")

        # check number of predictions and targets
        length(testtargets) == ntest ||
            throw(DimensionMismatch("number of test predictions and targets must be equal"))

        new{K,TP,TT}(kernel, testpredictions, testtargets)
    end
end

function UCME(kernel1::Kernel, kernel2::Kernel, data...)
    return UCME(TensorProduct(kernel1, kernel2), data...)
end

function UCME(kernel::Kernel, data...)
    predictions, targets = predictions_targets(data...)

    return UCME{typeof(kernel),typeof(predictions),typeof(targets)}(
        kernel, predictions, targets
    )
end

function _calibrationerror(
    estimator::UCME,
    predictions::AbstractVector,
    targets::AbstractVector
)
    @unpack kernel, testpredictions, testtargets = estimator

    # obtain number of samples
    nsamples = length(predictions)
    nsamples ≥ 1 || error("there must be at least one sample")

    # obtain number of test locations
    ntest = length(testpredictions)

    # evaluate statistic for the first test location
    tp = testpredictions[1]
    ty = testtargets[1]
    x = unsafe_ucme_eval_testlocation(kernel, predictions, targets, tp, ty)

    # initialize the estimate
    estimate = x / 1

    # for all other test locations
    for j in 2:ntest
        # evaluate statistic
        tp = testpredictions[j]
        ty = testtargets[j]
        x = unsafe_ucme_eval_testlocation(kernel, predictions, targets, tp, ty)

        # update the estimate
        estimate += (x - estimate) / j
    end

    return estimate
end

function unsafe_ucme_eval_testlocation(
    kernel::Kernel,
    predictions::AbstractVector,
    targets::AbstractVector,
    testprediction,
    testtarget
)
    # obtain number of samples
    nsamples = length(predictions)

    # evaluate statistic for the first sample
    p = predictions[1]
    y = targets[1]
    x = unsafe_ucme_eval(kernel, p, y, testprediction, testtarget)

    # initialize the estimate
    estimate = x / 1

    # for all other samples
    for i in 2:nsamples
        # evaluate statistic
        p = predictions[i]
        y = targets[i]
        x = unsafe_ucme_eval(kernel, p, y, testprediction, testtarget)

        # update the estimate
        estimate += (x - estimate) / i
    end

    return estimate^2
end

function unsafe_ucme_eval(
    kernel::Kernel,
    p::AbstractVector{<:Real},
    y::Integer,
    testp::AbstractVector{<:Real},
    testy::Integer
)
    a = kernel((p, y), (testp, testy))
    b = sum(p[z] * kernel((p, z), (testp, testy)) for z in 1:length(p))

    return a - b
end

function unsafe_ucme_eval(kernel::TensorProduct, p, y, testp, testy)
    κpredictions, κtargets = kernel.kernels
    return unsafe_ucme_eval_targets(κtargets, p, y, testp, testy) * κpredictions(p, testp)
end
# resolve method ambiguity
function unsafe_ucme_eval(
    kernel::TensorProduct,
    p::AbstractVector{<:Real},
    y::Integer,
    testp::AbstractVector{<:Real},
    testy::Integer
)
    κpredictions, κtargets = kernel.kernels
    return unsafe_ucme_eval_targets(κtargets, p, y, testp, testy) * κpredictions(p, testp)
end

function unsafe_ucme_eval_targets(
    κtargets::WhiteKernel,
    p::AbstractVector{<:Real},
    y::Integer,
    testp::AbstractVector{<:Real},
    testy::Integer
)
    @inbounds res = (y == testy) - p[testy]
    return res
end
