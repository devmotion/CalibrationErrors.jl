@doc raw"""
    UCME(k, testpredictions, testtargets)

Estimator of the unnormalized calibration mean embedding (UCME) with kernel `k` and sets of
`testpredictions` and `testtargets`.

Kernel `k` on the product space of predictions and targets has to be a `Kernel` from the
Julia package
[KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl)
that can be evaluated for inputs that are tuples of predictions and targets.

The number of test predictions and test targets must be the same and at least one.

# Details

The estimator is biased and guaranteed to be non-negative. Its sample complexity
is ``O(mn)``, where ``m`` is the number of test locations and ``n`` is the total number of
samples.

Let ``(T_i)_{i=1,\ldots,m}`` be the set of test locations, i.e., test predictions and
corresponding targets, and let ``(P_{X_j}, Y_j)_{j=1,\ldots,n}`` be a data set of
predictions and corresponding targets. The plug-in estimator of ``\mathrm{UCME}_{k,m}^2``
is defined as
```math
m^{-1} \sum_{i=1}^{m} {\bigg(n^{-1} \sum_{j=1}^n k\big(T_i, (P_{X_j}, Y_j)\big)
- \mathbb{E}_{Z \sim P_{X_j}} k\big(T_i, (P_{X_j}, Z)\big)\bigg)}^2.
```

# References

Widmann, D., Lindsten, F., & Zachariah, D. (2021).
[Calibration tests beyond classification](https://openreview.net/forum?id=-bxf89v3Nx).
To be presented at *ICLR 2021*.
"""
struct UCME{K<:Kernel,TP,TT} <: CalibrationErrorEstimator
    """Kernel."""
    kernel::K
    """Test predictions."""
    testpredictions::TP
    """Test targets."""
    testtargets::TT

    function UCME{K,TP,TT}(kernel::K, testpredictions::TP, testtargets::TT) where {K,TP,TT}
        check_nsamples(testpredictions, testtargets)
        return new{K,TP,TT}(kernel, testpredictions, testtargets)
    end
end

function UCME(kernel::Kernel, testpredictions, testtargets)
    return UCME{typeof(kernel),typeof(testpredictions),typeof(testtargets)}(
        kernel, testpredictions, testtargets
    )
end

function (estimator::UCME)(predictions::AbstractVector, targets::AbstractVector)
    @unpack kernel, testpredictions, testtargets = estimator

    # obtain number of samples
    nsamples = check_nsamples(predictions, targets)

    # compute average over test locations
    estimate = mean(zip(testpredictions, testtargets)) do (tp, ty)
        unsafe_ucme_eval_testlocation(kernel, predictions, targets, tp, ty)
    end

    return estimate
end

function unsafe_ucme_eval_testlocation(
    kernel::Kernel,
    predictions::AbstractVector,
    targets::AbstractVector,
    testprediction,
    testtarget,
)
    # compute average over predictions and targets for the given test location
    estimate = mean(zip(predictions, targets)) do (p, y)
        unsafe_ucme_eval(kernel, p, y, testprediction, testtarget)
    end

    return estimate^2
end

function unsafe_ucme_eval(
    kernel::Kernel,
    p::AbstractVector{<:Real},
    y::Integer,
    testp::AbstractVector{<:Real},
    testy::Integer,
)
    a = kernel((p, y), (testp, testy))
    b = sum(p[z] * kernel((p, z), (testp, testy)) for z in 1:length(p))

    return a - b
end

function unsafe_ucme_eval(kernel::KernelTensorProduct, p, y, testp, testy)
    κpredictions, κtargets = kernel.kernels
    return unsafe_ucme_eval_targets(κtargets, p, y, testp, testy) * κpredictions(p, testp)
end
# resolve method ambiguity
function unsafe_ucme_eval(
    kernel::KernelTensorProduct,
    p::AbstractVector{<:Real},
    y::Integer,
    testp::AbstractVector{<:Real},
    testy::Integer,
)
    κpredictions, κtargets = kernel.kernels
    return unsafe_ucme_eval_targets(κtargets, p, y, testp, testy) * κpredictions(p, testp)
end

function unsafe_ucme_eval_targets(
    κtargets::WhiteKernel,
    p::AbstractVector{<:Real},
    y::Integer,
    testp::AbstractVector{<:Real},
    testy::Integer,
)
    @inbounds res = (y == testy) - p[testy]
    return res
end
