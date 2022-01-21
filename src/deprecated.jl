@deprecate calibrationerror(estimator, data...) estimator(predictions_targets(data...)...)

@deprecate BiasedSKCE(k1::Kernel, k2::Kernel) BiasedSKCE(k1 ⊗ k2)
@deprecate UnbiasedSKCE(k1::Kernel, k2::Kernel) UnbiasedSKCE(k1 ⊗ k2)
@deprecate BlockUnbiasedSKCE(k1::Kernel, k2::Kernel) BlockUnbiasedSKCE(k1 ⊗ k2)
@deprecate BlockUnbiasedSKCE(k1::Kernel, k2::Kernel, blocksize::Int) BlockUnbiasedSKCE(
    k1 ⊗ k2, blocksize
)
@deprecate UCME(k1::Kernel, k2::Kernel, data...) UCME(k1 ⊗ k2, data...)
@deprecate UCME(kernel::Kernel, testdata...) UCME(
    kernel, predictions_targets(testdata...)...
)
@deprecate UCME(kernel::Kernel, testdata::Tuple{<:Any,<:Any}) UCME(
    kernel, predictions_targets(testdata...)...
)
@deprecate UCME(kernel::Kernel, testdata::AbstractVector{<:Tuple{<:Any,<:Any}}) UCME(
    kernel, predictions_targets(map(first, testdata), map(last, testdata))...
)
@deprecate UCME(
    kernel::Kernel, predictions::AbstractMatrix{<:Real}, targets::AbstractVector
) UCME(kernel, predictions_targets(predictions, targets)...)

@deprecate predictions_targets((predictions, targets)::Tuple{<:Any,<:Any}) predictions_targets(
    predictions, targets
) false
@deprecate predictions_targets(data::AbstractVector{<:Tuple{<:Any,<:Any}}) predictions_targets(
    map(first, data), map(last, data)
) false
@deprecate predictions_targets(predictions::AbstractMatrix{<:Real}, targets::AbstractVector) predictions_targets(
    [predictions[:, i] for i in axes(predictions, 2)], targets
) false
@deprecate predictions_targets(predictions::AbstractVector, targets::AbstractVector) tuple(
    predictions, targets
) false

@deprecate unbiasedskce(kernel, predictions, targets) UnbiasedSKCE(kernel)(
    predictions, targets
) false

@deprecate TVExponentialKernel() ExponentialKernel(; metric=TotalVariation())

@deprecate WassersteinExponentialKernel() ExponentialKernel(; metric=Wasserstein())
@deprecate MixtureWassersteinExponentialKernel() ExponentialKernel(;
    metric=MixtureWasserstein()
)
