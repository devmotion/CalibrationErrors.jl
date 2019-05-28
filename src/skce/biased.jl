struct BiasedSKCE{K<:MatrixKernel} <: SKCE
    """Kernel of estimator."""
    kernel::K
end

function _calibrationerror(skce::BiasedSKCE, predictions::AbstractMatrix{<:Real},
                           labels::AbstractVector{<:Integer})
    @unpack kernel = skce

    # obtain number of samples
    nsamples = size(predictions, 2)

    # initialize estimate
    s = zero(skce_result_type(skce, predictions))

    # for all samples
    @inbounds for j in axes(predictions, 2)
        # obtain predictions and labels
        p̃ = view(predictions, :, j)
        ỹ = labels[j]

        # for all previous samples
        for i in 1:(j-1)
            # obtain predictions and labels
            p = view(predictions, :, i)
            y = labels[i]

            # evaluate kernel function and update estimate
            s += 2 * skce_kernel(kernel, p, y, p̃, ỹ)
        end

        # evaluate kernel function and update estimate
        s += skce_kernel(kernel, p̃, ỹ, p̃, ỹ)
    end

    # normalize estimate
    inv(nsamples^2) * s
end
