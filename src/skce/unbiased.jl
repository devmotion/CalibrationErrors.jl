struct QuadraticUnbiasedSKCE{K<:MatrixKernel} <: SKCE
    """Kernel of estimator."""
    kernel::K
end

function _calibrationerror(skce::QuadraticUnbiasedSKCE, predictions::AbstractMatrix{<:Real},
                           labels::AbstractVector{<:Integer})
    @unpack kernel = skce

    # obtain number of samples
    nsamples = size(predictions, 2)

    # initialize estimate
    s = zero(skce_result_type(skce, predictions))

    # for all but the first sample
    @inbounds for j in 2:nsamples
        # obtain predictions and labels
        p̃ = view(predictions, :, j)
        ỹ = labels[j]

        # for all previous samples
        for i in 1:(j-1)
            # obtain predictions and labels
            p = view(predictions, :, i)
            y = labels[i]

            # evaluate kernel function and update estimate
            s += skce_kernel(kernel, p, y, p̃, ỹ)
        end
    end

    # normalize estimate
    2 * inv(nsamples * (nsamples - 1)) * s
end

struct LinearUnbiasedSKCE{K<:MatrixKernel} <: SKCE
    """Kernel of estimator."""
    kernel::K
end

function _calibrationerror(skce::LinearUnbiasedSKCE, predictions::AbstractMatrix{<:Real},
                           labels::AbstractVector{<:Integer})
    @unpack kernel = skce

    # obtain number of samples
    nsamples = size(predictions, 2)

    # initialize estimate
    s = zero(skce_result_type(skce, predictions))

    # for all pairs of subsequent samples
    @inbounds for i in 1:2:(nsamples-1)
        # evaluate kernel of next two samples
        j = i + 1
        s += skce_kernel(kernel, view(predictions, :, i), labels[i],
                         view(predictions, :, j), labels[j])
    end

    # normalize estimate
    inv(div(nsamples, 2)) * s
end
