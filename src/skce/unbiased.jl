struct QuadraticUnbiasedSKCE{K<:MatrixKernel} <: SKCE
    """Kernel of estimator."""
    kernel::K
end

function _calibrationerror(skce::QuadraticUnbiasedSKCE,
                           predictions::AbstractVector{<:AbstractVector{<:Real}},
                           targets::AbstractVector{<:Integer})
    @unpack kernel = skce

    # obtain number of samples
    nsamples = length(predictions)
    nsamples ≥ 2 || error("there must be at least two samples")

    @inbounds begin
        # evaluate kernel function for the first pair of samples
        result = skce_kernel(kernel, predictions[1], targets[1], predictions[2],
                             targets[2])

        # add evaluations of all other pairs of samples
        for j in 3:nsamples
            predictionj = predictions[j]
            targetj = targets[j]

            for i in 1:(j - 1)
                predictioni = predictions[i]
                targeti = targets[i]

                # evaluate kernel function and update estimate
                result += skce_kernel(kernel, predictioni, targeti, predictionj, targetj)
            end
        end
    end

    # normalize estimate
    result / div(nsamples * (nsamples - 1), 2)
end

struct LinearUnbiasedSKCE{K<:MatrixKernel} <: SKCE
    """Kernel of estimator."""
    kernel::K
end

function _calibrationerror(skce::LinearUnbiasedSKCE,
                           predictions::AbstractVector{<:AbstractVector{<:Real}},
                           targets::AbstractVector{<:Integer})
    @unpack kernel = skce

    # obtain number of samples
    nsamples = length(predictions)
    nsamples ≥ 2 || error("there must be at least two samples")

    @inbounds begin
        # evaluate kernel function for the first pair of samples
        result = skce_kernel(kernel, predictions[1], targets[1], predictions[2],
                             targets[2])

        # add evaluations of all subsequent pairs of samples
        for i in 3:2:(nsamples - 1)
            j = i + 1

            # evaluate kernel function for next two samples and update estimate
            result += skce_kernel(kernel, predictions[i], targets[i], predictions[j],
                                  targets[j])
        end
    end

    # normalize estimate
    result / div(nsamples, 2)
end
