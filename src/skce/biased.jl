struct BiasedSKCE{K<:MatrixKernel} <: SKCE
    """Kernel of estimator."""
    kernel::K
end

function _calibrationerror(skce::BiasedSKCE,
                           predictions::AbstractVector{<:AbstractVector{<:Real}},
                           targets::AbstractVector{<:Integer})
    @unpack kernel = skce

    # obtain number of samples
    nsamples = length(predictions)
    nsamples > 1 || error("there must be at least one sample")

    @inbounds begin
        # evaluate kernel function for the first sample
        prediction = predictions[1]
        target = targets[1]
        result = skce_kernel(kernel, prediction, target, prediction, target)

        # add evaluations of all other pairs of samples
        for i in 2:nsamples
            predictioni = predictions[i]
            targeti = targets[i]

            for j in 1:(i - 1)
                predictionj = predictions[j]
                targetj = targets[j]

                # evaluate kernel function and update estimate
                result += 2 * skce_kernel(kernel, predictioni, targeti, predictionj, targetj)
            end

            result += skce_kernel(kernel, predictioni, targeti, predictioni, targeti)
        end
    end

    # normalize estimate
    result / nsamples^2
end
