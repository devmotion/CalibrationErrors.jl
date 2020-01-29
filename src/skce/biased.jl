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
    nsamples ≥ 1 || error("there must be at least one sample")

    @inbounds begin
        # evaluate kernel function for the first sample
        prediction = predictions[1]
        target = targets[1]
        hij = skce_kernel(kernel, prediction, target, prediction, target)

        # initialize the calibration error estimate
        estimate = hij / 1

        # for all other pairs of samples
        n = 1
        for i in 2:nsamples
            predictioni = predictions[i]
            targeti = targets[i]

            for j in 1:(i - 1)
                predictionj = predictions[j]
                targetj = targets[j]

                # evaluate the kernel function
                hij = skce_kernel(kernel, predictioni, targeti, predictionj, targetj)

                # update the estimate (add two terms due to symmetry!)
                n += 2
                estimate += 2 * (hij - estimate) / n
            end

            # evaluate the kernel function
            hij = skce_kernel(kernel, predictioni, targeti, predictioni, targeti)

            # update the estimate
            n += 1
            estimate += (hij - estimate) / n
        end
    end

    estimate
end
