struct BiasedSKCE{K<:Kernel} <: SKCE
    """Kernel of estimator."""
    kernel::K
end

BiasedSKCE(kernel1::Kernel, kernel2::Kernel) =
    BiasedSKCE(TensorProductKernel(kernel1, kernel2))

function _calibrationerror(skce::BiasedSKCE,
                           predictions::AbstractVector,
                           targets::AbstractVector)
    @unpack kernel = skce

    # obtain number of samples
    nsamples = length(predictions)
    nsamples ≥ 1 || error("there must be at least one sample")

    @inbounds begin
        # evaluate kernel function for the first sample
        prediction = predictions[1]
        target = targets[1]
        hij = unsafe_skce_eval(kernel, prediction, target, prediction, target)

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
                hij = unsafe_skce_eval(kernel, predictioni, targeti, predictionj, targetj)

                # update the estimate (add two terms due to symmetry!)
                n += 2
                estimate += 2 * (hij - estimate) / n
            end

            # evaluate the kernel function
            hij = unsafe_skce_eval(kernel, predictioni, targeti, predictioni, targeti)

            # update the estimate
            n += 1
            estimate += (hij - estimate) / n
        end
    end

    estimate
end
