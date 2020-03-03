struct QuadraticUnbiasedSKCE{K<:Kernel} <: SKCE
    """Kernel of estimator."""
    kernel::K
end

QuadraticUnbiasedSKCE(kernel1::Kernel, kernel2::Kernel) =
    QuadraticUnbiasedSKCE(TensorProductKernel(kernel1, kernel2))

function _calibrationerror(skce::QuadraticUnbiasedSKCE,
                           predictions::AbstractVector,
                           targets::AbstractVector)
    @unpack kernel = skce

    # obtain number of samples
    nsamples = length(predictions)
    nsamples ≥ 2 || error("there must be at least two samples")

    @inbounds begin
        # evaluate the kernel function for the first pair of samples
        hij = unsafe_skce_eval(kernel, predictions[1], targets[1], predictions[2], targets[2])

        # initialize the estimate
        estimate = hij / 1

        # for all other pairs of samples
        n = 1
        for j in 3:nsamples
            predictionj = predictions[j]
            targetj = targets[j]

            for i in 1:(j - 1)
                predictioni = predictions[i]
                targeti = targets[i]

                # evaluate the kernel function
                hij = unsafe_skce_eval(kernel, predictioni, targeti, predictionj, targetj)

                # update the estimate
                n += 1
                estimate += (hij - estimate) / n
            end
        end
    end

    estimate
end

struct LinearUnbiasedSKCE{K<:Kernel} <: SKCE
    """Kernel of estimator."""
    kernel::K
end

LinearUnbiasedSKCE(kernel1::Kernel, kernel2::Kernel) =
    LinearUnbiasedSKCE(TensorProductKernel(kernel1, kernel2))

function _calibrationerror(skce::LinearUnbiasedSKCE,
                           predictions::AbstractVector,
                           targets::AbstractVector)
    @unpack kernel = skce

    # obtain number of samples
    nsamples = length(predictions)
    nsamples ≥ 2 || error("there must be at least two samples")

    @inbounds begin
        # evaluate the kernel function for the first pair of samples
        hij = unsafe_skce_eval(kernel, predictions[1], targets[1], predictions[2], targets[2])

        # initialize the estimate
        estimate = hij / 1

        # for all subsequent pairs of samples
        n = 1
        for i in 3:2:(nsamples - 1)
            j = i + 1

            # evaluate the kernel function
            hij = unsafe_skce_eval(kernel, predictions[i], targets[i], predictions[j],
                              targets[j])

            # update the estimate
            n += 1
            estimate += (hij - estimate) / n
        end
    end

    estimate
end
