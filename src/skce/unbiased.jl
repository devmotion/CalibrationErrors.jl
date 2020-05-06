struct UnbiasedSKCE{K<:Kernel} <: SKCE
    """Kernel of estimator."""
    kernel::K
end

function UnbiasedSKCE(kernel1::Kernel, kernel2::Kernel)
    return UnbiasedSKCE(TensorProduct(kernel1, kernel2))
end

function _calibrationerror(
    skce::UnbiasedSKCE,
    predictions::AbstractVector,
    targets::AbstractVector
)
    @unpack kernel = skce

    # obtain number of samples
    nsamples = length(predictions)
    nsamples ≥ 2 || error("there must be at least two samples")

    # compute the estimate
    estimate = unsafe_unbiasedskce(kernel, predictions, targets, 1, nsamples)

    return estimate
end

struct BlockUnbiasedSKCE{K<:Kernel} <: SKCE
    """Kernel of estimator."""
    kernel::K
    """Number of samples per block."""
    blocksize::Int

    function BlockUnbiasedSKCE{K}(kernel::K, blocksize::Int) where K
        blocksize ≥ 2 || error("there must be at least two samples per block")
        new{K}(kernel, blocksize)
    end
end

function BlockUnbiasedSKCE(kernel1::Kernel, kernel2::Kernel, blocksize::Int = 2)
    return BlockUnbiasedSKCE(TensorProduct(kernel1, kernel2), blocksize)
end
function BlockUnbiasedSKCE(kernel::Kernel, blocksize::Int = 2)
    return BlockUnbiasedSKCE{typeof(kernel)}(kernel, blocksize)
end

function _calibrationerror(
    skce::BlockUnbiasedSKCE,
    predictions::AbstractVector,
    targets::AbstractVector
)
    @unpack kernel, blocksize = skce

    # obtain number of samples
    nsamples = length(predictions)
    nsamples ≥ blocksize || error("there must be at least ", blocksize, " samples")

    # compute number of blocks
    nblocks = nsamples ÷ blocksize

    # evaluate U-statistic of the first block
    istart = 1
    iend = blocksize
    x = unsafe_unbiasedskce(kernel, predictions, targets, istart, iend)

    # initialize the estimate
    estimate = x / 1

    # for all other blocks
    for b in 2:nblocks
        # evaluate U-statistic
        istart += blocksize
        iend += blocksize
        x = unsafe_unbiasedskce(kernel, predictions, targets, istart, iend)

        # update the estimate
        estimate += (x - estimate) / b
    end

    return estimate
end

function unsafe_unbiasedskce(kernel, predictions, targets, istart, iend)
    # obtain number of samples
    nsamples = iend - istart + 1

    @inbounds begin
        # evaluate the kernel function for the first pair of samples
        hij = unsafe_skce_eval(kernel, predictions[istart], targets[istart], predictions[istart + 1], targets[istart + 1])

        # initialize the estimate
        estimate = hij / 1

        # for all other pairs of samples
        n = 1
        for j in (istart + 2):iend
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

    return estimate
end
