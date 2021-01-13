@doc raw"""
    UnbiasedSKCE(k)

Unbiased estimator of the squared kernel calibration error (SKCE) with kernel `k`.

Kernel `k` on the product space of predictions and targets has to be a `Kernel` from the
Julia package
[KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl)
that can be evaluated for inputs that are tuples of predictions and targets.

# Details

The estimator is unbiased and not guaranteed to be non-negative. Its sample complexity
is ``O(n^2)``, where ``n`` is the total number of samples.

Let ``(P_{X_i}, Y_i)_{i=1,\ldots,n}`` be a data set of predictions and corresponding
targets. The estimator is defined as
```math
\frac{2}{n(n-1)} \sum_{1 \leq i < j \leq n} h_k\big((P_{X_i}, Y_i), (P_{X_j}, Y_j)\big),
```
where
```math
\begin{aligned}
h_k\big((μ, y), (μ', y')\big) ={}&   k\big((μ, y), (μ', y')\big)
                                   - 𝔼_{Z ∼ μ} k\big((μ, Z), (μ', y')\big) \\
                                 & - 𝔼_{Z' ∼ μ'} k\big((μ, y), (μ', Z')\big)
                                   + 𝔼_{Z ∼ μ, Z' ∼ μ'} k\big((μ, Z), (μ', Z')\big).
\end{aligned}
```

# References

Widmann, D., Lindsten, F., & Zachariah, D. (2019). [Calibration tests in multi-class
classification: A unifying framework](https://proceedings.neurips.cc/paper/2019/hash/1c336b8080f82bcc2cd2499b4c57261d-Abstract.html).
In: Advances in Neural Information Processing Systems (NeurIPS 2019) (pp. 12257–12267).

Widmann, D., Lindsten, F., & Zachariah, D. (2021). [Calibration tests beyond
classification](https://openreview.net/forum?id=-bxf89v3Nx).

See also: [`BiasedSKCE`](@ref), [`BlockUnbiasedSKCE`](@ref)
"""
struct UnbiasedSKCE{K<:Kernel} <: SKCE
    """Kernel of estimator."""
    kernel::K
end

"""
    UnbiasedSKCE(k₁, k₂)

Unbiased estimator of the squared kernel calibration error (SKCE) with a tensor product
kernel ``k = k_1 \\otimes k_2``.

See also: [`UnbiasedSKCE`](@ref)
"""
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

@doc raw"""
    BlockUnbiasedSKCE(k[, blocksize = 2])

Unbiased estimator of the squared kernel calibration error (SKCE) with kernel `k` that
uses blocks with `blocksize` samples.

Kernel `k` on the product space of predictions and targets has to be a `Kernel` from the
Julia package
[KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl)
that can be evaluated for inputs that are tuples of predictions and targets.

The number of samples per block must be at least 2 and at most the total number of samples.
Note that samples in the last block are discarded if it is incomplete (see details below).

# Details

The estimator is unbiased and not guaranteed to be non-negative. Its sample complexity
is ``O(Bn)``, where ``B`` is the block size and ``n`` is the total number of samples.

Let ``(P_{X_i}, Y_i)_{i=1,\ldots,n}`` be a data set of predictions and corresponding
targets. The estimator with block size ``B`` is defined as
```math
{\bigg\lfloor \frac{n}{B} \bigg\rfloor}^{-1} \sum_{b=1}^{\lfloor n/B \rfloor}
\frac{2}{B(B-1)} \sum_{(b - 1) B < i < j \leq bB} h_k\big((P_{X_i}, Y_i), (P_{X_j}, Y_j)\big),
```
where
```math
\begin{aligned}
h_k\big((μ, y), (μ', y')\big) ={}&   k\big((μ, y), (μ', y')\big)
                                   - 𝔼_{Z ∼ μ} k\big((μ, Z), (μ', y')\big) \\
                                 & - 𝔼_{Z' ∼ μ'} k\big((μ, y), (μ', Z')\big)
                                   + 𝔼_{Z ∼ μ, Z' ∼ μ'} k\big((μ, Z), (μ', Z')\big).
\end{aligned}
```

# References

Widmann, D., Lindsten, F., & Zachariah, D. (2019). [Calibration tests in multi-class
classification: A unifying framework](https://proceedings.neurips.cc/paper/2019/hash/1c336b8080f82bcc2cd2499b4c57261d-Abstract.html).
In: Advances in Neural Information Processing Systems (NeurIPS 2019) (pp. 12257–12267).

Widmann, D., Lindsten, F., & Zachariah, D. (2021). [Calibration tests beyond
classification](https://openreview.net/forum?id=-bxf89v3Nx).

See also: [`BiasedSKCE`](@ref), [`UnbiasedSKCE`](@ref)
"""
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

"""
    BlockUnbiasedSKCE(k₁, k₂)

Unbiased estimator of the squared kernel calibration error (SKCE) with a tensor product
kernel ``k = k_1 \\otimes k_2`` that uses blocks with `blocksize` samples.

See also: [`BlockUnbiasedSKCE`](@ref)
"""
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
