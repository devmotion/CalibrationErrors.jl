@doc raw"""
    SKCE(k; unbiased::Bool=true, blocksize=identity)

Estimator of the squared kernel calibration error (SKCE) with kernel `k`.

Kernel `k` on the product space of predictions and targets has to be a `Kernel` from the
Julia package
[KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl)
that can be evaluated for inputs that are tuples of predictions and targets.

One can choose an unbiased or a biased variant with `unbiased=true` or `unbiased=false`,
respectively (see details below).

The SKCE is estimated as the average estimate of different blocks of samples. The number of
samples per block is set by `blocksize`:
- If `blocksize` is a function `blocksize(n::Int)`, then the number of samples per block is
  set to `blocksize(n)` where `n` is the total number of samples.
- If `blocksize` is an integer, then the number of samplers per block is set to `blocksize`,
  indepedent of the total number of samples.
The default setting `blocksize=identity` implies that a single block with all samples is
used.

The number of samples per block must be at least 1 if `unbiased=false` and 2 if
`unbiased=true`. Additionally, it must be at most the total number of samples. Note that the
last block is neglected if it is incomplete (see details below).

# Details

The unbiased estimator is not guaranteed to be non-negative whereas the biased estimator is
always non-negative.

The sample complexity of the estimator is ``O(mn)``, where ``m`` is the block size and ``n``
is the total number of samples. In particular, with the default setting `blocksize=identity`
the estimator has a quadratic sample complexity.

Let ``(P_{X_i}, Y_i)_{i=1,\ldots,n}`` be a data set of predictions and corresponding
targets. The estimator with block size ``m`` is defined as
```math
{\bigg\lfloor \frac{n}{m} \bigg\rfloor}^{-1} \sum_{b=1}^{\lfloor n/m \rfloor}
|B_b|^{-1} \sum_{(i, j) \in B_b} h_k\big((P_{X_i}, Y_i), (P_{X_j}, Y_j)\big),
```
where
```math
\begin{aligned}
h_k\big((μ, y), (μ', y')\big) ={}&   k\big((μ, y), (μ', y')\big)
                                   - 𝔼_{Z ∼ μ} k\big((μ, Z), (μ', y')\big) \\
                                 & - 𝔼_{Z' ∼ μ'} k\big((μ, y), (μ', Z')\big)
                                   + 𝔼_{Z ∼ μ, Z' ∼ μ'} k\big((μ, Z), (μ', Z')\big)
\end{aligned}
```
and blocks ``B_b`` (``b = 1, \ldots, \lfloor n/m \rfloor``) are defined as
```math
B_b = \begin{cases}
\{(i, j): (b - 1) m < i < j \leq bm \} & \text{(unbiased)}, \\
\{(i, j): (b - 1) m < i, j \leq bm \} & \text{(biased)}.
\end{cases}
```

# References

Widmann, D., Lindsten, F., & Zachariah, D. (2019). [Calibration tests in multi-class
classification: A unifying framework](https://proceedings.neurips.cc/paper/2019/hash/1c336b8080f82bcc2cd2499b4c57261d-Abstract.html).
In: Advances in Neural Information Processing Systems (NeurIPS 2019) (pp. 12257–12267).

Widmann, D., Lindsten, F., & Zachariah, D. (2021). [Calibration tests beyond
classification](https://openreview.net/forum?id=-bxf89v3Nx).
"""
struct SKCE{K<:Kernel,B} <: CalibrationErrorEstimator
    """Kernel of estimator."""
    kernel::K
    """Whether the unbiased estimator is used."""
    unbiased::Bool
    """Number of samples per block."""
    blocksize::B

    function SKCE{K,B}(kernel::K, unbiased::Bool, blocksize::B) where {K,B}
        if blocksize isa Integer
            blocksize ≥ 1 + unbiased || throw(
                ArgumentError(
                    "there must be at least $(1 + unbiased) $(unbiased ? "samples" : "sample") per block",
                ),
            )
        end
        return new{K,B}(kernel, unbiased, blocksize)
    end
end

function SKCE(kernel::Kernel; unbiased::Bool=true, blocksize::B=identity) where {B}
    return SKCE{typeof(kernel),B}(kernel, unbiased, blocksize)
end

## estimators without blocks
function (skce::SKCE{<:Kernel,typeof(identity)})(
    predictions::AbstractVector, targets::AbstractVector
)
    @unpack kernel, unbiased = skce
    return if unbiased
        unbiasedskce(kernel, predictions, targets)
    else
        biasedskce(kernel, predictions, targets)
    end
end

### unbiased estimator (no blocks)
function unbiasedskce(kernel::Kernel, predictions::AbstractVector, targets::AbstractVector)
    # obtain number of samples
    nsamples = check_nsamples(predictions, targets, 2)

    @inbounds begin
        # evaluate the kernel function for the first pair of samples
        hij = unsafe_skce_eval(
            kernel, predictions[1], targets[1], predictions[2], targets[2]
        )

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

    return estimate
end

### biased estimator (no blocks)
function biasedskce(kernel::Kernel, predictions::AbstractVector, targets::AbstractVector)
    # obtain number of samples
    nsamples = check_nsamples(predictions, targets, 1)

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

    return estimate
end

## estimators with blocks
function (skce::SKCE)(predictions::AbstractVector, targets::AbstractVector)
    @unpack kernel, unbiased, blocksize = skce

    # obtain number of samples
    nsamples = check_nsamples(predictions, targets, 1 + unbiased)

    # compute number of blocks
    _blocksize = blocksize isa Integer ? blocksize : blocksize(nsamples)
    (_blocksize isa Integer && _blocksize >= 1 + unbiased) ||
        error("number of samples per block must be an integer >= $(1 + unbiased)")
    nblocks = nsamples ÷ _blocksize
    nblocks >= 1 || error("at least one block of samples is required")

    # create iterator of partitions
    blocks = Iterators.take(
        zip(
            Iterators.partition(predictions, _blocksize),
            Iterators.partition(targets, _blocksize),
        ),
        nblocks,
    )

    # compute average estimate
    estimator = SKCE(kernel; unbiased=unbiased)
    estimate = mean(
        estimator(_predictions, _targets) for (_predictions, _targets) in blocks
    )

    return estimate
end

"""
    unsafe_skce_eval(k, p, y, p̃, ỹ)

Evaluate
```math
k((p, y), (p̃, ỹ)) - E_{z ∼ p}[k((p, z), (p̃, ỹ))] - E_{z̃ ∼ p̃}[k((p, y), (p̃, z̃))] + E_{z ∼ p, z̃ ∼ p̃}[k((p, z), (p̃, z̃))]
```
for kernel `k` and predictions `p` and `p̃` with corresponding targets `y` and `ỹ`.

This method assumes that `p`, `p̃`, `y`, and `ỹ` are valid and specified correctly, and
does not perform any checks.
"""
function unsafe_skce_eval end

# default implementation for classification
# we do not use the symmetry of `kernel` since it seems unlikely that `(p, y) == (p̃, ỹ)`
function unsafe_skce_eval(
    kernel::Kernel,
    p::AbstractVector{<:Real},
    y::Integer,
    p̃::AbstractVector{<:Real},
    ỹ::Integer,
)
    # precomputations
    n = length(p)

    @inbounds py = p[y]
    @inbounds p̃ỹ = p̃[ỹ]
    pym1 = py - 1
    p̃ỹm1 = p̃ỹ - 1

    tuple_p_y = (p, y)
    tuple_p̃_ỹ = (p̃, ỹ)

    # i = y, j = ỹ
    result = kernel((p, y), (p̃, ỹ)) * (1 - py - p̃ỹ + py * p̃ỹ)

    # i < y
    for i in 1:(y - 1)
        @inbounds pi = p[i]
        tuple_p_i = (p, i)

        # j < ỹ
        @inbounds for j in 1:(ỹ - 1)
            result += kernel(tuple_p_i, (p̃, j)) * pi * p̃[j]
        end

        # j = ỹ
        result += kernel(tuple_p_i, tuple_p̃_ỹ) * pi * p̃ỹm1

        # j > ỹ
        @inbounds for j in (ỹ + 1):n
            result += kernel(tuple_p_i, (p̃, j)) * pi * p̃[j]
        end
    end

    # i = y, j < ỹ
    @inbounds for j in 1:(ỹ - 1)
        result += kernel(tuple_p_y, (p̃, j)) * pym1 * p̃[j]
    end

    # i = y, j > ỹ
    @inbounds for j in (ỹ + 1):n
        result += kernel(tuple_p_y, (p̃, j)) * pym1 * p̃[j]
    end

    # i > y
    for i in (y + 1):n
        @inbounds pi = p[i]
        tuple_p_i = (p, i)

        # j < ỹ
        @inbounds for j in 1:(ỹ - 1)
            result += kernel(tuple_p_i, (p̃, j)) * pi * p̃[j]
        end

        # j = ỹ
        result += kernel(tuple_p_i, tuple_p̃_ỹ) * pi * p̃ỹm1

        # j > ỹ
        @inbounds for j in (ỹ + 1):n
            result += kernel(tuple_p_i, (p̃, j)) * pi * p̃[j]
        end
    end

    return result
end

# for binary classification with probabilities (corresponding to parameters of Bernoulli
# distributions) and boolean targets the expression simplifies to
# ```math
# k((p, y), (p̃, ỹ)) = (y(1-p) + (1-y)p)(ỹ(1-p̃) + (1-ỹ)p̃)(k((p, y), (p̃, ỹ)) - k((p, 1-y), (p̃, ỹ)) - k((p, y), (p̃, 1-ỹ)) + k((p, 1-y), (p̃, 1-ỹ)))
# ```
function unsafe_skce_eval(kernel::Kernel, p::Real, y::Bool, p̃::Real, ỹ::Bool)
    noty = !y
    notỹ = !ỹ
    z =
        kernel((p, y), (p̃, ỹ)) - kernel((p, noty), (p̃, ỹ)) -
        kernel((p, y), (p̃, notỹ)) + kernel((p, noty), (p̃, notỹ))
    return (y ? 1 - p : p) * (ỹ ? 1 - p̃ : p̃) * z
end

# evaluation for tensor product kernels
function unsafe_skce_eval(kernel::KernelTensorProduct, p, y, p̃, ỹ)
    κpredictions, κtargets = kernel.kernels
    return κpredictions(p, p̃) * unsafe_skce_eval_targets(κtargets, p, y, p̃, ỹ)
end

# resolve method ambiguity
function unsafe_skce_eval(
    kernel::KernelTensorProduct,
    p::AbstractVector{<:Real},
    y::Integer,
    p̃::AbstractVector{<:Real},
    ỹ::Integer,
)
    κpredictions, κtargets = kernel.kernels
    return κpredictions(p, p̃) * unsafe_skce_eval_targets(κtargets, p, y, p̃, ỹ)
end
function unsafe_skce_eval(kernel::KernelTensorProduct, p::Real, y::Bool, p̃::Real, ỹ::Bool)
    κpredictions, κtargets = kernel.kernels
    return κpredictions(p, p̃) * unsafe_skce_eval_targets(κtargets, p, y, p̃, ỹ)
end

function unsafe_skce_eval_targets(
    κtargets::Kernel,
    p::AbstractVector{<:Real},
    y::Integer,
    p̃::AbstractVector{<:Real},
    ỹ::Integer,
)
    # ensure that y ≤ ỹ (simplifies the implementation)
    y > ỹ && return unsafe_skce_eval_targets(κtargets, p̃, ỹ, p, y)

    # precomputations
    n = length(p)

    @inbounds begin
        py = p[y]
        pỹ = p[ỹ]
        p̃y = p̃[y]
        p̃ỹ = p̃[ỹ]
    end
    pym1 = py - 1
    pỹm1 = pỹ - 1
    p̃ym1 = p̃y - 1
    p̃ỹm1 = p̃ỹ - 1

    # i = y, j = ỹ
    result = κtargets(y, ỹ) * (1 - py - p̃ỹ + py * p̃ỹ)

    # i < y
    for i in 1:(y - 1)
        @inbounds pi = p[i]
        @inbounds p̃i = p̃[i]

        # i = j < y ≤ ỹ
        result += κtargets(i, i) * pi * p̃i

        # i < j < y ≤ ỹ
        @inbounds for j in (i + 1):(y - 1)
            result += κtargets(i, j) * (pi * p̃[j] + p[j] * p̃i)
        end

        # i < y < j < ỹ
        @inbounds for j in (y + 1):(ỹ - 1)
            result += κtargets(i, j) * (pi * p̃[j] + p[j] * p̃i)
        end

        # i < y ≤ ỹ < j
        @inbounds for j in (ỹ + 1):n
            result += κtargets(i, j) * (pi * p̃[j] + p[j] * p̃i)
        end
    end

    # y < i < ỹ
    for i in (y + 1):(ỹ - 1)
        @inbounds pi = p[i]
        @inbounds p̃i = p̃[i]

        # y < i = j < ỹ
        result += κtargets(i, i) * pi * p̃i

        # y < i < j < ỹ
        @inbounds for j in (i + 1):(ỹ - 1)
            result += κtargets(i, j) * (pi * p̃[j] + p[j] * p̃i)
        end

        # y < i < ỹ < j
        @inbounds for j in (ỹ + 1):n
            result += κtargets(i, j) * (pi * p̃[j] + p[j] * p̃i)
        end
    end

    # ỹ < i
    for i in (ỹ + 1):n
        @inbounds pi = p[i]
        @inbounds p̃i = p̃[i]

        # ỹ < i = j
        result += κtargets(i, i) * pi * p̃i

        # ỹ < i < j
        @inbounds for j in (i + 1):n
            result += κtargets(i, j) * (pi * p̃[j] + p[j] * p̃i)
        end
    end

    # handle special case y = ỹ
    if y == ỹ
        # i < y = ỹ, j = y = ỹ
        @inbounds for i in 1:(y - 1)
            result += κtargets(i, y) * (p[i] * p̃ym1 + pym1 * p̃[i])
        end

        # i = y = ỹ, j > y = ỹ
        @inbounds for j in (y + 1):n
            result += κtargets(y, j) * (pym1 * p̃[j] + p[j] * p̃ym1)
        end
    else
        # i < y
        for i in 1:(y - 1)
            @inbounds pi = p[i]
            @inbounds p̃i = p̃[i]

            # j = y < ỹ
            result += κtargets(i, y) * (pi * p̃y + pym1 * p̃i)

            # y < j = ỹ
            result += κtargets(i, ỹ) * (pi * p̃ỹm1 + pỹ * p̃i)
        end

        # i = y = j < ỹ
        result += κtargets(y, y) * pym1 * p̃y

        # i = y < j < ỹ and y < i < j = ỹ
        for ij in (y + 1):(ỹ - 1)
            @inbounds pij = p[ij]
            @inbounds p̃ij = p̃[ij]

            # i = y < j < ỹ
            result += κtargets(y, ij) * (pym1 * p̃ij + pij * p̃y)

            # y < i < j = ỹ
            result += κtargets(ij, ỹ) * (pij * p̃ỹm1 + pỹ * p̃ij)
        end

        # i = ỹ = j
        result += κtargets(ỹ, ỹ) * pỹ * (p̃ỹ - 1)

        # i = y < ỹ < j and i = ỹ < j
        for j in (ỹ + 1):n
            @inbounds pj = p[j]
            @inbounds p̃j = p̃[j]

            # i = y < ỹ < j
            result += κtargets(y, j) * (pym1 * p̃j + pj * p̃y)

            # i = ỹ < j
            result += κtargets(ỹ, j) * (p̃ỹm1 * pj + p̃j * pỹ)
        end
    end

    return result
end

function unsafe_skce_eval_targets(
    ::WhiteKernel,
    p::AbstractVector{<:Real},
    y::Integer,
    p̃::AbstractVector{<:Real},
    ỹ::Integer,
)
    @inbounds res = (y == ỹ) - p[ỹ] - p̃[y] + dot(p, p̃)
    return res
end

function unsafe_skce_eval_targets(κtargets::Kernel, p::Real, y::Bool, p̃::Real, ỹ::Bool)
    noty = !y
    notỹ = !ỹ
    z = κtargets(y, ỹ) - κtargets(noty, ỹ) - κtargets(y, notỹ) + κtargets(noty, notỹ)
    return (y ? 1 - p : p) * (ỹ ? 1 - p̃ : p̃) * z
end
function unsafe_skce_eval_targets(::WhiteKernel, p::Real, y::Bool, p̃::Real, ỹ::Bool)
    return 2 * (y - p) * (ỹ - p̃)
end
