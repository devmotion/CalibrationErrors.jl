"""
    MedianVarianceBinning([minsize::Int = 10, maxbins::Int = typemax(Int)])

Dynamic binning scheme of the probability simplex with at most `maxbins` bins that each
contain at least `minsize` samples.

The data set is split recursively as long as it is possible to split the bins while
satisfying these conditions. In each step, the bin with the maximum variance of predicted
probabilities for any component is selected and split at the median of the predicted
probability of the component with the largest variance.
"""
struct MedianVarianceBinning <: AbstractBinningAlgorithm
    minsize::Int
    maxbins::Int

    function MedianVarianceBinning(minsize, maxbins)
        minsize ≥ 1 || error("minimum number of samples must be positive")
        maxbins ≥ 1 || error("maximum number of bins must be positive")

        return new(minsize, maxbins)
    end
end

MedianVarianceBinning(minsize::Int=10) = MedianVarianceBinning(minsize, typemax(Int))

function perform(
    alg::MedianVarianceBinning,
    predictions::AbstractVector{<:AbstractVector{<:Real}},
    targets::AbstractVector{<:Integer},
)
    @unpack minsize, maxbins = alg

    # check if binning is not possible
    nsamples = length(predictions)
    nsamples < minsize && error("at least $minsize samples are required")

    # check if only trivial binning is possible
    minsplit = 2 * minsize
    (nsamples < minsplit || maxbins == 1) && return [Bin(predictions, targets)]

    # find dimension with maximum variance
    idxs_predictions = collect(1:nsamples)
    GC.@preserve idxs_predictions begin
        max_var_predictions, argmax_var_predictions = max_argmax_var(
            predictions, idxs_predictions
        )

        # create priority queue and empty set of bins
        queue = PriorityQueue(
            (idxs_predictions, argmax_var_predictions) => max_var_predictions,
            Base.Order.Reverse,
        )
        bins = Vector{typeof(Bin(predictions, targets))}(undef, 0)

        nbins = 1
        while nbins < maxbins && !isempty(queue)
            # pick the set with the largest variance
            idxs, argmax_var = dequeue!(queue)

            # compute indices of the two subsets when splitting at the median
            idxsbelow, idxsabove = unsafe_median_split!(idxs, predictions, argmax_var)

            # add a bin of all indices if one of the subsets is too small
            # can happen if there are many samples that are equal to the median
            if length(idxsbelow) < minsize || length(idxsabove) < minsize
                push!(bins, Bin(predictions[idxs], targets[idxs]))
                continue
            end

            for newidxs in (idxsbelow, idxsabove)
                if length(newidxs) < minsplit
                    # add a new bin if the subset can not be split further
                    push!(bins, Bin(predictions[newidxs], targets[newidxs]))
                else
                    # otherwise update the queue with the new subsets
                    max_var_newidxs, argmax_var_newidxs = max_argmax_var(
                        predictions, newidxs
                    )
                    enqueue!(queue, (newidxs, argmax_var_newidxs), max_var_newidxs)
                end
            end

            # in total one additional bin was created
            nbins += 1
        end

        # add remaining bins
        while !isempty(queue)
            # pop queue
            idxs, _ = dequeue!(queue)

            # create bin
            push!(bins, Bin(predictions[idxs], targets[idxs]))
        end
    end

    return bins
end

function max_argmax_var(x::AbstractVector{<:AbstractVector{<:Real}}, idxs)
    # compute variance along the first dimension
    maxvar = unsafe_variance_welford(x, idxs, 1)

    maxdim = 1
    for d in 2:length(x[1])
        # compute variance along the d-th dimension
        vard = unsafe_variance_welford(x, idxs, d)

        # update current optimum if required
        if vard > maxvar
            maxvar = vard
            maxdim = d
        end
    end

    return maxvar, maxdim
end

# use Welford algorithm to compute the unbiased sample variance
# taken from: https://github.com/JuliaLang/Statistics.jl/blob/da6057baf849cbc803b952ef7adf979ae3a9f9d2/src/Statistics.jl#L184-L199
# this function is unsafe since it does not perform any bounds checking
function unsafe_variance_welford(
    x::AbstractVector{<:AbstractVector{<:Real}}, idxs::Vector{Int}, dim::Int
)
    n = length(idxs)

    @inbounds begin
        M = x[idxs[1]][dim] / 1
        S = zero(M)
        for i in 2:n
            value = x[idxs[i]][dim]

            new_M = M + (value - M) / i
            S += (value - M) * (value - new_M)
            M = new_M
        end
    end

    return S / (n - 1)
end

# this function is unsafe since it leads to undefined behaviour if the
# outputs are accessed afer `idxs` has been garbage collected
function unsafe_median_split!(
    idxs::Vector{Int}, x::AbstractVector{<:AbstractVector{<:Real}}, dim::Int
)
    n = length(idxs)

    if length(idxs) < 2
        cutoff = 0
    else
        # partially sort the indices `idxs` according to the corresponding values in the
        # `d`th component of`x`
        m = div(n, 2) + 1
        f = let x = x, dim = dim
            idx -> x[idx][dim]
        end
        partialsort!(idxs, 1:m; by=f)

        # figure out all values < median
        # the median is `x[idxs[m]][dim]`` for vectors of odd length
        # and `(x[idxs[m - 1]][dim] + x[idxs[m]][dim]) / 2` for vectors of even length
        if x[idxs[m - 1]][dim] < x[idxs[m]][dim]
            cutoff = m - 1
        else
            # otherwise obtain the last value < median
            firstidxs = unsafe_wrap(Array, pointer(idxs, 1), m - 1)
            cutoff = searchsortedfirst(firstidxs, idxs[m]; by=f) - 1
        end
    end

    # create two new arrays that refer to the two subsets of indices
    idxsbelow = unsafe_wrap(Array, pointer(idxs, 1), cutoff)
    idxsabove = unsafe_wrap(Array, pointer(idxs, cutoff + 1), n - cutoff)

    return idxsbelow, idxsabove
end
