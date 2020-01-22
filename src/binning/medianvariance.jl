struct MedianVarianceBinning <: AbstractBinningAlgorithm
    minsize::Int
    maxbins::Int

    function MedianVarianceBinning(minsize, maxbins)
        maxbins ≥ 1 || error("maximum number of bins must be positive")

        new(minsize, maxbins)
    end
end

MedianVarianceBinning(minsize::Int = 10) = MedianVarianceBinning(minsize, typemax(Int))

function perform(alg::MedianVarianceBinning,
                 predictions::AbstractVector{<:AbstractVector{T}},
                 targets::AbstractVector{<:Integer}) where {T<:Real}
    @unpack minsize, maxbins = alg

    # check if binning is not possible
    nsamples = length(predictions)
    nsamples < minsize && error("at least $minsize samples are required")

    # check if only trivial binning is possible
    (nsamples == minsize || maxbins == 1) && return [Bin(predictions, targets)]

    # find dimension with maximum variance
    idxs_predictions = collect(1:nsamples)
    max_var_predictions, argmax_var_predictions = max_argmax_var(predictions, idxs_predictions)

    # create priority queue and empty set of bins
    queue = PriorityQueue(
        (idxs_predictions, argmax_var_predictions) => max_var_predictions,
        Base.Order.Reverse)
    bins = Vector{Bin{T}}(undef, 0)

    nbins = 1
    while nbins < maxbins && !isempty(queue)
        # pick the set with the largest variance
        idxs, argmax_var = dequeue!(queue)

        # compute the median along this dimension
        mid = median(predictions[i][argmax_var] for i in idxs)

        # compute indices of the two subsets when splitting at the median
        idxsbelow = [i for i in idxs if predictions[i][argmax_var] < mid]
        idxsabove = [i for i in idxs if predictions[i][argmax_var] ≥ mid]

        # if splitting is not possible, just create one new bin
        if length(idxsbelow) < minsize || length(idxsabove) < minsize
            push!(bins, Bin(predictions[idxs], targets[idxs]))
            continue
        end

        # otherwise update the queue with the new subsets
        for newidxs in (idxsbelow, idxsabove)
            max_var_newidxs, argmax_var_newidxs = max_argmax_var(predictions, newidxs)
            enqueue!(queue, (newidxs, argmax_var_newidxs), max_var_newidxs)
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

    bins::Vector{Bin{T}}
end

function max_argmax_var(x::AbstractVector{<:AbstractVector{<:Real}}, idxs)
    # compute variance along the first dimension
    maxvar = var(x[idx][1] for idx in idxs)

    maxdim = 1
    for d in 1:length(x[1])
        # compute variance along the d-th dimension
        vard = var(x[idx][d] for idx in idxs)

        # update current optimum if required
        if vard > maxvar
            maxvar = vard
            maxdim = d
        end
    end

    maxvar, maxdim
end

