struct MedianVarianceBinning <: AbstractBinningAlgorithm
    minsplit::Int
    maxbins::Int
end

MedianVarianceBinning() = MedianVarianceBinning(10)
MedianVarianceBinning(minsplit::Int) = MedianVarianceBinning(minsplit, typemax(Int))

function perform(alg::MedianVarianceBinning, predictions::AbstractMatrix{<:Real},
                 labels::AbstractVector{<:Integer})
    @unpack maxbins, minsplit = alg

    # check arguments
    nclasses, nsamples = check_size(predictions, labels)

    # check first if binning is possible
    (nsamples < minsplit || maxbins <= 1) &&
        return Dict{Int,Bin{eltype(predictions)}}(1 => Bin(predictions, labels))

    # create empty priority queue and set of bins
    queue = PriorityQueue{Tuple{typeof(predictions),typeof(labels),Int},
                          Float64}(Base.Order.Reverse)
    bins = Dict{Int,Bin{eltype(predictions)}}()

    # add data set to priority queue
    varfull = vec(var(predictions; dims = 2))
    maxvar, maxvardim = findmax(varfull)
    enqueue!(queue, (predictions, labels, maxvardim), maxvar)
    nbins = 1

    while nbins < maxbins && !isempty(queue)
        # pick set with largest variance
        predictions, labels, maxvardim = dequeue!(queue)

        # check if minimum split size is reached
        nsamples = size(predictions, 2)

        # compute index of median element
        mid = median(view(predictions, maxvardim, :))

        # compute indices of subsets
        idxsbelow = filter(i -> predictions[maxvardim, i] < mid, 1:nsamples)
        idxsabove = filter(i -> predictions[maxvardim, i] >= mid, 1:nsamples)

        # if splitting is not possible fix remaining partition
        if isempty(idxsbelow)
            bins[length(bins)+1] = Bin(predictions[:, idxsabove], labels[idxsabove])

            # there was only one set left, so we are done
            continue
        end
        if isempty(idxsabove)
            bins[length(bins)+1] = Bin(predictions[:, idxsbelow], labels[idxsbelow])

            # there was only one set left, so we are done
            continue
        end

        # otherwise create subsets and update the queue or list of bins
        for idxs in (idxsbelow, idxsabove)
            # create subsets
            newpredictions = predictions[:, idxs]
            newlabels = labels[idxs]

            # try to avoid needless calculations
            if length(newlabels) < minsplit
                bins[length(bins)+1] = Bin(newpredictions, newlabels)
            else
                newvar = vec(var(newpredictions; dims = 2))
                newmaxvar, newmaxvardim = findmax(newvar)
                enqueue!(queue, (newpredictions, newlabels, newmaxvardim), newmaxvar)
            end

            # we created one additional partition in total
            nbins += 1
        end
    end

    # add remaining bins
    while !isempty(queue)
        # pop queue
        predictions, labels = dequeue!(queue)

        # create bin
        bins[length(bins)+1] = Bin(predictions, labels)
    end

    bins
end
