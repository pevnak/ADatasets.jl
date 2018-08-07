function roccurve(scores, truelabels)
    # trualabels ϵ[0, 1]

    descendingidx = sortperm(scores, rev = true)
    scores = scores[descendingidx]
    truelabels = truelabels[descendingidx]

    distincvalueidx = find(diff(scores) .!= 0)
    thresholdidx = vcat(distincvalueidx, length(truelabels))

    tps = cumsum(truelabels)[thresholdidx]
    fps = thresholdidx .- tps

    if length(tps) == 0 || fps[1] != 0 || tps[1] != 0
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        tps = vcat(0, tps)
        fps = vcat(0, fps)
    end

    if fps[end] <= 0
        warn("No negative samples in y_true, false positive value should be meaningless")
        fpr = nothing
    else
        fpr = fps ./ fps[end]
    end

    if tps[end] <= 0
        warn("No positive samples in y_true, true positive value should be meaningless")
        tpr = nothing
    else
        tpr = tps ./ tps[end]
    end

    return fpr, tpr
end

function auc(x, y)
    dx = diff(x)
    dy = y[2:end] + y[1:end - 1]
    return dx' * dy / 2
end
