function roccurve(scores, truelabels)
    # truelabels l ϵ {0, 1}
    if all(unique(truelabels) .== [1, 2])
        truelabels .-= 1
    end

    descendingidx = sortperm(scores, rev = true)
    scores = scores[descendingidx]
    truelabels = truelabels[descendingidx]

    distincvalueidx = findall(diff(scores) .!= 0)
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

auc(ŷ::Vector{T}, y::Vector{Int}) where {T<: Real} = auc(roccurve(ŷ, y)...)

function auc(x::Vector{T}, y::Vector{T}) where {T<:Real}
    if all(unique(y) == [0,1]) || all(unique(y) == [1, 2])
        @warn "it seems like you are passing a function labels, invoking roccurve"
        x, y = roccurve(x, y)
    end

    dx = diff(x)
    dy = y[2:end] + y[1:end - 1]
    return dx' * dy / 2
end
