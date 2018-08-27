"""
    standartize(x...)

    returned standartized matrices with mean and standard deviation calculated jointly across all datasets

"""
function standartize(x...)
    xx = hcat(x...)
    mn = mean(xx, dims = 2);
    sd = std(xx, dims = 2)
    sd[sd.<1e-6] .= 1;
    map(z -> (z .- mn)./sd,x)
end

"""
    catwithlabels(x...)

    horizontally concatenate matrices `x` while return label vector, starting with first matrix with one

"""
catwithlabels(x...) = hcat(x...), mapreduce(i -> i[1]*ones(Int,i[2]),vcat,enumerate(size.(x,2)))

"""
    typedread(filename,T,transposed = true)

    load CSV file such that all columns are initiated to type T
"""
typedread(filename,T) = T.(Matrix(hcat(TextParse.csvread(filename,' ',header_exists=false,spacedelim = true)[1]...)'))

loaddataset(name,difficulty,idir,T=Float32) = (typedread(joinpath(idir,name,"normal.txt"),T),typedread(joinpath(idir,name,difficulty*".txt"),T))


"""
    trData, tstData, clusterdness = makeset(dataset, alpha, frequency, variation, [normalize, seed])

Sample a given dataset, return training and testing subsets and a measure of clusterdness.
See Emmott, Andrew F., et al. "Systematic construction of anomaly detection benchmarks from
real data.", 2013 for details.

alpha - the ratio of training to all data\n
difficulty - easy/medium/hard/very_hard problem based on similarity of anomalous measurements to normal\n
variation - low/high - should anomalies be clustered or not\n
seed - random seed
"""
function makeset(normal, anomalous, alpha, variation, seed=time_ns())
    # test correct parameters size
    !(0 <= alpha <= 1) && error("alpha must be in the interval [0,1]")

    # problem dimensions
    n = size(normal,2)
    trn_n = Int(round(n*alpha))
    tst_n = n - trn_n

    # how many anomalous points to be sampled
    a = size(anomalous,2)
    trn_a = Int(round(a*alpha))
    tst_a = a - trn_a


    # set seed
    Random.seed!(seed)
    # normalize the data to zero mean and unit variance
    normal, anomalous = standartize(normal, anomalous)

    # randomly sample the training and testing normal data
    inds = randperm(n)[1:trn_n]
    trn_n_data = normal[:, inds]
    tst_n_data = normal[:, setdiff(1:n,inds)]

    # now sample the anomalous data
    if variation == "low"
        # in this setting, simply sample trn_a anomalous points
        inds = randperm(a)[1:trn_a]
    elseif variation == "high"
        error("not impleemnted yet")
        # # in this setting, randomly choose a point and then trn_a-1 nearest points to it as a cluster
        # x = anomalous[:, sample(1:a)]
        # x = reshape(x, : , 1) # promote the vector to a 2D array
        # # here maybe other metrics could be used?
        # dists = pairwise(Euclidean(), x, anomalous) # get the distance vector
        # inds = sortperm(reshape(dists, length(dists))) # get the sorted indices
        # inds = inds[1:trn_a] # get the nearest ones
        # inds = inds[randperm(length(inds))] # scramble them
    end
    trn_a_data = anomalous[:, inds]
    tst_a_data = anomalous[:,setdiff(1:a,inds)]

    # compute the clusterdness - sample variance of normal vs anomalous instances
    varN = mean(pairwise(Euclidean(), trn_n_data[:, sample(1:size(trn_n_data,2), min(1000, size(trn_n_data,2)), replace=false)]))/2
    varA = (trn_n > 0) ? mean(pairwise(Euclidean(), trn_a_data[:, sample(1:size(trn_a_data,2), min(1000, size(trn_a_data,2)), replace=false)]))/2 : 0.0

    clusterdness =  (varA>0) ? varN/varA : clusterdness = Inf
    catwithlabels(trn_n_data, trn_a_data), catwithlabels(tst_n_data, tst_a_data), clusterdness
end

"""
    subsampleanomalous(x, α, seed = time_ns())
    subsampleanomalous(x, n, seed = time_ns())

    removes all but α-fraction (or n) non-anomalous samples from dataset `x = (data,labels)`. Unless
    there is no negative sample and `α` is zero, at least positive (anomalous) sample will be returned
"""
subsampleanomalous(x,α::AbstractFloat,seed = time_ns()) = subsampleanomalous(x,Int(ceil(α*sum(x[2] .== 1)/0.95)),seed)
function subsampleanomalous(x::Tuple{A,B},n::Int,seed = time_ns()) where {A<:AbstractMatrix, B<:Vector}
    inds =  _subsampleanomalous(x[2], n, seed)
    x[1][:,inds], x[2][inds]
end

function subsampleanomalous(x::Tuple{A,B},n::Int,seed = time_ns()) where {A<:AbstractVector, B<:Vector}
    inds =  _subsampleanomalous(x[2], n, seed)
    x[1][inds], x[2][inds]
end

function _subsampleanomalous(labels, n, seed)
    n == 0 && findall(labels .== 1)
    a = findall(labels .> 1)
    inds = sample(a,min(n,length(a)),replace=false)
    vcat(findall(labels .== 1), inds)
end

