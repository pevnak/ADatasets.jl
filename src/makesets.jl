using Distances
using CSV 
using StatsBase

function standartize(x...)
    xx = hcat(x...)
    mn = mean(xx,2);
    sd = std(xx,2)
    sd[sd.<1e-6] = 1;
    map(z -> (z .- mn)./sd,x)
end


loaddataset(name,difficulty,idir,T=Float32) = (Matrix{T}(CSV.read(joinpath(idir,name,"normal.txt"),header=false,delim=" "))',Matrix{T}(CSV.read(joinpath(idir,name,difficulty*".txt"),header=false,delim=" "))')


"""
    trData, tstData, clusterdness = makeset(dataset, alpha, frequency, variation, [normalize, seed])

Sample a given dataset, return training and testing subsets and a measure of clusterdness. 
See Emmott, Andrew F., et al. "Systematic construction of anomaly detection benchmarks from 
real data.", 2013 for details.

alpha - the ratio of training to all data\n
difficulty - easy/medium/hard/very_hard problem based on similarity of anomalous measurements to normal\n
frequency - ratio of anomalous to normal data\n
variation - low/high - should anomalies be clustered or not\n
seed - random seed
"""
function makeset(normal, anomalous, alpha, frequency, variation, seed=time_ns())
    # test correct parameters size
    !(0 <= alpha <= 1) && error("alpha must be in the interval [0,1]")
    !(0 <= frequency <= 1) && error("frequency must be in the interval [0,1]")

    # problem dimensions
    M, N = size(normal)
    trN = Int(floor(N*alpha))
    tstN = N - trN

    # how many anomalous points to be sampled 
    aM, aN = size(anomalous)
    trK = minimum(Int.(round.([trN*frequency, aN*alpha])))
    # tstK = minimum(Int.(round.([tstN*frequency, aN*(1-alpha)])))
    tstK = minimum(Int.(round.([tstN*frequency, aN*(1-alpha)])))

    # set seed
    srand(seed)
    # normalize the data to zero mean and unit variance    
    normal, anomalous = standartize(normal, anomalous)

    # randomly sample the training and testing normal data
    inds = sample(1:N, N, replace = false)
    trNdata = normal[:, inds[1:trN]]
    tstNdata = normal[:, inds[trN+1:end]]

    # now sample the anomalous data
    K = trK
    if variation == "low"
        # in this setting, simply sample trK and tstK anomalous points
        # is this done differently in the original paper?
        inds = sample(1:aN, K, replace = false)
    elseif variation == "high"
        # in this setting, randomly choose a point and then K-1 nearest points to it as a cluster
        ind = sample(1:aN, 1)
        x = anomalous[:, ind]
        x = reshape(x, length(x), 1) # promote the vector to a 2D array
        # here maybe other metrics could be used?
        dists = pairwise(Euclidean(), x, anomalous) # get the distance vector
        inds = sortperm(reshape(dists, length(dists))) # get the sorted indices
        inds = inds[1:K] # get the nearest ones
        inds = inds[sample(1:K, K, replace=false)] # scramble them
    end
    trAdata = anomalous[:, inds]
    tstAdata = anomalous[:,setdiff(1:size(anomalous,2),inds)]

    # compute the clusterdness - sample variance of normal vs anomalous instances
    varN = mean(pairwise(Euclidean(), trNdata[:, sample(1:size(trNdata,2), min(1000, N), replace=false)]))/2
    varA = (K > 0) ? mean(pairwise(Euclidean(), trAdata[:, sample(1:K, min(1000, K), replace=false)]))/2 : 0.0

    clusterdness =  (varA>0) ? varN/varA : clusterdness = Inf
    (hcat(trNdata, trAdata),vcat(zeros(Int,trN), ones(Int,trK))), (hcat(tstNdata, tstAdata),vcat(zeros(Int,tstN), ones(Int,tstK))), clusterdness
end
