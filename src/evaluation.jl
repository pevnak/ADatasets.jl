filterclass(x::Matrix, y::Vector{Int}, c::Int) = x[:,y .== c]
filterclass(x::Vector, y::Vector{Int}, c::Int) = x[y.== c]
filterclass(x::Tuple{A,B}, c::Int) where {A,B} = filterclass(x..., c)

"""
    detacc_at_fp(n, p, α = [0.005,0.01,0.05])

    detection accuracy at particular false positive rates

"""
detacc_at_fp(n::Vector{T}, p::Vector{T}, α = [0.005,0.01,0.05]) where {T<:Real} = mean(p' .> quantile(n,1 .- α), dims = 2)
detacc_at_fp(x::Vector{T}, y::Vector{Int}, α = [0.005,0.01,0.05]) where {T<:Real} = detacc_at_fp(x[y .== 1], x[y .== 2], α)

"""
    runtest(fit, ps, predicts, prnames,  dataset, anomaly_type, polution, variation, idir, odir, repetition = 1, steps = 50000)


    fit ---
"""
function runtest(fit, ps, predicts, prnames,  dataset, anomaly_type, polution, variation, idir, odir, name, repetition = 1, steps = 50000)
  println("processing knn ",dataset, anomaly_type, polution, variation)
  train, test, clusterdness = makeset(loaddataset(dataset,anomaly_type,idir)..., 0.75, variation)
  idim = size(train[1],1)
  data = RandomBatches((subsampleanomalous(train,polution)[1],), 100, steps)

  results = mapreduce(vcat, ps) do p
    m,info = fit(data,p...)

    #calculate areas under the curve on training and testing data for different prediction functions
    aucs = mapreduce(vcat,zip(predicts, prnames)) do i 
      f, prname = i 
      df = DataFrame(
        prname = prname,
      	auc_test = auc(f, m, test),
      	auc_train = auc(f, m, train),
      	auc_train005 = auc(f, m, subsampleanomalous(train,0.005,repetition)),
      	auc_train01 = auc(f, m, subsampleanomalous(train,0.01,repetition))
    	)
      hcat(df,fprstats(f, m, train,test,[0.005,0.01,0.05]))
    end
    aucs = join(info, aucs, kind = :cross)
  end

  results[:repetition] = repetition
  results[:clusterdness] = clusterdness
  ofname = joinpath(odir,dataset,@sprintf("%s_%s_%g_%s",name,anomaly_type,polution,variation))
  append2file(ofname*".jld2","auc",results)
  append2file(ofname*".csv",results)
end


"""
    anomalyexperiment(fit, trainstats, teststats, dataset; aparam, dataparts, repetition)


    trainstats --- statistics to be extracted on training data
    teststats --- statistics to be extracted on testing data
    aparam --- (type = "east", polution = "0.0", variation = "low") , variation parameters of generator of anomalies 
    dataparts --- (0.75, 0.25) fraction of samples used for training and for testing
    repetition --- used as a seed for random number generator to initialize the data
"""
function anomalyexperiment(fit, trainstats, teststats, dataset; aparam = (type = "easy", polution = 0.0, variation = "low"), dataparts = (0.75, 0.25), repetition = 1)
  println("parameters of anomalies: ", aparam)
  trndata, tstdata, clusterdness = makeset(loaddataset(dataset, a.type ,idir)..., dataparts[1], aparam.variation)

  model = fit(subsampleanomalous(trndata, aparam.polution)[1])
end

"""
  fprstats(f, m, train,test,α)

  puts threshold on false positive rate on training data (assuming most of them are normal)
  and calculates on this threshold false positive rate and detection accuracy

"""
function fprstats(f, m, train,test,α::T) where {T<:Real}
  τ = quantile(f(m,train[1]), 1 - α)
  o = f(m, test[1])
  tstfp = mean( o[test[2][:] .== 1] .- τ .> 0)
  dacc = mean(o[test[2][:] .== 2] .- τ .> 0)
  npscore = max(0,tstfp - α)/α + mean(o[test[2][:] .== 2] .- τ .<= 0)
  names = map(s -> Symbol(replace(@sprintf("%s_%g",s,α),"." => "")),["fpr", "dacc", "dacctst", "npscore"]) 
  DataFrame([tstfp dacc detacc_at_fp(o,test[2],[α])[1] npscore] , names)
end

fprstats(f, m, train,test,α::Vector) = mapreduce(i -> fprstats(f, m, train, test, i), hcat, α)

auc(predict::Function, data) = auc(roccurve(predict(data[1]), data[2] .- 1)...)
auc(predict::Function, m, data) = auc(roccurve(predict(m, data[1]), data[2] .- 1)...)
