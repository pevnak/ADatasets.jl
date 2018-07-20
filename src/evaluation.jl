using EvalCurves
using MLDataPattern
using FileIO


"""
    detacc_at_fp(n, p, α = [0.005,0.01,0.05])

    detection accuracy at particular false positive rates

"""
detacc_at_fp(n::Vector{T}, p::Vector{T}, α = [0.005,0.01,0.05]) where {T<:Real} = mean(p' .> quantile(n,1 - α),2)
detacc_at_fp(x::Vector{T}, y::Vector{Int}, α = [0.005,0.01,0.05]) where {T<:Real} = detacc_at_fp(x[y .== 1], x[y .== 2],α)

"""
    runtest(fit, ps, predicts, prnames,  dataset, anomaly_type, polution, variation, idir, odir, repetition = 1, steps = 50000)


    fit ---
"""
function runtest(fit, ps, predicts, prnames,  dataset, anomaly_type, polution, variation, idir, odir, name, repetition = 1, steps = 50000)
  println(@sprintf("processing knn %s %s %g %s",dataset, anomaly_type, polution, variation))
  train, test, clusterdness = makeset(loaddataset(dataset,anomaly_type,idir)..., 0.75,variation)
  idim = size(train[1],1)
  data = RandomBatches((train[1],),100,steps)

  results = mapreduce(vcat,ps) do p
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
      df = hcat(df,fprstats(f, m, train,test,0.005))
      df = hcat(df,fprstats(f, m, train,test,0.01))
      hcat(df,fprstats(f, m, train,test,0.05))
    end
    aucs = join(info, aucs, kind = :cross)

  end

  ofname = joinpath(odir,dataset,@sprintf("%s_%s_%g_%s.jld",name,anomaly_type,polution,variation))
  results[:repetition] = repetition
  results[:clusterdness] = clusterdness
  append2file(ofname,"auc",results)
end

"""
  fprstats(f, m, train,test,α)

  puts threshold on false positive rate on training data (assuming most of them are normal)
  and calculates on this threshold false positive rate and detection accuracy

"""
function fprstats(f, m, train,test,α)
  τ = quantile(f(m,train[1]), 1 - α)
  o = f(m, test[1])
  names = map(s -> Symbol(replace(@sprintf("%s_%g",s,α),".","")),["fpr","dacc"]) 
  DataFrame([mean( o[test[2][:] .== 1] .- τ .> 0) detacc_at_fp(o,test[2],[α])[1]], names)
end

auc(predict::Function, data) = EvalCurves.auc(EvalCurves.roccurve(predict(data[1]), data[2] - 1)...)
auc(predict::Function, m, data) = EvalCurves.auc(EvalCurves.roccurve(predict(m, data[1]), data[2] - 1)...)
