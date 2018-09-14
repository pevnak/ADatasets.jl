"""
    runtest(fit, ps, predicts, prnames,  dataset, anomaly_type, polution, variation, idir, odir, repetition = 1, steps = 50000)


    fit ---
"""
function densitylevel(fit, ps, dataset, α, idir, odir, name, repetition = 1; bs = 100, steps = 50000)
  println(name,"  ",dataset)

  #load data and reserve 50% for training, 25% for validation, 25% for testing
  normal = typedread(joinpath(idir,dataset,"normal.txt"),Float32)
  train, val, test  = splitobs(shuffleobs(normal), at = (0.5, 0.25))
  data = RandomBatches((train,),bs,steps)


  results = mapreduce(vcat,ps) do p
    m, info = fit(data,p...)

    #calculate areas under the curve on training and testing data for different prediction functions
    df = DataFrame(
      τ_train = quantile(m(getobs(train))[:], α),
      τ_val = quantile(m(getobs(val))[:], α),
      τ_test = quantile(m(getobs(test))[:], α),
      fp_train = mean(m(getobs(train)) .< 0),
      fp_val = mean(m(getobs(val)) .< 0),
      fp_test = mean(m(getobs(test)) .< 0)
  	)
    hcat(info, df)
  end

  ofname = joinpath(odir,dataset,@sprintf("%s_densitylevel.jld2",name))
  results[:repetition] = repetition
  append2file(ofname,"results",results)
end