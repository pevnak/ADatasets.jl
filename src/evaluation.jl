using EvalCurves

"""
    runtest(fit, ps, predicts, prnames,  dataset, anomaly_type, polution, variation, idir, odir, repetition = 1, steps = 50000)


    fit ---
"""
function runtest(fit, ps, predicts, prnames,  dataset, anomaly_type, polution, variation, idir, odir, repetition = 1, steps = 50000)
  println(@sprintf("processing knn %s %s %g %s",dataset, anomaly_type, polution, variation))
  train, test, clusterdness = makeset(loaddataset(dataset,anomaly_type,idir)..., 0.75,variation)
  idim = size(train[1],1)
  data = RandomBatches((train[1],),100,steps)

  results = mapreduce(vcat,ps) do p
    mf,info = fit(data,p...)
    aucs = DataFrame(prediction = prnames, 
    	test_aucs = evaluate(predicts,mf,test),
    	train_aucs = evaluate(predicts,mf,train),
    	train_005 = evaluate(predicts,mf,subsampleanomalous(train,0.005,repetition)),
    	train_01 = evaluate(predicts,mf,subsampleanomalous(train,0.01,repetition)),
    	)
    join(info, aucs, kind = :cross)
  end

  ofname = joinpath(odir,dataset,@sprintf("vae_%s_%g_%s.jld",anomaly_type,polution,variation))
  results[:repetition] = repetition
  results[:clusterdness] = clusterdness
  append2file(ofname,"auc",results)
end



evaluate(predict::Function, data) = EvalCurves.auc(EvalCurves.roccurve(predict(data[1]), data[2] - 1)...)
evaluate(predict::Function, m, data) = EvalCurves.auc(EvalCurves.roccurve(predict(m, data[1]), data[2] - 1)...)
evaluate(predict::AbstractArray, m, data) = map(p -> evaluate(p, m, data),predict)
