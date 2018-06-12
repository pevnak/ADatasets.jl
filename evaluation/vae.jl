using Revise
using ADatasets
using Iterators
using StatsBase
using EvalCurves
using Iterators
using Lazy
using DataFrames
using SemiSupervised
using FluxExtensions
using Flux
using MLDataPattern

import ADatasets: makeset, loaddataset, evaluate, append2file, surveydatasets, subsampleanomalous
import FluxExtensions: layerbuilder, freeze



idir = filter(isdir,["/Users/tpevny/Work/Data/datasets/numerical","/mnt/output/data/datasets/numerical"])[1];
odir = filter(isdir,["/Users/tpevny/Work/Julia/results/datasets","/mnt/output/results/datasets"])[1];


function fit(data,layers,hidden,zdim,β)
  idim = size(first(data)[1],1)
  m = SemiSupervised.VAE(layerbuilder(idim,hidden,2*zdim,layers,"relu","linear","Dense"),
      layerbuilder(zdim,hidden,idim,layers,"relu","linear","Dense"),β)
  m = Adapt.adapt(Float32,m);
  mf = freeze(m)
  opt = Flux.Optimise.ADAM(params(m));

  FluxExtensions.learn((x) -> SemiSupervised.loss(m,getobs(x)),opt,data,() -> (),10000)
  mf, DataFrame(layers = layers, hidden = hidden, zdim = zdim, β = β)
end

predicts = [(m, x) -> SemiSupervised.px(m,x,100,m.β)[:], (m, x) -> SemiSupervised.pxis(m,x,100,m.β)[:], (m, x) -> SemiSupervised.pxvita(m,x,m.β)[:]]
prnames = ["px","pxis","vita"]
ps = [(3,2^zdim,2^hidden,β) for zdim in 1:4 for hidden in 1:zdim for β in [0.1,0.5,1.0,2.0]]

function runtest(fit, ps, predicts, prnames,  dataset, anomaly_type, polution, variation,repetition = 1, steps = 50000)
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


map(d -> runtest(fit, ps, predicts, prnames, d,"easy",0.05,"low"),surveydatasets)
