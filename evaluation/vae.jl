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

import ADatasets: runtest
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

map(d -> runtest(fit, ps, predicts, prnames, d,"easy",0.05,"low",idir, odir, i),surveydatasets)
