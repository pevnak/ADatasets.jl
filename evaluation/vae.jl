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
using IPMeasures
using MLDataPattern

import ADatasets: runtest, surveydatasets
import FluxExtensions: layerbuilder, freeze

idir = filter(isdir,["/Users/tpevny/Work/Data/datasets/numerical","/mnt/output/data/datasets/numerical","/opt/output/data/datasets/numerical"])[1];
odir = filter(isdir,["/Users/tpevny/Work/Julia/results/datasets","/mnt/output/results/datasets","/opt/output/results/datasets"])[1];


function fit(data,layers,hidden,zdim,β)
  idim = size(first(data)[1],1)
  σ2 = SemiSupervised.bandwidthofkde(randn(zdim,1000),20)
  m = SemiSupervised.VAE(layerbuilder(idim,hidden,2*zdim,layers,"relu","linear","Dense"),
      layerbuilder(zdim,hidden,idim,layers,"relu","linear","Dense"),β)
  m = Adapt.adapt(Float32,m);
  mf = freeze(m)
  opt = Flux.Optimise.ADAM(params(m));

  FluxExtensions.learn((x) -> SemiSupervised.loss(m,getobs(x)),opt,data,() -> (),10000)

  # let's estimate the distance between the prior and posterior using MMD (zmmd) 
  # and calibration distance on prior and prior
  x = IPMeasures.samplecolumns(data.data[1],1000)
  n = size(x,2)
  z = SemiSupervised.gaussiansample(SemiSupervised.hsplitsoftp(mf.q(x))...)
	kσ2 = SemiSupervised.bandwidthofkde(z,20)
	kde = x -> SemiSupervised.kde(x,z,kσ2)
  df = DataFrame(
		layers = layers, 
		hidden = hidden, 
		zdim = zdim,
		β = β,
		zmmd = IPMeasures.mmdg(z, randn(zdim,n), 0.5/σ2),
		ummd = IPMeasures.mmdg(randn(zdim,n), randn(zdim,n), 0.5/σ2),
		recerror =  Flux.mse(x,mf.g(z)) / n,
		likelihood =  - mean(SemiSupervised.log_normal(x, mf.g(z), β)),
		train_px = mean(SemiSupervised.px(mf,x, 100,mf.β)),
		train_pxis = mean(SemiSupervised.pxis(mf, x , 100, mf.β)),
		train_kde_pxis = mean(SemiSupervised.pxis(mf, x, kde, 100, mf.β)),
		train_pxvita = mean(SemiSupervised.pxvita(mf,x,mf.β))
		)
  (mf, kde), df
end

predicts = [(m, x) -> - SemiSupervised.px(m[1],x,100,m[1].β)[:], 
	(m, x) -> - SemiSupervised.pxis(m[1],x,100, m[1].β)[:], 
	(m, x) -> - SemiSupervised.pxis(m[1], x, m[2], 100, m[1].β)[:],
	(m, x) -> - SemiSupervised.pxvita(m[1],x,m[1].β)[:]]
prnames = ["px","pxis","kde_pxis","vita"]
ps = [(3,2^zdim,2^hidden,β) for zdim in 2:5 for hidden in 1:zdim for β in [0.25,0.5,1.0,2.0]]

datasets = (length(ARGS) == 0) ? ADatasets.surveydatasets : ARGS
map(d -> runtest(fit, ps, predicts, prnames, d[1],"easy",0.05,"low",idir, odir, "mvae", d[2],10000),product(datasets,1:10))