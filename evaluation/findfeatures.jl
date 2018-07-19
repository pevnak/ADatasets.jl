using ADatasets
using Iterators
using StatsBase
using EvalCurves
using Iterators
using Lazy
using DataFrames
using IPMeasures
using SemiSupervised
using FluxExtensions
using Flux
using kNN
using MLDataPattern
import ADatasets: runtest
import FluxExtensions: layerbuilder, freeze


idir = filter(isdir,["/Users/tpevny/Work/Data/datasets/numerical","/mnt/output/data/datasets/numerical","/opt/output/data/datasets/numerical"])[1];
odir = filter(isdir,["/Users/tpevny/Work/Julia/results/datasets","/mnt/output/results/datasets","/opt/output/results/datasets"])[1];


function fit(data,layers,hidden,zdim,β)
  idim = size(data.data[1],1)
  σ2 = SemiSupervised.bandwidthofkde(randn(zdim,1000),20)
	m = SemiSupervised.VAE(FluxExtensions.layerbuilder(idim,hidden,2*zdim,3,"relu","linear","Dense"),
		FluxExtensions.layerbuilder(zdim,hidden,idim,3,"relu","linear","Dense"), β)
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
		likelihood =  - mean(SemiSupervised.lognormal(x, mf.g(z), β)),
		train_px = mean(SemiSupervised.px(mf,x, 100,mf.β)),
		train_pxis = mean(SemiSupervised.pxis(mf, x , 100, mf.β)),
		train_kde_pxis = mean(SemiSupervised.pxis(mf, x, kde, 100, mf.β)),
		train_pxvita = mean(SemiSupervised.pxvita(mf,x,mf.β))
		)
	(m, kde), df
end

predicts = [(m, x) -> - SemiSupervised.px(m[1],x,100,m[1].β)[:], 
	(m, x) -> - SemiSupervised.pxis(m[1],x,100, m[1].β)[:], 
	(m, x) -> - SemiSupervised.pxis(m[1], x, m[2], 100, m[1].β)[:],
	(m, x) -> - SemiSupervised.pxvita(m[1],x,m[1].β)[:]]
prnames = Symbol.(["px","pxis","kde_pxis","vita"])

layers, hidden, zdim, β = 3, 32, 32, 2.0

train, test, clusterdness = ADatasets.makeset(ADatasets.loaddataset("breast-cancer-wisconsin","easy",idir)..., 0.75,"low")

subspace(x,ft) = (x[1][ft,:],x[2])
d = size(train[1],1)
df = mapreduce(vcat, [[i,j] for i in 2:d for j in 1:i-1]) do ft
	trn, tst = subspace(train,ft), subspace(test,ft)
	data = RandomBatches((trn[1],),100,10000)
	m,info = fit(data, layers, hidden, zdim, β)
	mf = FluxExtensions.freeze(m)
	df = DataFrame(ADatasets.evaluate(predicts, mf, tst)') 
	names!(df,prnames)

	model = kNN.KNNAnomaly(trn[1],:kappa)
	ks = [1, 3, 5, 9, 13, 17, 21]
	aucs = map(k -> ADatasets.evaluate(x -> kNN.predict(model,x,k),tst), ks)
	i = indmax(aucs)
	df[:knn] = aucs[i]
	df[:k] = ks[i]
	df[:i] = ft[1]
	df[:j] = ft[2]
	display(df)
	df 
end

function rankvec(x) 
	y = zeros(Int,size(x))
	y[sortperm(x, rev = true)] .= 1:length(y)
	y ./ length(y)
end 

ft = [1,2]
trn, tst = subspace(train,ft), subspace(test,ft)
data = RandomBatches((trn[1],),100,10000)
m,info = fit(data, layers, hidden, zdim, β)
mf = FluxExtensions.freeze(m)
z1 = SemiSupervised.pxis(mf[1], tst[1], mf[2], 100, m[1].β)[:]
model = kNN.KNNAnomaly(trn[1],:kappa)
z2 = kNN.predict(model,tst[1],21)
scatter(tst[1][1,:],tst[1][2,:],zcolor = rankvec(z1), marker = 3*tst[2])
scatter(tst[1][1,:],tst[1][2,:],zcolor = rankvec(z2), marker = 3*tst[2])

