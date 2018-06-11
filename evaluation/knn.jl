using Revise
using ADatasets
using Iterators
using StatsBase
using EvalCurves
using Iterators
using kNN
using Lazy
using DataFrames


const idir = filter(isdir,["/Users/tpevny/Work/Data/datasets/numerical","/mnt/output/data/datasets/numerical"])[1];
const odir = filter(isdir,["/Users/tpevny/Work/Anomaly/results","/mnt/output/results/datasets"])[1];

function runtest(dataset, anomaly_type, polution, variation,repetition = 1)
  println(@sprintf("processing knn %s %s %g %s",dataset, anomaly_type,polution,variation))
  train, test, clusterdness = ADatasets.makeset(ADatasets.loaddataset(dataset,anomaly_type,idir)..., 0.75,polution,variation)
  model = kNN.KNNAnomaly(train[1],:kappa)
  ks = @>> 2.^(1:6) filter(i -> i<size(train[1],2))
  results = mapreduce(p -> DataFrame(variant = p[1], k = p[2],auc = ADatasets.evaluate((x) -> kNN.predict(model,x,p[2],p[1]),test)),vcat,product([:kappa,:delta,:gamma], ks))

  ofname = joinpath(odir,dataset,@sprintf("knn_%s_%g_%s.jld",anomaly_type,polution,variation))
  results[:repetition] = repetition
  results[:clusterdness] = clusterdness
  ADatasets.append2file(ofname,"auc",results)
end

map(d -> runtest(d,"easy",0.05,"low"),ADatasets.surveydatasets)
