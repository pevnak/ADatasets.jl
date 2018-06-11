using ADatasets
using FileIO
using DataFrames


idir = filter(isdir,["/mnt/output/results/datasets","/Users/tpevny/Work/Julia/results/datasets"])[]

extractknnauc(df) = mean(by(df,:repetition,dff -> DataFrame(auc = maximum(dff[:auc]))))

parseprefix(f) = f[1:search(f,'_')-1]
function extractvaeauc(df::DataFrame,criterion,prefix::String) 
	dff = by(df,[:prediction,:repetition]) do dff
		i = indmin(dff[criterion])
		DataFrame(aucs = dff[i,:aucs])
	end
	dff = by(df,[:prediction]) do dff
		DataFrame(aucs = mean(dff[:aucs]))
	end
	hcat(map(i -> DataFrame([dff[i,:aucs]],[Symbol("$(prefix)_"*dff[i,:prediction])]),1:size(dff,1))...)
end
extractvaeauc(filename::String,criterion)  = extractvaeauc(load(filename,"auc"),criterion,parseprefix(basename(filename)))

mapreduce( d-> DataFrame(problem = d, knn = extractauc(load("/mnt/output/results/datasets/$(d)/knn_easy_0.05_low.jld","auc"))), vcat, ADatasets.surveydatasets)
mapreduce( d-> DataFrame(problem = d, knn = extractvaeauctauc(load("/mnt/output/results/datasets/$(d)/knn_easy_0.05_low.jld","auc"))), vcat, ADatasets.surveydatasets)

rfile = "vae_easy_0.05_low.jld"

mapreduce(vcat,readdir(idir)) do d
	dff = mapreduce(f -> extractvaeauc(joinpath(idir,d,f),:train_aucs), hcat, ["vae_easy_0.05_low.jld","iwae_easy_0.05_low.jld"])
	hcat(DataFrame(problem = d),dff)
end