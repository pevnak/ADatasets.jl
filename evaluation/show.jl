using ADatasets
using FileIO
using DataFrames
using Lazy


idir = filter(isdir,["/mnt/output/results/datasets","/Users/tpevny/Work/Julia/results/datasets"])[]

# extractknnauc(df) = mean(by(df,:repetition,dff -> DataFrame(auc = maximum(dff[:auc]))))
# mapreduce( d-> DataFrame(problem = d, knn = extractauc(load("/mnt/output/results/datasets/$(d)/knn_easy_0.05_low.jld","auc"))), vcat, ADatasets.surveydatasets)
# mapreduce( d-> DataFrame(problem = d, knn = extractvaeauctauc(load("/mnt/output/results/datasets/$(d)/knn_easy_0.05_low.jld","auc"))), vcat, ADatasets.surveydatasets)

parseprefix(f) = f[1:search(f,'_')-1]

function extractvaeauc(df::DataFrame,criterion,prefix::String) 
	dff = by(df,[:prediction,:repetition]) do dff
		i = indmax(dff[criterion])
		DataFrame(aucs = dff[i,:test_aucs])
	end
	dff = by(dff,[:prediction]) do dff
		DataFrame(aucs = mean(dff[:aucs]))
	end
	hcat(map(i -> DataFrame([dff[i,:aucs]],[Symbol("$(prefix)_"*dff[i,:prediction])]),1:size(dff,1))...)
end
extractvaeauc(filename::String,criterion)  = extractvaeauc(load(filename,"auc"),criterion,parseprefix(basename(filename)))

function loadresults(f,idir,criterion = :test_aucs)
	files = @>> readdir(idir) map(s -> joinpath(idir,s,f)) filter(isfile)
	mapreduce(vcat,files) do f
		dff = extractvaeauc(f,criterion)
		d = replace(replace(dirname(f),idir,""),"/","")
		hcat(DataFrame(problem = d),dff)
	end
end

function showheatmap(df::DataFrame)
	dff = filter(row -> row[:prediction] == "kde_pxis",df)[[:zmmd,:recerror,:test_aucs]]
	scatter(dff[:zmmd],dff[:recerror],zcolor = dff[:test_aucs])
end

function showheatmap(df::DataFrame)
	dff = filter(row -> row[:prediction] == "kde_pxis",df)[[:zmmd,:recerror,:test_aucs]]
	scatter(dff[:zmmd],dff[:recerror],zcolor = dff[:test_aucs])
end

function rankresults(df)
	algs = setdiff(names(df),[:problem])
	dff = Matrix(df[algs])
	r = mapslices(dff,2) do x 
		y = zeros(Int,size(x))
		y[sortperm(x, rev = true)] .= 1:length(y)
		y 
	end
	r = vcat(r,mean(r,1))
	problems = vcat(df[:problem],"mean")
	dff = DataFrame(hcat(problems,r));
	names!(dff, [:problem,algs...])
	dff
end

dfipmae = loadresults("ipmae_easy_0.05_low.jld",idir)
dfvae = loadresults("vae_easy_0.05_low.jld",idir)
df = join(dfvae,dfipmae,on = :problem)[[:problem,:vae_px,:ipmae_pxis]]
rankresults(df)