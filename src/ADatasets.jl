module ADatasets
using DataFrames, MLDataPattern, FileIO, Distances, StatsBase, Printf, Statistics, Random, JLD2, CSV


surveydatasets = ["breast-cancer-wisconsin", "cardiotocography", "ecoli", "magic-telescope", "waveform-1", "statlog-segment", "wall-following-robot", "yeast", "sonar"]
easydatasets = ["abalone", "blood-transfusion", "breast-cancer-wisconsin", "breast-tissue", "cardiotocography", "ecoli", "glass", "haberman", "ionosphere", "iris", "isolet", "letter-recognition", "libras", "magic-telescope", "miniboone", "multiple-features", "musk-2", "page-blocks", "parkinsons", "pendigits", "pima-indians", "sonar", "spect-heart", "statlog-satimage", "statlog-segment", "statlog-shuttle", "statlog-vehicle", "synthetic-control-chart", "wall-following-robot", "waveform-1", "waveform-2", "wine", "yeast"]

include("roc.jl")
include("perfmeasures.jl")
include("densitylevel.jl")
include("makesets.jl")
include("evaluation.jl")

"""
		function append2file(fname::String,dname::String,d::DataFrame)

		add dataframe `d` to the file `fname` under the name `dname`.
		If dataframe exists, it is appended to the end
"""
function append2file(fname::String,dname::String,d::DataFrame)
  !isdir(dirname(fname)) && mkpath(dirname(fname))
  if isfile(fname)
    a = load(fname)
    if haskey(a,dname)
    	a[dname] = vcat(a[dname],d)
    else 
    	a[dname] = d;
    end
  	save(fname,a)
  else 
  	save(fname,dname,d)
  end
end

function append2file(fname::String,d::DataFrame)
  !isdir(dirname(fname)) && mkpath(dirname(fname))
  d = isfile(fname) ? vcat(load(fname), d) : d
  save(fname,d)
end

end # module
