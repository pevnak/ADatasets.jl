module ADatasets
using DataFrames
surveydatasets = ["breast-cancer-wisconsin", "cardiotocography", "ecoli", "magic-telescope", "waveform-1", "statlog-segment", "wall-following-robot", "yeast", "sonar"]

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

end # module
