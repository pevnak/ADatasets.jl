module ADatasets
using DataFrames
surveydatasets = ["breast-cancer-wisconsin", "cardiotocography", "ecoli", "magic-telescope", "waveform-1", "statlog-segment", "wall-following-robot", "yeast", "sonar"]

include("makesets.jl")

evaluate(predict::Function, data) = EvalCurves.auc(EvalCurves.roccurve(predict(data[1]), data[2] - 1)...)
evaluate(predict::Function, m, data) = EvalCurves.auc(EvalCurves.roccurve(predict(m, data[1]), data[2] - 1)...)
evaluate(predict::AbstractArray, m, data) = map(p -> evaluate(p, m, data),predict)


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
