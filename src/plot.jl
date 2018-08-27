using Plots
function plotheatmap(x::Matrix,f)
	xl = range(minimum(x[1,:]), stop = maximum(x[1,:]), length = 100)
	yl = range(minimum(x[2,:]), stop = maximum(x[2,:]), length = 100)
	xx = hcat([[i,j] for i in xl for j in yl]...);

	z =  reshape(f(xx),length(xl),length(yl));
	# z = reshape(model(xx).data,length(xl),length(yl))
	contour(xl,yl,z)
end

function plotheatmap(x::Tuple,f)
	x, y = x[1], x[2]
	plotheatmap(x::Matrix,f)
	xx = filterclass((x,y), 1)
	mask = f(xx) .> 0
	scatter!(xx[1,mask], xx[2,mask], marker = [:cross], label = "false positives", color = :red)
	scatter!(xx[1,.!mask], xx[2,.!mask], marker = [:cross], label = "true negatives", color = :blue)
	xx = filterclass((x,y), 2)
	mask = f(xx) .> 0
	scatter!(xx[1,mask], xx[2,mask], marker = [:circle], label = "true positives", color = :red)
	scatter!(xx[1,.!mask], xx[2,.!mask], marker = [:circle], label = "false negatives", color = :blue)
end

