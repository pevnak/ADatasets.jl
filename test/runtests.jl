using ADatasets
using Base.Test

@testset "degenerative case" begin
	x = ones(3)
	for y in [[1, 0, 1], [0, 1, 1], [1, 1, 0]]
		fpr, tpr = ADatasets.roccurve(x, y)
		@test all(fpr .== [0.0, 1.0]) && all(tpr .== [0.0, 1.0])
		@test ADatasets.auc(fpr,tpr) == 0.5
	end
end

# Compare the these functions to SciKit Learn functions
using PyCall
@pyimport sklearn.metrics as sm

function compareskauc(labels, ascores)
	pyfpr, pytpr, _ = sm.roc_curve(labels, ascores, drop_intermediate = false)
	pyauc = sm.auc(pyfpr, pytpr)

	fpr, tpr = ADatasets.roccurve(ascores, labels)
	auc = ADatasets.auc(fpr, tpr)

	@test pyauc ≈ auc
end

@testset "Scikit learn auc comparison" begin
	# rand dataset
	for i in 1:10
		counts = rand(1:1000, 2)
		labels = vcat(zeros(counts[1]), ones(counts[2]))
		ascores = rand(size(labels))
		compareskauc(labels, ascores)
	end
end
