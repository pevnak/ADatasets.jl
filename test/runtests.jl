using ADatasets
using Test
using ADatasets: sampleindices, subsampletrainvalidation

@testset "test functions for construction of datasets" begin 
	@test length(sampleindices([1,2,3,4,5], 0.2)) == 1
	@test length(sampleindices([1,2,3,4,5], 1.2)) == 5
	@test length(sampleindices([1,2,3,4,5], 10)) == 5
	@test length(sampleindices([1,2,3,4,5], 0)) == 0
	@test length(sampleindices([1,2,3,4,5], 1)) == 1
end

@testset "test functions for construction of datasets" begin
	x = (randn(2,10), fill(0,10))
	xtrn, xval = subsampletrainvalidation(x, 0.1,0.8)
	labels = xtrn[2]
	@test length(xtrn[2]) == 8
	@test size(xtrn[1],2) == 8
	@test length(xval[2]) == 2
	@test size(xval[1],2) == 2


	x = (randn(2,20), vcat(fill(0,10),fill(1,10)))
	xtrn, xval = subsampletrainvalidation(x, 0.1,0.8)
	@test length(xtrn[2]) == 9
	@test size(xtrn[1],2) == 9
	@test sum(xtrn[2] .== 1) == 1
	@test length(xval[2]) == 11
	@test size(xval[1],2) == 11
	@test sum(xval[2] .== 1) == 9

	@test length(sampleindices([1,2,3,4,5], 1.2)) == 5
	@test length(sampleindices([1,2,3,4,5], 10)) == 5
	@test length(sampleindices([1,2,3,4,5], 0)) == 0
	@test length(sampleindices([1,2,3,4,5], 1)) == 1
end

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
