function perfsummary(train, test, α::Real, name = nothing)
	τtrn = quantile(subsampleanomalous(train,α)[1], 1 - α) # quantile on positive  data
	τtst = quantile(subsampleanomalous(test,α)[1], 1 - α) # quantile on positive testing

	df = DataFrame(
		τtrn = τtrn,
		τtst = τtst,
		fprate_τtrn = mean(filterclass(test,1) .>= τtrn),
		tprate_τtrn = mean(filterclass(test,2) .>= τtrn),
		fprate_τtst = mean(filterclass(test,1) .>= τtst),
		tprate_τtst = mean(filterclass(test,2) .>= τtst),
		auc_trn = auc(roccurve(train[1], train[2] .- 1)...),
		auc_tst = auc(roccurve(test[1], test[2] .- 1)...)
		)
	if name != nothing
		df = hcat(DataFrame(name = name), df)
	end 
	df
end

function perfsummary(train, test, name = nothing)
	df = DataFrame(
		fprate_trn = mean(filterclass(test,1) .>= 0),
		tprate_trn = mean(filterclass(test,2) .>= 0),
		fprate_tst = mean(filterclass(test,1) .>= 0),
		tprate_tst = mean(filterclass(test,2) .>= 0),
		auc_trn = auc(roccurve(train[1], train[2] .- 1)...),
		auc_tst = auc(roccurve(test[1], test[2] .- 1)...)
		)
	if name != nothing
		df = hcat(DataFrame(name = name), df)
	end 
	df
end
