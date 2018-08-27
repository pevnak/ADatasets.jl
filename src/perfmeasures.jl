function perfsummary(train, test, α)
	τtrn = quantile(subsampleanomalous(train,α)[1], 1 - α) # quantile on positive  data
	τtst = quantile(subsampleanomalous(test,α)[1], 1 - α) # quantile on positive testing

	DataFrame(
		τtrn = τtrn,
		τtst = τtst,
		fprate_τtrn = mean(filterclass(test,1) .> τtrn),
		tprate_τtrn = mean(filterclass(test,2) .> τtrn),
		fprate_τtst = mean(filterclass(test,1) .> τtst),
		tprate_τtst = mean(filterclass(test,2) .> τtst),
		auc_trn = auc(roccurve(train[1], train[2] .- 1)...),
		auc_tst = auc(roccurve(test[1], test[2] .- 1)...)
		)
end
