using ADatasets


idir = filter(isdir,["/mnt/output/results/datasets","/Users/tpevny/Work/Data/datasets/numerical"])[1]
odir = filter(isdir,["/mnt/output/results/datasets","/Users/tpevny/Work/Julia/results/datasets"])[1]






dname = "abalone"
difficulty = "easy"
ADatasets.loaddataset(dname,difficulty,idir)
train, test, variation = ADatasets.makeset(ADatasets.loaddataset("abalone","easy",idir)..., 0.75, "low")



udata = RandomBatches((twomoon(Float64,200)[1][:,1:200],),200,steps)