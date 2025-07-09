using Printf

# ROC curve for adjacency lists
# A0 is the inferred adjacency, A is the ground truth (assumed boolean adjacency list)
# hyperedges are distinct by their first index, the ordering of the other indices does not matter.
function my_ROC(A0::Matrix{Float64}, A::Matrix{Float64}, n::Int64; verbosity=0)
	function logmsg(level, msg, verbosity)
		if verbosity >= level
			println(msg)
		end
	end
	
	m,o = size(A0)
	mm,o = size(A)
	o -= 1 # one col of A0 is inferred coeff

	# max_edges = n*binomial(n-1,o-1) # number of edges involving exactly o distinct nodes
	max_edges = n*sum(binomial(n-i,o-i) for i in 1:o)

	A1 = sortslices([A0[:,o+1] A0[:,1:o]],dims=1,rev=true) # sorts inferred edges by inferred coeff
	I1 = [A1[i,2:o+1] for i in 1:m] # vec of inferred edges
	V1 = [A1[i,1] for i in 1:m] # vec of inferred coeffs
	I = [A[i,1:o] for i in 1:mm] # vec of ground truth edges 

	tp = [0,]
	fp = [0,]
	
	for i in I1
		if i in I
			# true positive
			push!(tp,tp[end]+1)
			push!(fp,fp[end])
		else
			# false positive
			push!(tp,tp[end])
			push!(fp,fp[end]+1)
		end
	end

	# complete the inference randomly
	mtp = length(I) - tp[end] # missing true positives
	mfp = max_edges-length(I)-fp[end] # missing false positives
	logmsg(1, @sprintf("%d, %d", mtp, mfp), verbosity)
	t = shuffle([ones(mtp);zeros(mfp)])
	tt = 1 .- t
	s = [sum(t[1:i]) for i in 1:length(t)] # true positives get inferred
	ss = [sum(tt[1:i]) for i in 1:length(tt)] # false positives get inferred

	tp = [tp;(tp[end] .+ s)]
	fp = [fp;(fp[end] .+ ss)]

	tpr = tp/length(I)
	fpr = fp/(max_edges-length(I))

	return tpr,fpr
end

function my_ROC(A01::Matrix{Float64}, A1::Matrix{Float64}, A02::Matrix{Float64}, A2::Matrix{Float64}, n::Int64; verbosity=0)
	function logmsg(level, msg, verbosity)
		if verbosity >= level
			println(msg)
		end
	end

	m1,o1 = size(A01)
	mm1,o1 = size(A1)
	o1 -= 1
	max_edges1 = n*sum(binomial(n-i,o1-i) for i in 1:o1)

	m2,o2 = size(A02)
	mm2,o2 = size(A2)
	o2 -= 1
	max_edges2 = n*sum(binomial(n-i,o2-i) for i in 1:o2) # = n * (binomial(n-1, 2) + binomial(n-1, 1)) when o2 = 3

	max_edges = max_edges1 + max_edges2

	a = sortslices([[A01[:,o1+1];A02[:,o2+1]] (1:(size(A01)[1]+size(A02)[1]))],dims=1,rev=true)
	ids = Int64.(a[:,2])
	I01 = [A01[i,1:o1] for i in 1:m1]
	I02 = [A02[i,1:o2] for i in 1:m2]
	I0 = [I01;I02][ids] # vec of inferred edges (pairwise and triadic)

	I1 = [A1[i,1:o1] for i in 1:mm1]
	I2 = [A2[i,1:o2] for i in 1:mm2]
	I = [I1;I2]

	tp = [0,]
	fp = [0,]
	
	for i in I0
		if i in I
			# true positive
			push!(tp,tp[end]+1)
			push!(fp,fp[end])
		else
			# false positive
			push!(tp,tp[end])
			push!(fp,fp[end]+1)
		end
	end

	# complete the inference randomly
	mtp = length(I) - tp[end] # missing true positives
	mfp = max_edges-length(I)-fp[end] # missing false positives
	logmsg(1, @sprintf("%d, %d", mtp, mfp), verbosity)
	t = shuffle([ones(mtp);zeros(mfp)])
	tt = 1 .- t
	s = [sum(t[1:i]) for i in 1:length(t)] # true positives get inferred
	ss = [sum(tt[1:i]) for i in 1:length(tt)] # false positives get inferred

	tp = [tp;(tp[end] .+ s)]
	fp = [fp;(fp[end] .+ ss)]

	tpr = tp/length(I)
	fpr = fp/(max_edges-length(I))

	return tpr,fpr
end

function get_auc(tpr::Vector{Float64}, fpr::Vector{Float64})
	return sum(tpr[2:end].*(fpr[2:end]-fpr[1:end-1]))
end