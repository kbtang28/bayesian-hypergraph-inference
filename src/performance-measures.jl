using Printf, Random

# internal logging helper
_logmsg(level, msg, verbosity, io) = verbosity >= level && println(io, msg)

# randomly complete tp/fp vectors to cover all unranked edges.
function _complete_tp_fp(tp, fp, n_true, max_edges, verbosity, io)
	mtp = n_true - tp[end] 			   # missing true positives
	mfp = max_edges - n_true - fp[end] # missing false positives
	_logmsg(1, @sprintf("%d, %d", mtp, mfp), verbosity, io)

	t = shuffle([ones(mtp); zeros(mfp)])
	tt = 1 .- t
	s = cumsum(t)
	ss = cumsum(tt)
	
	tp = [tp; tp[end] .+ s]
	fp = [fp; fp[end] .+ ss]

	return tp, fp
end

# sort combined inferred edges from two orders by descending score
function _merge_and_sort(A01, A02, o1, o2, m1, m2)
    scores = [A01[:, o1+1]; A02[:, o2+1]]
    orig_ids = 1:(m1 + m2)
    a = sortslices([scores orig_ids], dims=1, rev=true)
    ids = Int.(a[:, 2])

    I01 = [A01[i, 1:o1] for i in 1:m1]
    I02 = [A02[i, 1:o2] for i in 1:m2]
    return [I01; I02][ids]
end

"""
    my_ROC(A0, A, n; verbosity, io) -> (tpr, fpr)

ROC curve for a single interaction order.
`A0`: inferred adjacency list (last column = score); `A`: ground-truth adjacency list.
Randomly completes the curve over all unranked edges.
"""
function my_ROC(A0::Matrix{Float64}, A::Matrix{Float64}, n::Int64; 
			    verbosity=0, io::IO=stdout)
	
	m, _  = size(A0)
	mm, _ = size(A)
	o     = size(A0, 2) - 1 # one col of A0 is inferred coeff

	max_edges = n * binomial(n-1, o-1) # number of edges involving exactly o distinct nodes

	# sort inferred edges by inferred coeff
	A1 = sortslices([A0[:, o+1] A0[:, 1:o]], dims=1, rev=true)
	I1 = [A1[i, 2:o+1] for i in 1:m] # vec of inferred edges
	V1 = [A1[i, 1] for i in 1:m]     # vec of inferred coeffs
	I  = [A[i, 1:o] for i in 1:mm] 	# vec of ground truth edges 

	tp = [0,]
	fp = [0,]
	for i in I1
		if i in I
			# true positive
			push!(tp, tp[end]+1)
			push!(fp, fp[end])
		else
			# false positive
			push!(tp, tp[end])
			push!(fp, fp[end]+1)
		end
	end

	tp, fp = _complete_tp_fp(tp, fp, length(I), max_edges, verbosity, io)

	tpr = tp ./ length(I)
	fpr = fp ./ (max_edges - length(I))

	return tpr, fpr
end

"""
    my_ROC(A01, A1, A02, A2, n; verbosity, io) -> (tpr, fpr)

ROC curve combining pairwise and triadic interaction orders, ranked jointly by score.
"""
function my_ROC(A01::Matrix{Float64}, A1::Matrix{Float64}, 
				A02::Matrix{Float64}, A2::Matrix{Float64}, n::Int64; 
				verbosity=0, io::IO=stdout)
	
	m1, _  = size(A01)
	mm1, _ = size(A1)
	o1     = size(A01, 2) - 1
	max_edges1 = o1 * binomial(n, o1)

	m2, _  = size(A02)
	mm2, _ = size(A2)
	o2     = size(A02, 2) - 1
	max_edges2 = o2 * binomial(n, o2)

	max_edges = max_edges1 + max_edges2

	I0 = _merge_and_sort(A01, A02, o1, o2, m1, m2) # inferred edges

	I1 = [A1[i, 1:o1] for i in 1:mm1]
	I2 = [A2[i, 1:o2] for i in 1:mm2]
	I  = [I1; I2] # true edges

	tp = [0,]
	fp = [0,]
	
	for i in I0
		if i in I
			# true positive
			push!(tp, tp[end]+1)
			push!(fp, fp[end])
		else
			# false positive
			push!(tp, tp[end])
			push!(fp, fp[end]+1)
		end
	end

	# complete the inference randomly
	tp, fp = _complete_tp_fp(tp, fp, length(I), max_edges, verbosity, io)

	tpr = tp ./ length(I)
	fpr = fp ./ (max_edges - length(I))

	return tpr, fpr
end

"""
    my_PRC(A0, A, n; verbosity, io) -> (precision, recall)

Precision-recall curve for a single interaction order.
`A0`: inferred adjacency list (last column = score); `A`: ground-truth adjacency list.
Randomly completes the curve over all unranked edges.
"""
function my_PRC(A0::Matrix{Float64}, A::Matrix{Float64}, n::Int64; 
			    verbosity=0, io::IO=stdout)
	
	m, _  = size(A0)
	mm, _ = size(A)
	o     = size(A0, 2) - 1 # one col of A0 is inferred coeff

	max_edges = n * binomial(n-1, o-1) # number of edges involving exactly o distinct nodes

	# sort inferred edges by inferred coeff
	A1 = sortslices([A0[:, o+1] A0[:, 1:o]], dims=1, rev=true)
	I1 = [A1[i, 2:o+1] for i in 1:m] # vec of inferred edges
	V1 = [A1[i, 1] for i in 1:m]     # vec of inferred coeffs
	I  = [A[i, 1:o] for i in 1:mm] 	# vec of ground truth edges

	tp = [0,]
	fp = [0,]
	
	for i in I1
		if i in I
			# true positive
			push!(tp, tp[end]+1)
			push!(fp, fp[end])
		else
			# false positive
			push!(tp, tp[end])
			push!(fp, fp[end]+1)
		end
	end

	# complete the inference randomly
	tp, fp = _complete_tp_fp(tp, fp, length(I), max_edges, verbosity, io)

	precision = tp[2:end] ./ (tp[2:end] .+ fp[2:end])
	recall    = tp[2:end] ./ length(I) # TPR

	first_prec = tp[2] == 1 ? 1.0 : 0.0
	pushfirst!(precision, first_prec)
	pushfirst!(recall, 0.0)
	
	return precision, recall
end

"""
    my_PRC(A01, A1, A02, A2, n; verbosity, io) -> (precision, recall)

Precision-recall curve combining pairwise and triadic orders, ranked jointly by score.
"""

function my_PRC(A01::Matrix{Float64}, A1::Matrix{Float64}, 
				A02::Matrix{Float64}, A2::Matrix{Float64}, n::Int64; 
				verbosity=0, io::IO=stdout)

	m1, _  = size(A01)
	mm1, _ = size(A1)
	o1     = size(A01, 2) - 1
	max_edges1 = o1 * binomial(n, o1)

	m2, _  = size(A02)
	mm2, _ = size(A2)
	o2     = size(A02, 2) - 1
	max_edges2 = o2 * binomial(n, o2)

	max_edges = max_edges1 + max_edges2

	I0 = _merge_and_sort(A01, A02, o1, o2, m1, m2) # inferred edges

	I1 = [A1[i, 1:o1] for i in 1:mm1]
	I2 = [A2[i, 1:o2] for i in 1:mm2]
	I  = [I1; I2] # vec of ground truth edges

	tp = [0,]
	fp = [0,]
	
	for i in I0
		if i in I
			# true positive
			push!(tp, tp[end]+1)
			push!(fp, fp[end])
		else
			# false positive
			push!(tp, tp[end])
			push!(fp, fp[end]+1)
		end
	end

	# complete the inference randomly
	tp, fp = _complete_tp_fp(tp, fp, length(I), max_edges, verbosity, io)

	precision = tp[2:end] ./ (tp[2:end] .+ fp[2:end])
	recall    = tp[2:end] ./ length(I) # TPR

	first_prec = tp[2] == 1 ? 1.0 : 0.0
	pushfirst!(precision, first_prec)
	pushfirst!(recall, 0.0)
	
	return precision, recall
end

"""
    my_F1(A0, A, n; verbosity, io, extra_out) -> F1 [, precision, recall]

F1 score for a single interaction order. `A0`: inferred adjacency list
(last column = score); `A`: ground-truth adjacency list. Set `extra_out=true`
to also return precision and recall.
"""
function my_F1(A0::Matrix{Float64}, A::Matrix{Float64}, n::Int64; 
			   verbosity=0, io::IO=stdout, extra_out=false)
	
	m, _  = size(A0)
	mm, _ = size(A)
	o     = size(A0, 2) - 1 # one col of A0 is inferred coeff

	max_edges = n * binomial(n-1, o-1) # number of edges involving exactly o distinct nodes

	I0 = [A0[i, 1:o] for i in 1:m] # vec of inferred edges
	I  = [A[i, 1:o] for i in 1:mm] # vec of ground truth edges 

	tp = sum(in(I0).(I))
	fp = length(I0) - tp
	fn = length(I)  - tp

	_logmsg(1, @sprintf("%d, %d", length(I) - tp, max_edges - length(I) - fp), verbosity, io)

	f1 = (2tp) / (2tp + fp + fn)
	if extra_out
		return f1, tp / (tp + fp), tp / length(I)
	else
		return f1
	end
end

"""
    my_F1(A01, A1, A02, A2, n; verbosity, io, extra_out) -> F1 [, precision, recall]

F1 score combining pairwise and triadic interaction orders.
"""
function my_F1(A01::Matrix{Float64}, A1::Matrix{Float64}, 
			   A02::Matrix{Float64}, A2::Matrix{Float64}, n::Int64; 
			   verbosity=0, io::IO=stdout, extra_out=false)

	m1, _  = size(A01)
	mm1, _ = size(A1)
	o1     = size(A01, 2) - 1
	max_edges1 = o1 * binomial(n, o1)

	m2, _  = size(A02)
	mm2, _ = size(A2)
	o2     = size(A02, 2) - 1
	max_edges2 = o2 * binomial(n, o2)

	max_edges = max_edges1 + max_edges2

	I01 = [A01[i, 1:o1] for i in 1:m1]
	I02 = [A02[i, 1:o2] for i in 1:m2]
	I0  = [I01; I02] # vec of inferred edges (pairwise and triadic)

	I1 = [A1[i, 1:o1] for i in 1:mm1]
	I2 = [A2[i, 1:o2] for i in 1:mm2]
	I  = [I1; I2] # vec of ground truth edges

	tp = sum(in(I0).(I))
	fp = length(I0) - tp
	fn = length(I)  - tp

	_logmsg(1, @sprintf("%d, %d", length(I) - tp, max_edges - length(I) - fp), verbosity, io)

	f1 = (2tp) / (2tp + fp + fn)
	if extra_out
		return f1, tp / (tp + fp), tp / length(I)
	else
		return f1
	end
end

"""
    get_auc(ys, xs; rule="RH") -> Float64

Area under a curve defined by points `(xs, ys)`.
`rule="RH"` uses the right-hand (forward) Riemann sum; `rule="T"` uses the trapezoid rule.
"""
function get_auc(ys::Vector{Float64}, xs::Vector{Float64}; rule="RH")
	if rule == "RH"
		return sum(ys[2:end] .* (xs[2:end] .- xs[1:end-1])) # right-hand rule
	elseif rule == "T"
		return sum(0.5 .* (xs[2:end] .- xs[1:end-1]) .* (ys[1:end-1] .+ ys[2:end]))
	end
end

"""
    get_aurocs(Ainf, n, A2l, A3l) -> Vector{Float64}

Area under ROC curve for (1) pooled pairwise and triadic interaction orders,
ranked jointly by score; (2) pairwise interactions only; (3) triadic interactions only.
"""
function get_aurocs(Ainf, n, A2l, A3l)
	aurocs = zeros(Float64, 3)

	tpr, fpr = my_ROC(abs.(Ainf[2]), A2l, abs.(Ainf[3]), A3l, n; verbosity=0)
	aurocs[1] = get_auc(tpr, fpr)

	tpr2, fpr2 = my_ROC(abs.(Ainf[2]), A2l, n)
	aurocs[2] = get_auc(tpr2, fpr2)

	tpr3, fpr3 = my_ROC(abs.(Ainf[3]), A3l, n)
	aurocs[3] = get_auc(tpr3, fpr3)

	return aurocs
end

"""
    get_auprcs(Ainf, n, A2l, A3l) -> Vector{Float64}

Area under precision-recall curve for (1) pooled pairwise and triadic interaction orders,
ranked jointly by score; (2) pairwise interactions only; (3) triadic interactions only.
"""
function get_auprcs(Ainf, n, A2l, A3l)
	auprcs = zeros(Float64, 3)

	prec, rec = my_PRC(abs.(Ainf[2]), A2l, abs.(Ainf[3]), A3l, n; verbosity=0)
	auprcs[1] = get_auc(prec, rec, rule="T")

	prec2, rec2 = my_PRC(abs.(Ainf[2]), A2l, n)
	auprcs[2] = get_auc(prec2, rec2, rule="T")

	prec3, rec3 = my_PRC(abs.(Ainf[3]), A3l, n)
	auprcs[3] = get_auc(prec3, rec3, rule="T")

	return auprcs
end