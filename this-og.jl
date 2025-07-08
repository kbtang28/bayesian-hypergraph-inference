include("this-tools.jl")

# ================================================================================
"""
	this(X::Matrix{Float64}, Y::Matrix{Float64}, ooi::Vector{Int64}, dmax::Int64, λ::Float64=.1, ρ::Float64=1., niter::Int64=10)

Infers the hypergraph underlying the dynamics of its vertices with knowledge of the states 'X' and of the derivatives 'Y' at each vertex.

_INPUT_:\\
`X`: Time series of the system's state. Each column is the time series of the state of one agent.\\
`Y`: Time series of the system's velocity. Each column is the time series of the velocity of on agent.\\
`ooi`: Orders of interest. Vector of integers listing the orders of interactions that we analyze.\\
`dmax`: Maximal degree to be considered in the Taylor expansion. Typically, dmax=maximum(ooi).\\
`λ`: SINDy's threshold deciding whether an hyperedge exists or not.\\
`ρ`: Regularization parameters promoting sparsity.\\
`niter`: Maximal number of iterations for THIS algorithm.

_OUTPUT_:\\
`Ainf`: Dictionary associating the inferred coefficient of the Taylor expansion to a pair (node,hyperedge). Namely, 'Ainf[(i,h)]' is the coefficient corresponding to the hyperedge 'h' in the Taylor expansion of the dynamics of node 'i'. If agents have internal dimensions larger than 1, `Ainf` is just a boolean.\\
`coeff`: Matrix of coefficents obtained by SINDy. The column indices are the agents' indices and the columns indices are the indices of the monomials in the Taylor series.\\
`relerr`: Relative error, i.e., `err` normalized by the magnitude of `Y`.
"""
function this(X::Matrix{Float64}, Y::Matrix{Float64}, ooi::Vector{Int64}, dmax::Int64, λ::Float64=.1, ρ::Float64=1., niter::Int64=10)
	# Size of the data matrix. 
	T, n = size(X)

	if size(X) != size(Y)
		@info "Dimensions of states and derivatives do not match."
		return nothing
	end

	# Running THIS ##############################################################
	# Retrieve the values of the monomials at each time step.
	θ, d = get_θd(X,dmax)
	idx_mon = Dict{Int64,Vector{Int64}}()
	for i in 1:size(d)[1]
		mon = d[i,:][d[i,:] .!= 0]
		if length(mon) == length(union(mon)) # skips monomials with repeated factors, e.g., xᵢxⱼ²
			idx_mon[i] = sort(mon)
		end
	end

	# Running SINDy
	coeff, err = mySINDy(θ,Y,λ,ρ,niter)
	relerr = err/norm(Y,1)

	#############################################################################

	# Reconstructing the adjacency tensors ######################################
	Ainf = Dict{Int64,Matrix{Float64}}(o => zeros(0,o+1) for o in 1:dmax+1) # For each order o, associates a dictionary associating the pair (agents, hyperedge) to the inferred weight, as seen from the agent.
	idx_coeff = Dict{Int64,Vector{Int64}}()
	nz_idx = Int64[]
	for i in keys(idx_mon)
		aaa = setdiff((1:n)[abs.(coeff[i,:]) .> 1e-8], idx_mon[i]) # ensures monomial involving xᵢ does not get inferred for xᵢ
		if !isempty(aaa)
			push!(nz_idx,i)
			idx_coeff[i] = aaa
		end
	end

	for id in nz_idx
		ii = idx_coeff[id]
		jj = idx_mon[id]
		o = length(jj)+1
		Ainf[o] = vcat(Ainf[o],[ii repeat(jj',length(ii),1) coeff[id,ii]])
	end

	#############################################################################

	return Ainf, coeff, relerr
end

# ================================================================================
"""
	mySINDy(θ::Matrix{Float64}, Y::Matrix{Float64}, λ::Float64=.1, ρ::Float64=1., niter::Int64=10)

Own implementation of SINDy. Adapted from [https://github.com/eurika-kaiser/SINDY-MPC/blob/master/utils/sparsifyDynamics.m], accessed on December 27, 2023.

_INPUT_:\\
`θ`: Values of the basis functions at each time steps. Here the basis functions are the monomials up to degree `dmax`.\\
`Y`: Time series of the system's velocity. Each row is the time series of the velocity of on agent.\\
`λ`: SINDy's threshold deciding whether an hyperedge exists or not.\\
`ρ`: Regularization parameters promoting sparsity.\\
`niter`: Maximal number of iterations for THIS algorithm.

_OUTPUT_:\\
`Ξ`: Matrix of coefficient inferred by SINDy. The index of each row is the index of an agent and the the columns corresponds to elements of the Taylor basis.\\
`err`: 1-norm of the difference between `Y` and the inferred dynamics.
"""
function mySINDy(θ::Matrix{Float64}, Y::Matrix{Float64}, λ::Float64=.1, ρ::Float64=1., niter::Int64=10)
	# Sizes of the data matrices.
	T,n = size(Y)
	T,m = size(θ)

	# Initialization of the coefficient matrix.
	Ξ = (θ'*θ + ρ*Id(m)) \ θ'*Y # Least square with Tikhonov regularization (ridge regression)
	
	nz = 1e6 # Number of nonzero elements in Ξ.
	k = 1 # Counter of iterations.

	while k < niter && sum(abs.(Ξ) .> 1e-6) != nz
		k += 1
		nz = sum(abs.(Ξ) .> 1e-6)

		# Getting rid of the small (< λ) components and re-optimizing.
		smallinds = (abs.(Ξ) .< λ)
		Ξ[smallinds] .= 0.
		for i in 1:n
			biginds = .~smallinds[:, i]
			Ξ[biginds, i] = @views (θ[:, biginds]'*θ[:, biginds] + ρ*Id(sum(biginds))) \ (θ[:, biginds]'*Y[:, i])
		end
	end

	# Absolute error
	err = norm(Y - θ*Ξ,1)

	return Ξ, err
end