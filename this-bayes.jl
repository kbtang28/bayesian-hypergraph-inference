using SparseBayes
include("this-tools.jl")

function this_bayes(X::Matrix{Float64}, Y::Matrix{Float64}, ooi::Vector{<:Integer}, dmax::Int64; opts=SBOpts(nitr=500, free_basis=[1]), settings=SBSettings(), ctrls=SBCtrlSettings(beta_update_frequency=3))
    T, n = size(X)

    if size(X) != size(Y)
        @info "Dimensions of states and derivatives do not match."
        return nothing
    end

    # retrieve values of monomials at each time step
    θ, d = get_θd(X, dmax)

    idx_mon = Dict{Int64, Vector{Int64}}() # (monomial index) => (nodes involved in monomial)
    for i in 1:size(d)[1]
		mon = d[i,:][d[i,:] .!= 0]
		if length(mon) == length(union(mon)) # skips monomials with repeated factors, e.g., xᵢxⱼ²
			idx_mon[i] = sort(mon)
		end
	end

    # run sparse Bayes
    out, diagnostics = sparse_bayes(θ, Y; opts=opts, settings=settings, ctrls=ctrls)

    # calculate relative error
    err = norm(Y - θ*out.value, 1)
    relerr = err/norm(Y, 1)

    # reconstruct adjacency tensors
    Ainf = Dict{Int64,Matrix{Float64}}(o => zeros(0,o+1) for o in vcat(1, ooi))

    idx_coeff = Dict{Int64,Vector{Int64}}() # (monomial index) => (nodes for which monomial coeff is nonzero)
    for id in keys(idx_mon)
		aaa = setdiff((1:n)[abs.(out.value[id, :]) .> 1e-8],idx_mon[id]) # ensures monomial involving xᵢ does not get inferred for xᵢ
		if !isempty(aaa)
			idx_coeff[id] = aaa
		end
	end

	for id in keys(idx_coeff)
		ii = idx_coeff[id]
		jj = idx_mon[id]
		o = length(jj)+1
		Ainf[o] = vcat(Ainf[o], [ii repeat(jj', length(ii), 1) out.value[id,ii]]) # last col is inferred coeffs
	end

    return Ainf, relerr, out, diagnostics
end

function this_bayes_whitened(X::Matrix{Float64}, Y::Matrix{Float64}, Σ, ooi::Vector{<:Integer}, dmax::Int64; verbosity=2)
    T, n = size(X)

    if size(X) != size(Y)
        @info "Dimensions of states and derivatives do not match."
        return nothing
    end

    # retrieve values of monomials at each time step
    θ, d = get_θd(X, dmax)

    idx_mon = Dict{Int64, Vector{Int64}}() # (monomial index) => (nodes involved in monomial)
    for i in 1:size(d)[1]
		mon = d[i,:][d[i,:] .!= 0]
		if length(mon) == length(union(mon)) # skips monomials with repeated factors, e.g., xᵢxⱼ²
			idx_mon[i] = sort(mon)
		end
	end

    # whiten
    F = cholesky(Σ)
    θw = F.L \ θ
    Yw = F.L \ Y
    
    # run sparse Bayes
    opts = SBOpts(verbosity=verbosity, nitr=1000, free_basis=[1], fixed_noise=true)
    settings = SBSettings(beta=1.0)
    out, diagnostics = sparse_bayes(θw, Yw; opts=opts, settings=settings)

    # calculate relative error
    err = norm(Y - θ*out.value, 1)
    relerr = err/norm(Y, 1)

    # reconstruct adjacency tensors
    Ainf = Dict{Int64,Matrix{Float64}}(o => zeros(0,o+1) for o in vcat(1, ooi))

    idx_coeff = Dict{Int64,Vector{Int64}}() # (monomial index) => (nodes for which monomial coeff is nonzero)
    for id in keys(idx_mon)
		aaa = setdiff((1:n)[abs.(out.value[id, :]) .> 1e-8],idx_mon[id]) # ensures monomial involving xᵢ does not get inferred for xᵢ
		if !isempty(aaa)
			idx_coeff[id] = aaa
		end
	end

	for id in keys(idx_coeff)
		ii = idx_coeff[id]
		jj = idx_mon[id]
		o = length(jj)+1
		Ainf[o] = vcat(Ainf[o], [ii repeat(jj', length(ii), 1) out.value[id,ii]]) # last col is inferred coeffs
	end

    return Ainf, relerr, out, diagnostics
end