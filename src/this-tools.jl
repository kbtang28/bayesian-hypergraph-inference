using LinearAlgebra, Combinatorics, Statistics

# based on code from https://github.com/TaylorBasedHypergraphInference/THIS

"""
    get_θd(X, dmax, i0=1) -> (θ, d)
    get_thetad(X, dmax, i0=1) -> (θ, d)

Return the matrix of monomial values up to degree `dmax` evaluated at each row of `X`,
together with `d`, a matrix encoding which agents appear in each monomial.

- `X`: state time series (T × n); columns = agents, rows = time steps
- `dmax`: maximum monomial degree
- `i0`: starting agent index (used internally for recursion)

`θ` has shape T × M (M = number of monomials); `d` has shape M × dmax,
where non-zero entries list the agents involved (repeated entries indicate
degree > 1).

    get_θ(X, dmax, i0=1) -> θ
    get_theta(X, dmax, i0=1) -> θ

Return only the monomial matrix.

    get_d(n, dmax, i0=1) -> d

Return only the agent-index table for `n` agents.
"""
function get_θd(X::Matrix{Float64}, dmax::Int64, i0::Int64=1)
	T, n = size(X)
	θ = ones(T)
	d = zeros(Int64, 1, dmax)

	if dmax == 0
		return θ, d
	end

	for i in 1:n
		θ0, d0 = get_θd(X[:, i:n], dmax-1, i)
		θ = hcat(θ, θ0 .* repeat(X[:, [i,]], 1, size(θ0, 2)))
		d = vcat(d, [d0 i*ones(Int64, size(d0)[1], 1)])
	end

	d += (i0 - 1) * (d .> 0)

	return θ, d
end

get_thetad(X::Matrix{Float64}, dmax::Int64, i0::Int64=1) = get_θd(X, dmax, i0)

function get_θ(X::Matrix{Float64}, dmax::Int64, i0::Int64=1)
	T, n = size(X)
	θ = ones(T)

	dmax == 0 && return θ

	for i in 1:n
		θ0 = get_θ(X[:, i:n], dmax - 1, i)
		θ = hcat(θ, θ0 .* repeat(X[:, [i,]], 1, size(θ0, 2)))
	end

	return θ
end

function get_theta(X::Matrix{Float64}, dmax::Int64, i0::Int64=1)
	return get_θ(X, dmax, i0)
end

function get_d(n::Int64, dmax::Int64, i0::Int64=1)
	d = zeros(Int64, 1, dmax)

	dmax == 0 && return d
	for i in 1:n
		d0 = get_d(n - i + 1, dmax - 1, i)
		d = vcat(d, [d0 i*ones(Int64, size(d0)[1], 1)])
	end

	d += (i0 - 1) * (d .> 0)

	return d
end

"""
    Id(n) -> Matrix{Float64}

Return the `n × n` identity matrix.
"""
Id(n::Int) = diagm(ones(n))

"""
    get_Ainf(coeff, ooi, dmax) -> Dict{Int, Matrix{Float64}}

Reconstruct the adjacency dictionary from a raw coefficient matrix.

- `coeff`: M × n coefficient matrix (output of sparse regression)
- `ooi`: orders of interaction to include
- `dmax`: maximum monomial degree used to build the library
"""
function get_Ainf(coeff, ooi, dmax)
    _, n = size(coeff)
    d = get_d(n, dmax)

    idx_mon = Dict{Int64, Vector{Int64}}() # (monomial index) => (nodes involved in monomial)
    for i in axes(d, 1)
		mon = filter(!iszero, d[i, :])

		if length(mon) == length(union(mon)) # skips monomials with repeated factors, e.g., xᵢxⱼ²
			idx_mon[i] = sort(mon)
		end
	end

    Ainf      = Dict{Int64,Matrix{Float64}}(o => zeros(0, o+1) for o in vcat(1, ooi))
    idx_coeff = Dict{Int64,Vector{Int64}}() # (monomial index) => (nodes for which monomial coeff is nonzero)
    
	for id in keys(idx_mon)
		aaa = setdiff((1:n)[abs.(coeff[id, :]) .> 1e-8], idx_mon[id]) # ensures monomial involving xᵢ does not get inferred for xᵢ
		if !isempty(aaa)
			idx_coeff[id] = aaa
		end
	end

	for id in keys(idx_coeff)
		ii = idx_coeff[id]
		jj = idx_mon[id]
		o  = length(jj)+1
		Ainf[o] = vcat(Ainf[o], [ii repeat(jj', length(ii), 1) coeff[id, ii]]) # last col is inferred coeffs
	end

    return Ainf
end