# based on code from https://github.com/TaylorBasedHypergraphInference/THIS

"""
    this(X, Y, ooi, dmax, λ, ρ, niter; with_scaling) -> (Ainf, coeff, relerr)

Infer higher-order interactions via SINDy with STLS.

- `X`: state time series (T × n); rows = time steps, columns = agents
- `Y`: derivative time series (T × n)
- `ooi`: orders of interaction to reconstruct (e.g. `[2, 3]`)
- `dmax`: maximum monomial degree (typically `maximum(ooi) - 1`)
- `λ`: sparsity threshold — coefficients below this are zeroed out
- `ρ`: Tikhonov regularization strength (ridge penalty)
- `niter`: maximum SINDy iterations
- `with_scaling`: if `true`, normalise library columns before regression

Returns the inferred adjacency dictionary `Ainf` (keyed by interaction order),
the full coefficient matrix `coeff`, and the relative 1-norm error `relerr`.
"""
function this(X::Matrix{Float64}, Y::Matrix{Float64}, 
			  ooi::Vector{Int}, dmax::Int, 
			  λ::Float64=.1, ρ::Float64=1., niter::Int64=10; 
			  with_scaling::Bool=false)
	
	if size(X) != size(Y)
		@info "Dimensions of states and derivatives do not match."
		return nothing
	end

	# build monomial library
	if with_scaling
		θ0 = get_θ(X, dmax)

		# normalize columns of θ0
		col_norms = norm.(eachcol(θ0))
		θ = θ0 ./ col_norms'
	else
		θ = get_θ(X, dmax)
	end

	# run SINDy with STLS
	coeff, err = mySINDy(θ, Y, λ, ρ, niter)

	if with_scaling
		coeff  = coeff ./ col_norms
		err    = norm(Y - θ0*coeff, 1)
	end
	relerr = err/norm(Y,1)

	# reconstruct adjacency tensors
	Ainf = get_Ainf(coeff, ooi, dmax)

	return Ainf, coeff, relerr
end

# ================================================================================
"""
    mySINDy(θ, Y, λ, ρ, niter) -> (Ξ, err)

Sparse identification of nonlinear dynamics (SINDy) with iterative hard thresholding.

- `θ`: library matrix of basis-function values (T × M)
- `Y`: target derivative matrix (T × n)
- `λ`: sparsity threshold
- `ρ`: ridge regularization strength
- `niter`: maximum iterations

Returns the coefficient matrix `Ξ` (M × n) and the 1-norm reconstruction error.

Adapted from [https://github.com/eurika-kaiser/SINDY-MPC](https://github.com/eurika-kaiser/SINDY-MPC/blob/master/utils/sparsifyDynamics.m).
"""
function mySINDy(θ::Matrix{Float64}, Y::Matrix{Float64}, 
				 λ::Float64=.1, ρ::Float64=1., niter::Int64=10)
	T,n = size(Y)
	T,m = size(θ)

	# ridge regression initialization
	Ξ = (θ' * θ + ρ * Id(m)) \ (θ' * Y) # least squares with Tikhonov regularization (ridge regression)
	
	nz = 1_000_000 
	k = 1

	while k < niter && sum(abs.(Ξ) .> 1e-6) != nz
		k += 1
		nz = sum(abs.(Ξ) .> 1e-6)

		# get rid of the small (< λ) components and re-optimizing.
		smallinds = (abs.(Ξ) .< λ)
		Ξ[smallinds] .= 0.

		for i in 1:n
			biginds = .~smallinds[:, i]
			Θi = @view θ[:, biginds]
			Ξ[biginds, i] = @views (Θi' * Θi + ρ*Id(sum(biginds))) \ (Θi' * Y[:, i])
		end
	end

	return Ξ, norm(Y - θ*Ξ, 1)
end