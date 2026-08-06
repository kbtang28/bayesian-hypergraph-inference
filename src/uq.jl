using LinearAlgebra, SparseBayes, Distributions

"""
    sample_posterior(relevant, D, t, α, β; nsamples) -> Matrix{Float64}

Draw `nsamples` samples from the posterior over coefficients for a single output node.

- `relevant`: Boolean mask selecting the active basis functions
- `D`:        full design matrix (T × M)
- `t`:        target time series (length T)
- `α`:        precisions for the active coefficients (length = sum(relevant))
- `β`:        noise precision

Returns an M × nsamples matrix (zero rows for inactive coefficients).
"""
function sample_posterior(relevant::BitVector, D, t, α::Vector{Float64}, β::Float64; 
                          nsamples::Int=1)
    T, M = size(D)
    m    = sum(relevant)
    @assert length(α) == m "α must have length $(m), got $(length(α))"

    # restrict design matrix to free coordinates
    Df = D[:, relevant]

    # build precision and factor
    Λ = β * (Df' * Df) + diagm(α)
    F = cholesky(Symmetric(Λ))

    # compute mean
    rhs = β * (Df' * t)
    μf  = F \ rhs

    # sample noise
    Z = randn(m, nsamples)
    ε = F.U \ Z

    # assemble
    ξ                 = zeros(Float64, M, nsamples)
    ξ[relevant, :]   .= (μf .+ ε)

    return ξ
end

"""
    sample_joint_posterior(out, D, Y; nsamples) -> Matrix or Vector{Matrix}

Sample from the posterior for every output node using a `SBOut` result.

- `out`: SBOut result from sparse Bayesian regression 
- `D`:   full design matrix (T × M)
- `Y`:   target time series (T x n)

Returns an M × n matrix when `nsamples == 1`, or a length `nsamples` vector
of M × n matrices otherwise.
"""
function sample_joint_posterior(out::SBOut, D, Y; nsamples::Int=1)
    T, n = size(Y)
    _, M = size(D)

    Ξ = nsamples > 1 ? zeros(Float64, M, nsamples, n) : zeros(Float64, M, n)

    for i in 1:n
        relevant = (out.value[:, i] .!= 0.0)
        β = out.beta[i]
        α = out.alpha[i]
        
        if nsamples > 1
            Ξ[:, :, i] = sample_posterior(relevant, D, Y[:, i], α, β; nsamples=nsamples)
        else
            Ξ[:, i] = sample_posterior(relevant, D, Y[:, i], α, β)
        end
    end

    return nsamples > 1 ? [Array(x) for x in eachslice(Ξ, dims=2)] : Ξ
end

# check whether 0 lies in the conditional credible interval
function _zero_in_conditional_CI(μ, Λii, level::Float64)
    dist = Normal(μ, sqrt(1 / Λii))
    γ    = (1 - level)/2
    lower, upper = quantile.(dist, [γ, 1-γ])

    return lower ≤ 0 ≤ upper
end

"""
    significant_coeff(μ, α, β, D, level) -> Vector{Bool}

Return a Boolean mask indicating which coefficients are significantly non-zero
at the given credible-interval `level`, based on the conditional posterior.

- `μ`:     posterior mean
- `α`:     precisions for the active coefficients (length ≤ M)
- `β`:     noise precision
- `D`:     full design matrix (T × M)
- `level`: credible interval level
"""
function significant_coeff(μ::Vector{Float64}, α::Vector{Float64}, β::Float64, 
                           D, level::Float64)
    _, M = size(D)
    
    relevant = (μ .!= 0.0)
    m        = sum(relevant)
    @assert length(α) == m "α must have length $(m), got $(length(α))"

    # restrict design matrix, mean to free coordinates
    Df = D[:, relevant]
    μf = μ[relevant]

    # build precision
    Λ      = β * (Df' * Df) + diagm(α)
    Λ_diag = diag(Λ)

    # check whether 0 in CI for conditional posterior
    sig              = Vector{Bool}(undef, M)
    sig[.!relevant] .= false
    sig_relevant     = [!_zero_in_conditional_CI(μf[j], Λ_diag[j], level) for j in 1:m]
    sig[relevant]    = sig_relevant

    return sig
end

"""
    significant_coeff(out, D, level) -> Matrix{Bool}

Apply `significant_coeff` to every output node in a `SBOut` result.
Returns an M × n Boolean matrix.
"""
function significant_coeff(out::SBOut, D, level::Float64)
    M, n = size(out.value)

    sig = Matrix{Bool}(undef, M, n)

    for i in 1:n
        sig[:, i] = significant_coeff(out.value[:, i], out.alpha[i], out.beta[i], D, level)
    end

    return sig
end