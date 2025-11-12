using LinearAlgebra, SparseBayes, Distributions

function sample_posterior(relevant::BitVector, D, t, α::Vector{Float64}, β::Float64; nsamples::Int=1)
    T, M = size(D)
    m = sum(relevant)
    @assert length(α) == m "α must be vector of length $(m), got $(length(α))"

    # restrict design matrix to free coordinates
    Df = D[:, relevant]

    # build precision and factor
    Λ = β * (Df' * Df) + diagm(α)
    F = cholesky(Symmetric(Λ))

    # compute mean
    rhs = β * (Df' * t)
    μf = F \ rhs

    # sample noise
    Z = randn(m, nsamples)
    ε = F.U \ Z

    # assemble
    ξ = zeros(Float64, M, nsamples)
    ξ[relevant, :] .= (μf .+ ε)
    ξ[.!relevant, :] .= 0.0

    return ξ
end

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

# function significant_coeffs(μ::Vector{Float64}, α::Vector{Float64}, β::Float64, D, level::Float64)
#     _, M = size(D)
    
#     relevant = (μ .!= 0.0)
#     m = sum(relevant)
#     @assert length(α) == m "α must be vector of length $(m), got $(length(α))"

#     # restrict design matrix to free coordinates
#     Df = D[:, relevant]

#     # build precision and factor
#     Λ = β * (Df' * Df) + diagm(α)
#     F = cholesky(Symmetric(Λ))

#     # compute marginal variances
#     vars = zeros(Float64, m)

#     for j in 1:m
#         ej = zeros(Float64, m); ej[j] = 1.0
#         x = F \ ej
#         vars[j] = x[j]
#     end

#     # compute credible intervals 
#     γ = 1-level
#     sig = Vector{Bool}(undef, M)
#     sig[.!relevant] .= 0
#     sig_relevant = Vector{Bool}(undef, m)
#     for j in 1:m
#         σ = sqrt(vars[j])
#         qlow = quantile(Normal(0,1), γ/2)
#         qhigh = quantile(Normal(0,1), 1-γ/2)
#         lower = μ[relevant][j] + σ*qlow
#         upper = μ[relevant][j] + σ*qhigh

#         sig_relevant[j] = !(lower <= 0 <= upper)
#     end
#     sig[relevant] = sig_relevant

#     return sig
# end

function _zero_in_conditional_CI(μ, Λii, level::Float64)
    dist = Normal(μ, sqrt(1/Λii))
    γ = (1 - level)/2
    lower, upper = quantile.(dist, [γ, 1-γ])

    return lower ≤ 0 ≤ upper
end

function significant_coeff(μ::Vector{Float64}, α::Vector{Float64}, β::Float64, D, level::Float64)
    _, M = size(D)
    
    relevant = (μ .!= 0.0)
    m = sum(relevant)
    @assert length(α) == m "α must be vector of length $(m), got $(length(α))"

    # restrict design matrix, mean to free coordinates
    Df = D[:, relevant]
    μf = μ[relevant]

    # build precision
    Λ = β * (Df' * Df) + diagm(α)
    Λ_diag = diag(Λ)

    # check whether 0 in CI for conditional posterior
    sig = Vector{Bool}(undef, M)
    sig[.!relevant] .= 0
    sig_relevant = Vector{Bool}(undef, m)
    for j in 1:m
        sig_relevant[j] = !_zero_in_conditional_CI(μf[j], Λ_diag[j], level)
    end
    sig[relevant] = sig_relevant

    return sig
end

function significant_coeff(out::SBOut, D, level::Float64)
    M, n = size(out.value)

    sig = Matrix{Bool}(undef, M, n)

    for i in 1:n
        sig[:, i] = significant_coeff(out.value[:, i], out.alpha[i], out.beta[i], D, level)
    end

    return sig
end