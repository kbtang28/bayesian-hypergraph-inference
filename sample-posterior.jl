using LinearAlgebra, SparseBayes

function sample_posterior(relevant::BitVector, D, t, α::Vector{Float64}, β::Float64; nsamples::Int=1)
    T, M = size(D)
    m = sum(relevant)
    @assert length(α) == m "α must be vector of length $(m), got $(length(α))"

    # restrict design matrix to free coordinates
    Df = D[:, relevant]

    # build precision and factorize
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

function sample_posterior_whitened(relevant::BitVector, D, t, α::Vector{Float64}, β::Float64; nsamples::Int=1)
    T, M = size(D)
    m = sum(relevant)
    @assert length(α) == m "α must be vector of length $(m), got $(length(α))"

    # restrict design matrix to free coordinates and white
    Df = D[:, relevant]
    A_sqrtinv = 1.0 ./ sqrt.(α)
    Dtilde = Df .* A_sqrtinv' # scales cols

    # build precision and factorize
    Λ = β * (Dtilde' * Dtilde) + I
    L = cholesky(Symmetric(Λ))

    # compute mean
    rhs = β * (Dtilde' * t)
    μf = A_sqrtinv .* (L \ rhs)

    # sample noise
    z = randn(m, nsamples)
    ε = A_sqrtinv .* (L.U \ z)

    # assemble
    ξ = zeros(Float64, M, nsamples)
    ξ[relevant, :] .= (μf .+ ε)
    ξ[.!relevant, :] .= 0.0

    return ξ
end

function sample_joint_posterior_whitened(out::SBOut, D, Y; nsamples::Int=1)
    T, n = size(Y)
    _, M = size(D)

    Ξ = nsamples > 1 ? zeros(Float64, M, nsamples, n) : zeros(Float64, M, n)

    for i in 1:n
        relevant = (out.value[:, i] .!= 0.0)
        β = out.beta[i]
        α = out.alpha[i]
        
        if nsamples > 1
            Ξ[:, :, i] = sample_posterior_whitened(relevant, D, Y[:, i], α, β; nsamples=nsamples)
        else
            Ξ[:, i] = sample_posterior_whitened(relevant, D, Y[:, i], α, β)
        end
    end

    return nsamples > 1 ? [Array(x) for x in eachslice(Ξ, dims=2)] : Ξ
end