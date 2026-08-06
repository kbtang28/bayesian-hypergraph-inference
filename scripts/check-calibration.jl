include("../src/BayesTHIS.jl")
using .BayesTHIS
using LinearAlgebra, Statistics, StatsBase, DataFrames, Random, Combinatorics, Dates, Parquet2

Random.seed!(20260802)

# experiment & model parameters (fixed)
const dmax = 2; const ooi = [2,3]
const nrep = 200
const n = 10
const L_grid = [round.(10 .^(range(-2, log10(3), 11)), digits=2); 1.0]
const sampling_grid  = [[(L, 300, 0.1) for L in L_grid]; (1.0, 100, 0.5); (1.0, 600, 0.05)]

"""
Data-generating process for a replicate.
 
- `KURAMOTO`: ẋ = F(x) + ε with F the full triadic Kuramoto vector field.
- `WELLSPEC`: ẋ = Θ(X) ξ* + ε with ξ* the exact second-order Taylor coefficient
  matrix of the same hypergraph.
"""
@enum Tier KURAMOTO WELLSPEC

"""
Role of a library column in the inference problem for node `i`, used to
stratify coverage and retention rates.
"""
@enum ColClass GENUINE_PAIRWISE CONTAMINATION COMBO GENUINE_TRIADIC NUISANCE TRUE_ZERO


# utility functions
"""
    monomial_sets(n, dmax) -> Vector{Vector{Int}}

Row m of `get_d` lists the agents in monomial m; zeros are padding. Returns the
agent multiset per column, sorted.
"""
function monomial_sets(n, dmax)
    d = get_d(n, dmax)
    return [sort(filter(!iszero, d[m, :])) for m in axes(d, 1)]
end

"""
    taylor_coeffs_kuramoto_3rd(A2, A3, P; φ2=0., φ3=0.) -> (c0, J, H)

Exact Taylor coefficients of triadic Kuramoto vector field at the origin:
- `c0[i]     = F_i(0)`
- `J[i,l]    = ∂_l F_i(0)`
- `H[i,l,m]  = ∂_{l,m} F_i(0)`
so that `F_i(θ) ≈ c0[i] + Σ_l J[i,l] θ_l + 0.5 * Σ_{l,m} H[i,l,m] θ_l θ_m`.
"""
function taylor_coeffs_kuramoto_3rd(A2::Array{<:Real,2}, A3::Array{<:Real,3}, P::Vector{<:Real}; φ2::Real=0., φ3::Real=0.)
	# ---------------------------------------------------------------------------------
	# Cross-check using automatic differentiation:
	#   using ForwardDiff
	#   f(θ) = f_kuramoto_3rd(θ, A2, A3, P, φ2, φ3)
	#   maximum(abs, ForwardDiff.jacobian(f, zeros(n)) .- J)
	#   maximum(abs, cat([ForwardDiff.hessian(θ -> f(θ)[i], zeros(n)) for i in 1:n]...,
	#                    dims=3) .- permutedims(H, (2, 3, 1)))
	# ---------------------------------------------------------------------------------
	
    n = length(P)
    @assert size(A2) == (n, n) && size(A3) == (n, n, n)

    c2, s2 = cos(φ2), sin(φ2)
    c3, s3 = cos(φ3), sin(φ3)

    k2 = vec(sum(A2, dims=2))               # k2[i]   = Σ_j  a2_ij
    T3 = dropdims(sum(A3, dims=3), dims=3)  # T3[i,l] = Σ_k  a3_ilk
    S3 = vec(sum(T3, dims=2))               # S3[i]   = Σ_{j,k} a3_ijk

    c0 = float.(P)
    J  = zeros(n, n)
    H  = zeros(n, n, n)

    for i in 1:n
        # first order
        for l in 1:n
            l == i && continue
            J[i, l] = c2 * A2[i, l] + 2c3 * T3[i, l]
        end
        J[i, i] = -c2 * k2[i] - 2c3 * S3[i]

        # second order
        for l in 1:n, m in 1:n
            (l == i || m == i) && continue
            H[i, l, m] = l == m ? -s2 * A2[i, l] - 2s3 * T3[i, l] :
                                  -2s3 * A3[i, l, m]
        end
        for l in 1:n
            l == i && continue
            H[i, i, l] = H[i, l, i] = s2 * A2[i, l] + 4s3 * T3[i, l]
        end
        H[i, i, i] = -s2 * k2[i] - 4s3 * S3[i]
    end

    return c0, J, H
end

"""
    taylor_coeff_matrix(A2, A3, P; φ2=0., φ3=0., dmax=2) -> Ξ

Ground-truth (M x n) coefficient matrix `Ξ` in the monomial ordering produced
by `get_θd(X, dmax)` from this-tools.jl, so that `ẋ ≈ get_θ(X, dmax) * Ξ`.

Monomials of degree > 2 receive zero.
"""
function taylor_coeff_matrix(A2, A3, P::Vector{<:Real}; φ2::Real=0., φ3::Real=0., dmax::Int=2)
    n = length(P)
    c0, J, H = taylor_coeffs_kuramoto_3rd(A2, A3, P; φ2, φ3)

    d = get_d(n, dmax)
    Ξ = zeros(size(d, 1), n)

    for r in axes(d, 1)
        mon = filter(!iszero, d[r, :])
        if isempty(mon)          # constant
            Ξ[r, :] = c0
        elseif length(mon) == 1  # x_l
            l = mon[1]
            Ξ[r, :] = J[:, l]
        elseif length(mon) == 2  # x_l x_m  (0.5*H on squares)
            l, m = mon
            Ξ[r, :] = l == m ? 0.5 * H[:, l, l] : H[:, l, m]
        end
    end

    return Ξ
end


"""
    proj_coeff_matrix(A2, A3, P, L; φ2=0., φ3=0., Nmc=10^5) -> Matrix (M × n)

Least-squares projection of Kuramoto RHS onto span(Θ) under X ~ Uniform([-L/2, L/2]^n).
Monte Carlo approximation.
"""
function proj_coeff_matrix(A2, A3, P::Vector{<:Real}, L::Real; φ2::Real=0., φ3::Real=0., Nmc=10^5)
    n  = size(A2, 1)
    X  = (rand(Nmc, n) .- 0.5) .* L
    Θ  = get_θ(X, dmax)
    Y = f_kuramoto_3rd(X, A2, A3, P, φ2, φ3)
    return qr(Θ, ColumnNorm()) \ Y # M × n
end

"""
    classify(i, mon, A2, A3) -> Vector{ColClass}

Classifies each library monomial for single node i.
`mon` is M-vector output of `monomial_sets`.
"""
function classify(i, mon, A2, A3)
    n = size(A2, 1)
    map(mon) do S
        if isempty(S) || i in S
            NUISANCE # constant, xᵢ, xᵢ², xᵢxⱼ
        elseif length(S) == 1
            j = S[1]
            has3 = any(A3[i, j, k] != 0 for k in 1:n)
            if A2[i,j] != 0
                has3 ? COMBO : GENUINE_PAIRWISE
            else
                has3 ? CONTAMINATION : TRUE_ZERO
            end
        elseif S[1] == S[2]
            NUISANCE  # xⱼ² — no hyperedge meaning
        else
            A3[i, S[1], S[2]] != 0 ? GENUINE_TRIADIC : TRUE_ZERO
        end
    end
end

"""
    marginal_sds(μ::Vector{Float64}, α::Vector{Float64}, β::Float64, D)

Marginal coefficient SD for all terms in active set (μ_m != 0.0), 0 else.
"""
function marginal_sds(μ::Vector{Float64}, α::Vector{Float64}, β::Float64, D)
    _, M = size(D)
    
    relevant = (μ .!= 0.0)
    m = sum(relevant)
    @assert length(α) == m "α must be vector of length $(m), got $(length(α))"

    # restrict design matrix to free coordinates
    Df = D[:, relevant]

    # build precision and factor
    Λ = β * (Df' * Df) + diagm(α)
    F = cholesky(Symmetric(Λ))

    # compute marginal variances
    vars = zeros(Float64, M)

    tmp = zeros(Float64, m)
    for j in 1:m
        ej = zeros(Float64, m); ej[j] = 1.0
        x = F \ ej
        tmp[j] = x[j]
    end
    vars[relevant] .= tmp

    return sqrt.(vars)
end

"""
    conditional_sds(μ::Vector{Float64}, α::Vector{Float64}, β::Float64, D) -> Dict{Int,Float64}

Conditional coefficient SD for all terms in active set (μ_m != 0.0), 0 else.
"""
function conditional_sds(μ::Vector{Float64}, α::Vector{Float64}, β::Float64, D)
    _, M = size(D)
    
    relevant = (μ .!= 0.0)
    m        = sum(relevant)
    @assert length(α) == m "α must have length $(m), got $(length(α))"

    # restrict design matrix to free coordinates
    Df = D[:, relevant]

    # build precision
    Λ      = β * (Df' * Df) + diagm(α)
    Λ_diag = diag(Λ)

    # compute conditional variances
    tmp = 1 ./ Λ_diag
    vars = zeros(Float64, M)
    vars[relevant] .= tmp

    return sqrt.(vars)
end

# fits one dataset and appends per-column rows
function fit_and_push!(rows, rep, tier, L, T, σ, X, Θ, Y, taylor_coeffs, proj_coeffs, classes)
    opts = SBOpts(verbosity=0, nitr=1000, free_basis=[1])
    ctrls = SBCtrlSettings(beta_update_frequency=3)
    _, coeffs, out, _ = this_bayes(X, Y, ooi, dmax; opts=opts, ctrls=ctrls)

    for i in 1:n
        σ_marg = marginal_sds(coeffs[:, i], out.alpha[i], out.beta[i], Θ)
        σ_cond = conditional_sds(coeffs[:, i], out.alpha[i], out.beta[i], Θ)
        
        active = (coeffs[:, i] .!= 0.0)
        for (m, a) in enumerate(active)
            # m is library term index (1..M), a is T/F
            push!(rows, (
                rep = rep,
                tier = tier,
                L = L,
                noise = σ,
                T = T,
                node = i,
                col = m,
                class = classes[i][m],
                coeff_taylor = taylor_coeffs[m, i],
                coeff_proj = proj_coeffs[m, i],
                mu = coeffs[m, i],
                active = a,
                sd_marg = σ_marg[m],
                sd_cond = σ_cond[m],
                β = out.beta[i]
            ))
        end
    end
end

# runs a replicate
function run_replicate!(rows, rep, L, T, σ, A2, A3, taylor_coeffs, proj_coeffs, classes)
    X = (rand(T, n) .- 0.5) .* L
    Θ = get_θ(X, dmax)
    E = σ .* randn(size(X))

    Y_kuramoto = f_kuramoto_3rd(X, A2, A3, zeros(n), pi/4, pi/4) .+ E
    Y_wellspec = Θ * taylor_coeffs .+ E

    # under WELLSPEC the projection target is exactly ξ*, so pass taylor_coeffs
    fit_and_push!(rows, rep, KURAMOTO, L, T, σ, X, Θ, Y_kuramoto, taylor_coeffs, proj_coeffs, classes)
    fit_and_push!(rows, rep, WELLSPEC, L, T, σ, X, Θ, Y_wellspec, taylor_coeffs, taylor_coeffs, classes)
end

# ----------- RUN EXPERIMENT -----------
timestamp = Dates.format(now(), "yyyy-mm-dd")

A2, A3, A2l, A3l = gnm_random_hyperg(n, 0.35, 0.05)

taylor_coeffs = taylor_coeff_matrix(A2, A3, zeros(n), φ2=pi/4, φ3=pi/4)

mon = monomial_sets(n, dmax)
classes = [classify(i, mon, A2, A3) for i in 1:n]

# cache projected targets (per sampling box size L)
L_dict = Dict(L => proj_coeff_matrix(A2, A3, zeros(n), L; φ2=pi/4, φ3=pi/4) for L in L_grid)

rows = NamedTuple[]
for (L, T, σ) in sampling_grid
    println("done with ($(L), $(T), $(σ))")
    proj_coeffs = L_dict[L]
    for r in 1:nrep
        run_replicate!(rows, r, L, T, σ, A2, A3, taylor_coeffs, proj_coeffs, classes)
    end
end

df = DataFrame(rows)

out = copy(df)
for col in names(out)
    if eltype(out[!, col]) <: Enum
        out[!, col] = string.(out[!, col])
    end
end

Parquet2.writefile(joinpath(@__DIR__, "..", "out", "calibration", "out_$(timestamp).parq"), out)