import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# load dependencies...
using Random, Printf, SparseBayes, LinearAlgebra, Dates, DelimitedFiles, ProgressMeter
import Statistics: mean, std
import StatsBase: sample, corspearman

include("gen-rand-hyperg.jl")
include("hyperg-kuramoto.jl")
include("this-og.jl")
include("this-bayes.jl")
include("performance-measures.jl")

Random.seed!(123)

# experiment and model parameters
n = 10
A2, A3, A2l, A3l = gnm_random_hyperg(n, 0.35, 0.05)
p = (A2, A3, zeros(n), π/4, π/4)

num_edges = size(A2l,1) + size(A3l,1)
K = round(Int64, 0.75*num_edges)

N = 500
σ = 0.2

d_array = 2:2:8 # dimension of underlying subspace
δ_array = [0.01, 0.02, 0.05, 0.1, 0.25] # sd of noise

n_itr=100
ooi = [2,3]
dmax = 2
λs = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0] # STLS sparsity parameter

function generate_low_dim_X(N, n, d, δ)
    # orthonormal basis (n x d)
    Q, _ = qr(randn(n, d))
    U = Matrix(Q)

    # sample d-dimensional hypercube
    Z = rand(N, d) .- 0.5

    # generate X
    X = Z*U' .+ δ*randn(N, n)

    # scale to lie in hypercube of sidelength 1.0
    X = 0.5 * (X ./ maximum(abs.(X)))

    # compute library conditioning number
    T = get_theta(X, 2)
    norms = norm.(eachcol(T))
    T = T ./ norms'

    s = svdvals(T)
    kappa = maximum(s) / minimum(s)

    return X, kappa
end

function jaccard_similarity(S1, S2)
    a = length(intersect(S1, S2))
    b = length(union(S1, S2))

    return a / b
end

function avg_jaccard_similarity(Ainfs::Vector{Dict{Int64, Matrix{Float64}}}, K, n_itr)
    # get top K terms by magnitude for each fit
    Ainfs_topK = Vector{Vector{Vector{Float64}}}[]
    for Ainf in Ainfs
        nz = size(Ainf[2], 1) + size(Ainf[3], 1)
        a = sortslices([abs.([Ainf[2][:, 3]; Ainf[3][:, 4]]) 1:nz], dims=1, rev=true)
        ids = Int64.(a[:, 2])

        E2 = [Ainf[2][i, 1:2] for i in 1:size(Ainf[2], 1)]
        E3 = [Ainf[3][i, 1:3] for i in 1:size(Ainf[3], 1)]
        K = minimum([K, nz])
        E = [E2; E3][ids[1:K]]

        Ainfs_topK = [Ainfs_topK; [E]]
    end
    
    sum = 0.0
    for i in 1:n_itr
        for j in (i+1):n_itr
            sum += jaccard_similarity(Ainfs_topK[i], Ainfs_topK[j])
        end
    end

    return 2*sum / (n_itr*(n_itr-1))
end

function _get_aurocs(Ainf, n, A2l, A3l)
    aurocs = zeros(Float64, 3)

    tpr, fpr = my_ROC(abs.(Ainf[2]), A2l, abs.(Ainf[3]), A3l, n; verbosity=0)
    aurocs[1] = get_auc(tpr, fpr)

    tpr2, fpr2 = my_ROC(abs.(Ainf[2]), A2l, n)
    aurocs[2] = get_auc(tpr2, fpr2)

    tpr3, fpr3 = my_ROC(abs.(Ainf[3]), A3l, n)
    aurocs[3] = get_auc(tpr3, fpr3)

    return aurocs
end

function _get_auprcs(Ainf, n, A2l, A3l)
    auprcs = zeros(Float64, 3)

    prec, rec = my_PRC(abs.(Ainf[2]), A2l, abs.(Ainf[3]), A3l, n; verbosity=0)
    auprcs[1] = get_auc(prec, rec, rule="T")

    prec2, rec2 = my_PRC(abs.(Ainf[2]), A2l, n)
    auprcs[2] = get_auc(prec2, rec2, rule="T")

    prec3, rec3 = my_PRC(abs.(Ainf[3]), A3l, n)
    auprcs[3] = get_auc(prec3, rec3, rule="T")

    return auprcs
end

function test_inference(d, δ, n_itr; N = N, n = n, λs = λs, p = p, σ = σ, A2l = A2l, A3l = A3l, K = K)
    # generate dataset sampling near d-dimensional manifold
    X, kappa = generate_low_dim_X(N, n, d, δ)

    # repeat inference
    bayes_aurocs = zeros(Float64, 3, n_itr)
    this_aurocs  = zeros(Float64, 3, length(λs), n_itr)
    bayes_auprcs = zeros(Float64, 3, n_itr)
    this_auprcs = zeros(Float64, 3, length(λs), n_itr)

    bayes_Ainfs = Dict{Int64, Matrix{Float64}}[]
    this_Ainfs = Vector{Dict{Int64, Matrix{Float64}}}[]

    bayes_coeffs = Matrix{Float64}[]
    this_coeffs = Vector{Matrix{Float64}}[]

    for itr in 1:n_itr
        # sample new noise realization
        Y = f_kuramoto_3rd(X, p...) .+ σ*randn(size(X))

        # inference with Bayes-THIS & measure performance
        opts = SBOpts(verbosity=0, nitr=1000, free_basis=[1])
        ctrls = SBCtrlSettings(beta_update_frequency=3)
        bayes_Ainf, bayes_coeff, _, _ = this_bayes(X, Y, ooi, dmax; opts=opts, ctrls=ctrls)
        bayes_Ainfs = [bayes_Ainfs; bayes_Ainf] # record
        bayes_coeffs = [bayes_coeffs; [bayes_coeff]] # record

        bayes_aurocs[:, itr] = _get_aurocs(bayes_Ainf, n, A2l, A3l)
        bayes_auprcs[:, itr] = _get_auprcs(bayes_Ainf, n, A2l, A3l)


        # inference with THIS & measure performance (looping over λs...)
        tmp_Ainfs = Dict{Int64, Matrix{Float64}}[]
        tmp_coeffs = Matrix{Float64}[]
        for (j, λ) in enumerate(λs)
            this_Ainf, this_coeff, _ = this(X, Y, ooi, dmax, λ, 1e-4, 500, with_scaling=true)
            tmp_Ainfs = [tmp_Ainfs; this_Ainf] # record
            tmp_coeffs = [tmp_coeffs; [this_coeff]] # record

            this_aurocs[:, j, itr] = _get_aurocs(this_Ainf, n, A2l, A3l)
            this_auprcs[:, j, itr] = _get_auprcs(this_Ainf, n, A2l, A3l)
        end
        this_Ainfs = [this_Ainfs; [tmp_Ainfs]] # record
        this_coeffs = [this_coeffs; [tmp_coeffs]] # record
    end

    bayes_jaccard = avg_jaccard_similarity(bayes_Ainfs, K, n_itr)
    this_jaccard = zeros(Float64, length(λs))
    for (i, _) in enumerate(λs)
        this_jaccard[i] = avg_jaccard_similarity([this_Ainfs[itr][i] for itr in 1:n_itr], K, n_itr)
    end

    return kappa, bayes_aurocs, this_aurocs, bayes_auprcs, this_auprcs, bayes_jaccard, this_jaccard
end

# ----------- RUN EXPERIMENT -----------
timestamp = Dates.format(now(), "yyyy-mm-dd")

# allocate and package results
kappas = zeros(Float64, length(d_array), length(δ_array))
bayes_aurocs = zeros(Float64, 3, n_itr, length(d_array), length(δ_array))
this_aurocs  = zeros(Float64, 3, length(λs), n_itr, length(d_array), length(δ_array))
bayes_auprcs = zeros(Float64, 3, n_itr, length(d_array), length(δ_array))
this_auprcs  = zeros(Float64, 3, length(λs), n_itr, length(d_array), length(δ_array))
bayes_jaccard = zeros(Float64, length(d_array), length(δ_array))
this_jaccard = zeros(Float64, length(λs), length(d_array), length(δ_array))
bayes_coeff_cor = zeros(Float64, length(d_array), length(δ_array))
this_coeff_cor = zeros(Float64, length(λs), length(d_array), length(δ_array))

for (i, d) in enumerate(d_array)
    for (j, δ) in enumerate(δ_array)
        println("d = $(d), δ = $(δ)")

        results = test_inference(d, δ, n_itr)

        kappas[i, j]               = results[1]
        bayes_aurocs[:, :, i, j]   = results[2]
        this_aurocs[:, :, :, i, j] = results[3]
        bayes_auprcs[:, :, i, j]   = results[4]
        this_auprcs[:, :, :, i, j] = results[5]
        bayes_jaccard[i, j]        = results[6]
        this_jaccard[:, i, j]      = results[7]
    end
end

out_dir = "out/collinearity/"
writedlm(joinpath(out_dir, "kappas-$(timestamp).txt"), kappas)
writedlm(joinpath(out_dir, "bayes-aurocs-$(timestamp).txt"), bayes_aurocs)
writedlm(joinpath(out_dir, "this-aurocs-$(timestamp).txt"), this_aurocs)
writedlm(joinpath(out_dir, "bayes-auprcs-$(timestamp).txt"), bayes_auprcs)
writedlm(joinpath(out_dir, "this-auprcs-$(timestamp).txt"), this_auprcs)
writedlm(joinpath(out_dir, "bayes-jaccard-$(timestamp).txt"), bayes_jaccard)
writedlm(joinpath(out_dir, "this-jaccard-$(timestamp).txt"), this_jaccard)