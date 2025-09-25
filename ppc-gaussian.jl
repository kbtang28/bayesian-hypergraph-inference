using Random, Printf, SparseBayes, LinearAlgebra, Dates, DelimitedFiles

include("gen-rand-hyperg.jl")
include("hyperg-kuramoto.jl")
include("this-bayes.jl")
include("sample-posterior.jl")
include("performance-measures.jl")

Random.seed!(1234)

# helper function
function ppc(X_og, ρ, σ; K=1000, nitr_noise=10)
    @printf(SB_logfile, "Testing ρ = %.2f, σ = %.2f...\n", ρ, σ)
    @printf(ROC_logfile, "Testing ρ = %.2f, σ = %.2f...\n", ρ, σ)    

    # state variables and derivatives
    X = ρ*X_og
    clean_Y = f_kuramoto_3rd(X, p...)
    
    # inference settings
    opts = SBOpts(verbosity=2, nitr=500, free_basis=[1], fixed_noise=true, io_list=[SB_logfile])
    settings = SBSettings(beta=1/(σ^2))

    pval = Vector{Float64}(undef, nitr_noise)
    auc = Vector{Float64}(undef, nitr_noise)
    auc3 = Vector{Float64}(undef, nitr_noise)

    for itr in 1:nitr_noise
        println.([SB_logfile, ROC_logfile], "Starting iteration $(itr) / $(nitr_noise)...")

        # noisy measurements of derivatives
        Y = clean_Y + σ*randn(size(X))

        # fit pairwise model
        _, _, out, diagnostics = this_bayes(X, Y, [2], 1; opts=opts, settings=settings)

        # replicate data and compute discrepancies
        Θ = get_theta(X, 1)
        Ξs = sample_joint_posterior(out, Θ, Y; nsamples=K)

        D_obs = [sum( norm.(eachrow( (Y - Θ*Ξ)/σ )).^2 ) for Ξ in Ξs]
        D_rep = [sum( norm.( eachrow( randn(size(Y)) )).^2 ) for _ in Ξs]
        pval[itr] = sum(D_obs .<= D_rep)/K

        # measure quality of inference - pairwise + triadic model
        Ainf, _, _, _ = this_bayes(X, Y, [2,3], 2; opts=opts, settings=settings)
        @printf(ROC_logfile, "Bayes-THIS --------------- (full) ")
        tpr, fpr = my_ROC(abs.(Ainf[2]), A2l, abs.(Ainf[3]), A3l, n; verbosity=1, io=ROC_logfile)
        auc[itr] = get_auc(tpr, fpr)

        @printf(ROC_logfile, "Bayes-THIS ---------------- (tri) ")
        tpr3, fpr3 = my_ROC(abs.(Ainf[3]), A3l, n; verbosity=1, io=ROC_logfile)
        auc3[itr] = get_auc(tpr3, fpr3)
    end

    flush(SB_logfile)
    flush(ROC_logfile)

    return mean(pval), mean(auc), mean(auc3)
end

# hypergraph model
n = 7
A2, A3, A2l, A3l = gnm_random_hyperg(n, 0.4, 0.1)
p = (A2, A3, zeros(n), π/4, π/4)

# experimental settings
ρs = 10 .^(range(-2, 0, 11))
σs = [0.1, 0.8]

# ----------- RUN EXPERIMENT -----------

# set up log files
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMM")

global SB_logfile = open("out/logs/ppcs-$(timestamp)-SB.txt", "w")
global ROC_logfile = open("out/logs/ppcs-$(timestamp)-ROC.txt", "w")

# generate state data (geometry fixed throughout experiments)
nitr = 10
Xs = [(rand(200, n) .- 0.5) for _ in 1:nitr]

pvals = zeros(Float64, nitr, length(ρs), length(σs))
aucs = zeros(Float64, nitr, length(ρs), length(σs))
auc3s = zeros(Float64, nitr, length(ρs), length(σs))

for (k, σ) in enumerate(σs)
    for (j, ρ) in enumerate(ρs)
        for i in 1:nitr
            pval, auc, auc3 = ppc(Xs[i], ρ, σ)
            pvals[i,j,k] = pval
            aucs[i,j,k] = auc
            auc3s[i,j,k] = auc3
        end
    end
end

close(SB_logfile)
close(ROC_logfile)

writedlm("out/ppc-pvals.txt", pvals)
writedlm("out/ppc-aucs.txt", aucs)
writedlm("out/ppc-auc3s.txt", auc3s)