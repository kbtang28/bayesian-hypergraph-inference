using Distributed

num_procs = parse(Int, ENV["SLURM_NTASKS"])
addprocs(max(num_procs-1, 0), topology=:master_worker)

# activate environment
@everywhere begin
    import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
end

# load dependencies...
@everywhere include(joinpath(@__DIR__, "..", "src", "BayesTHIS.jl"))
@everywhere begin
    using .BayesTHIS
    using Random, Printf, SparseBayes, LinearAlgebra, Dates, DelimitedFiles, Distributions, ProgressMeter
end

Random.seed!(1234)

# experiment & model parameters (master)
n = 10
A2, A3, A2l, A3l = gnm_random_hyperg(n, 0.35, 0.05)
p = (A2, A3, zeros(n), π/4, π/4)

levels = [cdf(Normal(), σ) - cdf(Normal(), -σ) for σ in [1.0, 2.0, 3.0]]
ρs = 10 .^(range(-2, 0, 11))
σs = [0.05, 0.2]
nitr = 100
K = 1000 # number of replicates for each iteration

# state data fixed throughout experiment
X = (rand(300, n) .- 0.5)

# broadcast globals to all workers
@everywhere begin
    const n_const      = $n
    const p_const      = $p
    const levels_const = $levels
    const A2l_const    = $A2l
    const A3l_const    = $A3l
end

# worker-side utility functions
@everywhere begin
    # partitions D into pairwise and triadic columns
    function partition_D(D, d, i)
        tri_cols = [(length(unique(r)) == 2) & (!in(r)(0)) & (!in(r)(i)) for r in eachrow(d)]
        # tri_cols = [(!in(r)(0)) for r in eachrow(d)]

        D1 = D[:, .!tri_cols]
        D2 = D[:, tri_cols]

        return D1, D2
    end

    # projects X onto the colspace of D2 after orthogonalizing wrt D1
    function project_colD2orth(D1, D2, X)
        
        # orthogonal basis for col(D1)
        QD1 = Matrix(qr(D1).Q)

        # remove D1-component from D2
        D2orth = D2 - QD1 * (QD1' * D2) # (I - P₁)D₂

        # orthogonal basis for col(D2orth)
        QD2orth = Matrix(qr(D2orth).Q)

        # project X onto col(D2orth)
        return QD2orth * (QD2orth' * X)
    end

    function ppc(X_og, ρ, σ, nitr, K; n = n_const, p = p_const, levels = levels_const, A2l = A2l_const, A3l = A3l_const)
        T, n = size(X_og)

        # state variables and derivatives
        X = ρ*X_og
        clean_Y = f_kuramoto_3rd(X, p...)

        # design matrix
        Θ = get_theta(X, 2); # full design matrix
        pw_Θ = get_theta(X, 1); # pairwise monomials only

        # inference settings for pairwise model
        opts = SBOpts(verbosity=0, nitr=1000, free_basis=[1], fixed_noise=true)
        settings = SBSettings(beta=1/(σ^2))

        # inference settings for full model
        full_opts = SBOpts(verbosity=0, nitr=1000, free_basis=[1])
        ctrls = SBCtrlSettings(beta_update_frequency=3)

        # repeat inference for different noise realizations
        pval = Vector{Float64}(undef, nitr)
        auc  = Vector{Float64}(undef, nitr)
        auc3 = Vector{Float64}(undef, nitr)
        F1   = Array{Float64}(undef, 3, length(levels), nitr)

        for itr in 1:nitr
            # noisy measurements of derivatives
            Y = clean_Y + σ*randn(size(X))

            # fit pairwise model
            _, _, pw_out, _ = this_bayes(X, Y, [2], 1; opts=opts, settings=settings)

            Ξs = sample_joint_posterior(pw_out, pw_Θ, Y; nsamples=K)
            T_obs = zeros(Float64, K)
            T_rep = zeros(Float64, K)
            for (k, Ξ) in enumerate(Ξs)
                err_obs = Y - pw_Θ*Ξ
                
                # replicate data
                # Y_rep = pw_Θ*Ξ + σ*randn(size(X))
                err_rep = σ*randn(size(X)) # err_rep = Y_rep - pw_Θ*Ξ
                
                for i in 1:n
                    Θ1, Θ2 = partition_D(Θ, get_d(n, 2), i) # partition design matrix
                    T_obs[k] += err_obs[:, i]' * project_colD2orth(Θ1, Θ2, err_obs[:, i])
                    T_rep[k] += err_rep[:, i]' * project_colD2orth(Θ1, Θ2, err_rep[:, i])
                end
            end
            
            # compute p-values
            pval[itr] = sum( T_rep .> T_obs ) / K

            # measure quality of inference - full model
            Ainf, coeff, tri_out, _ = this_bayes(X, Y, [2,3], 2; opts=full_opts, ctrls=ctrls)
            
            tpr, fpr = my_ROC(abs.(Ainf[2]), A2l, abs.(Ainf[3]), A3l, n)
            auc[itr] = get_auc(tpr, fpr)

            pre, rec = my_PRC(abs.(Ainf[3]), A3l, n)
            auc3[itr] = get_auc(pre, rec)

            F1[:, :, itr] = F1_filter_by_CI(tri_out, Θ, coeff, A2l, A3l, levels)
        end

        return pval, auc, auc3, F1
    end
end

# ----------- RUN EXPERIMENT -----------
timestamp = Dates.format(now(), "yyyy-mm-dd")

# allocate space for results

pvals = Array{Float64}(undef, nitr, length(ρs), length(σs))
aucs  = Array{Float64}(undef, nitr, length(ρs), length(σs))
auc3s = Array{Float64}(undef, nitr, length(ρs), length(σs))
F1s   = Array{Float64}(undef, 3, length(levels), nitr, length(ρs), length(σs))

for (k, σ) in enumerate(σs)
    results = @showprogress pmap(ρs) do ρ
        ppc(X, ρ, σ, nitr, K)
    end

    for (j, _) in enumerate(ρs)
        pvals[:, j, k]     .= results[j][1]
        aucs[:, j, k]      .= results[j][2]
        auc3s[:, j, k]     .= results[j][3]
        F1s[:, :, :, j, k] .= results[j][4]
    end
end

out_dir = joinpath(@__DIR__, "..", "out", "ppc")
writedlm(joinpath(out_dir, "scatter-pvals-$(timestamp).txt"), pvals)
writedlm(joinpath(out_dir, "scatter-aucs-$(timestamp).txt"), aucs)
writedlm(joinpath(out_dir, "scatter-auc3s-$(timestamp).txt"), auc3s)
writedlm(joinpath(out_dir, "scatter-F1s-$(timestamp).txt"), F1s)

rmprocs(workers())