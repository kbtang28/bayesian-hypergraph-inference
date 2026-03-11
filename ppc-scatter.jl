using Distributed

num_procs = parse(Int, ENV["SLURM_NTASKS"])
addprocs(max(num_procs-1, 0), topology=:master_worker)

# activate environment
@everywhere begin
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

# load dependencies...
@everywhere begin
    using Random, Printf, SparseBayes, LinearAlgebra, Dates, DelimitedFiles, Distributions, ProgressMeter

    include("gen-rand-hyperg.jl")
    include("hyperg-kuramoto.jl")
    include("this-bayes.jl")
    include("sample-posterior.jl")
    include("performance-measures.jl")
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
    function F1_filter_by_CI(out, D, coeff, levels; A2l = A2l_const, A3l = A3l_const)
        _, n = size(coeff)

        F1s = zeros(Float64, 3, length(levels))

        for (i, level) in enumerate(levels)
            # filter out coeffs with 0 in (level)% CI for conditional posterior
            sig = significant_coeff(out, D, level)
            Ainf = get_Ainf(coeff .* sig, [2,3], 2)

            # compute F1 scores
            F1s[1, i] = my_F1(abs.(Ainf[2]), A2l, abs.(Ainf[3]), A3l, n)
            F1s[2, i] = my_F1(abs.(Ainf[2]), A2l, n)
            F1s[3, i] = my_F1(abs.(Ainf[3]), A3l, n)
        end

        return F1s
    end

    function partition_D(D, d, i)
        tri_cols = [(length(unique(r)) == 2) & (!in(r)(0)) & (!in(r)(1)) for r in eachrow(d)]
        # tri_cols = [(!in(r)(0)) for r in eachrow(d)]

        D1 = D[:, .!tri_cols]
        D2 = D[:, tri_cols]

        return D1, D2
    end

    function project_colD2orth(D1, D2, X)
        # projects X onto the colspace of D2 after orthogonalizing wrt D1
        
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

        # inference settings
        opts = SBOpts(verbosity=0, nitr=1000, free_basis=[1], fixed_noise=true)
        settings = SBSettings(beta=1/(σ^2))

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
            pval[itr] = sum( T_rep .>= T_obs ) / K

            # measure quality of inference - full model
            full_opts = SBOpts(verbosity=0, nitr=1000, free_basis=[1])
            ctrls = SBCtrlSettings(beta_update_frequency=3)
            Ainf, coeff, tri_out, _ = this_bayes(X, Y, [2,3], 2; opts=full_opts, ctrls=ctrls)
            
            tpr, fpr = my_ROC(abs.(Ainf[2]), A2l, abs.(Ainf[3]), A3l, n)
            auc[itr] = get_auc(tpr, fpr)

            pre, rec = my_PRC(abs.(Ainf[3]), A3l, n)
            auc3[itr] = get_auc(pre, rec)

            F1[:, :, itr] = F1_filter_by_CI(tri_out, Θ, coeff, levels)
        end

        return pval, auc, auc3, F1
    end
end

# ----------- RUN EXPERIMENT -----------

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

writedlm("out/ppc/ppc-scatter-pvals.txt", pvals)
writedlm("out/ppc/ppc-scatter-aucs.txt", aucs)
writedlm("out/ppc/ppc-scatter-auc3s.txt", auc3s)
writedlm("out/ppc/ppc-scatter-F1s.txt", F1s)

rmprocs(workers())