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
    using Random, Printf, LinearAlgebra, Dates, DelimitedFiles, ProgressMeter, Statistics
end

Random.seed!(117)

# experiment & model parameters (master)
n = 10; t2 = 0.35; t3 = 0.06
A2, A3, A2l, A3l = gnm_random_hyperg(n, t2, t3)
p = (A2, A3, zeros(n), π/4, π/4)

ooi = [2,3]
dmax = 2

N_array = 60:5:600
σ_array = range(0.05, 1.0, 20)
n_itr = 200
λs = [round.([d * 10. ^ exp for exp in [-3, -2, -1, 0] for d in 1:9]; digits=3); 10.0] # STLS sparsity parameter

# SLURM array index
array_id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
σ = σ_array[array_id]

# broadcast globals to all workers
@everywhere begin
    const n_const    = $n
    const A2l_const  = $A2l
    const A3l_const  = $A3l
    const ooi_const  = $ooi
    const dmax_const = $dmax
    const λs_const   = $λs
    const p_const    = $p
end

# worker-side utility functions
@everywhere begin
    function select_λ_cv(X, Y, ooi, dmax, λs; K = 5, one_se = false)
        N = size(X, 1)
        K = min(K, N)

        perm  = randperm(N)
        folds = [perm[k:K:N] for k in 1:K] # interleaved for balanced-ish sizes

        errs = zeros(Float64, length(λs), K)

        for k in 1:K
            test_idx  = folds[k]
            train_idx = reduce(vcat, folds[[1:k-1; k+1:K]])

            Xtr, Ytr = X[train_idx, :], Y[train_idx, :]
            Xte, Yte = X[test_idx, :],  Y[test_idx, :]

            Θte = get_theta(Xte, dmax)

            for (j, λ) in enumerate(λs)
                _, Ξ, _ = this(Xtr, Ytr, ooi, dmax, λ, 1e-4, 500, with_scaling=true)
                errs[j, k] = sum(abs2, Yte .- (Θte * Ξ)) / length(Yte) # held-out MSE
            end
        end

        mse = vec(mean(errs, dims = 2))

        jsel = if one_se
            se   = vec(std(errs, dims = 2)) ./ sqrt(K)
            jmin = argmin(mse)
            maximum(findall(j -> mse[j] <= mse[jmin] + se[jmin], eachindex(λs)))
        else
            argmin(mse)
        end

        return λs[jsel], mse
    end

    function test_inference(N, σ, n_itr; n = n_const, A2l = A2l_const, A3l = A3l_const, ooi = ooi_const, dmax = dmax_const, λs = λs_const, p = p_const)        
        # repeat inference
        bayes_aurocs = zeros(Float64, 3, n_itr)
        this_aurocs  = zeros(Float64, 3, length(λs), n_itr)
        bayes_auprcs = zeros(Float64, 3, n_itr)
        this_auprcs  = zeros(Float64, 3, length(λs), n_itr)

        cv_aurocs = zeros(Float64, 3, n_itr)
        cv_auprcs = zeros(Float64, 3, n_itr)
        λs_cv     = zeros(Float64, n_itr)

        for itr in 1:n_itr
            X = rand(N, n) .- 0.5
            Y = f_kuramoto_3rd(X, p...) .+ σ*randn(size(X))

            # inference with Bayes-THIS & measure performance
            opts  = SBOpts(verbosity=0, nitr=1000, free_basis=[1])
            ctrls = SBCtrlSettings(beta_update_frequency=3)
            bayes_Ainf, _, _, _ = this_bayes(X, Y, ooi, dmax; opts=opts, ctrls=ctrls)
            
            bayes_aurocs[:, itr] = get_aurocs(bayes_Ainf, n, A2l, A3l)
            bayes_auprcs[:, itr] = get_auprcs(bayes_Ainf, n, A2l, A3l)

            # inference with THIS & measure performance (looping over λs...)
            for (j, λ) in enumerate(λs)
                this_Ainf, _, _ = this(X, Y, ooi, dmax, λ, 1e-4, 500, with_scaling=true)

                this_aurocs[:, j, itr] = get_aurocs(this_Ainf, n, A2l, A3l)
                this_auprcs[:, j, itr] = get_auprcs(this_Ainf, n, A2l, A3l)
            end

            λ_cv, _ = select_λ_cv(X, Y, ooi, dmax, λs; K=5)
            cv_Ainf, _, _ = this(X, Y, ooi, dmax, λ_cv, 1e-4, 500, with_scaling=true)

            cv_aurocs[:, itr] = get_aurocs(cv_Ainf, n, A2l, A3l)
            cv_auprcs[:, itr] = get_auprcs(cv_Ainf, n, A2l, A3l)
            λs_cv[itr]        = λ_cv
        end
        
        return bayes_aurocs, this_aurocs, bayes_auprcs, this_auprcs, cv_aurocs, cv_auprcs, λs_cv
    end
end

# ----------- RUN EXPERIMENT -----------
timestamp = Dates.format(now(), "yyyy-mm-dd")

# set up log files
println("Running array_id=$(array_id) with σ=$(σ), nworkers()=$(nworkers())")
println("Running array_id=$(array_id) with σ=$(σ), nworkers()=$(nworkers())")

results = @showprogress pmap(collect(N_array)) do N
    test_inference(N, σ, n_itr)
end

# allocate and package results
bayes_aurocs   = zeros(Float64, 3, n_itr, length(N_array))
this_aurocs    = zeros(Float64, 3, length(λs), n_itr, length(N_array))
bayes_auprcs   = zeros(Float64, 3, n_itr, length(N_array))
this_auprcs    = zeros(Float64, 3, length(λs), n_itr, length(N_array))
this_cv_aurocs = zeros(Float64, 3, n_itr, length(N_array))
this_cv_auprcs = zeros(Float64, 3, n_itr, length(N_array))
λs_cv          = zeros(Float64, n_itr, length(N_array))

for (i, _) in enumerate(N_array)
    bayes_aurocs[:, :, i]   .= results[i][1]
    this_aurocs[:, :, :, i] .= results[i][2]
    bayes_auprcs[:, :, i]   .= results[i][3]
    this_auprcs[:, :, :, i] .= results[i][4]
    this_cv_aurocs[:, :, i] .= results[i][5]
    this_cv_auprcs[:, :, i] .= results[i][6]
    λs_cv[:, i]             .= results[i][7]
end

out_dir = joinpath(@__DIR__, "..", "out", "robustness")
if !isdir(out_dir)
    mkpath(out_dir)
end
writedlm(joinpath(out_dir, "bayes-aurocs-$(array_id)-$(timestamp).txt"), bayes_aurocs)
writedlm(joinpath(out_dir, "this-aurocs-$(array_id)-$(timestamp).txt"), this_aurocs)
writedlm(joinpath(out_dir, "bayes-auprcs-$(array_id)-$(timestamp).txt"), bayes_auprcs)
writedlm(joinpath(out_dir, "this-auprcs-$(array_id)-$(timestamp).txt"), this_auprcs)
writedlm(joinpath(out_dir, "this-cv-aurocs-$(array_id)-$(timestamp).txt"), this_cv_aurocs)
writedlm(joinpath(out_dir, "this-cv-auprcs-$(array_id)-$(timestamp).txt"), this_cv_auprcs)
writedlm(joinpath(out_dir, "this-lambda-cv-$(array_id)-$(timestamp).txt"),  λs_cv)

rmprocs(workers())