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
    using Random, Printf, SparseBayes, LinearAlgebra, Dates, DelimitedFiles, ProgressMeter
end

# experiment & model parameters (master)
n = 10
t2_array = 0.05:0.05:0.75
t3_array = 0.05:0.05:0.75

ooi = [2,3]
dmax = 2

N = 250
σ = 0.2
n_itr = 80
λs = [0.01, 0.05, 0.1, 0.5, 1.0] # STLS sparsity parameter

# broadcast globals to all workers
@everywhere begin
    const n_const    = $n
    const ooi_const  = $ooi
    const dmax_const = $dmax
    const λs_const   = $λs
    const N_const    = $N
    const σ_const    = $σ
end

# worker-side utility functions
@everywhere begin
    function test_inference(t2, t3; n = n_const, ooi = ooi_const, dmax = dmax_const, λs = λs_const, N = N_const, σ = σ_const)        
        # generate random hypergraph
        A2, A3, A2l, A3l = gnm_random_hyperg(n, t2, t3)
        p = (A2, A3, zeros(n), π/4, π/4)
            
        # sample data
        X = rand(N, n) .- 0.5
        Y = f_kuramoto_3rd(X, p...) .+ σ*randn(size(X))

        # inference with Bayes-THIS & measure performance
        opts = SBOpts(verbosity=0, nitr=2000, free_basis=[1])
        ctrls = SBCtrlSettings(beta_update_frequency=3)
        bayes_Ainf, _, _, _ = this_bayes(X, Y, ooi, dmax; opts=opts, ctrls=ctrls)
            
        bayes_aurocs = get_aurocs(bayes_Ainf, n, A2l, A3l)
        bayes_auprcs = get_auprcs(bayes_Ainf, n, A2l, A3l)

        # inference with THIS & measure performance (looping over λs...)
        this_aurocs = zeros(Float64, 3, length(λs))
        this_auprcs = zeros(Float64, 3, length(λs))
        for (j, λ) in enumerate(λs)
            this_Ainf, _, _ = this(X, Y, ooi, dmax, λ, 1e-4, 1000, with_scaling=true)

            this_aurocs[:, j] = get_aurocs(this_Ainf, n, A2l, A3l)
            this_auprcs[:, j] = get_auprcs(this_Ainf, n, A2l, A3l)
        end
        
        return bayes_aurocs, this_aurocs, bayes_auprcs, this_auprcs
    end
end

# ----------- RUN EXPERIMENT -----------
timestamp = Dates.format(now(), "yyyy-mm-dd")

# allocate storage for results
bayes_aurocs = zeros(Float64, 3, n_itr, length(t2_array), length(t3_array))
this_aurocs  = zeros(Float64, 3, length(λs), n_itr, length(t2_array), length(t3_array))
bayes_auprcs = zeros(Float64, 3, n_itr, length(t2_array), length(t3_array))
this_auprcs  = zeros(Float64, 3, length(λs), n_itr, length(t2_array), length(t3_array))

for (j, t2) in enumerate(t2_array)
    for (k, t3) in enumerate(t3_array)
        results = @showprogress pmap(1:n_itr) do _
            test_inference(t2, t3)
        end

        for i in 1:n_itr
            bayes_aurocs[:, i, j, k]   .= results[i][1]
            this_aurocs[:, :, i, j, k] .= results[i][2]
            bayes_auprcs[:, i, j, k]   .= results[i][3]
            this_auprcs[:, :, i, j, k] .= results[i][4]
        end
    end
end

out_dir = joinpath(@__DIR__, "..", "out", "sparsity")
writedlm(joinpath(out_dir, "bayes-aurocs-$(timestamp)-sparsity.txt"), bayes_aurocs)
writedlm(joinpath(out_dir, "this-aurocs-$(timestamp)-sparsity.txt"), this_aurocs)
writedlm(joinpath(out_dir, "bayes-auprcs-$(timestamp)-sparsity.txt"), bayes_auprcs)
writedlm(joinpath(out_dir, "this-auprcs-$(timestamp)-sparsity.txt"), this_auprcs)

rmprocs(workers())