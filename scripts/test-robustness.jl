using Distributed

num_procs = parse(Int, ENV["SLURM_NTASKS"])
addprocs(max(num_procs-1, 0), topology=:master_worker)

# activate environment
@everywhere begin
    import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
end

# load dependencies...
@everywhere begin
    include(joinpath(@__DIR__, "..", "BayesTHIS.jl"))
    using .BayesTHIS
    using Random, Printf, SparseBayes, LinearAlgebra, Dates, DelimitedFiles, ProgressMeter
end

# experiment & model parameters (master)
n = 10

hyperg_models_dir = joinpath(@__DIR__, "..", "hyperg-models")
A2 = readdlm(joinpath(hyperg_models_dir, "toy-hyperg-n10v1-A2.txt"))
A2l = readdlm(joinpath(hyperg_models_dir, "toy-hyperg-n10v1-A2l.txt"))
A3 = readdlm(joinpath(hyperg_models_dir, "toy-hyperg-n10v1-A3.txt")); A3 = reshape(A3, n, n, n)
A3l = readdlm(joinpath(hyperg_models_dir, "toy-hyperg-n10v1-A3l.txt"))

p = (A2, A3, zeros(n), π/4, π/4)

ooi = [2,3]
dmax = 2

N_array = 60:5:600
σ_array = range(0.05, 1.0, 20)
n_itr = 300
λs = [round.([d * 10. ^ exp for exp in [-2, -1] for d in 1:9]; digits=2); 1.0] # STLS sparsity parameter

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
    function test_inference(N, σ, n_itr; n = n_const, A2l = A2l_const, A3l = A3l_const, ooi = ooi_const, dmax = dmax_const, λs = λs_const, p = p_const)        
        # repeat inference
        bayes_aurocs = zeros(Float64, 3, n_itr)
        this_aurocs  = zeros(Float64, 3, length(λs), n_itr)
        bayes_auprcs = zeros(Float64, 3, n_itr)
        this_auprcs  = zeros(Float64, 3, length(λs), n_itr)

        for itr in 1:n_itr
            X = rand(N, n) .- 0.5
            Y = f_kuramoto_3rd(X, p...) .+ σ*randn(size(X))

            # inference with Bayes-THIS & measure performance
            opts = SBOpts(verbosity=0, nitr=1000, free_basis=[1])
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
        end
        
        return bayes_aurocs, this_aurocs, bayes_auprcs, this_auprcs
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
bayes_aurocs = zeros(Float64, 3, n_itr, length(N_array))
this_aurocs  = zeros(Float64, 3, length(λs), n_itr, length(N_array))
bayes_auprcs = zeros(Float64, 3, n_itr, length(N_array))
this_auprcs  = zeros(Float64, 3, length(λs), n_itr, length(N_array))

for (i, _) in enumerate(N_array)
    bayes_aurocs[:, :, i]   .= results[i][1]
    this_aurocs[:, :, :, i] .= results[i][2]
    bayes_auprcs[:, :, i]   .= results[i][3]
    this_auprcs[:, :, :, i] .= results[i][4]
end

out_dir = joinpath(@__DIR__, "..", "out", "robustness")
writedlm(joinpath(out_dir, "bayes-aurocs-$(array_id)-$(timestamp).txt"), bayes_aurocs)
writedlm(joinpath(out_dir, "this-aurocs-$(array_id)-$(timestamp).txt"), this_aurocs)
writedlm(joinpath(out_dir, "bayes-auprcs-$(array_id)-$(timestamp).txt"), bayes_auprcs)
writedlm(joinpath(out_dir, "this-auprcs-$(array_id)-$(timestamp).txt"), this_auprcs)

rmprocs(workers())