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
array_id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])

n_array = [7, 10, 15, 20, 30]
n = n_array[array_id] # num nodes

# fix average pairwise and triadic degree
fixed_d2 = 3
fixed_d3 = 2
t2 = fixed_d2 / (n-1) # proportion of pairwise edges that exist
t3 = fixed_d3 / binomial(n-1, 2) # proportion of triadic edges that exist

ooi = [2,3]
dmax = 2

# scaling factor for number of data points (as proportion of library size)
ρ_array = 0.5:0.5:4.0

n_itr = 50
λs = [0.01, 0.05, 0.1, 0.5, 1.0] # STLS sparsity parameter

# broadcast globals to all workers
@everywhere begin
    const ooi_const  = $ooi
    const dmax_const = $dmax
    const λs_const   = $λs
end

# worker-side utility functions
@everywhere begin
    function test_inference(n, t2, t3, ρ; ooi = ooi_const, dmax = dmax_const, λs = λs_const)        
        # generate random hypergraph
        A2, A3, A2l, A3l = gnm_random_hyperg(n, t2, t3)
        p = (A2, A3, zeros(n), π/4, π/4)
            
        # sample data
        M = size(get_theta(ones(1, n), 2), 2) # library size
        X = rand(round(Int64, ρ*M), n) .- 0.5
        Y = f_kuramoto_3rd(X, p...) .+ 0.2*randn(size(X))

        # inference with Bayes-THIS & measure performance
        opts = SBOpts(verbosity=0, nitr=15000, free_basis=[1])
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

# set up log files
println("Running array_id=$(array_id) with n=$(n), nworkers()=$(nworkers())")
println("Running array_id=$(array_id) with n=$(n), nworkers()=$(nworkers())")

# allocate storage for results
bayes_aurocs = zeros(Float64, 3, n_itr, length(ρ_array))
this_aurocs  = zeros(Float64, 3, length(λs), n_itr, length(ρ_array))
bayes_auprcs = zeros(Float64, 3, n_itr, length(ρ_array))
this_auprcs  = zeros(Float64, 3, length(λs), n_itr, length(ρ_array))

for (j, ρ) in enumerate(ρ_array)
    results = @showprogress pmap(1:n_itr) do _
        test_inference(n, t2, t3, ρ)
    end

    for i in 1:n_itr
        bayes_aurocs[:, i, j]   .= results[i][1]
        this_aurocs[:, :, i, j] .= results[i][2]
        bayes_auprcs[:, i, j]   .= results[i][3]
        this_auprcs[:, :, i, j] .= results[i][4]
    end
end

out_dir = joinpath(@__DIR__, "..", "out", "num-nodes")
if !isdir(out_dir)
    mkpath(out_dir)
end
writedlm(joinpath(out_dir, "bayes-aurocs-$(array_id)-$(timestamp)-num-nodes.txt"), bayes_aurocs)
writedlm(joinpath(out_dir, "this-aurocs-$(array_id)-$(timestamp)-num-nodes.txt"), this_aurocs)
writedlm(joinpath(out_dir, "bayes-auprcs-$(array_id)-$(timestamp)-num-nodes.txt"), bayes_auprcs)
writedlm(joinpath(out_dir, "this-auprcs-$(array_id)-$(timestamp)-num-nodes.txt"), this_auprcs)

rmprocs(workers())