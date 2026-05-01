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
    using Random, Printf, SparseBayes, LinearAlgebra, Dates, DelimitedFiles, ProgressMeter

    include("gen-rand-hyperg.jl")
    include("hyperg-kuramoto.jl")
    include("this-og.jl")
    include("this-bayes.jl")
    include("performance-measures.jl")
end

# experiment & model parameters (master)
array_id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])

n_array = [7, 10, 15, 20, 30]
n = n_array[array_id] # num nodes

fixed_d2 = 3
fixed_d3 = 2
t2 = fixed_d2 / (n-1) # proportion of pairwise edges that exist
t3 = fixed_d3 / binomial(n-1, 2) # proportion of triadic edges that exist

ooi = [2,3]
dmax = 2

ρ_array = 0.5:0.5:4.0 # scaling factor for number of data points

n_itr = 50
λs = [0.01, 0.05, 0.1, 0.5, 1.0]

# broadcast globals to all workers
@everywhere begin
    const ooi_const  = $ooi
    const dmax_const = $dmax
    const λs_const   = $λs
end

# worker-side utility functions
@everywhere begin
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

    # helper function to do inference and measure performance
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
            
        bayes_aurocs = _get_aurocs(bayes_Ainf, n, A2l, A3l)
        bayes_auprcs = _get_auprcs(bayes_Ainf, n, A2l, A3l)

        # inference with THIS & measure performance (looping over λs...)
        this_aurocs = zeros(Float64, 3, length(λs))
        this_auprcs = zeros(Float64, 3, length(λs))
        for (j, λ) in enumerate(λs)
            this_Ainf, _, _ = this(X, Y, ooi, dmax, λ, 1e-4, 1000, with_scaling=true)

            this_aurocs[:, j] = _get_aurocs(this_Ainf, n, A2l, A3l)
            this_auprcs[:, j] = _get_auprcs(this_Ainf, n, A2l, A3l)
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

writedlm("out/num-nodes/bayes-aurocs-$(array_id)-$(timestamp)-num-nodes.txt", bayes_aurocs)
writedlm("out/num-nodes/this-aurocs-$(array_id)-$(timestamp)-num-nodes.txt", this_aurocs)
writedlm("out/num-nodes/bayes-auprcs-$(array_id)-$(timestamp)-num-nodes.txt", bayes_auprcs)
writedlm("out/num-nodes/this-auprcs-$(array_id)-$(timestamp)-num-nodes.txt", this_auprcs)

rmprocs(workers())