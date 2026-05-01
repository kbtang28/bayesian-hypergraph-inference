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

    # Random.seed!(1) # for reproducibility
end

# experiment & model parameters (master)
n = 10

A2 = readdlm("hyperg-models/toy-hyperg-n10-A2.txt")
A2l = readdlm("hyperg-models/toy-hyperg-n10-A2l.txt")
A3 = readdlm("hyperg-models/toy-hyperg-n10-A3.txt"); A3 = reshape(A3, n, n, n)
A3l = readdlm("hyperg-models/toy-hyperg-n10-A3l.txt")

p = (A2, A3, zeros(n), π/4, π/4)

ooi = [2,3]
dmax = 2

N_array = 60:5:600
σ_array = range(0.05, 1.0, 20)
n_itr = 300
λs = [round.([d * 10. ^ exp for exp in [-2, -1] for d in 1:9]; digits=2); 1.0]

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
    function test_inference(N, σ, n_itr; n = n_const, A2l = A2l_const, A3l = A3l_const, ooi = ooi_const, dmax = dmax_const, λs = λs_const, p = p_const)        
        # repeat inference
        bayes_aurocs = zeros(Float64, 3, n_itr)
        this_aurocs  = zeros(Float64, 3, length(λs), n_itr)
        bayes_auprcs = zeros(Float64, 3, n_itr)
        this_auprcs = zeros(Float64, 3, length(λs), n_itr)

        for itr in 1:n_itr
            X = rand(N, n) .- 0.5
            Y = f_kuramoto_3rd(X, p...) .+ σ*randn(size(X))

            # inference with Bayes-THIS & measure performance
            opts = SBOpts(verbosity=0, nitr=1000, free_basis=[1])
            ctrls = SBCtrlSettings(beta_update_frequency=3)
            bayes_Ainf, _, _, _ = this_bayes(X, Y, ooi, dmax; opts=opts, ctrls=ctrls)
            
            bayes_aurocs[:, itr] = _get_aurocs(bayes_Ainf, n, A2l, A3l)
            bayes_auprcs[:, itr] = _get_auprcs(bayes_Ainf, n, A2l, A3l)

            # inference with THIS & measure performance (looping over λs...)
            for (j, λ) in enumerate(λs)
                this_Ainf, _, _ = this(X, Y, ooi, dmax, λ, 1e-4, 500, with_scaling=true)

                this_aurocs[:, j, itr] = _get_aurocs(this_Ainf, n, A2l, A3l)
                this_auprcs[:, j, itr] = _get_auprcs(this_Ainf, n, A2l, A3l)
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

writedlm("out/robustness/bayes-aurocs-$(array_id)-$(timestamp).txt", bayes_aurocs)
writedlm("out/robustness/this-aurocs-$(array_id)-$(timestamp).txt", this_aurocs)
writedlm("out/robustness/bayes-auprcs-$(array_id)-$(timestamp).txt", bayes_auprcs)
writedlm("out/robustness/this-auprcs-$(array_id)-$(timestamp).txt", this_auprcs)

rmprocs(workers())