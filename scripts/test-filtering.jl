using Distributed

num_procs = parse(Int, ENV["SLURM_NTASKS"])
addprocs(max(num_procs-1, 0), topology=:master_worker)

# activate environment
@everywhere begin
    import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
end

# load dependencies
@everywhere begin
    include(joinpath(@__DIR__, "..", "BayesTHIS.jl"))
    using .BayesTHIS
    using Random, Printf, SparseBayes, Dates, DelimitedFiles, Distributions, ProgressMeter
end

# experiment & model parameters (master)
n = 10

hyperg_models_dir = joinpath(@__DIR__, "..", "hyperg-models")
A2 = readdlm(joinpath(hyperg_models, "toy-hyperg-n10v2-A2.txt"))
A2l = readdlm(joinpath(hyperg_models, "toy-hyperg-n10v2-A2l.txt"))
A3 = readdlm(joinpath(hyperg_models, "toy-hyperg-n10v2-A3.txt")); A3 = reshape(A3, n, n, n)
A3l = readdlm(joinpath(hyperg_models, "toy-hyperg-n10v2-A3l.txt"))

p = (A2, A3, zeros(n), π/4, π/4)

ooi = [2,3]
dmax = 2

τs = round.([d * 10. ^ exp for exp in -2:0 for d in 1:9]; digits=2)
levels = [cdf(Normal(), σ) - cdf(Normal(), -σ) for σ in [1.0, 2.0, 3.0]]
experiment_settings = [(40:2:400, 0.1), (40:5:500, 0.5)]
n_itr = 300

# SLURM array index
array_id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
N_array, σ = experiment_settings[array_id]

# broadcast globals to all workers
@everywhere begin
    const n_const      = $n
    const A2l_const    = $A2l
    const A3l_const    = $A3l
    const p_const      = $p
    const ooi_const    = $ooi
    const dmax_const   = $dmax
    const τs_const     = $τs
    const levels_const = $levels
end

# worker-side utility function
@everywhere begin
    function test_inference(N, σ, n_itr; n = n_const, p = p_const, ooi = ooi_const, dmax = dmax_const, τs = τs_const, levels = levels_const)
        # repeat inference
        F1s_coeff_mags = zeros(Float64, 3, length(τs), n_itr)
        F1s_coeff_CIs = zeros(Float64, 3, length(levels), n_itr)

        for itr in 1:n_itr
            X = rand(N, n) .- 0.5
            Y = f_kuramoto_3rd(X, p...) .+ σ*randn(size(X))

            # inference with Bayes-THIS
            opts = SBOpts(verbosity=0, nitr=1000, free_basis=[1])
            ctrls = SBCtrlSettings(beta_update_frequency=3)
            _, coeff, out, _ = this_bayes(X, Y, ooi, dmax; opts=opts, ctrls=ctrls)

            # measure performance
            F1s_coeff_mags[:, :, itr] = F1_filter_by_coeff_mag(coeff, A2l_const, A3l_const, τs)

            D = get_theta(X, dmax)
            F1s_coeff_CIs[:, :, itr] = F1_filter_by_CI(out, D, coeff, A2l_const, A3l_const, levels)
        end

        return F1s_coeff_mags, F1s_coeff_CIs
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

# allocate and unpack results
F1s_coeff_mags = zeros(Float64, 3, length(τs), n_itr, length(N_array))
F1s_coeff_CIs  = zeros(Float64, 3, length(levels), n_itr, length(N_array))

for (i, _) in enumerate(N_array)
    F1s_coeff_mags[:, :, :, i] .= results[i][1]
    F1s_coeff_CIs[:, :, :, i]  .= results[i][2]
end

out_dir = joinpath(@__DIR__, "..", "out", "filtering")
writedlm(joinpath(out_dir, "coeff-mags-$(σ)-F1s-$(timestamp).txt"), F1s_coeff_mags)
writedlm(joinpath(out_dir, "coeff-CIs-$(σ)-F1s-$(timestamp).txt"), F1s_coeff_CIs)

rmprocs(workers())