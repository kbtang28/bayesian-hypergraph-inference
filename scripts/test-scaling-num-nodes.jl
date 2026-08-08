using Distributed

num_procs   = parse(Int, get(ENV, "SLURM_NTASKS", "1"))
max_workers = parse(Int, get(ENV, "MAX_WORKERS", string(max(num_procs - 1, 0))))
addprocs(min(max(num_procs - 1, 0), max_workers), topology = :master_worker)

# activate environment
@everywhere begin
    import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
end

# load dependencies...
@everywhere include(joinpath(@__DIR__, "..", "src", "BayesTHIS.jl"))
@everywhere begin
    using .BayesTHIS
    using Random, Printf, LinearAlgebra, Dates, DelimitedFiles, ProgressMeter
end

# shared parameters, instance construction, and test_inference
@everywhere include(joinpath(@__DIR__, "set-up-scaling.jl"))

# experiment & model parameters (master)
array_id = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "1"))
n = n_array[array_id]

cfg = settings_for(n)
n_itr, ρ_array, λs_local = cfg.n_itr, cfg.ρ_array, cfg.λs

# ----------- RUN EXPERIMENT -----------
timestamp = Dates.format(now(), "yyyy-mm-dd")

M         = lib_size(n)
max_N     = round(Int, maximum(ρ_array) * M)
design_gb = 8 * max_N * M / 2^30

println("Running array_id=$(array_id) with n=$(n), nworkers()=$(nworkers())")
println("  M=$(M), max N=$(max_N), n_itr=$(n_itr), rho in $(ρ_array), lambda in $(λs_local)")
@printf("  design matrix Theta(X) ~ %.2f GB per worker\n", design_gb)
flush(stdout)

# allocate storage for results
bayes_aurocs = zeros(Float64, 3, n_itr, length(ρ_array))
this_aurocs  = zeros(Float64, 3, length(λs_local), n_itr, length(ρ_array))
bayes_auprcs = zeros(Float64, 3, n_itr, length(ρ_array))
this_auprcs  = zeros(Float64, 3, length(λs_local), n_itr, length(ρ_array))

for (j, ρ) in enumerate(ρ_array)
    results = @showprogress pmap(1:n_itr) do _
        test_inference(n, ρ; λs_local = λs_local)
    end

    for i in 1:n_itr
        bayes_aurocs[:, i, j]   .= results[i][1]
        this_aurocs[:, :, i, j] .= results[i][2]
        bayes_auprcs[:, i, j]   .= results[i][3]
        this_auprcs[:, :, i, j] .= results[i][4]
    end
    GC.gc()
end

out_dir = joinpath(@__DIR__, "..", "out", "num-nodes")
isdir(out_dir) || mkpath(out_dir)

tag = "$(array_id)-$(timestamp)-num-nodes"
writedlm(joinpath(out_dir, "bayes-aurocs-$(tag).txt"), bayes_aurocs)
writedlm(joinpath(out_dir, "this-aurocs-$(tag).txt"),  this_aurocs)
writedlm(joinpath(out_dir, "bayes-auprcs-$(tag).txt"), bayes_auprcs)
writedlm(joinpath(out_dir, "this-auprcs-$(tag).txt"),  this_auprcs)

# grid metadata, since rho_array and lambdas now depend on n
writedlm(joinpath(out_dir, "grid-$(tag).txt"),
         ["n" n; "n_itr" n_itr; "rho" join(ρ_array, ",");
          "lambdas" join(λs_local, ","); "M" M])

println("Wrote results to $(out_dir) with tag $(tag)")

rmprocs(workers())
