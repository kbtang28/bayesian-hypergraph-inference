using Distributed

num_procs = parse(Int, get(ENV, "SLURM_NTASKS", "4"))
addprocs(max(num_procs - 1, 0), topology = :master_worker)

@everywhere begin
    import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
end

@everywhere include(joinpath(@__DIR__, "..", "src", "BayesTHIS.jl"))
@everywhere begin
    using .BayesTHIS
    using Dates, Random, DelimitedFiles
    import StatsBase: std, mean

    @enum EnsembleKind FLAG RELABEL RESAMPLE
end

# experiment & model parameters (main)
n         = 15
t2        = 0.35 # expect ~19 triangles
n_graphs  = 40
n_reps    = 3 # independent ensemble draws per target
alphas    = [0.5, 1.0, 2.0, 4.0, 8.0]

dc_targets = collect(-0.9:0.15:0.9)   # relabel family: fixed DC, overlap free
fa_targets = collect(0.0:0.1:0.8)     # resample family: fixed overlap, DC free

n_triads   = 12 # fixed triad count per graph

@everywhere begin
    const n_const   = $n
end

Random.seed!(271828) # governs graph draws, X, and all task seeds below

# worker-side utility functions
@everywhere begin
    # experimental unit: ensemble draw at target DC, all couplings
    function run_task(X, A2, A2l, T0, K2, pw_coeff, task, alphas; n = n_const)
        # noise_seed shared across targets and ensembles at matched (graph, rep)
        rng_struct = Xoshiro(task.struct_seed)
        rng_noise  = Xoshiro(task.noise_seed)

        K3 = triadic_degrees(T0, n)

        # realize triadic structure
        if task.ensemble == FLAG
            T, acc, turn = T0, NaN, NaN
        elseif task.ensemble == RELABEL
            p, _, acc = anneal_to_dc(K2, K3, task.dc_target; rng = rng_struct)
            T, turn = permute_triads(T0, p), NaN
        elseif task.ensemble == RESAMPLE
            T, _, acc, turn = anneal_triads(A2, T0, task.fa_target; rng = rng_struct)
        else
            error("unknown ensemble $(task.ensemble)")
        end

        A3, A3l = triadic_from_list(T, n)

        dc_ach    = degree_corr(A2, A3)
        hetr      = degree_hetero_ratio(A2, A3)
        pa        = pair_alignment(A2, T)
        tri_coeff = 1 / (2 * mean(degrees(A3)))

        rows = Vector{Vector{Any}}()

        for alpha in alphas
            kuramoto_p = (pw_coeff * A2, alpha * tri_coeff * A3, zeros(n), π/4, π/4)

            Y = f_kuramoto_3rd(X, kuramoto_p...)
            Y = Y .+ 0.01 * randn(rng_noise, size(Y))

            ooi = [2, 3]; dmax = 2
            opts  = SBOpts(verbosity = 0, nitr = 2000, free_basis = [1])
            ctrls = SBCtrlSettings(beta_update_frequency = 3)

            Ainf, coeff, out, _ = this_bayes(X, Y, ooi, dmax; opts = opts, ctrls = ctrls)

            D      = get_theta(X, dmax)
            f1     = F1_filter_by_CI(out, D, coeff, A2l, A3l, [0.9545])
            pr, rc = precision_recall_filter_by_CI(out, D, coeff, A2l, A3l, [0.9545])
            auroc  = get_aurocs(Ainf, n, A2l, A3l)
            auprc  = get_auprcs(Ainf, n, A2l, A3l)

            push!(rows, vcat(
                Any[task.graph, string(task.ensemble), task.dc_target, task.fa_target, 
                    task.rep, alpha],
                Any[dc_ach, hetr, acc, turn],
                Any[pa.n_covered, pa.n_absent, pa.frac_absent,
                    pa.w_total, pa.w_absent, pa.w_frac_absent],
                collect(Any, vec(f1)), collect(Any, vec(pr)), collect(Any, vec(rc)),
                collect(Any, vec(auroc)), collect(Any, vec(auprc)),
                Any[mean(out.beta), task.struct_seed, task.noise_seed]))
        end

        return rows
    end
end

# ----------- RUN EXPERIMENT -----------
header = ["graph" "ensemble" "target_dc" "target_fa" "rep" "alpha" "dc" "hetero_ratio" "accept_rate" "turnovers" "n_covered" "n_absent" "frac_absent" "w_total" "w_absent" "w_frac_absent" "f1_pooled" "f1_pw" "f1_tri" "pre_pooled" "pre_pw" "pre_tri" "rec_pooled" "rec_pw" "rec_tri" "auroc_pooled" "auroc_pw" "auroc_tri" "auprc_pooled" "auprc_pw" "auprc_tri" "beta_mean" "struct_seed" "noise_seed"]

all_rows = Vector{Vector{Any}}()

for g in 1:n_graphs

    # generate base graph, flag complex; reject if degenerate
    local A2, A3, A2l, A3l, T0, K2, K3
    while true
        A2, _, A2l, _ = gnm_random_hyperg(n, t2, 0.0)
        fc = flag_complex_fixed(A2, A2l, 12)
        fc === nothing && continue
        _, A3, _, A3l = fc
        T0 = triads(A3l)
        K2 = degrees(A2)
        K3 = triadic_degrees(T0, n)
        (std(K3) > 0 && std(K2) > 0) && break
    end

    # feasible DC interval for this realization, uniform-relabelling null
    lo, hi = dc_bounds(K2, K3)
    @info "graph $g" ntriads=length(T0) dc_flag=degree_corr(A2, A3) dc_lo=lo dc_hi=hi

    # sample dataset X
    X = rand(600, n) .- 0.5

    # compute mean pairwise degree, pairwise coeff
    avgK2    = mean(degrees(A2))
    pw_coeff = 1 / avgK2

    # one noise seed per replicate
    noise_seeds = rand(UInt64, n_reps+1)

    # build task list - starting with flag complex
    tasks = Any[(graph = g, ensemble = FLAG, dc_target = degree_corr(A2, A3), fa_target = 0.0,
                 rep = 0, struct_seed = rand(UInt64), noise_seed = noise_seeds[end])]

    # relabel to hit DC targets
    for tgt in dc_targets, r in 1:n_reps
        (tgt < lo || tgt > hi) && continue
        push!(tasks, (graph = g, ensemble = RELABEL, dc_target = tgt, fa_target = NaN,
                      rep = r, struct_seed = rand(UInt64), noise_seed = noise_seeds[r]))
    end

    # resample to hit face-overlap targets
    for tgt in fa_targets, r in 1:n_reps
        push!(tasks, (graph = g, ensemble = RESAMPLE, dc_target = NaN, fa_target = tgt,
                      rep = r, struct_seed = rand(UInt64), noise_seed = noise_seeds[r]))
    end

    results = pmap(tasks) do task
        run_task(X, A2, A2l, T0, K2, pw_coeff, task, alphas)
    end

    for res in results, row in res
        push!(all_rows, row)
    end
end

# write results
timestamp = Dates.format(now(), "yyyy-mm-dd")
out_dir   = joinpath(@__DIR__, "..", "out", "dc-ensemble")
isdir(out_dir) || mkpath(out_dir)

M = permutedims(hcat(all_rows...))
open(joinpath(out_dir, "dc-ensemble-$(timestamp).csv"), "w") do io
    writedlm(io, header, ',')
    writedlm(io, M, ',')
end

rmprocs(workers())
