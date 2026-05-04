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
    using Dates, Random, SparseBayes, DelimitedFiles
    import StatsBase: std, mean
end

Random.seed!(271828)

# experiment & model parameters (master)
n = 15
t2 = 0.35 # expect ∼19 triangles
n_graphs = 40
n_swaps = collect(0:7)
alphas = [0.5, 1.0, 2.0, 4.0, 8.0]

# broadcast globals to all workers
@everywhere begin
    const n_const = $n
end

# worker-side utility functions
@everywhere begin
    function test_inference(X, A2, A3, A2l, A3l, pw_coeff, tri_coeff, n_swap, alphas; n = n_const)
        # to store results
        F1s    = zeros(Float64, 3, length(alphas))
        pres   = zeros(Float64, 3, length(alphas))
        recs   = zeros(Float64, 3, length(alphas))
        aurocs = zeros(Float64, 3, length(alphas))
        auprcs = zeros(Float64, 3, length(alphas))
        betas  = zeros(Float64, n, length(alphas))

        # swap nodes
        sA3, sA3l = swap_nodes(A2, A3, A3l, n_swap)

        # build adjacency list dict
        Es = Dict(2 => A2l, 3 => A3l)

        for (j, alpha) in enumerate(alphas)
            # params for Kuramoto
            kuramoto_p = (pw_coeff * A2, alpha * tri_coeff * sA3, zeros(n), π/4, π/4)

            # measure vector field
            Y = f_kuramoto_3rd(X, kuramoto_p...)
        
            # add noise
            Y = Y .+ 0.001*randn(size(Y))

            # inference
            ooi = [2,3]; dmax = 2
            opts = SBOpts(verbosity=0, nitr=2000, free_basis=[1])
            ctrls = SBCtrlSettings(beta_update_frequency=3)
        
            Ainf, coeff, out, _ = this_bayes(X, Y, ooi, dmax; opts=opts, ctrls=ctrls)

            # measure quality of inference
            D = get_theta(X, dmax)
            F1s[:, j] .= F1_filter_by_CI(out, D, coeff, A2l, sA3l, [0.9545])
            A, B = precision_recall_filter_by_CI(out, D, coeff, A2l, sA3l, [0.9545])
            pres[:, j] .= A
            recs[:, j] .= B
            aurocs[:, j] .= get_aurocs(Ainf, n, A2l, sA3l)
            auprcs[:, j] .= get_auprcs(Ainf, n, A2l, sA3l)

            # record inferred inverse noise variance
            betas[:, j] .= out.beta
        end

        return F1s, pres, recs, aurocs, auprcs, betas, degree_corr(A2, sA3), degree_hetero_ratio(A2, sA3)
    end
end

# ----------- RUN EXPERIMENT -----------
timestamp = Dates.format(now(), "yyyy-mm-dd")

# allocate space for results
F1s    = zeros(Float64, (3, n_graphs, length(n_swaps), length(alphas)))
pres   = zeros(Float64, (3, n_graphs, length(n_swaps), length(alphas)))
recs   = zeros(Float64, (3, n_graphs, length(n_swaps), length(alphas)))
aurocs = zeros(Float64, (3, n_graphs, length(n_swaps), length(alphas)))
auprcs = zeros(Float64, (3, n_graphs, length(n_swaps), length(alphas)))
betas  = zeros(Float64, (n, n_graphs, length(n_swaps), length(alphas)))

deg_corr = zeros(Float64, n_graphs, length(n_swaps))
deg_hetr = zeros(Float64, n_graphs, length(n_swaps))

for i in 1:n_graphs
    # generate base graph, corresponding flag complex
    A2, _, A2l, _ = gnm_random_hyperg(n, t2, 0.0)
    _, A3, _, A3l = flag_complex(A2, A2l, ptri=1.0)

    # sample dataset X
    X = rand(600, n) .- 0.5

    # compute mean pairwise and triadic degree, pairwise and triadic coeffs
    avgK2 = mean(degrees(A2))
    pw_coeff = 1 / avgK2

    avgK3 = mean(degrees(A3))
    tri_coeff = 1 / (2*avgK3)

    results = pmap(n_swaps) do n_swap
        test_inference(X, A2, A3, A2l, A3l, pw_coeff, tri_coeff, n_swap, alphas)
    end

    # unpack results
    for (j, _) in enumerate(n_swaps)
        F1s[:, i, j, :]     .= results[j][1]
        pres[:, i, j, :]    .= results[j][2]
        recs[:, i, j, :]    .= results[j][3]
        aurocs[:, i, j, :]  .= results[j][4]
        auprcs[:, i, j, :]  .= results[j][5]
        betas[:, i, j, :]   .= results[j][6]
        deg_corr[i, j]       = results[j][7]
        deg_hetr[i, j]       = results[j][8]
    end
end

out_dir = joinpath(@__DIR__, "..", "out", "node-swap")
if !isdir(out_dir)
    mkpath(out_dir)
end
writedlm(joinpath(out_dir, "F1s-fixed-noise-$(timestamp).txt"), F1s)
writedlm(joinpath(out_dir, "pres-fixed-noise-$(timestamp).txt"), pres)
writedlm(joinpath(out_dir, "recs-fixed-noise-$(timestamp).txt"), recs)
writedlm(joinpath(out_dir, "aurocs-fixed-noise-$(timestamp).txt"), aurocs)
writedlm(joinpath(out_dir, "auprcs-fixed-noise-$(timestamp).txt"), auprcs)
writedlm(joinpath(out_dir, "betas-fixed-noise-$(timestamp).txt"), betas)
writedlm(joinpath(out_dir, "deg-corr-fixed-noise-$(timestamp).txt"), deg_corr)
writedlm(joinpath(out_dir, "deg-hetr-fixed-noise-$(timestamp).txt"), deg_hetr)

rmprocs(workers())