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
    import StatsBase: std

    include("gen-rand-hyperg.jl")
    include("hyperg-kuramoto.jl")
    include("this-bayes.jl")
    include("performance-measures.jl")
    include("sample-posterior.jl")
    include("multiorder-laplacian.jl")
    include("structure-utils.jl")
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
    function F1s_filter_by_CI(out, D, coeff, level, A2l, A3l; extra_out=false)
        _, n = size(coeff)

        sig = significant_coeff(out, D, level)
        Ainf = get_Ainf(coeff .* sig, [2,3], 2)
        
        F1s = zeros(Float64, 3)

        if extra_out
            pres = zeros(Float64, 3)
            recs = zeros(Float64, 3)

            a,  b,  c  = my_F1(abs.(Ainf[2]), A2l, abs.(Ainf[3]), A3l, n; extra_out=true)
            a2, b2, c2 = my_F1(abs.(Ainf[2]), A2l, n; extra_out=true)
            a3, b3, c3 = my_F1(abs.(Ainf[3]), A3l, n; extra_out=true)

            F1s[1]  = a; F1s[2]  = a2; F1s[3]  = a3
            pres[1] = b; pres[2] = b2; pres[3] = b3
            recs[1] = c; recs[2] = c2; recs[3] = c3
            return F1s, pres, recs
        else
            F1s[1] = my_F1(abs.(Ainf[2]), A2l, abs.(Ainf[3]), A3l, n)
            F1s[2] = my_F1(abs.(Ainf[2]), A2l, n)
            F1s[3] = my_F1(abs.(Ainf[3]), A3l, n)
            return F1s
        end
        
    end

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

    function test_inference(X, A2, A3, A2l, A3l, pw_coeff, tri_coeff, n_swap, alphas; n = n_const)
        # to store results
        F1s    = zeros(Float64, 3, length(alphas))
        pres   = zeros(Float64, 3, length(alphas))
        recs   = zeros(Float64, 3, length(alphas))
        aurocs = zeros(Float64, 3, length(alphas))
        auprcs = zeros(Float64, 3, length(alphas))
        lyaps  = zeros(Float64, n, length(alphas))
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
            A, B, C = F1s_filter_by_CI(out, D, coeff, 0.9545, A2l, sA3l; extra_out=true)
            F1s[:, j]    .= A
            pres[:, j]   .= B
            recs[:, j]   .= C
            aurocs[:, j] .= _get_aurocs(Ainf, n, A2l, sA3l)
            auprcs[:, j] .= _get_auprcs(Ainf, n, A2l, sA3l)

            # record Lyapunov exponents of multiorder Laplacian
            weights = Dict(2 => 1.0, 3 => alpha)
            lyaps[:, j] .= compute_lyap_multi(Es, n, weights)

            # record inferred inverse noise variance
            betas[:, j] .= out.beta
        end

        return F1s, pres, recs, aurocs, auprcs, lyaps, betas, degree_corr(A2, sA3), degree_hetero_ratio(A2, sA3)
    end
end

# ----------- RUN EXPERIMENT -----------

# allocate space for results
F1s    = zeros(Float64, (3, n_graphs, length(n_swaps), length(alphas)))
pres   = zeros(Float64, (3, n_graphs, length(n_swaps), length(alphas)))
recs   = zeros(Float64, (3, n_graphs, length(n_swaps), length(alphas)))
aurocs = zeros(Float64, (3, n_graphs, length(n_swaps), length(alphas)))
auprcs = zeros(Float64, (3, n_graphs, length(n_swaps), length(alphas)))
lyaps  = zeros(Float64, (n, n_graphs, length(n_swaps), length(alphas)))
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
        lyaps[:, i, j, :]   .= results[j][6]
        betas[:, i, j, :]   .= results[j][7]
        deg_corr[i, j]       = results[j][8]
        deg_hetr[i, j]       = results[j][9]
    end
end

writedlm("out/node-swap/node-swap-F1s-fixed-noise.txt", F1s)
writedlm("out/node-swap/node-swap-pres-fixed-noise.txt", pres)
writedlm("out/node-swap/node-swap-recs-fixed-noise.txt", recs)
writedlm("out/node-swap/node-swap-aurocs-fixed-noise.txt", aurocs)
writedlm("out/node-swap/node-swap-auprcs-fixed-noise.txt", auprcs)
writedlm("out/node-swap/node-swap-lyaps-fixed-noise.txt", lyaps)
writedlm("out/node-swap/node-swap-betas-fixed-noise.txt", betas)
writedlm("out/node-swap/node-swap-deg-corr-fixed-noise.txt", deg_corr)
writedlm("out/node-swap/node-swap-deg-hetr-fixed-noise.txt", deg_hetr)

rmprocs(workers())