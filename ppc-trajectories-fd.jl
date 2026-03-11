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
    using Random, ProgressMeter, OrdinaryDiffEq, SparseBayes, DelimitedFiles, Distributions, Dates, Printf

    include("gen-rand-hyperg.jl")
    include("hyperg-kuramoto.jl")
    include("finite-diffs.jl")
    include("this-bayes.jl")
    include("sample-posterior.jl")
    include("performance-measures.jl")
end

Random.seed!(18)

# experiment & model parameters (master)
n = 8
A2, A3, A2l, A3l = gnm_random_hyperg(n, 0.4, 0.1)
p = (A2, 2.0*A3, zeros(n), π/4, π/4)

tspan = (0.0, 0.70) # over which to integrate trajectories
n_ics = 30 # number of ICs to sample
dt = 0.01 # fine resolution timestep
t_sample = tspan[1] : dt : tspan[2] # timesteps at which to sample

levels = [cdf(Normal(), σ) - cdf(Normal(), -σ) for σ in [1.0, 2.0, 3.0]]
ρs = 10 .^ range(-2, 0.0, 11)
σxs = [0.0005, 0.001]
nitr = 100
K = 1000 # number of replicates for each iteration

# broadcast globals to all workers
@everywhere begin
    const tspan_const    = $tspan
    const t_sample_const = $t_sample
    const n_const        = $n
    const p_const        = $p
    const levels_const   = $levels
    const A2l_const      = $A2l
    const A3l_const      = $A3l
end

# worker-side utility functions
@everywhere begin
    function F1_filter_by_CI(out, D, coeff, levels; A2l = A2l_const, A3l = A3l_const)
        _, n = size(coeff)

        F1s = zeros(Float64, 3, length(levels))

        for (i, level) in enumerate(levels)
            # filter out coeffs with 0 in (level)% CI for conditional posterior
            sig = significant_coeff(out, D, level)
            Ainf = get_Ainf(coeff .* sig, [2,3], 2)

            # compute F1 scores
            F1s[1, i] = my_F1(abs.(Ainf[2]), A2l, abs.(Ainf[3]), A3l, n)
            F1s[2, i] = my_F1(abs.(Ainf[2]), A2l, n)
            F1s[3, i] = my_F1(abs.(Ainf[3]), A3l, n)
        end

        return F1s
    end

    function partition_D(D, d, i)
        tri_cols = [(length(unique(r)) == 2) & (!in(r)(0)) & (!in(r)(1)) for r in eachrow(d)]
        # tri_cols = [(!in(r)(0)) for r in eachrow(d)]

        D1 = D[:, .!tri_cols]
        D2 = D[:, tri_cols]

        return D1, D2
    end

    function project_colD2orth(D1, D2, X)
        # projects X onto the colspace of D2 after orthogonalizing wrt D1
        
        # orthogonal basis for col(D1)
        QD1 = Matrix(qr(D1).Q)

        # remove D1-component from D2
        D2orth = D2 - QD1 * (QD1' * D2)

        # orthogonal basis for col(D2orth)
        QD2orth = Matrix(qr(D2orth).Q)

        # project X onto col(D2orth)
        return QD2orth * (QD2orth' * X)
    end

    function ppc(u0s, ρ, σx, L_I, L_D1, nitr, K; tspan = tspan_const, t_sample = t_sample_const, n = n_const, p = p_const, levels = levels_const, A2l = A2l_const, A3l = A3l_const)
        # generate trajectories
        clean_Xs = Matrix{Float64}[]
        for u0 in u0s
            prob = ODEProblem(f_kuramoto_3rd!, (ρ/2)*u0, tspan, p)
            sol = solve(prob)
            X = stack(sol(t_sample).u; dims=1)
            push!(clean_Xs, X)
        end

        # inference settings
        opts = SBOpts(verbosity=0, nitr=1000, free_basis=[1], fixed_noise=true)
        σ = norm(L_D1[1,:]) * σx # approximate σx^2(L_D1 * L_D1') by its diagonal
        settings = SBSettings(beta=1/(σ^2))

        # repeat inference for different noise realizations
        pval = Vector{Float64}(undef, nitr)
        auc  = Vector{Float64}(undef, nitr)
        auc3 = Vector{Float64}(undef, nitr)
        F1   = Array{Float64}(undef, 3, length(levels), nitr)

        for itr in 1:nitr
            # add noise and estimate derivatives
            noisy_Xs = [X + σx*randn(size(X)) for X in clean_Xs]
            Xmat = vcat([L_I*X for X in noisy_Xs]...)  # stack trajectories
            Ymat = vcat([L_D1*X for X in noisy_Xs]...) # stack trajectories

            T, n = size(Xmat)

            # design matrix
            Θ = get_theta(Xmat, 2); # full design matrix
            pw_Θ = get_theta(Xmat, 1); # pairwise monomials only

            # fit pairwise model
            _, _, pw_out, _ = this_bayes(Xmat, Ymat, [2], 1; opts=opts, settings=settings)

            # replicate data and compute discrepancies
            Ξs = sample_joint_posterior(pw_out, pw_Θ, Ymat; nsamples=K)
            T_obs = zeros(Float64, K)
            T_rep = zeros(Float64, K)
            for (k, Ξ) in enumerate(Ξs)
                err_obs = Ymat - pw_Θ*Ξ
            
                # replicate data
                # Y_rep = pw_Θ*Ξ + σ*randn(size(Xmat))
                err_rep = σ*randn(size(Xmat)) # err_rep = Y_rep - pw_Θ*Ξ
                
                for i in 1:n
                    Θ1, Θ2 = partition_D(Θ, get_d(n, 2), i) # partition design matrix
                    T_obs[k] += err_obs[:, i]' * project_colD2orth(Θ1, Θ2, err_obs[:, i])
                    T_rep[k] += err_rep[:, i]' * project_colD2orth(Θ1, Θ2, err_rep[:, i])
                end
            end
            
            # compute p-values
            pval[itr] = sum( T_rep .>= T_obs ) / K

            # measure quality of inference - full model
            full_opts = SBOpts(verbosity=0, nitr=1000, free_basis=[1])
            ctrls = SBCtrlSettings(beta_update_frequency=3)
            Ainf, coeff, tri_out, _ = this_bayes(Xmat, Ymat, [2,3], 2; opts=full_opts, ctrls=ctrls)

            tpr, fpr = my_ROC(abs.(Ainf[2]), A2l, abs.(Ainf[3]), A3l, n)
            auc[itr] = get_auc(tpr, fpr)

            pre, rec = my_PRC(abs.(Ainf[3]), A3l, n)
            auc3[itr] = get_auc(pre, rec)

            F1[:, :, itr] = F1_filter_by_CI(tri_out, Θ, coeff, levels)
        end

        return pval, auc, auc3, F1
    end
end

# ----------- RUN EXPERIMENT -----------

# generate n_ics initial conditions
ic_ct = 1
u0s = Vector{Float64}[] # to store initial conditions

while ic_ct <= n_ics
    println("Sampling IC $(ic_ct)/$(n_ics)...")
    u0 = rand(n) .- 0.5
    u0 = u0 / maximum(abs.(u0)) # maximum IC for any node is ±1.0

    # check we synchronize(ish)...
    prob = ODEProblem(f_kuramoto_3rd!, u0, tspan, p_const)
    sol = solve(prob)
    if maximum( abs.( sol(t_sample[end]) ) ) > 1.0
        continue
    end
    
    push!(u0s, u0)
    global ic_ct += 1
end

# generate linear operators for finite differencing
L_I, L_D1 = FD(length(t_sample), 6, dt, 4)

# allocate space for results
pvals = Array{Float64}(undef, nitr, length(ρs), length(σxs))
aucs  = Array{Float64}(undef, nitr, length(ρs), length(σxs))
auc3s = Array{Float64}(undef, nitr, length(ρs), length(σxs))
F1s   = Array{Float64}(undef, 3, length(levels), nitr, length(ρs), length(σxs))

for (k, σx) in enumerate(σxs)
    results = @showprogress pmap(ρs) do ρ
        ppc(u0s, ρ, σx, L_I, L_D1, nitr, K)
    end

    for (j, _) in enumerate(ρs)
        pvals[:, j, k]     .= results[j][1]
        aucs[:, j, k]      .= results[j][2]
        auc3s[:, j, k]     .= results[j][3]
        F1s[:, :, :, j, k] .= results[j][4]
    end
end

writedlm("out/ppc/ppc-traj-fd-pvals.txt", pvals)
writedlm("out/ppc/ppc-traj-fd-aucs.txt", aucs)
writedlm("out/ppc/ppc-traj-fd-auc3s.txt", auc3s)
writedlm("out/ppc/ppc-traj-fd-F1s.txt", F1s)

rmprocs(workers())