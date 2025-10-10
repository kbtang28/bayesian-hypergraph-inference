# import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()

using Random, CairoMakie, OrdinaryDiffEq, SparseBayes, DelimitedFiles, Distributions, Dates, Printf

include("gen-rand-hyperg.jl")
include("hyperg-kuramoto.jl")
include("finite-diffs.jl")
include("this-bayes.jl")
include("sample-posterior.jl")
include("performance-measures.jl")

Random.seed!(18)

# helper function
function ppc(u0s, ρ, σx, L_I, L_D1; K=1000, nitr_noise=100)
    @printf(SB_logfile, "Testing ρ = %.2f, σx = %.4f...\n", ρ, σx)
    @printf(ROC_logfile, "Testing ρ = %.2f, σx = %.4f...\n", ρ, σx)

    # generate trajectories
    clean_Xs = Matrix{Float64}[]
    for u0 in u0s
        prob = ODEProblem(f_kuramoto_3rd!, (ρ/2)*u0, tspan, p)
        sol = solve(prob)
        X = stack(sol(t_sample).u; dims=1)
        push!(clean_Xs, X)
    end

    # inference settings
    opts = SBOpts(verbosity=2, nitr=500, free_basis=[1], fixed_noise=true, io_list=[SB_logfile])
    σ = norm(L_D1[1,:]) * σx # approximate σx^2(L_D1 * L_D1') by its diagonal
    settings = SBSettings(beta=1/(σ^2))

    pval = Vector{Float64}(undef, nitr_noise)
    auc = Vector{Float64}(undef, nitr_noise)
    auc3 = Vector{Float64}(undef, nitr_noise)

    for itr in 1:nitr_noise
        println.([SB_logfile, ROC_logfile], "Starting iteration $(itr) / $(nitr_noise)...")

        # add noise and estimate derivatives
        noisy_Xs = [X + σx*randn(size(X)) for X in clean_Xs]
        Xmat = vcat([L_I*X for X in noisy_Xs]...)
        Ymat = vcat([L_D1*X for X in noisy_Xs]...)

        T, n = size(Xmat)

        # fit pairwise model
        _, _, out, diagnostics = this_bayes(Xmat, Ymat, [2], 1; opts=opts, settings=settings)

        # replicate data and compute discrepancies
        Θ = get_theta(Xmat, 1)
        Ξs = sample_joint_posterior(out, Θ, Ymat; nsamples=K)

        D_obs = [sum( norm.(eachrow( (Ymat - Θ*Ξ)/σ )).^2 ) for Ξ in Ξs]
        pval[itr] = sum( ccdf.( Chisq(T*n),  D_obs ) )/K

        # measure quality of inference - pairwise + triadic model
        Ainf, _, _, _ = this_bayes(Xmat, Ymat, [2,3], 2; opts=opts, settings=settings)
        @printf(ROC_logfile, "Bayes-THIS --------------- (full) ")
        tpr, fpr = my_ROC(abs.(Ainf[2]), A2l, abs.(Ainf[3]), A3l, n; verbosity=1, io=ROC_logfile)
        auc[itr] = get_auc(tpr, fpr)

        @printf(ROC_logfile, "Bayes-THIS ---------------- (tri) ")
        tpr3, fpr3 = my_ROC(abs.(Ainf[3]), A3l, n; verbosity=1, io=ROC_logfile)
        auc3[itr] = get_auc(tpr3, fpr3)
    end

    flush(SB_logfile)
    flush(ROC_logfile)

    return pval, auc, auc3
end

# hypergraph model
n = 7
global A2, A3, A2l, A3l = gnm_random_hyperg(n, 0.4, 0.1)
global p = (A2, 0.5*A3, zeros(n), π/4, π/4)

# experimental settings
global tspan = (0.0, 0.70) # over which to integrate trajectories
n_ics = 30 # number of ICs to sample
dt = 0.01 # fine resolution timestep
global t_sample = tspan[1] : dt : tspan[2]

ρs = 10 .^ range(log10(0.05), log10(1.0), 11)
σxs = [0.0005, 0.001]

# ----------- RUN EXPERIMENT -----------

# set up log files
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMM")

global SB_logfile = open("out/logs/ppcs-traj-$(timestamp)-SB.txt", "w")
global ROC_logfile = open("out/logs/ppcs-traj-$(timestamp)-ROC.txt", "w")

# generate initial conditions
ic_ct = 1
u0s = Vector{Float64}[]
while ic_ct <= n_ics
    println("Sampling IC $(ic_ct)/$(n_ics)...")
    u0 = rand(n) .- 0.5
    u0 = u0 / maximum(abs.(u0))

    # check we synchronize...
    prob = ODEProblem(f_kuramoto_3rd!, u0, tspan, p)
    sol = solve(prob)
    if maximum( abs.( sol(t_sample[end]) ) ) > 1.0
        continue
    end
    
    # sampled_X = stack(sol(t_sample).u; dims=1)
    
    # inch = 96; pt = 4/3
    # fig1 = Figure(size=(5inch,5inch),fontsize=8pt)
    # ax1 = Axis(fig1[1,1], xgridvisible=false, ygridvisible=false)
    # lines!(ax1, cos.(sampled_X[:, 1]), cos.(sampled_X[:, 2]))
    # lines!(ax1, cos.(sampled_X[:, 3]), cos.(sampled_X[:, 4]))
    # lines!(ax1, cos.(sampled_X[:, 5]), cos.(sampled_X[:, 6]))
    # colsize!(fig1.layout, 1, Aspect(1, 1.0))
    # resize_to_layout!(fig1)
    # display(fig1)
    
    # fig2 = Figure(size=(3inch,3inch),fontsize=8pt)
    # ax2 = Axis(fig2[1,1], limits=(tspan, nothing), xgridvisible=false, ygridvisible=false)
    # for j in 1:n
    #     lines!(ax2, t_sample, sampled_X[:, j])
    # end
    # colsize!(fig2.layout, 1, Aspect(1, 3.0))
    # resize_to_layout!(fig2)
    # display(fig2)

    push!(u0s, u0)
    global ic_ct += 1
end

# generate linear operators for finite differencing
L_I, L_D1 = FD(length(t_sample), 4, dt, 5)

nitr = 100
pvals = zeros(Float64, nitr, length(ρs), length(σxs))
aucs = zeros(Float64, nitr, length(ρs), length(σxs))
auc3s = zeros(Float64, nitr, length(ρs), length(σxs))

for (k, σx) in enumerate(σxs)
    for (j, ρ) in enumerate(ρs)
        pval, auc, auc3 = ppc(u0s, ρ, σx, L_I, L_D1; K=1000, nitr_noise=nitr)
        pvals[:,j,k] = pval
        aucs[:,j,k] = auc
        auc3s[:,j,k] = auc3
    end
end

close(SB_logfile)
close(ROC_logfile)

writedlm("out/ppc-traj-pvals-$(n_ics).txt", pvals)
writedlm("out/ppc-traj-aucs-$(n_ics).txt", aucs)
writedlm("out/ppc-traj-auc3s-$(n_ics).txt", auc3s)