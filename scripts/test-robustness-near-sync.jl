import Pkg
Pkg.activate(@__DIR__)

# load dependencies...
using Random, Printf, SparseBayes, LinearAlgebra, Dates, DelimitedFiles, OrdinaryDiffEq, CairoMakie
import Statistics: mean, std
import StatsBase: sample

include("gen-rand-hyperg.jl")
include("hyperg-kuramoto.jl")
include("this-og.jl")
include("this-bayes.jl")
include("performance-measures.jl")
include("structure-utils.jl")

Random.seed!(123)

# experiment & model parameters (master)
n = 10

A2 = readdlm("hyperg-models/toy-hyperg-n10-A2.txt")
A2l = readdlm("hyperg-models/toy-hyperg-n10-A2l.txt")
A3 = readdlm("hyperg-models/toy-hyperg-n10-A3.txt"); A3 = reshape(A3, n, n, n)
A3l = readdlm("hyperg-models/toy-hyperg-n10-A3l.txt")

# compute mean pairwise and triadic degree, pairwise and triadic coeffs
avgK2 = mean(degrees(A2))
pw_coeff = 1 / avgK2

avgK3 = mean(degrees(A3))
tri_coeff = 1 / (2*avgK3)

c = 2.0
Ks = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 16.0]
couplings = [(K*pw_coeff, c*K*tri_coeff) for K in Ks]
σ = 0.4
n_itr = 100

ooi = [2,3]
dmax = 2
λs = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0] # STLS sparsity parameter

tspan = (0.0, 0.70) # over which to integrate trajectories
n_ics = 50 # number of ICs to sample
dt = 0.1 # fine resolution timestep
t_sample = tspan[1] : dt : tspan[2] # timesteps at which to sample

# utility functions
function theta_cond(X)
    T = get_theta(X, 2)
    norms = norm.(eachcol(T))
    T = T ./ norms'

    Ttrunc = T[:, 2:end]

    s = svdvals(Ttrunc)
    kappa = maximum(s) / minimum(s)

    return kappa
end

function theta_coherence(X)
    T = get_theta(X, 2)
    norms = norm.(eachcol(T))
    T = T ./ norms'

    Ttrunc = T[:, 2:end]

    _, M = size(Ttrunc)
    utri = triu!(trues(M, M), 1)

    return maximum(abs.(Ttrunc' * Ttrunc)[utri])
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

function test_inference(u0s, coupling; n = n, M = M, A2 = A2, A3 = A3, A2l = A2l, A3l = A3l, tspan = tspan, t_sample = t_sample, σ = σ, λs = λs, n_itr = n_itr)
    # dynamical parameters
    p = (coupling[1]*A2, coupling[2]*A3, zeros(n), π/4, π/4)

    # simulate trajectories
    Xs = Matrix{Float64}[]
    for u0 in u0s
        prob = ODEProblem(f_kuramoto_3rd!, u0, tspan, p)
        sol = solve(prob, Tsit5())
        X = stack(sol(t_sample).u; dims=1)
        push!(Xs, X)
    end
    Xmat = vcat(Xs...)

    # compute derivatives analytically
    Ymat = f_kuramoto_3rd(Xmat, p...)

    # inference settings
    ooi = [2,3]
    dmax = 2
    opts = SBOpts(verbosity=0, nitr=1000, free_basis=[1])
    ctrls = SBCtrlSettings(beta_update_frequency=3)
        
    # repeat inference
    bayes_aurocs = zeros(Float64, 3, n_itr)
    this_aurocs  = zeros(Float64, 3, length(λs), n_itr)
    bayes_auprcs = zeros(Float64, 3, n_itr)
    this_auprcs  = zeros(Float64, 3, length(λs), n_itr)

    bayes_coeffs = zeros(Float64, M, n, n_itr)
    this_coeffs  = zeros(Float64, M, n, length(λs), n_itr)

    for itr in 1:n_itr
        # add noise to derivatives
        noisy_Ymat = Ymat .+ σ*randn(size(Ymat))

        # inference with Bayes-THIS & measure performance
        bayes_Ainf, bayes_coeff, _, _ = this_bayes(Xmat, noisy_Ymat, ooi, dmax; opts=opts, ctrls=ctrls)
        
        bayes_aurocs[:, itr] = _get_aurocs(bayes_Ainf, n, A2l, A3l)
        bayes_auprcs[:, itr] = _get_auprcs(bayes_Ainf, n, A2l, A3l)
        
        bayes_coeffs[:, :, itr] = bayes_coeff # record inferred coefficients

        # inference with THIS & measure performance (looping over λs...)
        for (j, λ) in enumerate(λs)
            this_Ainf, this_coeff, _ = this(Xmat, noisy_Ymat, ooi, dmax, λ, 1e-4, 500, with_scaling=true)

            this_aurocs[:, j, itr] = _get_aurocs(this_Ainf, n, A2l, A3l)
            this_auprcs[:, j, itr] = _get_auprcs(this_Ainf, n, A2l, A3l)

            this_coeffs[:, :, j, itr] = this_coeff
        end
    end

    return theta_cond(Xmat), bayes_aurocs, this_aurocs, bayes_auprcs, this_auprcs, bayes_coeffs, this_coeffs
end

# ----------- RUN EXPERIMENT -----------
timestamp = Dates.format(now(), "yyyy-mm-dd")

# generate n_ics initial conditions
ic_ct = 1
u0s = Vector{Float64}[] # to store initial conditions

while ic_ct <= n_ics
    println("Sampling IC $(ic_ct)/$(n_ics)...")
    u0 = rand(n) .- 0.5
    u0 = u0 / maximum(abs.(u0)) # maximum IC for any node is ±1.0

    # check we synchronize(ish) with weakest coupling...
    p = (couplings[1][1]*A2, couplings[1][1]*A3, zeros(n), π/4, π/4)
    prob = ODEProblem(f_kuramoto_3rd!, u0, tspan, p)
    sol = solve(prob, Tsit5())
    if any(( abs.(sol(t_sample[end])) - abs.(sol(t_sample[1])) ) .> 0.0)
        continue
    end

    # fig = Figure()
    # ax = Axis(fig[1,1])
    # ts = range(tspan[1], tspan[2], 100)
    # for i in 1:n
    #     lines!(ax, ts, sol(ts, idxs=i).u)
    # end
    # display(fig)
    
    push!(u0s, u0)
    global ic_ct += 1
end

# allocate space to store results
M = size(get_theta(ones(1, n), dmax), 2)

kappas       = zeros(Float64, length(couplings))
bayes_aurocs = zeros(Float64, 3, n_itr, length(couplings))
this_aurocs  = zeros(Float64, 3, length(λs), n_itr, length(couplings))
bayes_auprcs = zeros(Float64, 3, n_itr, length(couplings))
this_auprcs  = zeros(Float64, 3, length(λs), n_itr, length(couplings))
bayes_coeffs = zeros(Float64, M, n, n_itr, length(couplings))
this_coeffs  = zeros(Float64, M, n, length(λs), n_itr, length(couplings))

println("Testing coupling strengths...")
for (k, coupling) in enumerate(couplings)
    println(round.(coupling, digits=2))

    results = test_inference(u0s, coupling)

    kappas[k]               = results[1]
    bayes_aurocs[:, :, k]   = results[2]
    this_aurocs[:, :, :, k] = results[3]
    bayes_auprcs[:, :, k]   = results[4]
    this_auprcs[:, :, :, k] = results[5]
    bayes_coeffs[:, :, :, k]   = results[6]
    this_coeffs[:, :, :, :, k] = results[7]
end

out_dir = "out/near-sync/"
writedlm(joinpath(out_dir, "u0s-$(timestamp).txt"), u0s)
writedlm(joinpath(out_dir, "kappas-$(timestamp).txt"), kappas)
writedlm(joinpath(out_dir, "bayes-aurocs-$(timestamp).txt"), bayes_aurocs)
writedlm(joinpath(out_dir, "this-aurocs-$(timestamp).txt"), this_aurocs)
writedlm(joinpath(out_dir, "bayes-auprcs-$(timestamp).txt"), bayes_auprcs)
writedlm(joinpath(out_dir, "this-auprcs-$(timestamp).txt"), this_auprcs)
writedlm(joinpath(out_dir, "bayes-coeffs-$(timestamp).txt"), bayes_coeffs)
writedlm(joinpath(out_dir, "this-coeffs-$(timestamp).txt"), this_coeffs)