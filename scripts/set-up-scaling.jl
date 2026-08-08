# shared experiment parameters, instance construction for scaling experiments:
# - test-scaling-num-nodes.jl   -- reconstruction quality vs. n (parallel)
# - benchmark-cost-scaling.jl   -- wall-clock and peak memory vs. n (serial)

# system sizes
const n_array = [7, 10, 15, 20, 30, 50, 75, 100]

# fixed average pairwise and triadic degree across all n, so that local
# connectivity is held constant and only combinatorial complexity grows
const fixed_d2 = 3
const fixed_d3 = 2

const ooi  = [2, 3]
const dmax = 2

const σ_noise = 0.2              # derivative noise standard deviation
const λs = [round.([d * 10. ^ exp for exp in [-3, -2, -1, 0] for d in 1:9]; digits=3); 10.0] # STLS sparsity parameter

# size of monomial library at dmax = 2: 1 + n + n(n+1)/2
lib_size(n) = 1 + n + n * (n + 1) ÷ 2

# edge probs that hold avg deg at (fixed_d2, fixed_d3)
edge_probs(n) = (fixed_d2 / (n - 1), fixed_d3 / binomial(n - 1, 2))

# replicate count, dataset sizes (as proportion of lib_size(n)), STLS sparsity parameters
function settings_for(n)
    if n <= 30
        return (n_itr = 50, ρ_array = collect(0.5:0.5:4.0), λs = λs)
    elseif n <= 50
        return (n_itr = 10, ρ_array = [1.0, 2.0, 4.0], λs = λs)
    else
        return (n_itr = 5,  ρ_array = [1.0, 2.0], λs = λs)
    end
end

# one synthetic problem instance
function make_instance(n, ρ)
    t2, t3 = edge_probs(n)
    A2, A3, A2l, A3l = gnm_random_hyperg(n, t2, t3)
    p = (A2, A3, zeros(n), π/4, π/4)

    M = lib_size(n)
    N = round(Int64, ρ * M)
    X = rand(N, n) .- 0.5
    Y = f_kuramoto_3rd(X, p...) .+ σ_noise * randn(size(X))

    return X, Y, A2l, A3l, M, N
end

# inference settings
bayes_settings() = (SBOpts(verbosity = 0, nitr = 30000, free_basis = [1]),
                    SBCtrlSettings(beta_update_frequency = 3))

# one experiment unit
function test_inference(n, ρ; λs_local = λs)
    X, Y, A2l, A3l, _, _ = make_instance(n, ρ)

    opts, ctrls = bayes_settings()
    bayes_Ainf, _, _, _ = this_bayes(X, Y, ooi, dmax; opts = opts, ctrls = ctrls)

    bayes_aurocs = get_aurocs(bayes_Ainf, n, A2l, A3l)
    bayes_auprcs = get_auprcs(bayes_Ainf, n, A2l, A3l)

    this_aurocs = zeros(Float64, 3, length(λs_local))
    this_auprcs = zeros(Float64, 3, length(λs_local))
    for (j, λ) in enumerate(λs_local)
        this_Ainf, _, _ = this(X, Y, ooi, dmax, λ, 1e-4, 1000, with_scaling = true)

        this_aurocs[:, j] = get_aurocs(this_Ainf, n, A2l, A3l)
        this_auprcs[:, j] = get_auprcs(this_Ainf, n, A2l, A3l)
    end

    return bayes_aurocs, this_aurocs, bayes_auprcs, this_auprcs
end
