import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "BayesTHIS.jl"))
using .BayesTHIS
using Random, SparseBayes, Printf, Distributions, LinearAlgebra, Dates, DelimitedFiles, CairoMakie
import ColorSchemes: cmr_prinsenvlag, get
psblues = get(cmr_prinsenvlag, range(0.7, 0.9, 3))
triteal = "#009988"
pwblue = "#0077BB"
leakcyan = "#33BBEE"
falseorg = "#EE7733"

Random.seed!(1) # for reproducibility

# helper functions to compute conditional posterior sds
function conditional_posterior_sds(μ, α, β, D)
    _, M = size(D)

    relevant = (μ .!= 0.0)
    m = sum(relevant)
    @assert length(α) == m

    Df = D[:, relevant]
    μf = μ[relevant]

    Λ = diagm(α) + β * (Df' * Df)
    Λ_diag = diag(Λ)

    sds = zeros(Float64, M)
    sds[relevant] .= 1 ./ sqrt.(Λ_diag)

    return sds
end

function conditional_posterior_sds(out, D)
    M, n = size(out.value)

    sds = zeros(Float64, M, n)
    for i in 1:n
        sds[:, i] .= conditional_posterior_sds(out.value[:, i], out.alpha[i], out.beta[i], D)
    end

    return sds
end

# helper function to collect valid monomial indices for i-th node
function valid_monomial(i, n, dmax)
    d = get_d(n, dmax)

    idx_mon = Dict{Int64, Vector{Int64}}() # (monomial index) => (nodes involved in monomial)
    for i in 1:size(d)[1]
        mon = d[i, :][d[i, :] .!= 0]
        if (length(mon) == length(union(mon))) && !isempty(mon)
            idx_mon[i] = sort(mon)
        end
    end

    valid = Int64[]
    for (key, value) in idx_mon
        if !in(value)(i)
            push!(valid, key)
        end
    end

    return valid
end

# hypergraph model
n = 10

hyperg_model_dir = joinpath(@__DIR__, "..", "hyperg-models")
A2 = readdlm(joinpath(hyperg_model_dir, "toy-hyperg-n10v1-A2.txt"))
A2l = readdlm(joinpath(hyperg_model_dir, "toy-hyperg-n10v1-A2l.txt"))
A3 = readdlm(joinpath(hyperg_model_dir, "toy-hyperg-n10v1-A3.txt")); A3 = reshape(A3, n, n, n)
A3l = readdlm(joinpath(hyperg_model_dir, "toy-hyperg-n10v1-A3l.txt"))
# A2, A3, A2l, A3l = gnm_random_hyperg(n, 0.35, 0.05)
p = (A2, A3, zeros(n), π/4, π/4)

true_monomial = Vector{Int64}[] # ids for true monomials
push!(true_monomial, [40, 53, 46])
push!(true_monomial, [53, 58, 65, 46, 60])
push!(true_monomial, [40, 47, 53, 58, 38])
push!(true_monomial, [40, 30, 61])
push!(true_monomial, [2, 23, 32, 47, 12, 22])
push!(true_monomial, [23, 40, 58, 57])
push!(true_monomial, [2, 13, 23, 65, 52])
push!(true_monomial, [13, 23, 47, 62, 21, 39])
push!(true_monomial, [58, 20, 25])
push!(true_monomial, [13, 53, 7, 17, 37, 49])

leak_monomial = Vector{Int64}[] # ids for "leaking" triadic → pairwise monomials
push!(leak_monomial, [65])
push!(leak_monomial, [40, 62])
push!(leak_monomial, [32, 62])
push!(leak_monomial, [23, 62, 58, 65])
push!(leak_monomial, [65, 13])
push!(leak_monomial, [53, 65])
push!(leak_monomial, [47])
push!(leak_monomial, [32, 65])
push!(leak_monomial, [13, 23, 32])
push!(leak_monomial, [2, 40, 32, 58, 47])

ooi = [2,3]
dmax = 2
levels = [cdf(Normal(), σ) - cdf(Normal(), -σ) for σ in [1.0, 2.0, 3.0]]
εs = round.([d * 10. ^ exp for exp in -2:0 for d in 1:9]; digits=2)

# build X
T = 300
X = (rand(T, n) .- 0.5)

# build monomial library, d
Theta = get_theta(X, dmax)
_, M = size(Theta)
d = get_d(n, dmax)

# figure set-up
pt = 4 ÷ 3; inch = 96
fig = Figure(size=(3.3inch, 4inch), fontsize=10pt)
ax1 = Axis(fig[1,1], xgridvisible=false, ygridvisible=false, yscale=log10)
ax2 = Axis(fig[2,1], xgridvisible=false, ygridvisible=false, yscale=log10)

# LOW NOISE SETTING
Y = f_kuramoto_3rd(X, p...) + 0.1*randn(size(X))

# inference with Bayes-THIS
opts = SBOpts(verbosity=0, nitr=1000, free_basis=[1])
ctrls = SBCtrlSettings(beta_update_frequency=3)
Ainf, coeff, out, _ = this_bayes(X, Y, ooi, dmax; opts=opts, ctrls=ctrls)

# print F1s
println("Low noise:")
display(F1_filter_by_CI(out, Theta, coeff, A2l, A3l, levels))

sds = conditional_posterior_sds(out, Theta)

coeffs_to_plot = Float64[]
sds_to_plot = Float64[]
true_to_plot = Bool[]
leak_to_plot = Bool[]
pwise_to_plot = Bool[]
for j in 1:n
    v = valid_monomial(j, n, dmax)
    t = true_monomial[j]
    l = leak_monomial[j]
    for i in v
        if (coeff[i, j] != 0.0)
            # plot valid monomials with non-zero coefficients only
            push!(coeffs_to_plot, coeff[i,j])
            push!(sds_to_plot, sds[i,j])

            if i in t
                # true monomial - true positives
                push!(true_to_plot, true)
                push!(leak_to_plot, false)
            else
                # false positives
                push!(true_to_plot, false)
                if i in l
                    # "leaking" pairwise interaction
                    push!(leak_to_plot, true)
                else
                    push!(leak_to_plot, false)
                end
            end

            if length(d[i, :][d[i, :] .!= 0]) == 1
                # pairwise interactions
                push!(pwise_to_plot, true)
            else
                # triadic interactions
                push!(pwise_to_plot, false)
            end
        end
    end
end

# plot panel (a)
xs = range(minimum(sds_to_plot), maximum(sds_to_plot), 100)
maxy = maximum(abs.(coeffs_to_plot))
xlims!(ax1, extrema(xs))
ylims!(ax1, nothing, maxy)

# color decision space
band!(ax1, xs, xs, maxy*ones(100), color=psblues[1], alpha=0.3)
band!(ax1, xs, 2*xs, maxy*ones(100), color=psblues[2], alpha=0.3)
band!(ax1, xs, 3*xs, maxy*ones(100), color=psblues[3], alpha=0.3)
lines!(ax1, xs[1:87], xs[1:87], color=:black, linestyle=:dash, linewidth=1.0)
text!(ax1, 0.89, 0.655, text="0.683", space=:relative, fontsize=8pt, rotation=pi/48, align=(:left, :center))
lines!(ax1, xs[1:87], 2*xs[1:87], color=:black, linestyle=:dash, linewidth=1.0)
text!(ax1, 0.89, 0.75, text="0.954", space=:relative, fontsize=8pt, rotation=pi/48, align=(:left, :center))
lines!(ax1, xs[1:87], 3*xs[1:87], color=:black, linestyle=:dash, linewidth=1.0)
text!(ax1, 0.89, 0.815, text="0.997", space=:relative, fontsize=8pt, rotation=pi/48, align=(:left, :center))

# plot false pairwise positives
scatter!(ax1, sds_to_plot[.!true_to_plot .& (pwise_to_plot .& .!leak_to_plot)], abs.(coeffs_to_plot[.!true_to_plot .& (pwise_to_plot .& .!leak_to_plot)]), color=falseorg, alpha=0.6, markersize=8)
# plot false triadic positives
scatter!(ax1, sds_to_plot[.!true_to_plot .& .!pwise_to_plot], abs.(coeffs_to_plot[.!true_to_plot .& .!pwise_to_plot]), color=falseorg, alpha=0.6, marker=:utriangle, markersize=8)
# plot leaking triadic → pairwise false positives
scatter!(ax1, sds_to_plot[leak_to_plot], abs.(coeffs_to_plot[leak_to_plot]), color=leakcyan, alpha=0.7, markersize=8)
# plot true pairwise positives
scatter!(ax1, sds_to_plot[true_to_plot .& pwise_to_plot], abs.(coeffs_to_plot[true_to_plot .& pwise_to_plot]), color=pwblue, alpha=0.7, markersize=10)
# plot true triadic positivies
scatter!(ax1, sds_to_plot[true_to_plot .& .!pwise_to_plot], abs.(coeffs_to_plot[true_to_plot .& .!pwise_to_plot]), color=triteal, alpha=1.0, marker=:utriangle, markersize=10)

# HIGH NOISE SETTING
Y = f_kuramoto_3rd(X, p...) + 0.5*randn(size(X))

# inference with Bayes-THIS
opts = SBOpts(verbosity=0, nitr=1000, free_basis=[1])
ctrls = SBCtrlSettings(beta_update_frequency=3)
Ainf, coeff, out, _ = this_bayes(X, Y, ooi, dmax; opts=opts, ctrls=ctrls)

# print F1s
println("High noise:")
display(F1_filter_by_CI(out, Theta, coeff, A2l, A3l, levels))

sds = conditional_posterior_sds(out, Theta)

coeffs_to_plot = Float64[]
sds_to_plot = Float64[]
true_to_plot = Bool[]
leak_to_plot = Bool[]
pwise_to_plot = Bool[]
for j in 1:n
    v = valid_monomial(j, n, dmax)
    t = true_monomial[j]
    l = leak_monomial[j]
    for i in v
        if (coeff[i, j] != 0.0)
            # plot valid monomials with non-zero coefficients only
            push!(coeffs_to_plot, coeff[i,j])
            push!(sds_to_plot, sds[i,j])

            if i in t
                # true monomial - true positives
                push!(true_to_plot, true)
                push!(leak_to_plot, false)
            else
                # false positives
                push!(true_to_plot, false)
                if i in l
                    # "leaking" pairwise interaction
                    push!(leak_to_plot, true)
                else
                    push!(leak_to_plot, false)
                end
            end

            if length(d[i, :][d[i, :] .!= 0]) == 1
                # pairwise interactions
                push!(pwise_to_plot, true)
            else
                # triadic interactions
                push!(pwise_to_plot, false)
            end
        end
    end
end

# plot panel (b)
xs = range(minimum(sds_to_plot), maximum(sds_to_plot), 100)
maxy = maximum(abs.(coeffs_to_plot))
xlims!(ax2, extrema(xs))
ylims!(ax2, nothing, maxy)

# color decision space 
band!(ax2, xs, xs, maxy*ones(100), color=psblues[1], alpha=0.3)
band!(ax2, xs, 2*xs, maxy*ones(100), color=psblues[2], alpha=0.3)
band!(ax2, xs, 3*xs, maxy*ones(100), color=psblues[3], alpha=0.3)
lines!(ax2, xs, xs, color=:black, linestyle=:dash, linewidth=1.0)
lines!(ax2, xs, 2*xs, color=:black, linestyle=:dash, linewidth=1.0)
lines!(ax2, xs, 3*xs, color=:black, linestyle=:dash, linewidth=1.0)

# plot false pairwise positives
sc1 = scatter!(ax2, sds_to_plot[.!true_to_plot .& (pwise_to_plot .& .!leak_to_plot)], abs.(coeffs_to_plot[.!true_to_plot .& (pwise_to_plot .& .!leak_to_plot)]), color=falseorg, alpha=0.6, markersize=8)
# plot false triadic positives
sc2 = scatter!(ax2, sds_to_plot[.!true_to_plot .& .!pwise_to_plot], abs.(coeffs_to_plot[.!true_to_plot .& .!pwise_to_plot]), color=falseorg, alpha=0.6, marker=:utriangle, markersize=8)
# plot leaking triadic → pairwise false positives
sc3 = scatter!(ax2, sds_to_plot[leak_to_plot], abs.(coeffs_to_plot[leak_to_plot]), color=leakcyan, alpha=0.7, markersize=8)
# plot true pairwise positives
sc4 = scatter!(ax2, sds_to_plot[true_to_plot .& pwise_to_plot], abs.(coeffs_to_plot[true_to_plot .& pwise_to_plot]), color=pwblue, alpha=0.7, markersize=10)
# plot true triadic positives
sc5 = scatter!(ax2, sds_to_plot[true_to_plot .& .!pwise_to_plot], abs.(coeffs_to_plot[true_to_plot .& .!pwise_to_plot]), color=triteal, alpha=1.0, marker=:utriangle, markersize=10)

# finishing touches
ax1.title = "(a) Low noise (σ = 0.1)"
ax2.title = "(b) High noise (σ = 0.5)"

ax2.xlabel = rich("Conditional posterior s.d. σ", subscript("m | –m"))
ax1.ylabel = "Coefficient magnitude |ξₘ|"
ax2.ylabel = "Coefficient magnitude |ξₘ|"

axislegend(ax2, [sc4, sc5, sc3, sc2, sc1], ["Pairwise TP", "Triadic TP", "Pairwise FP (from Δ)", "Triadic FP", "Pairwise FP"], nbanks=2, labelsize=8pt, rowgap=-7, padding=(0,4,-2,-2), position=:rb, patchlabelgap=1)

save(joinpath(@__DIR__, "..", "figs", "decision-space.png"), fig, dpi=300)
display(fig)