using DelimitedFiles, CairoMakie
import Statistics: mean, median, quantile
import ColorSchemes: cmr_ember, cmr_prinsenvlag
bblue = "#4D8DDE"
torange = "#CD6825"
tyellow = "#DAA71C"

include("this-tools.jl")

λs = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
c = 2.0
Ks = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 16.0]
n_itr = 100
M = 66
n = 10

results_dir = "out/near-sync/"
timestamp = "2026-04-13"
kappas = readdlm(joinpath(results_dir, "kappas-$(timestamp).txt"))
bayes_aurocs = readdlm(joinpath(results_dir, "bayes-aurocs-$(timestamp).txt"))
this_aurocs = readdlm(joinpath(results_dir, "this-aurocs-$(timestamp).txt"))
bayes_auprcs = readdlm(joinpath(results_dir, "bayes-auprcs-$(timestamp).txt"))
this_auprcs = readdlm(joinpath(results_dir, "this-auprcs-$(timestamp).txt"))
bayes_coeffs = readdlm(joinpath(results_dir, "bayes-coeffs-$(timestamp).txt"))
this_coeffs = readdlm(joinpath(results_dir, "this-coeffs-$(timestamp).txt"))

kappas = reshape(kappas, length(Ks))
bayes_aurocs = reshape(bayes_aurocs, 3, n_itr, length(Ks))
this_aurocs  = reshape(this_aurocs, 3, length(λs), n_itr, length(Ks))
bayes_auprcs = reshape(bayes_auprcs, 3, n_itr, length(Ks))
this_auprcs  = reshape(this_auprcs, 3, length(λs), n_itr, length(Ks))
bayes_coeffs = reshape(bayes_coeffs, M, n, n_itr, length(Ks))
this_coeffs = reshape(this_coeffs, M, n, length(λs), n_itr, length(Ks))

# figure set-up
lo_q = 0.1
hi_q = 0.9

pt = 4 ÷ 3; inch = 96; cm = inch/2.54
fig = Figure(size=(3.3inch, 3.3inch), fontsize=10pt)

ax1 = Axis(fig[1,1], xlabel="Condition number κ of Θ(X)", ylabel="Pooled AUROC", xscale=log10, xgridvisible=false, ygridvisible=false, limits=(nothing, nothing, nothing, 0.85))

median_bayes_aurocs = median(bayes_aurocs[1, :, :], dims=1)[1, :]
bayes_aurocs_lo = [quantile(bayes_aurocs[1, :, k], lo_q) for k in 1:length(Ks)]
bayes_aurocs_hi = [quantile(bayes_aurocs[1, :, k], hi_q) for k in 1:length(Ks)]

median_this_aurocs_1 = median(this_aurocs[1, 3, :, :, :], dims=1)[1, :, 1]
this_aurocs_1_lo = [quantile(this_aurocs[1, 3, :, k], lo_q) for k in 1:length(Ks)]
this_aurocs_1_hi = [quantile(this_aurocs[1, 3, :, k], hi_q) for k in 1:length(Ks)]

median_this_aurocs_2 = median(this_aurocs[1, 5, :, :, :], dims=1)[1, :, 1]
this_aurocs_2_lo = [quantile(this_aurocs[1, 5, :, k], lo_q) for k in 1:length(Ks)]
this_aurocs_2_hi = [quantile(this_aurocs[1, 5, :, k], hi_q) for k in 1:length(Ks)]

band!(ax1, kappas[1:(end-1)], this_aurocs_1_lo[1:(end-1)], this_aurocs_1_hi[1:(end-1)], alpha=0.3, color=tyellow)
sc1 = scatterlines!(ax1, kappas[1:(end-1)], median_this_aurocs_1[1:(end-1)], marker=:rect, markersize=10, color=tyellow)
band!(ax1, kappas[1:(end-1)], this_aurocs_2_lo[1:(end-1)], this_aurocs_2_hi[1:(end-1)], alpha=0.3, color=torange)
sc2 = scatterlines!(ax1, kappas[1:(end-1)], median_this_aurocs_2[1:(end-1)], marker=:utriangle, markersize=10, color=torange)
band!(ax1, kappas[1:(end-1)], bayes_aurocs_lo[1:(end-1)], bayes_aurocs_hi[1:(end-1)], alpha=0.3, color=bblue)
sc3 = scatterlines!(ax1, kappas[1:(end-1)], median_bayes_aurocs[1:(end-1)], markersize=10, color=bblue)
emptysc = scatter!(ax1, NaN, NaN, marker=:circle, markersize=0)

axislegend(ax1, [sc3, sc1, emptysc, sc2], ["B-THIS", "THIS (λ = 0.1)", "", "THIS (λ = 1.0)"], nbanks=2, rowgap=-9, colgap=8, padding=(5,5,-1,-1), position=:rt, markersize=8, linepoints=[Point2f(0.2, 0.5), Point2f(0.8, 0.5)], patchlabelgap=1)
# axislegend(ax1, [sc3, sc2, sc1], ["Bayes-THIS", "THIS (λ = 1.0)", "THIS (λ = 0.1)"], labelsize=7pt, rowgap=-9, padding=(0,4,-2,-2), position=:lb, linepoints=[Point2f(0.2, 0.5), Point2f(0.8, 0.5)], patchlabelgap=1)

gl = fig[2,1] = GridLayout()
ax11 = Axis(gl[1,1], limits=(nothing, nothing, 0.0, nothing), xgridvisible=false, ygridvisible=false, ylabelfont=:bold, ylabel="True", title="κ = $(round(kappas[4], digits=2))")
hidexdecorations!(ax11)
hideydecorations!(ax11, label=false)

density!(ax11, this_coeffs[46, 1, 5, :, 4], color=torange, alpha=0.65)
density!(ax11, bayes_coeffs[46, 1, :, 4], color=bblue, alpha=0.65)
vlines!(ax11, [0.0], color=:black, linewidth=1.0)

ax21 = Axis(gl[2,1], limits=(nothing, nothing, 0.0, nothing), xgridvisible=false, ygridvisible=false, ylabelfont=:bold, ylabel="False", xlabel="Inferred ξ")
linkxaxes!(ax11, ax21)
hideydecorations!(ax21, label=false)

density!(ax21, this_coeffs[50, 1, 5, :, 4], color=torange, alpha=0.65)
density!(ax21, bayes_coeffs[50, 1, :, 4], color=bblue, alpha=0.65)
vlines!(ax21, [0.0], color=:black, linewidth=1.0)

ax12 = Axis(gl[1,2], limits=(nothing, nothing, 0.0, nothing), xgridvisible=false, ygridvisible=false, title="κ = $(round(kappas[8], digits=2))")
hidexdecorations!(ax12)
hideydecorations!(ax12)

density!(ax12, this_coeffs[46, 1, 5, :, 8], color=torange, alpha=0.65)
density!(ax12, bayes_coeffs[46, 1, :, 8], color=bblue, alpha=0.65)
vlines!(ax12, [0.0], color=:black, linewidth=1.0)

ax22 = Axis(gl[2,2], limits=(nothing, nothing, 0.0, nothing), xgridvisible=false, ygridvisible=false, xlabel="Inferred ξ")
linkxaxes!(ax12, ax22)
hideydecorations!(ax22)

density!(ax22, this_coeffs[50, 1, 5, :, 8], color=torange, alpha=0.65)
density!(ax22, bayes_coeffs[50, 1, :, 8], color=bblue, alpha=0.65)
vlines!(ax22, [0.0], color=:black, linewidth=1.0)

rowgap!(gl, 1, 8.0)
colgap!(gl, 1, 5.0)

rowsize!(fig.layout, 2, Aspect(1, 0.4))

text!(ax1, 0.02, 0.98, text="(a)", space=:relative, font=:bold, align=(:left, :top))
text!(ax11, 0.03, 0.95, text="(b)", space=:relative, font=:bold, align=(:left, :top))
text!(ax12, 0.03, 0.95, text="(c)", space=:relative, font=:bold, align=(:left, :top))
text!(ax21, 0.03, 0.95, text="(d)", space=:relative, font=:bold, align=(:left, :top))
text!(ax22, 0.03, 0.95, text="(e)", space=:relative, font=:bold, align=(:left, :top))

save("figs/robustness-near-sync.png", fig, dpi=300)
fig