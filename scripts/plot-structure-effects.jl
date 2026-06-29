using CairoMakie, DelimitedFiles

import StatsBase: mean, std, quantile
import ColorSchemes: cmr_ember, cmr_prinsenvlag, ColorScheme, get, GnBu_9
import Colors

# experiment settings
n = 15
n_graphs = 40
n_swaps = collect(0:7)
alphas = [0.5, 1.0, 2.0, 4.0, 8.0]

# colors
triteal = "#009988"
pwblue = "#0077BB"
# alpha_palette = ColorScheme([get(cmr_ember, i) for i in reverse(range(0.33, 1.0, length(alphas)))])
alpha_palette = ColorScheme([GnBu_9[i] for i in 4:8])
bb = get(cmr_prinsenvlag, 0.88)

# read in data from node swap experiment
results_dir = joinpath(@__DIR__, "..", "out", "node-swap")
timestamp = "2026-05-02"
F1s = readdlm(joinpath(results_dir, "F1s-fixed-noise-$(timestamp).txt")); F1s = reshape(F1s, (3, n_graphs, length(n_swaps), length(alphas)))
pres = readdlm(joinpath(results_dir, "pres-fixed-noise-$(timestamp).txt")); pres = reshape(pres, (3, n_graphs, length(n_swaps), length(alphas)))
recs = readdlm(joinpath(results_dir, "recs-fixed-noise-$(timestamp).txt")); recs = reshape(recs, (3, n_graphs, length(n_swaps), length(alphas)))
auprcs = readdlm(joinpath(results_dir, "auprcs-fixed-noise-$(timestamp).txt")); auprcs = reshape(auprcs, (3, n_graphs, length(n_swaps), length(alphas)))
aurocs = readdlm(joinpath(results_dir, "aurocs-fixed-noise-$(timestamp).txt")); aurocs = reshape(aurocs, (3, n_graphs, length(n_swaps), length(alphas)))
betas = readdlm(joinpath(results_dir, "betas-fixed-noise-$(timestamp).txt")); betas = reshape(betas, (n, n_graphs, length(n_swaps), length(alphas)))
deg_corr = readdlm(joinpath(results_dir, "deg-corr-fixed-noise-$(timestamp).txt")); deg_corr = reshape(deg_corr, n_graphs, length(n_swaps))

avg_auprcs = mean(auprcs, dims=2)[:, 1, :, :]
avg_aurocs = mean(aurocs, dims=2)[:, 1, :, :]
avg_F1s = mean(F1s, dims=2)[:, 1, :, :]

pt = 4 ÷ 3; inch = 96; cm = inch/2.54
fig = Figure(size=(15cm, 5.5cm), fontsize=10pt)
ax1 = Axis(fig[1,1], xlabel="Cross-order degree correlation", ylabel="Pairwise F1", ylabelpadding=8, xgridvisible=false, ygridvisible=false)
# for (j, _) in Iterators.reverse(enumerate(alphas))
#     scatter!(ax1, vec(deg_corr), vec(F1s[2, :, :, j]), color=(alpha_palette[j], 0.7))
# end
scatter!(ax1, vec(deg_corr), vec(F1s[2, :, :, 2]), color=(alpha_palette[2], 0.8))
# hidexdecorations!(ax1)

axinset = Axis(fig[1,1], width=Relative(0.4), height=Relative(0.45), halign=0.94, valign=0.09, limits=(nothing, nothing, nothing, 1.0), ylabel="Pairwise F1", yticks=0.7:0.1:1.0, xgridvisible=false, ygridvisible=false)
for (j, _) in Iterators.reverse(enumerate(alphas))
    scatter!(axinset, vec(deg_corr), vec(F1s[2, :, :, j]), color=(alpha_palette[j], 0.8), markersize=4)
end
linkyaxes!(ax1, axinset)
hidexdecorations!(axinset)

ax2 = Axis(fig[1,2], xlabel="Cross-order degree correlation", ylabel="Pairwise AUPRC", ylabelpadding=8, xgridvisible=false, ygridvisible=false)
for j in length(alphas):-1:1
    scatter!(ax2, vec(deg_corr), vec(auprcs[2, :, :, j]), color=(alpha_palette[j], 0.9))
end

Colorbar(fig[1,3], width=8.0, limits=(-1.5, 3.5), colormap=cgrad(alpha_palette, 5, categorical=:true), ticks=log2.(alphas), label="log₂(c)", labelpadding=1, tickformat="{:d}", vertical=true)

# Colorbar(fig.layout[2,1], limits=(-1.5, 3.5), alignmode=Outside(3), colormap=cgrad(blues, 5, categorical=:true), ticks=log2.(alphas), tellwidth=false, tellheight=false, flipaxis=true, height=Relative(0.35), width=Relative(0.6), halign=:right, valign=:bottom, label="log₂(c)", labelpadding=1, tickformat="{:d}", vertical=false)

text!(ax1, 0.05, 0.94, text="(a)", font=:bold, align=(:center, :center), space=:relative)
text!(axinset, 0.12, 0.88, text="(b)", font=:bold, align=(:center, :center), space=:relative)
text!(ax2, 0.05, 0.94, text="(c)", font=:bold, align=(:center, :center), space=:relative)

colsize!(fig.layout, 1, Aspect(1, 1.6))
colsize!(fig.layout, 2, Aspect(1, 1.6))
# colsize!(fig.layout, 2, Aspect(1, 0.7))
colgap!(fig.layout, 1, 8.0)
colgap!(fig.layout, 2, 5.0)
Label(fig[2,1], "⟵ Increasing num. node swaps", valign=:top, padding=(0,0,-10,-10), tellwidth=false)
Label(fig[2,2], "⟵ Increasing num. node swaps", valign=:top, padding=(0,0,-10,-10), tellwidth=false)

save(joinpath(@__DIR__, "..", "figs", "pairwise-inference-vs-codc.png"), fig, dpi=300)
fig