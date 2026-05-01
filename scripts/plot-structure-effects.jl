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
F1s = readdlm("out/node-swap/node-swap-F1s-fixed-noise.txt"); F1s = reshape(F1s, (3, n_graphs, length(n_swaps), length(alphas)))
pres = readdlm("out/node-swap/node-swap-pres-fixed-noise.txt"); pres = reshape(pres, (3, n_graphs, length(n_swaps), length(alphas)))
recs = readdlm("out/node-swap/node-swap-recs-fixed-noise.txt"); recs = reshape(recs, (3, n_graphs, length(n_swaps), length(alphas)))
auprcs = readdlm("out/node-swap/node-swap-auprcs-fixed-noise.txt"); auprcs = reshape(auprcs, (3, n_graphs, length(n_swaps), length(alphas)))
aurocs = readdlm("out/node-swap/node-swap-aurocs-fixed-noise.txt"); aurocs = reshape(aurocs, (3, n_graphs, length(n_swaps), length(alphas)))
lyaps = readdlm("out/node-swap/node-swap-lyaps-fixed-noise.txt"); lyaps = reshape(lyaps, (n, n_graphs, length(n_swaps), length(alphas)))
lyap2 = lyaps[2, :, :, :];
betas = readdlm("out/node-swap/node-swap-betas-fixed-noise.txt"); betas = reshape(betas, (n, n_graphs, length(n_swaps), length(alphas)))
deg_corr = readdlm("out/node-swap/node-swap-deg-corr-fixed-noise.txt"); deg_corr = reshape(deg_corr, n_graphs, length(n_swaps))

avg_auprcs = mean(auprcs, dims=2)[:, 1, :, :]
avg_aurocs = mean(aurocs, dims=2)[:, 1, :, :]
avg_F1s = mean(F1s, dims=2)[:, 1, :, :]

pt = 4 ÷ 3; inch = 96
fig = Figure(size=(3.35inch, 3.7inch), fontsize=10pt)
ax1 = Axis(fig[1,1], ylabel="Pairwise F1", ylabelpadding=8, xgridvisible=false, ygridvisible=false)
# for (j, _) in Iterators.reverse(enumerate(alphas))
#     scatter!(ax1, vec(deg_corr), vec(F1s[2, :, :, j]), color=(alpha_palette[j], 0.7))
# end
scatter!(ax1, vec(deg_corr), vec(F1s[2, :, :, 2]), color=(alpha_palette[2], 0.8))
hidexdecorations!(ax1)

axinset = Axis(fig[1,1], width=Relative(0.4), height=Relative(0.45), halign=0.94, valign=0.09, limits=(nothing, nothing, nothing, 1.0), ylabel="Pairwise F1", yticks=0.7:0.1:1.0, xgridvisible=false, ygridvisible=false)
for (j, _) in Iterators.reverse(enumerate(alphas))
    scatter!(axinset, vec(deg_corr), vec(F1s[2, :, :, j]), color=(alpha_palette[j], 0.5), markersize=6)
end
linkyaxes!(ax1, axinset)
hidexdecorations!(axinset)

ax2 = Axis(fig[2,1], xlabel="Cross-order degree correlation", ylabel="Pairwise AUPRC", ylabelpadding=8, xgridvisible=false, ygridvisible=false)
for j in length(alphas):-1:1
    scatter!(ax2, vec(deg_corr), vec(auprcs[2, :, :, j]), color=(alpha_palette[j], 0.7))
end

Colorbar(fig[1,2], width=8.0, limits=(-1.5, 3.5), colormap=cgrad(alpha_palette, 5, categorical=:true), ticks=log2.(alphas), label="log₂(c)", labelpadding=1, tickformat="{:d}", vertical=true)
Colorbar(fig[2,2], width=8.0, limits=(-1.5, 3.5), colormap=cgrad(alpha_palette, 5, categorical=:true), ticks=log2.(alphas), label="log₂(c)", labelpadding=1, tickformat="{:d}", vertical=true)

# Colorbar(fig.layout[2,1], limits=(-1.5, 3.5), alignmode=Outside(3), colormap=cgrad(blues, 5, categorical=:true), ticks=log2.(alphas), tellwidth=false, tellheight=false, flipaxis=true, height=Relative(0.35), width=Relative(0.6), halign=:right, valign=:bottom, label="log₂(c)", labelpadding=1, tickformat="{:d}", vertical=false)

text!(ax1, 0.05, 0.94, text="(a)", font=:bold, align=(:center, :center), space=:relative)
text!(axinset, 0.12, 0.88, text="(b)", font=:bold, align=(:center, :center), space=:relative)
text!(ax2, 0.05, 0.94, text="(c)", font=:bold, align=(:center, :center), space=:relative)

colsize!(fig.layout, 1, Aspect(1, 1.6))
# colsize!(fig.layout, 2, Aspect(1, 0.7))
rowgap!(fig.layout, 1, 10.0)
colgap!(fig.layout, 1, 5.0)
Label(fig[3,1], "⟵ Increasing num. node swaps", valign=:top, padding=(0,0,-10,-10))

save("figs/pairwise-inference-vs-codc.png", fig, dpi=300)
fig