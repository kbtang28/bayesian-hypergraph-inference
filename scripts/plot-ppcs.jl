using CairoMakie, DelimitedFiles, Random

import ColorSchemes: cmr_ember, get, ColorScheme
pv_blue = "#5bb4e9"
pv_palette = ColorScheme(get(cmr_ember, Iterators.reverse(range(0.0, 1.0, 100))))

Random.seed!(1)

# top row - scattered data, Gaussian noise on analytical derivatives
ρs = 10 .^ range(-2.0, 0.0, 11)
σs = [0.05, 0.2]
nlevels = 3
nitr = 100
lowclip = 0.05
file_prefix = "scatter"
timestamp = "2026-05-02"
xlabel = "sampling box size"

# read in data
results_dir = joinpath(@__DIR__, "..", "out", "ppc")
pvals = readdlm(joinpath(results_dir, file_prefix*"-pvals-$(timestamp).txt")); pvals = reshape(pvals, nitr, length(ρs), length(σs))
aucs = readdlm(joinpath(results_dir, file_prefix*"-aucs-$(timestamp).txt")); aucs = reshape(aucs, nitr, length(ρs), length(σs))
auc3s = readdlm(joinpath(results_dir, file_prefix*"-auc3s-$(timestamp).txt")); auc3s = reshape(auc3s, nitr, length(ρs), length(σs))
F1s = readdlm(joinpath(results_dir, file_prefix*"-F1s-$(timestamp).txt")); F1s = reshape(F1s, 3, nlevels, nitr, length(ρs), length(σs))

# figure set-up
pt = 4 ÷ 3; inch = 96
fig = Figure(size=(3.3inch, 3.5inch), fontsize=10pt)
axs = [Axis(fig[row, col], xgridvisible=false, ygridvisible=false, limits=(nothing, (0.0, 1.0)), xscale=log10, xticks=LogTicks([-2, -1, 0]), yticks=0.0:0.2:1.0, ytickformat="{:.1f}") for row in 1:2, col in 1:2]
linkyaxes!(axs[1,1], axs[1,2])
linkyaxes!(axs[1,1], axs[2,1])
linkyaxes!(axs[1,1], axs[2,2])

jittered_ρ_logscale = log10.(repeat(ρs, inner=nitr)) .+ randn(length(ρs)*nitr).*0.015
jittered_ρ = 10 .^ jittered_ρ_logscale

scatter!(axs[1,1], jittered_ρ, vec(F1s[3, 3, :, :, 1]), color=vec(pvals[:, :, 1]), colorscale=log10, colormap=pv_palette, colorrange=(lowclip, 1.0), lowclip=pv_blue, markersize=5.0)
scatter!(axs[1,2], jittered_ρ, vec(F1s[3, 3, :, :, 2]), color=vec(pvals[:, :, 2]), colorscale=log10, colormap=pv_palette, colorrange=(lowclip, 1.0), lowclip=pv_blue, markersize=5.0)

Colorbar(fig[1,3], size=8, limits=(lowclip, 1.0), scale=log10, colormap=pv_palette, lowclip=pv_blue, ticklabelrotation=pi/6, label="Post. pred. p-value")

# bottom row - trajectory data, Gaussian noise on trajectory + finite difference
ρs = (10 .^ range(-2.0, 0.0, 11)) ./ 2
σs = [0.0005, 0.001]
nlevels = 3
nitr = 100
lowclip = 0.01
file_prefix = "traj-fd"
xlabel = "I.C. scale"

# read in data
pvals = readdlm(joinpath(results_dir, file_prefix*"-pvals-$(timestamp).txt")); pvals = reshape(pvals, nitr, length(ρs), length(σs))
aucs = readdlm(joinpath(results_dir, file_prefix*"-aucs-$(timestamp).txt")); aucs = reshape(aucs, nitr, length(ρs), length(σs))
auc3s = readdlm(joinpath(results_dir, file_prefix*"-auc3s-$(timestamp).txt")); auc3s = reshape(auc3s, nitr, length(ρs), length(σs))
F1s = readdlm(joinpath(results_dir, file_prefix*"-F1s-$(timestamp).txt")); F1s = reshape(F1s, 3, nlevels, nitr, length(ρs), length(σs))

jittered_ρ_logscale = log10.(repeat(ρs, inner=nitr)) .+ randn(length(ρs)*nitr).*0.015
jittered_ρ = 10 .^ jittered_ρ_logscale

scatter!(axs[2,1], jittered_ρ, vec(F1s[3, 3, :, :, 1]), color=vec(pvals[:, :, 1]), colorscale=log10, colormap=pv_palette, colorrange=(lowclip, 1.0), lowclip=pv_blue, markersize=5.0)
scatter!(axs[2,2], jittered_ρ, vec(F1s[3, 3, :, :, 2]), color=vec(pvals[:, :, 2]), colorscale=log10, colormap=pv_palette, colorrange=(lowclip, 1.0), lowclip=pv_blue, markersize=5.0)

Colorbar(fig[2,3], size=8, limits=(lowclip, 1.0), scale=log10, colormap=pv_palette, lowclip=pv_blue, ticklabelrotation=pi/6, label="Post. pred. p-value")

# finishing touches
axs[1,1].ylabel = "Triadic F1"
axs[2,1].ylabel = "Triadic F1"
axs[1,1].xlabel = "Sampling box size"
axs[1,2].xlabel = "Sampling box size"
axs[2,1].xlabel = "Init. cond. scale"
axs[2,2].xlabel = "Init. cond. scale"

axs[1,1].title = "Low noise"
axs[1,2].title = "High noise"

hideydecorations!(axs[1,2])
hideydecorations!(axs[2,2])

# row1label = Label(fig[1,0], "Scattered data", rotation=pi/2, tellheight=false, tellwidth=false, font=:bold)
# row2label = Label(fig[2,0], "Trajectory data", rotation=pi/2, tellheight=false, tellwidth=false, font=:bold)

for (ax, label) in zip(axs, ["(a)", "(c)", "(b)", "(d)"])
    text!(ax, 0.03, 0.93, text=label, font=:bold, align=(:left, :center), space=:relative)
end

colsize!(fig.layout, 1, Aspect(1, 0.95))
colsize!(fig.layout, 2, Aspect(1, 0.95))
colgap!(fig.layout, 1, 8)
colgap!(fig.layout, 2, 8)
rowgap!(fig.layout, 10)

# resize_to_layout!(fig)
save(joinpath(@__DIR__, "..", "figs", "ppc-F1s.png"), fig, dpi=500)
display(fig)