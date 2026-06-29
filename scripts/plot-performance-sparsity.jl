using DelimitedFiles, CairoMakie
import Statistics: mean, median, quantile
import ColorSchemes: cmr_ember, cmr_prinsenvlag 

n = 10
t2_array = 0.05:0.05:0.75
t3_array = 0.05:0.05:0.75

ooi = [2,3]
dmax = 2

N = 500
σ = 0.2
n_itr = 80
λs = [0.01, 0.05, 0.1, 0.5, 1.0]

# read in results
results_dir = joinpath(@__DIR__, "..", "out", "sparsity")
timestamp = "2026-05-02"

bayes_aurocs = readdlm(joinpath(results_dir, "bayes-aurocs-$(timestamp)-sparsity.txt"))
bayes_aurocs = reshape(bayes_aurocs, 3, n_itr, length(t2_array), length(t3_array))
this_aurocs  = readdlm(joinpath(results_dir, "this-aurocs-$(timestamp)-sparsity.txt"))
this_aurocs  = reshape(this_aurocs, 3, length(λs), n_itr, length(t2_array), length(t3_array))
bayes_auprcs = readdlm(joinpath(results_dir, "bayes-auprcs-$(timestamp)-sparsity.txt"))
bayes_auprcs = reshape(bayes_auprcs, 3, n_itr, length(t2_array), length(t3_array))
this_auprcs  = readdlm(joinpath(results_dir, "this-auprcs-$(timestamp)-sparsity.txt"))
this_auprcs  = reshape(this_auprcs, 3, length(λs), n_itr, length(t2_array), length(t3_array))

# figure set-up
pt = 4 ÷ 3; inch = 96; cm = inch/2.54
fig = Figure(size=(15cm, 7.2cm), fontsize=10pt)
axs = [Axis(fig[row, col], xlabel="ρ₂", xlabelpadding=2.0, xticks=(5:5:length(t2_array), string.(t2_array[5:5:end])), yticks=(5:5:length(t3_array), string.(t3_array[5:5:end])), ylabel="ρ₃", limits=(nothing, nothing), xticklabelrotation=pi/4) for row in 2:3, col in [2,3,6,7]]

hideydecorations!(axs[1, 2])
hideydecorations!(axs[1, 4])
hidedecorations!(axs[2, 1])
hidedecorations!(axs[2, 3])
hidespines!(axs[2, 1])
hidespines!(axs[2, 3])

axs[2,1].xgridvisible=false
axs[2,1].ygridvisible=false
axs[2,3].xgridvisible=false
axs[2,3].ygridvisible=false

λ_id = 3 # λ = 0.1

auc = mean(bayes_aurocs[1, :, :, :], dims=1)[1, :, :]
this_auc = mean(this_aurocs[1, λ_id, :, :, :], dims=1)[1, :, :] 
tri_auc = mean(bayes_auprcs[3, :, :, :], dims=1)[1, :, :]
this_tri_auc = mean(this_auprcs[3, λ_id, :, :, :], dims=1)[1, :, :] 

# crange_auc = extrema(vcat(auc, this_auc))
crange_auc = (0.0, 1.0)
heatmap!(axs[1,1], 1:length(t2_array), 1:length(t3_array), auc, colormap=:cmr_ember, colorrange=crange_auc)
heatmap!(axs[1,2], 1:length(t2_array), 1:length(t3_array), this_auc, colormap=:cmr_ember, colorrange=crange_auc)

mean_diff = auc .- this_auc

cmin, cmax = extrema(mean_diff)
crange = (-maximum(abs.([cmin, cmax])), maximum(abs.([cmin, cmax])))
heatmap!(axs[2,2], 1:length(t2_array), 1:length(t3_array), mean_diff, colormap=:cmr_prinsenvlag, colorrange=crange)

# crange_tri_auc = extrema(vcat(tri_auc, this_tri_auc))
crange_tri_auc = (0.0, 1.0)
heatmap!(axs[1,3], 1:length(t2_array), 1:length(t3_array), tri_auc, colormap=:cmr_ember, colorrange=crange_tri_auc)
heatmap!(axs[1,4], 1:length(t2_array), 1:length(t3_array), this_tri_auc, colormap=:cmr_ember, colorrange=crange_tri_auc)

mean_tri_diff = tri_auc .- this_tri_auc

tri_cmin, tri_cmax = extrema(mean_tri_diff)
crange_tri = (-maximum(abs.([tri_cmin, tri_cmax])), maximum(abs.([tri_cmin, tri_cmax])))
heatmap!(axs[2,4], 1:length(t2_array), 1:length(t3_array), mean_tri_diff, colormap=:cmr_prinsenvlag, colorrange=crange_tri)

# finishing touches
Label(fig[1,2], "Bayes-THIS", font=:bold, tellwidth=false, padding=(0,0,-5,0))
Label(fig[1,3], "THIS (λ = 0.1)", font=:bold, tellwidth=false, padding=(0,0,-5,0))
Label(fig[1,6], "Bayes-THIS", font=:bold, tellwidth=false, padding=(0,0,-5,0))
Label(fig[1,7], "THIS (λ = 0.1)", font=:bold, tellwidth=false, padding=(0,0,-5,0))

Label(fig[2:3,1], "Pairwise + triadic interactions", tellheight=false, font=:bold, rotation=pi/2, padding=(0,-13,-25,0))
Label(fig[2:3,5], "Triadic interactions", tellheight=false, font=:bold, rotation=pi/2, padding=(0,-13,-25,0))

Box(fig[2:3, 2:4, Makie.GridLayoutBase.Outer()], color=:transparent, alignmode=Outside(-5,-10,-10,-10), cornerradius=4, strokewidth=1)
Box(fig[2:3, 6:8, Makie.GridLayoutBase.Outer()], color=:transparent, alignmode=Outside(-5,-10,-10,-10), cornerradius=4, strokewidth=1)

Colorbar(fig[2,4], size=6, label="AUROC", colormap=:cmr_ember, colorrange=crange_auc, ticks=range(crange_auc[1], crange_auc[2], 3), tickformat="{:.2f}")
Colorbar(fig[3,4], size=6, label="Mean difference", colormap=:cmr_prinsenvlag, colorrange=crange, tickformat="{:.2f}")

Colorbar(fig[2,8], size=6, label="AUPRC", colormap=:cmr_ember, colorrange=crange_tri_auc, ticks=range(crange_tri_auc[1], crange_tri_auc[2], 3), tickformat="{:.2f}")
Colorbar(fig[3,8], size=6, label="Mean difference", colormap=:cmr_prinsenvlag, colorrange=(crange_tri), ticks=-0.02:0.02:0.02, tickformat="{:.2f}")

rowsize!(fig.layout, 2, Aspect(2, 1.0))
rowsize!(fig.layout, 3, Aspect(3, 1.0))
rowgap!(fig.layout, 2, 10.0)
colgap!(fig.layout, 2, -32.0)
colgap!(fig.layout, 3, 5.0)
colgap!(fig.layout, 6, -32.0)
colgap!(fig.layout, 7, 5.0)

text!(axs[1,1], 0.9, 0.95, text="(a)", space=:relative, font=:bold, align=(:right, :top))
text!(axs[1,2], 0.9, 0.95, text="(b)", space=:relative, font=:bold, align=(:right, :top))
text!(axs[2,2], 0.9, 0.95, text="(c)", space=:relative, font=:bold, align=(:right, :top))
text!(axs[1,3], 0.9, 0.95, text="(d)", space=:relative, font=:bold, align=(:right, :top))
text!(axs[1,4], 0.9, 0.95, text="(e)", space=:relative, font=:bold, align=(:right, :top))
text!(axs[2,4], 0.9, 0.95, text="(f)", space=:relative, font=:bold, align=(:right, :top))

save(joinpath(@__DIR__, "..", "figs", "compare-robustness-sparsity.png"), fig, dpi=300)
fig