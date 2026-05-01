using DelimitedFiles, CairoMakie
import Statistics: mean, median, quantile
import ColorSchemes: cmr_ember, cmr_prinsenvlag 

include("performance-measures.jl")

N_array = 60:5:600
σ_array = range(0.05, 1.0, 20)
n_itr = 300
λs = [round.([d * 10. ^ exp for exp in [-2, -1] for d in 1:9]; digits=2); 1.0]

# read in and package results
results_dir = "out/robustness/"
date = "2026-03-02"

# bayes_aurocs = zeros(Float64, 3, n_itr, length(N_array), length(σ_array))
# bayes_auprcs = zeros(Float64, 3, n_itr, length(N_array), length(σ_array))
# this_aurocs  = zeros(Float64, 3, length(λs), n_itr, length(N_array), length(σ_array))
# this_auprcs  = zeros(Float64, 3, length(λs), n_itr, length(N_array), length(σ_array))

# for (j, _) in enumerate(σ_array)
#     tmp_bayes_aurocs = readdlm(joinpath(results_dir, "bayes_aurocs_$(j)-$(date).txt"))
#     tmp_bayes_aurocs = reshape(tmp_bayes_aurocs, 3, n_itr, length(N_array))
#     bayes_aurocs[:, :, :, j] .= tmp_bayes_aurocs

#     tmp_bayes_auprcs = readdlm(joinpath(results_dir, "bayes_auprcs_$(j)-$(date).txt"))
#     tmp_bayes_auprcs = reshape(tmp_bayes_auprcs, 3, n_itr, length(N_array))
#     bayes_auprcs[:, :, :, j] .= tmp_bayes_auprcs

#     tmp_this_aurocs = readdlm(joinpath(results_dir, "this_aurocs_$(j)-$(date).txt"))
#     tmp_this_aurocs = reshape(tmp_this_aurocs, 3, length(λs), n_itr, length(N_array))
#     this_aurocs[:, :, :, :, j] .= tmp_this_aurocs

#     tmp_this_auprcs = readdlm(joinpath(results_dir, "this_auprcs_$(j)-$(date).txt"))
#     tmp_this_auprcs = reshape(tmp_this_auprcs, 3, length(λs), n_itr, length(N_array))
#     this_auprcs[:, :, :, :, j] .= tmp_this_auprcs
# end

# writedlm(joinpath(results_dir, "bayes_aurocs.txt"), bayes_aurocs)
# writedlm(joinpath(results_dir, "bayes_auprcs.txt"), bayes_auprcs)
# writedlm(joinpath(results_dir, "this_aurocs.txt"), this_aurocs)
# writedlm(joinpath(results_dir, "this_auprcs.txt"), this_auprcs)

bayes_aurocs = readdlm(joinpath(results_dir, "bayes_aurocs.txt"))
bayes_aurocs = reshape(bayes_aurocs, 3, n_itr, length(N_array), length(σ_array));
bayes_auprcs = readdlm(joinpath(results_dir, "bayes_auprcs.txt"))
bayes_auprcs = reshape(bayes_auprcs, 3, n_itr, length(N_array), length(σ_array));
this_aurocs  = readdlm(joinpath(results_dir, "this_aurocs.txt"))
this_aurocs  = reshape(this_aurocs, 3, length(λs), n_itr, length(N_array), length(σ_array));
this_auprcs  = readdlm(joinpath(results_dir, "this_auprcs.txt"))
this_auprcs  = reshape(this_auprcs, 3, length(λs), n_itr, length(N_array), length(σ_array));

# figure set-up
pt = 4 ÷ 3; inch = 96; cm = inch/2.54
fig = Figure(size=(17cm, 10.75cm), fontsize=10pt)
axs = [Axis(fig[row, col], xlabel="N", xlabelpadding=2.0, ylabel="σ", limits=(nothing, (0.05, 1.0)), yticks=[0.05, 0.5, 1.0], ytickformat="{:.2f}") for row in 2:5, col in 2:4]

hideydecorations!(axs[1, 2])
hideydecorations!(axs[3, 2])
hideydecorations!(axs[1, 3])
hideydecorations!(axs[3, 3])
hideydecorations!(axs[2, 3])
hideydecorations!(axs[4, 3])
hidedecorations!(axs[2, 1])
hidedecorations!(axs[4, 1])
hidespines!(axs[2, 1])
hidespines!(axs[4, 1])

axs[2,1].xgridvisible=false
axs[2,1].ygridvisible=false
axs[4,1].xgridvisible=false
axs[4,1].ygridvisible=false

auc = mean(bayes_aurocs[1, :, :, :], dims=1)[1, :, :]
this_auc_1 = mean(this_aurocs[1, 10, :, :, :], dims=1)[1, :, :] # λ = 0.1
this_auc_2 = mean(this_aurocs[1, 19, :, :, :], dims=1)[1, :, :] # λ = 1.0

tri_auc = mean(bayes_auprcs[3, :, :, :], dims=1)[1, :, :]
this_tri_auc_1 = mean(this_auprcs[3, 10, :, :, :], dims=1)[1, :, :] # λ = 0.1
this_tri_auc_2 = mean(this_auprcs[3, 19, :, :, :], dims=1)[1, :, :] # λ = 1.0

# crange_auc = extrema(vcat(auc, this_auc_1, this_auc_2))
crange_auc = (0.0, 1.0)
heatmap!(axs[1,1], N_array, σ_array, auc, colormap=:cmr_ember, colorrange=crange_auc)
heatmap!(axs[1,2], N_array, σ_array, this_auc_1, colormap=:cmr_ember, colorrange=crange_auc)
heatmap!(axs[1,3], N_array, σ_array, this_auc_2, colormap=:cmr_ember, colorrange=crange_auc)

mean_diff_1 = auc .- this_auc_1
mean_diff_2 = auc .- this_auc_2

cmin, cmax = extrema(vcat(mean_diff_1, mean_diff_2))
crange = (-maximum(abs.([cmin, cmax])), maximum(abs.([cmin, cmax])))
heatmap!(axs[2,2], N_array, σ_array, mean_diff_1, colormap=:cmr_prinsenvlag, colorrange=crange)
heatmap!(axs[2,3], N_array, σ_array, mean_diff_2, colormap=:cmr_prinsenvlag, colorrange=crange)

# crange_tri_auc = extrema(vcat(tri_auc, this_tri_auc_1, this_tri_auc_2))
crange_tri_auc = (0.0, 1.0)
heatmap!(axs[3,1], N_array, σ_array, tri_auc, colormap=:cmr_ember, colorrange=crange_tri_auc)
heatmap!(axs[3,2], N_array, σ_array, this_tri_auc_1, colormap=:cmr_ember, colorrange=crange_tri_auc)
heatmap!(axs[3,3], N_array, σ_array, this_tri_auc_2, colormap=:cmr_ember, colorrange=crange_tri_auc)

mean_tri_diff_1 = tri_auc .- this_tri_auc_1
mean_tri_diff_2 = tri_auc .- this_tri_auc_2

tri_cmin, tri_cmax = extrema(vcat(mean_tri_diff_1, mean_tri_diff_2))
crange_tri = (-maximum(abs.([tri_cmin, tri_cmax])), maximum(abs.([tri_cmin, tri_cmax])))
heatmap!(axs[4,2], N_array, σ_array, mean_tri_diff_1, colormap=:cmr_prinsenvlag, colorrange=crange_tri)
heatmap!(axs[4,3], N_array, σ_array, mean_tri_diff_2, colormap=:cmr_prinsenvlag, colorrange=crange_tri)

# finishing touches
Label(fig[1,2], "Bayes-THIS", font=:bold, tellwidth=false, padding=(0,0,-5,0))
Label(fig[1,3], "THIS (λ = 0.1)", font=:bold, tellwidth=false, padding=(0,0,-5,0))
Label(fig[1,4], "THIS (λ = 1.0)", font=:bold, tellwidth=false, padding=(0,0,-5,0))

Label(fig[2:3,1], "Pairwise + triadic interactions", tellheight=false, font=:bold, rotation=pi/2, padding=(0,-5,-25,0))
Label(fig[4:5,1], "Triadic interactions", tellheight=false, font=:bold, rotation=pi/2, padding=(0,-5,-25,0))

Box(fig[2:3, 2:5, Makie.GridLayoutBase.Outer()], color=:transparent, alignmode=Outside(-5,-10,-10,-10), cornerradius=4, strokewidth=1)
Box(fig[4:5, 2:5, Makie.GridLayoutBase.Outer()], color=:transparent, alignmode=Outside(-5,-10,-10,-10), cornerradius=4, strokewidth=1)

Colorbar(fig[2,5], label="AUROC", colormap=:cmr_ember, colorrange=crange_auc, ticks=range(crange_auc[1], crange_auc[2], 3), tickformat="{:.2f}")
Colorbar(fig[3,5], label="Mean difference", colormap=:cmr_prinsenvlag, colorrange=crange)

Colorbar(fig[4,5], label="AUPRC", colormap=:cmr_ember, colorrange=crange_tri_auc, ticks=range(crange_tri_auc[1], crange_tri_auc[2], 3), tickformat="{:.2f}")
Colorbar(fig[5,5], label="Mean difference", colormap=:cmr_prinsenvlag, colorrange=crange_tri)

rowsize!(fig.layout, 2, Aspect(2, 0.3))
rowsize!(fig.layout, 3, Aspect(3, 0.3))
rowsize!(fig.layout, 4, Aspect(2, 0.3))
rowsize!(fig.layout, 5, Aspect(3, 0.3))
rowgap!(fig.layout, 2, 10.0)
rowgap!(fig.layout, 3, 25.0)
rowgap!(fig.layout, 4, 10.0)
colgap!(fig.layout, 2, -34.0)
colgap!(fig.layout, 3, 10.0)
colgap!(fig.layout, 4, 10.0)

text!(axs[1,1], 590, 0.1, text="(a)", font=:bold, align=(:right, :bottom))
text!(axs[1,2], 590, 0.1, text="(b)", font=:bold, align=(:right, :bottom))
text!(axs[1,3], 590, 0.1, text="(c)", font=:bold, align=(:right, :bottom))
text!(axs[2,2], 590, 0.1, text="(d)", font=:bold, align=(:right, :bottom))
text!(axs[2,3], 590, 0.1, text="(e)", font=:bold, align=(:right, :bottom))
text!(axs[3,1], 590, 0.1, text="(f)", font=:bold, align=(:right, :bottom))
text!(axs[3,2], 590, 0.1, text="(g)", font=:bold, align=(:right, :bottom))
text!(axs[3,3], 590, 0.1, text="(h)", font=:bold, align=(:right, :bottom))
text!(axs[4,2], 590, 0.1, text="(i)", font=:bold, align=(:right, :bottom))
text!(axs[4,3], 590, 0.1, text="(j)", font=:bold, align=(:right, :bottom))

save("figs/compare-robustness-sweeps.png", fig, dpi=300)
fig