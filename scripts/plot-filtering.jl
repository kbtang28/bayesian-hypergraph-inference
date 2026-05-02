using CairoMakie, DelimitedFiles, Distributions

import Statistics: mean, median, quantile
import ColorSchemes: cmr_prinsenvlag, get
psblues = get(cmr_prinsenvlag, range(0.7, 0.9, 3))
psoranges = get(cmr_prinsenvlag, range(0.0, 0.3, 6))[end:-1:1]

# settings from experiment
τs = round.([d * 10. ^ exp for exp in -2:0 for d in 1:9]; digits=2)
levels = [cdf(Normal(), σ) - cdf(Normal(), -σ) for σ in [1.0, 2.0, 3.0]]
experiment_settings = [(40:2:400, 0.1), (40:5:500, 0.5)]
n_itr = 300

# figure set-up
pt = 4 ÷ 3; inch = 96
fig = Figure(size=(6.6inch, 3.0inch), fontsize=8pt)
axs = [Axis(fig[row, col], xgridvisible=false, ygridvisible=false, ylabel="F1", limits=(nothing, (0.0, 1.0))) for row in 1:2, col in 1:2]
linkyaxes!(vec(axs))
hideydecorations!(axs[1,2])
hideydecorations!(axs[2,2])
axs[1,1].title = "Pairwise + triadic"
axs[1,2].title = "Triadic"
axs[2,1].xlabel = "N"
axs[2,2].xlabel = "N"

τs_to_plot = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

# low noise settings
N_array, σ = experiment_settings[1]
xlims!(axs[1,1], N_array[1], N_array[end])
xlims!(axs[1,2], N_array[1], N_array[end])

results_dir = joinpath(@__DIR__, "..", "out", "filtering")
F1s_coeff_mags = readdlm(joinpath(results_dir, "coeff-mags-$(σ)-F1s.txt")); F1s_coeff_mags = reshape(F1s_coeff_mags, 3, length(τs), n_itr, length(N_array))
F1s_coeff_CIs = readdlm(joinpath(results_dir, "coeff-CIs-$(σ)-F1s.txt")); F1s_coeff_CIs = reshape(F1s_coeff_CIs, 3, length(levels), n_itr, length(N_array))

for (col, i) in enumerate([1,3])
    for (l, τ) in enumerate(τs_to_plot)
        j = findfirst(τs .== τ)
        mean_F1 = vec(mean(F1s_coeff_mags[i, j, :, :], dims=1))
        lines!(axs[1, col], N_array, mean_F1, color=psoranges[l], linewidth=1.5, linestyle=(:dash, :dense))
    end
end
for (col, i) in enumerate([1,3])
    for (j, level) in enumerate(levels)
        mean_F1 = vec(mean(F1s_coeff_CIs[i, j, :, :], dims=1))
        lines!(axs[1, col], N_array, mean_F1, color=psblues[j], linewidth=3.0, alpha=0.9)
    end
end

# high noise settings
N_array, σ = experiment_settings[2]
xlims!(axs[2,1], N_array[1], N_array[end])
xlims!(axs[2,2], N_array[1], N_array[end])

F1s_coeff_mags = readdlm(joinpath(results_dir, "coeff-mags-$(σ)-F1s.txt")); F1s_coeff_mags = reshape(F1s_coeff_mags, 3, length(τs), n_itr, length(N_array))
F1s_coeff_CIs = readdlm(joinpath(results_dir, "coeff-CIs-$(σ)-F1s.txt")); F1s_coeff_CIs = reshape(F1s_coeff_CIs, 3, length(levels), n_itr, length(N_array))

for (col, i) in enumerate([1,3])
    for (l, τ) in enumerate(τs_to_plot)
        j = findfirst(τs .== τ)
        mean_F1 = vec(mean(F1s_coeff_mags[i, j, :, :], dims=1))
        lines!(axs[2, col], N_array, mean_F1, color=psoranges[l], linewidth=1.5, linestyle=(:dash, :dense))
    end
end
for (col, i) in enumerate([1,3])
    for (j, level) in enumerate(levels)
        mean_F1 = vec(mean(F1s_coeff_CIs[i, j, :, :], dims=1))
        lines!(axs[2, col], N_array, mean_F1, color=psblues[j], linewidth=3.0, alpha=0.9)
    end
end

# finishing touches
Label(fig[1, 0], "Low noise", font=:bold, rotation=pi/2, tellheight=false)
Label(fig[2, 0], "High noise", font=:bold, rotation=pi/2, tellheight=false)

# pts = Point2f.(range(0,1; length=80), 0.5)
# grad_dash = LineElement(points=pts, color=range(0.3, 0.0; length=80), linecolorrange=(0, 1.0), colormap=cmr_prinsenvlag, linewidth=3.0)
# grad_line = LineElement(points=pts, color=range(0.7, 0.9; length=80), linecolorrange=(0, 1.0), colormap=cmr_prinsenvlag, linewidth=3.0)

# Legend(fig[3, 1:2], [grad_dash, grad_line], ["coefficient magnitude", "conditional posterior CI"], orientation=:horizontal, tellwidth=false, tellheight=false, padding=(5.0, 5.0, 0.0, 0.0))

Colorbar(fig[3,1], size=8, label = "Coefficient magnitude τ", colormap=cgrad(psoranges, 6, categorical=true), vertical=false, ticks=(range(1/12, 11/12, 6), string.(τs_to_plot)), flipaxis=false)
Colorbar(fig[3,2], size=8, label = "γ-credible interval", colormap=cgrad(psblues, 3, categorical=true), vertical=false, ticks=(range(1/6, 5/6, 3), string.(round.(levels, digits=3))), flipaxis=false)

text!(axs[1,1], 0.035, 0.9, text="(a)", font=:bold, align=(:center, :center), space=:relative)
text!(axs[1,2], 0.035, 0.9, text="(b)", font=:bold, align=(:center, :center), space=:relative)
text!(axs[2,1], 0.035, 0.9, text="(c)", font=:bold, align=(:center, :center), space=:relative)
text!(axs[2,2], 0.035, 0.9, text="(d)", font=:bold, align=(:center, :center), space=:relative)

colgap!(fig.layout, 1, 5.0)
colsize!(fig.layout, 1, Aspect(1, 4.0))
colsize!(fig.layout, 2, Aspect(1, 4.0))
rowgap!(fig.layout, 2, 6.0)
resize_to_layout!(fig)
save(joinpath(@__DIR__, "..", "figs", "compare-filtering.png"), fig)
display(fig)