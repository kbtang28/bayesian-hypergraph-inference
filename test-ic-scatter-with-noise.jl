using Random, CairoMakie, Printf, DelimitedFiles, Dates

import ColorSchemes: seaborn_colorblind

include("gen-rand-hyperg.jl")
include("hyperg-kuramoto.jl")
include("this-og.jl")
include("this-bayes.jl")
include("performance-measures.jl")

Random.seed!(1234)

# generate hypergraph
n = 7
t2 = .4
t3 = n > 30 ? .01 : .05

A2, A3, A2l, A3l = gnm_random_hyperg(n, t2, t3)

# set up figure
inch = 96; pt = 4/3
fig = Figure(size=(3.5inch, 3.5inch), fontsize=8pt)
axs = [Axis(fig[row, col], 
            xgridvisible=false, ygridvisible=false, 
            xticks=0.0:0.2:1.0, yticks=0.0:0.2:1.0, 
            limits=((-0.05, 1.05), (-0.05, 1.05)), 
            xlabel="FPR", ylabel="TPR") for row in 1:2, col in 1:2
]

# set up log file
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMM")
logfile = open("out/test-ic-scatter-with-noise-$(timestamp).txt", "w")

# inference settings for Bayes-THIS
opts = SBOpts(nitr=1e3, verbosity=2, free_basis=[1], io_list=[logfile]) # first col of θ is "bias" - don't want to penalize fitting to data here
settings = SBSettings()
ctrls = SBCtrlSettings(beta_update_frequency=3)

# hold constant through experiment
amplitude = 1.
ooi = [2,3]
dmax = 2

# low noise, low data setting
println("--- low noise, low data ---")
println(logfile, "---- low noise, low data ----")

σ = 0.2; T = 40
X = amplitude*(rand(T,n) .- .5)
Y = f_kuramoto_3rd(X, A2, A3, zeros(n), π/4, π/4) + σ*randn(size(X))

λ = 0.6; ρ = 0.0 # from scaled experiment
this1_xxx = this(X, Y, ooi, dmax, λ, ρ, 500, with_scaling=false)
println("THIS results:")
this1_tpr, this1_fpr = my_ROC(abs.(this1_xxx[1][2]), A2l, abs.(this1_xxx[1][3]), A3l, n; verbosity=1)
println(get_auc(this1_tpr, this1_fpr))
println(my_F1(abs.(this1_xxx[1][2]), A2l, abs.(this1_xxx[1][3]), A3l, n))
lines!(axs[1,1], this1_fpr, this1_tpr, color=seaborn_colorblind[1], linewidth=1.5)

bayes1_xxx = this_bayes(X, Y, ooi, dmax; opts=opts, settings=settings)
println("Bayes-THIS results:")
bayes1_tpr, bayes1_fpr = my_ROC(abs.(bayes1_xxx[1][2]), A2l, abs.(bayes1_xxx[1][3]), A3l, n; verbosity=1)
println(get_auc(bayes1_tpr, bayes1_fpr))
println(my_F1(abs.(bayes1_xxx[1][2]), A2l, abs.(bayes1_xxx[1][3]), A3l, n))
lines!(axs[1,1], bayes1_fpr, bayes1_tpr, color=seaborn_colorblind[2], linewidth=1.5)

# low noise, high data setting
println("--- low noise, high data ---")
println(logfile, "\n---- low noise, high data ----")

σ = 0.2; T = 200
X = amplitude*(rand(T,n) .- .5)
Y = f_kuramoto_3rd(X, A2, A3, zeros(n), π/4, π/4) + σ*randn(size(X))

λ = 0.4; ρ = 0.0 # from scaled experiment
this2_xxx = this(X, Y, ooi, dmax, λ, ρ, 500, with_scaling=false)
println("THIS results:")
this2_tpr, this2_fpr = my_ROC(abs.(this2_xxx[1][2]), A2l, abs.(this2_xxx[1][3]), A3l, n; verbosity=1)
println(get_auc(this2_tpr, this2_fpr))
println(my_F1(abs.(this2_xxx[1][2]), A2l, abs.(this2_xxx[1][3]), A3l, n))
lines!(axs[1,2], this2_fpr, this2_tpr, color=seaborn_colorblind[1], linewidth=1.5)

bayes2_xxx = this_bayes(X, Y, ooi, dmax; opts=opts, settings=settings)
println("Bayes-THIS results:")
bayes2_tpr, bayes2_fpr = my_ROC(abs.(bayes2_xxx[1][2]), A2l, abs.(bayes2_xxx[1][3]), A3l, n; verbosity=1)
println(get_auc(bayes2_tpr, bayes2_fpr))
println(my_F1(abs.(bayes2_xxx[1][2]), A2l, abs.(bayes2_xxx[1][3]), A3l, n))
lines!(axs[1,2], bayes2_fpr, bayes2_tpr, color=seaborn_colorblind[2], linewidth=1.5)

# high noise, low data setting
println("--- high noise, low data ---")
println(logfile, "\n---- high noise, low data ----")

σ = 0.4; T = 40
X = amplitude*(rand(T,n) .- .5)
Y = f_kuramoto_3rd(X, A2, A3, zeros(n), π/4, π/4) + σ*randn(size(X))

λ = 1.0; ρ = 0.0 # from scaled experiment
this3_xxx = this(X, Y, ooi, dmax, λ, ρ, 500, with_scaling=false)
println("THIS results:")
this3_tpr, this3_fpr = my_ROC(abs.(this3_xxx[1][2]), A2l, abs.(this3_xxx[1][3]), A3l, n; verbosity=1)
println(get_auc(this3_tpr, this3_fpr))
println(my_F1(abs.(this3_xxx[1][2]), A2l, abs.(this3_xxx[1][3]), A3l, n))
lines!(axs[2,1], this3_fpr, this3_tpr, color=seaborn_colorblind[1], linewidth=1.5)

bayes3_xxx = this_bayes(X, Y, ooi, dmax; opts=opts, settings=settings)
println("Bayes-THIS results:")
bayes3_tpr, bayes3_fpr = my_ROC(abs.(bayes3_xxx[1][2]), A2l, abs.(bayes3_xxx[1][3]), A3l, n; verbosity=1)
println(get_auc(bayes3_tpr, bayes3_fpr))
println(my_F1(abs.(bayes3_xxx[1][2]), A2l, abs.(bayes3_xxx[1][3]), A3l, n))
lines!(axs[2,1], bayes3_fpr, bayes3_tpr, color=seaborn_colorblind[2], linewidth=1.5)

# high noise, high data setting
println("--- high noise, high data ---")
println(logfile, "\n---- high noise, high data ----")

σ = 0.8; T = 200
X = amplitude*(rand(T,n) .- .5)
Y = f_kuramoto_3rd(X, A2, A3, zeros(n), π/4, π/4) + σ*randn(size(X))

λ = 0.3; ρ = 0.0 # from scaled experiment
this4_xxx = this(X, Y, ooi, dmax, λ, ρ, 500, with_scaling=false)
println("THIS results:")
this4_tpr, this4_fpr = my_ROC(abs.(this4_xxx[1][2]), A2l, abs.(this4_xxx[1][3]), A3l, n; verbosity=1)
println(get_auc(this4_tpr, this4_fpr))
println(my_F1(abs.(this4_xxx[1][2]), A2l, abs.(this4_xxx[1][3]), A3l, n))
lines!(axs[2,2], this4_fpr, this4_tpr, color=seaborn_colorblind[1], linewidth=1.5)

bayes4_xxx = this_bayes(X, Y, ooi, dmax; opts=opts, settings=settings)
println("Bayes-THIS results:")
bayes4_tpr, bayes4_fpr = my_ROC(abs.(bayes4_xxx[1][2]), A2l, abs.(bayes4_xxx[1][3]), A3l, n; verbosity=1)
println(get_auc(bayes4_tpr, bayes4_fpr))
println(my_F1(abs.(bayes4_xxx[1][2]), A2l, abs.(bayes4_xxx[1][3]), A3l, n))
lines!(axs[2,2], bayes4_fpr, bayes4_tpr, color=seaborn_colorblind[2], linewidth=1.5)

close(logfile)

# finish formatting figure
hidedecorations!(axs[1,2])
hidexdecorations!(axs[1,1])
hideydecorations!(axs[2,2])
colsize!(fig.layout, 1, Aspect(1, 1.0))
colsize!(fig.layout, 2, Aspect(1, 1.0))
Label(fig[0, 1], "Low data", font=:bold, tellheight=true, tellwidth=false)
Label(fig[0, 2], "High data", font=:bold, tellheight=true, tellwidth=false)
Label(fig[1, 0], "Low noise", rotation=π/2, font=:bold, tellheight=false, tellwidth=false)
Label(fig[2, 0], "High noise", rotation=π/2, font=:bold, tellheight=false, tellwidth=false)
resize_to_layout!(fig)
save("figs/test-noise-data-regimes-rocs.png", fig)