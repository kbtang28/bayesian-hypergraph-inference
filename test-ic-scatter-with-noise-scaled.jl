using Random, CairoMakie, Printf, DelimitedFiles, Dates

import ColorSchemes: seaborn_colorblind

include("gen-rand-hyperg.jl")
include("hyperg-kuramoto.jl")
include("this-og.jl")
include("this-bayes.jl")
include("performance-measures.jl")

Random.seed!(11) # for reproducibility

# helper function
function test_inference(σ, T, nitr, SB_logfile, ROC_logfile)
    # settings for Bayes-THIS
    opts = SBOpts(nitr=1e3, free_basis=[1], verbosity=2, io_list=[SB_logfile]) # first col of θ is "bias" - don't want to penalize fitting to data here

    # settings for reconstruction
    ooi = [2,3]
    dmax = 2

    # to store results
    this_aucs  = zeros(Float64, nitr, length(λs))
    this_aucs2 = zeros(Float64, nitr, length(λs))
    this_aucs3 = zeros(Float64, nitr, length(λs))

    this_F1s  = zeros(Float64, nitr, length(λs))
    this_F1s2 = zeros(Float64, nitr, length(λs))
    this_F1s3 = zeros(Float64, nitr, length(λs))

    bayes_aucs  = zeros(Float64, nitr)
    bayes_aucs2 = zeros(Float64, nitr)
    bayes_aucs3 = zeros(Float64, nitr)

    bayes_F1s  = zeros(Float64, nitr)
    bayes_F1s2 = zeros(Float64, nitr)
    bayes_F1s3 = zeros(Float64, nitr)

    for i in 1:nitr
        println(SB_logfile, "ITERATION $(i) / $(nitr)")
        println(ROC_logfile, "** ITERATION $(i) / $(nitr) **")

        # draw random hypergraph
        A2, A3, A2l, A3l = gnm_random_hyperg(n, t2, t3)

        # generate ICs
        amplitude = 1.0 # hypercube of sidelength 1
        X = amplitude*(rand(T,n) .- .5)
        
        # generate noisy measurements of time-derivatives
        Y = f_kuramoto_3rd(X, A2, A3, zeros(n), π/4, π/4) + σ*randn(size(X))

        # inference with Bayes-THIS & measure performance
        bayes_xxx = this_bayes(X, Y, ooi, dmax; opts=opts)
        
        @printf(ROC_logfile, "Bayes-THIS ---------------------- full ")
        bayes_tpr, bayes_fpr = my_ROC(abs.(bayes_xxx[1][2]), A2l, abs.(bayes_xxx[1][3]), A3l, n; verbosity=1, io=ROC_logfile)
        bayes_aucs[i] = get_auc(bayes_tpr, bayes_fpr)
        bayes_F1s[i] = my_F1(abs.(bayes_xxx[1][2]), A2l, abs.(bayes_xxx[1][3]), A3l, n; ε=0.0)
        
        @printf(ROC_logfile, "%38s ", "pairwise")
        bayes_tpr2, bayes_fpr2 = my_ROC(abs.(bayes_xxx[1][2]), A2l, n; verbosity=1, io=ROC_logfile)
        bayes_aucs2[i] = get_auc(bayes_tpr2, bayes_fpr2)
        bayes_F1s2[i] = my_F1(abs.(bayes_xxx[1][2]), A2l, n; ε=0.0)
        
        @printf(ROC_logfile, "%38s ", "triadic")
        bayes_tpr3, bayes_fpr3 = my_ROC(abs.(bayes_xxx[1][3]), A3l, n; verbosity=1, io=ROC_logfile)
        bayes_aucs3[i] = get_auc(bayes_tpr3, bayes_fpr3)
        bayes_F1s3[i] = my_F1(abs.(bayes_xxx[1][3]), A3l, n; ε=0.0)

        # inference with original THIS & measure performance (looping over λ...)
        for (j, λ) in enumerate(λs)
            this_xxx = this(X, Y, ooi, dmax, λ, 0.0, 500, with_scaling=false) # plain STLS (not STRidge)
            
            @printf(ROC_logfile, "THIS (λ = %.2f) ----------------- full ", λ)
            this_tpr, this_fpr = my_ROC(abs.(this_xxx[1][2]), A2l, abs.(this_xxx[1][3]), A3l, n; verbosity=1, io=ROC_logfile)
            this_aucs[i, j] = get_auc(this_tpr, this_fpr)
            this_F1s[i, j] = my_F1(abs.(this_xxx[1][2]), A2l, abs.(this_xxx[1][3]), A3l, n; ε=0.0)

            @printf(ROC_logfile, "%38s ", "pairwise")
            this_tpr2, this_fpr2 = my_ROC(abs.(this_xxx[1][2]), A2l, n; verbosity=1, io=ROC_logfile)
            this_aucs2[i, j] = get_auc(this_tpr2, this_fpr2)
            this_F1s2[i, j] = my_F1(abs.(this_xxx[1][2]), A2l, n; ε=0.0)

            @printf(ROC_logfile, "%38s ", "triadic")
            this_tpr3, this_fpr3 = my_ROC(abs.(this_xxx[1][3]), A3l, n; verbosity=1, io=ROC_logfile)
            this_aucs3[i, j] = get_auc(this_tpr3, this_fpr3)
            this_F1s3[i, j] = my_F1(abs.(this_xxx[1][3]), A3l, n; ε=0.0)
        end

        flush(SB_logfile)
        flush(ROC_logfile)
    end

    this_auc_dict = Dict(0 => this_aucs,
                         2 => this_aucs2,
                         3 => this_aucs3
    )
    this_F1_dict = Dict(0 => this_F1s,
                        2 => this_F1s2,
                        3 => this_F1s3
    )
    bayes_auc_dict = Dict(0 => bayes_aucs,
                          2 => bayes_aucs2,
                          3 => bayes_aucs3
    )
    bayes_F1_dict = Dict(0 => bayes_F1s,
                         2 => bayes_F1s2,
                         3 => bayes_F1s3
    )

    return this_auc_dict, this_F1_dict, bayes_auc_dict, bayes_F1_dict
end

# parameters for hypergraph generation
global n = 7
global t2 = .4
global t3 = n > 30 ? .01 : .05

# λs to consider for SINDy sparsity parameter
global λs = [round.([d * 10. ^ exp for exp in [-2, -1] for d in 1:9]; digits=2); 1.0]

nitr = 50

# set up log file
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMM")
SB_logfile = open("out/test-ic-scatter-with-noise-scaled-$(timestamp)-SB.txt", "w")
ROC_logfile = open("out/test-ic-scatter-with-noise-scaled-$(timestamp)-ROC.txt", "w")

# set up figure
inch = 96; pt = 4/3
fig = Figure(size=(3inch, 3inch), fontsize=8pt)
xticklabels = ["low noise,\nlow data", "low noise,\nhigh data", "high noise,\nlow data", "high noise,\nhigh data"]
ax = Axis(fig[1,1], 
          xgridvisible=false, ygridvisible=false, 
          xticks=(1:4, xticklabels), yticks=0.0:0.2:1.0,
          xticksvisible=false,
          limits=(nothing, (0.0, 1.0)),
          ylabel="AUC"
)
dodge=repeat([1,2], inner=nitr)

# low noise, low data regime
println(SB_logfile, "LOW NOISE, LOW DATA REGIME")
println(ROC_logfile, "LOW NOISE, LOW DATA REGIME")
this1_auc_dict, this1_F1_dict, bayes1_auc_dict, bayes1_F1_dict = test_inference(0.2, 40, nitr, SB_logfile, ROC_logfile);
optimal_λ1 = argmax(mean.([this1_auc_dict[0][:, i] for i in 1:length(λs)]))
println("Low noise, low data regime: optimal λ is $(λs[optimal_λ1])")
boxplot!(ax, ones(nitr*2), [bayes1_auc_dict[0]; this1_auc_dict[0][:, optimal_λ1]], dodge=dodge, color=map(d -> d == 1 ? seaborn_colorblind[2] : seaborn_colorblind[1], dodge))

# low noise, high data regime
println(SB_logfile, "\n\nLOW NOISE, HIGH DATA REGIME")
println(ROC_logfile, "\n\nLOW NOISE, HIGH DATA REGIME")
this2_auc_dict, this2_F1_dict, bayes2_auc_dict, bayes2_F1_dict = test_inference(0.2, 200, nitr, SB_logfile, ROC_logfile);
optimal_λ2 = argmax(mean.([this2_auc_dict[0][:, i] for i in 1:length(λs)]))
println("Low noise, high data regime: optimal λ is $(λs[optimal_λ2])")
boxplot!(ax, 2*ones(nitr*2), [bayes2_auc_dict[0]; this2_auc_dict[0][:, optimal_λ2]], dodge=dodge, color=map(d -> d == 1 ? seaborn_colorblind[2] : seaborn_colorblind[1], dodge))

# high noise, low data regime
println(SB_logfile, "\n\nHIGH NOISE, LOW DATA REGIME")
println(ROC_logfile, "\n\nHIGH NOISE, LOW DATA REGIME")
this3_auc_dict, this3_F1_dict, bayes3_auc_dict, bayes3_F1_dict = test_inference(0.4, 40, nitr, SB_logfile, ROC_logfile);
optimal_λ3 = argmax(mean.([this3_auc_dict[0][:, i] for i in 1:length(λs)]))
println("High noise, low data regime: optimal λ is $(λs[optimal_λ3])")
boxplot!(ax, 3*ones(nitr*2), [bayes3_auc_dict[0]; this3_auc_dict[0][:, optimal_λ3]], dodge=dodge, color=map(d -> d == 1 ? seaborn_colorblind[2] : seaborn_colorblind[1], dodge))

# high noise, high data regime
println(SB_logfile, "\n\nHIGH NOISE, HIGH DATA REGIME")
println(ROC_logfile, "\n\nHIGH NOISE, HIGH DATA REGIME")
this4_auc_dict, this4_F1_dict, bayes4_auc_dict, bayes4_F1_dict = test_inference(0.8, 200, nitr, SB_logfile, ROC_logfile);
optimal_λ4 = argmax(mean.([this4_auc_dict[0][:, i] for i in 1:length(λs)]))
println("High noise, high data regime: optimal λ is $(λs[optimal_λ4])")
boxplot!(ax, 4*ones(nitr*2), [bayes4_auc_dict[0]; this4_auc_dict[0][:, optimal_λ4]], dodge=dodge, color=map(d -> d == 1 ? seaborn_colorblind[2] : seaborn_colorblind[1], dodge))

close(SB_logfile)
close(ROC_logfile)

colsize!(fig.layout, 1, Aspect(1, 1.5))
resize_to_layout!(fig)
save("figs/test-noise-data-regimes-aucs.png", fig)

fig = Figure(size=(3inch, 3inch), fontsize=8pt)
xticklabels = ["low noise,\nlow data", "low noise,\nhigh data", "high noise,\nlow data", "high noise,\nhigh data"]
ax = Axis(fig[1,1], 
          xgridvisible=false, ygridvisible=false, 
          xticks=(1:4, xticklabels), yticks=0.0:0.2:1.0,
          xticksvisible=false,
          limits=(nothing, (0.0, 1.0)),
          ylabel="F1"
)

# low noise, low data regime
boxplot!(ax, ones(nitr*2), [bayes1_F1_dict[0]; this1_F1_dict[0][:, optimal_λ1]], dodge=dodge, color=map(d -> d == 1 ? seaborn_colorblind[2] : seaborn_colorblind[1], dodge))

# low noise, high data regime
boxplot!(ax, 2*ones(nitr*2), [bayes2_F1_dict[0]; this2_F1_dict[0][:, optimal_λ2]], dodge=dodge, color=map(d -> d == 1 ? seaborn_colorblind[2] : seaborn_colorblind[1], dodge))

# high noise, low data regime
boxplot!(ax, 3*ones(nitr*2), [bayes3_F1_dict[0]; this3_F1_dict[0][:, optimal_λ3]], dodge=dodge, color=map(d -> d == 1 ? seaborn_colorblind[2] : seaborn_colorblind[1], dodge))

# high noise, high data regime
boxplot!(ax, 4*ones(nitr*2), [bayes4_F1_dict[0]; this4_F1_dict[0][:, optimal_λ4]], dodge=dodge, color=map(d -> d == 1 ? seaborn_colorblind[2] : seaborn_colorblind[1], dodge))

colsize!(fig.layout, 1, Aspect(1, 1.5))
resize_to_layout!(fig)
save("figs/test-noise-data-regimes-F1s-eps0.0.png", fig)
