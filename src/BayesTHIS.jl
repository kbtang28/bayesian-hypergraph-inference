module BayesTHIS

export F1_filter_by_CI, F1_filter_by_coeff_mag, precision_recall_filter_by_CI
export centralFDcoeffs, FD
export gnm_random_hyperg, hyperg_connected
export f_kuramoto_3rd, f_kuramoto_3rd!
export my_ROC, my_PRC, my_F1, get_auc, get_aurocs, get_auprcs
export sample_posterior, sample_joint_posterior, significant_coeff
export degrees, degree_corr, degree_hetero_ratio, flag_complex, shuffle_hyperedges, shuffle_hyperedges!, swap_nodes
export this_bayes
export this
export get_θd, get_thetad, get_θ, get_theta, get_d, get_Ainf
export SBOpts, SBSettings, SBCtrlSettings, SBOut, SBDiagnostics # from SparseBayes.jl

include("f1s-with-filtering.jl")
include("finite-diffs.jl")
include("gen-rand-hyperg.jl")
include("hyperg-kuramoto.jl")
include("performance-measures.jl")
include("sample-posterior.jl")
include("structure-utils.jl")
include("this-bayes.jl")
include("this-tools.jl")
include("this.jl")

end