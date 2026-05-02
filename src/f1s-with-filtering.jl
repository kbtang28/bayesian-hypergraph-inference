# compute the three F1 scores (combined, pairwise-only, triadic-only) for given coefficient mask
function _f1_from_mask(coeff::Matrix{Float64}, sig::AbstractMatrix{Bool},
                       A2l::Matrix{Float64}, A3l::Matrix{Float64})
    _, n = size(coeff)

    Ainf = get_Ainf(coeff .* sig, [2, 3], 2)

    f1_combined = my_F1(abs.(Ainf[2]), A2l, abs.(Ainf[3]), A3l, n)
    f1_pairwise = my_F1(abs.(Ainf[2]), A2l, n)
    f1_triadic  = my_F1(abs.(Ainf[3]), A3l, n)

    return f1_combined, f1_pairwise, f1_triadic
end

"""
    F1_filter_by_CI(out, D, coeff, A2l, A3l, levels) -> Matrix{Float64}
 
Compute F1 scores after filtering coefficients by credible-interval significance.
 
For each significance `level` in `levels`, coefficients whose conditional posterior
credible interval contains zero are zeroed out before computing F1.
 
Returns a `3 × length(levels)` matrix with rows:
  1. combined pairwise + triadic F1
  2. pairwise-only F1
  3. triadic-only F1
"""
function F1_filter_by_CI(out, D, coeff::Matrix{Float64},
                         A2l::Matrix{Float64}, A3l::Matrix{Float64},
                         levels::AbstractVector{Float64})
    F1s = zeros(Float64, 3, length(levels))

    for (i, level) in enumerate(levels)
        sig = significant_coeff(out, D, level)
        F1s[:, i] .= _f1_from_mask(coeff, sig, A2l, A3l)
    end
    return F1s
end

"""
    F1_filter_by_coeff_mag(coeff, A2l, A3l, τs) -> Matrix{Float64}
 
Compute F1 scores after filtering coefficients by magnitude threshold.
 
For each threshold `τ` in `τs`, coefficients with `|coeff| < τ` are zeroed out
before computing F1.
 
Returns a `3 × length(τs)` matrix with rows:
  1. combined pairwise + triadic F1
  2. pairwise-only F1
  3. triadic-only F1
"""
function F1_filter_by_coeff_mag(coeff::Matrix{Float64},
                                A2l::Matrix{Float64}, A3l::Matrix{Float64},
                                τs::AbstractVector{Float64})
    F1s = zeros(Float64, 3, length(τs))

    for (i, τ) in enumerate(τs)
        sig = abs.(coeff) .> τ
        F1s[:, i] .= _f1_from_mask(coeff, sig, A2l, A3l)
    end
    return F1s
end