using SparseBayes

"""
    this_bayes(X, Y, ooi, dmax; opts, settings, ctrls) -> (Ainf, coeff, out, diagnostics)

Infer higher-order interactions via sparse Bayesian regression (SparseBayes).

- `X`: state time series (T × n); rows = time steps, columns = agents
- `Y`: derivative time series (T × n)
- `ooi`: orders of interaction to reconstruct (e.g. `[2, 3]`)
- `dmax`: maximum monomial degree (typically `maximum(ooi) - 1`)

Returns the inferred adjacency dictionary `Ainf` (keyed by interaction order), 
the full coefficient matrix `coeff`, the full `SBOut` object, and solver diagnostics.
"""
function this_bayes(X::Matrix{Float64}, Y::Matrix{Float64}, 
                    ooi::Vector{<:Integer}, dmax::Int64; 
                    opts=SBOpts(nitr=500, free_basis=[1]), 
                    settings=SBSettings(), 
                    ctrls=SBCtrlSettings(beta_update_frequency=3))

    if size(X) != size(Y)
        @info "Dimensions of states and derivatives do not match."
        return nothing
    end

    # build monomial library
    θ = get_θ(X, dmax)

    # run sparse Bayesian regression
    out, diagnostics = sparse_bayes(θ, Y; opts=opts, settings=settings, ctrls=ctrls)

    # reconstruct adjacency tensors
    Ainf = get_Ainf(out.value, ooi, dmax)

    return Ainf, out.value, out, diagnostics
end