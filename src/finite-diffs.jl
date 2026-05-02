using LinearAlgebra, SparseArrays

"""
    centralFDcoeffs(n) -> Vector{Float64}

Return the `n`-th order central finite difference coefficients for the first derivative.
Small coefficients (< 1e-10) are rounded to zero.
"""
function centralFDcoeffs(n::Int)
    # 1st order central finite difference coefficients

    x = -(n/2) : (n/2) # stencil points
    mat = (x') .^ (0:n)

    rhs = [0; 1; zeros(n-1)]
    coeffD1 = mat \ rhs

    # round small coeffs to zero
    coeffD1[abs.(coeffD1) .< 1e-10] .= 0.0

    return vec(coeffD1')
end

# internal helper: return (identity_coeffs, derivative_coeffs) for a given stencil width.
function _fd_coefficients(N_dom::Int)
    if N_dom == 2
        return [0; 1; 0],           
               [-1/2; 0; 1/2]
    elseif N_dom == 4
        return [0; 0; 1; 0; 0],     
               [1/12; -2/3; 0; 2/3; -1/12]
    elseif N_dom == 6
        return [0; 0; 0; 1; 0; 0; 0],     
               [-1/60; 3/20; -3/4; 0; 3/4; -3/20; 1/60]
    elseif N_dom == 8
        return [0; 0; 0; 0; 1; 0; 0; 0; 0], 
               [1/280; -4/105; 1/5; -4/5; 0; 4/5; -1/5; 4/105; -1/280]
    elseif N_dom == 10
        return [0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0],
               [-1/1260; 5/504; -5/84; 5/21; -5/6; 0; 5/6; -5/21; 5/84; -5/504; 1/1260]
    else
        d_coeffs = centralFDcoeffs(N_dom)
        i_coeffs = zeros(N_dom + 1); i_coeffs[N_dom ÷ 2 + 1] = 1.0
        return i_coeffs, d_coeffs
    end
end

"""
    FD(N_pt, N_dom, dt) -> (L_I, L_D1)

Build sparse central finite difference matrices for a uniform time series.

- `N_pt`:  number of data points
- `N_dom`: stencil width (order of accuracy); must be a positive even integer
- `dt`:    time step size

Returns the interpolation matrix `L_I` and the first-derivative matrix `L_D1`.
"""
function FD(N_pt::Int, N_dom::Int, dt::Float64)
    @assert N_dom >= 2 "N_dom must be a positive even integer."
    @assert iseven(N_dom) "Only even N_dom supported."

    i_coeffs, d_coeffs = _fd_coefficients(N_dom)
    nrows = N_pt - N_dom

    function stencil(coeffs)
        diags = [(k - 1) => fill(c, nrows) for (k, c) in enumerate(coeffs)]
        return spdiagm(nrows, N_pt, diags...)
    end

    L_I  = stencil(i_coeffs)
    L_D1 = stencil(d_coeffs) / dt
    return L_I, L_D1
end

"""
    FD(N_pt, N_dom, dt, m) -> (L_I, L_D1)

Strided variant of `FD`: evaluates the stencil every `m` points rather than every point.
Useful for subsampling a dense time series onto a coarser grid.
"""
function FD(N_pt::Int, N_dom::Int, dt::Float64, m::Int)
    @assert N_dom >= 2 "N_dom must be a positive even integer."
    @assert iseven(N_dom) "Only even N_dom supported."

    i_coeffs, d_coeffs = _fd_coefficients(N_dom)
    w = N_dom ÷ 2 # stencil half-width
    coarse_idx = Int.(collect((w + 1):m:(N_pt - w)))
    N_coarse = length(coarse_idx)

    function stencil(coeffs)
        rows, cols, vals = Int[], Int[], Float64[]
        for (r, center) in enumerate(coarse_idx)
            idxs = (center - w):(center + w)
            append!(rows, fill(r, length(coeffs)))
            append!(cols, idxs)
            append!(vals, coeffs)
        end
        return sparse(rows, cols, vals, N_coarse, N_pt)
    end

    L_I  = stencil(i_coeffs)
    L_D1 = stencil(d_coeffs) / dt
    return L_I, L_D1
end