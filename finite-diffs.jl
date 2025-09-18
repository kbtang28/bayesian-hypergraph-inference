using SparseArrays

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

function FD(N_pt::Int, N_dom::Int, dt::Float64)
    # N_pt  = number of data pts
    # N_dom = order of accuracy (num of pts used - 1)
    # dt    = size of timestep (assuming regularly sampled)

    @assert N_dom >= 2 "N_dom must be positive even integer."
    @assert mod(N_dom, 2) == 0 "Only even N_dom supported."

    function stencil(coeffs, N_pt, N_dom)
        nrows = N_pt - N_dom
        diags = [(k-1) => fill(c, nrows) for (k,c) in enumerate(coeffs)]
        return spdiagm(N_pt - N_dom, N_pt, diags...)
    end

    if N_dom == 2
        L_I = stencil([0; 1; 0], N_pt, N_dom)

        coeffs = [-1/2; 0; 1/2]
        L_D1 = stencil(coeffs, N_pt, N_dom)/dt
    elseif N_dom == 4
        L_I = stencil([0; 0; 1; 0; 0], N_pt, N_dom)

        coeffs = [1/12; -2/3; 0; 2/3; -1/12]
        L_D1 = stencil(coeffs, N_pt, N_dom)/dt
    elseif N_dom == 6
        L_I = stencil([0; 0; 0; 1; 0; 0; 0], N_pt, N_dom)

        coeffs = [-1/60; 3/20; -3/4; 0; 3/4; -3/20; 1/60]
        L_D1 = stencil(coeffs, N_pt, N_dom)/dt
    elseif N_dom == 8
        L_I = stencil([0; 0; 0; 0; 1; 0; 0; 0; 0], N_pt, N_dom)

        coeffs = [1/280; -4/105; 1/5; -4/5; 0; 4/5; -1/5; 4/105; -1/280]
        L_D1 = stencil(coeffs, N_pt, N_dom)/dt
    elseif N_dom == 10
        L_I = stencil([0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0], N_pt, N_dom)

        coeffs = [-1/1260; 5/504; -5/84; 5/21; -5/6; 0; 5/6; -5/21; 5/84; -5/504; 1/1260]
        L_D1 = stencil(coeffs, N_pt, N_dom)/dt
    else
        coeffs = centralFDcoeffs(N_dom)
        L_D1 = stencil(coeffs, N_pt, N_dom)/dt
    end

    return L_I, L_D1
end