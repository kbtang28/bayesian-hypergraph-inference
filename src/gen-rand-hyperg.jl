using Combinatorics

"""
    gnm_random_hyperg(n, t2, t3) -> (A2, A3, A2l, A3l)

Generate a random hypergraph on `n` nodes with pairwise and triadic interactions.

- `t2`: fraction of all possible pairwise edges to include
- `t3`: fraction of all possible triadic hyperedges to include

Returns the pairwise adjacency matrix `A2`, the triadic adjacency tensor `A3`,
and their edge-list representations `A2l`, `A3l`.
"""
function gnm_random_hyperg(n, t2, t3)
    A2  = zeros(n, n) # adjacency matrix for pairwise interactions
    A2l = zeros(0, 3) # edge list for pairwise interactinos
    
    n2 = round(Int64, t2*binomial(n,2)) # number of pairwise interactions to add
    i2 = Set(shuffle(1:binomial(n,2))[1:n2])
    E2 = Vector{Vector{Int64}}()

    for (count, c) in enumerate(combinations(1:n, 2))
        if count in i2
            push!(E2,c)
            A2[c[1], c[2]] = 1.0
            A2[c[2], c[1]] = 1.0
            A2l = [A2l; [c[1] c[2] 1.0; c[2] c[1] 1.0]]
        end
    end

    A3  = zeros(n, n, n) # adjacency tensor for triadic interactions
    A3l = zeros(0, 4)    # edge list for triadic interactions

    n3 = round(Int64,t3*binomial(n,3)) # number of triadic interactions to add
    i3 = Set(shuffle(1:binomial(n,3))[1:n3])
    E3 = Vector{Vector{Int64}}()

    for (count, c) in enumerate(combinations(1:n, 3))
        if count in i3
            push!(E3, c)
            A3[c[1],c[2],c[3]] = 1.0
			A3[c[1],c[3],c[2]] = 1.0
			A3[c[2],c[1],c[3]] = 1.0
			A3[c[2],c[3],c[1]] = 1.0
			A3[c[3],c[1],c[2]] = 1.0
			A3[c[3],c[2],c[1]] = 1.0
            A3l = [A3l; [c[1] c[2] c[3] 1.0; c[2] c[1] c[3] 1.0; c[3] c[1] c[2] 1.0]]
        end
    end

    return A2, A3, A2l, A3l
end

"""
    hyperg_connected(A2, A3) -> Bool

Return `true` if every node has at least one pairwise or triadic interaction.
"""
function hyperg_connected(A2::Array{<:Real,2}, A3::Array{<:Real,3})
    deg2 = vec(sum(A2, dims=2))
    deg3 = vec(sum(A3, dims=[1, 2]))
    return all((deg2 .> 0.0) .|| (deg3 .> 0.0))
end