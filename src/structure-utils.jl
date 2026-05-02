using LinearAlgebra
import StatsBase: sample
import Statistics: cor, mean

"""
    degrees(T::AbstractArray) -> Vector

Node degrees from a dense adjacency matrix (order 2) or tensor (order > 2).
For tensors, divides by `(s-1)!` to avoid counting each hyperedge multiple times.
"""
function degrees(T::AbstractArray)
    s = ndims(T) # order of interaction
    n = size(T,1) # number of nodes
    @assert all(size(T,i) == n for i in 1:s) "Adjacency tensor must be n × ⋯ × n"

    if s == 2
        return vec(sum(T, dims=2))
    end

    return vec(sum(T, dims=2:s)) ./ factorial(s - 1)
end

"""
    degrees(E::AbstractMatrix, n) -> Vector

Node degrees from an edge-list matrix `E` (one hyperedge per row) over `n` nodes.
Duplicate and mirrored hyperedges are deduplicated before counting.
"""
function degrees(E::AbstractMatrix, n::Integer)
    E = unique(sort(E, dims=2), dims=1)

    K = zeros(n)
    for row in eachrow(E)        
        for u in row
            @inbounds K[Int(u)] += 1
        end
    end

    return K
end

"""
    degree_corr(A2, A3) -> Float64
    degree_corr(A2l, A3l, n) -> Float64

Pearson correlation between pairwise and triadic node degrees.
"""
function degree_corr(A2, A3)
    return cor(degrees(A2), degrees(A3))
end

function degree_corr(A2l, A3l, n)
    return cor(degrees(A2l, n), degrees(A3l, n))
end

"""
    degree_hetero_ratio(A2, A3) -> Float64
    degree_hetero_ratio(A2l, A3l, n) -> Float64

Ratio of triadic to pairwise degree heterogeneity, where heterogeneity is
`(max(K) - mean(K)) / mean(K)`.
"""
function degree_hetero_ratio(A2, A3)
    K2 = degrees(A2)
    K3 = degrees(A3)

    h(K) = (maximum(K) - mean(K)) / mean(K)

    return h(K3) / h(K2)
end

function degree_hetero_ratio(A2l, A3l, n)
    K2 = degrees(A2l, n)
    K3 = degrees(A3l, n)

    h(K) = (maximum(K) - mean(K)) / mean(K)

    return h(K3) / h(K2)
end

# enumerate all triangles in pairwise adjacency matrix `A2`
function _find_triangles(A2)
    N, _ = size(A2)
    
    ntri = round(Int, tr(A2 * A2 * A2) / 6)

    triangles = zeros(Int, 3, 0)

    for i in 1:N, j in i:N, k in j:N
        if all(x -> x .== 1.0, [A2[i, j], A2[i, k], A2[j, k]])
            triangles = [triangles [i; j; k]]
        end
    end

    @assert ntri == size(triangles, 2)

    return triangles
end

# fill all 6 symmetric permutations of hyperedge (i,j,k) in `A3`
function _add_triangle!(A3, i, j, k)
    i, j, k = Int(i), Int(j), Int(k)

    for (a, b, c) in [(i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)]
        A3[a, b, c] = 1.0
    end
end

"""
    flag_complex(A2, A2l; ptri=1.0) -> (A2, A3, A2l, A3l)

Lift a pairwise graph with adjacency matrix `A2` and edge-list matrix `A2l` to a 
flag complex by promoting each triangle to a triadic hyperedge with probability `ptri`.
"""
function flag_complex(A2, A2l; ptri = 1.0)
    N = size(A2, 1)

    all_triangles = _find_triangles(A2)
    ntri = size(all_triangles, 2)

    to_add = rand(ntri) .<= ptri

    A3  = zeros(N, N, N)
    A3l = zeros(0, 4)
    for tri in 1:ntri
        if to_add[tri]
            i, j, k = Int.(all_triangles[:, tri])
            _add_triangle!(A3, i, j, k)
            A3l = [A3l; [i j k 1.0; j i k 1.0; k i j 1.0]]
        end
    end

    return A2, A3, A2l, A3l
end

"""
    shuffle_hyperedges!(A3, A3l, p)

In-place random rewiring of triadic hyperedges: each unique hyperedge is
replaced with a uniformly random non-existing one with probability `p`.
"""
function shuffle_hyperedges!(A3, A3l, p)
    N = size(A3, 1)

    sA3l = sort(A3l[:, 1:3], dims=2)
    uid = unique(i -> sA3l[i, :], 1:size(sA3l, 1))
    uhedges = sA3l[uid, :]

    for (row, hedge) in zip(uid, eachrow(uhedges))
        rand() > p && continue

        i, j, k = Int.(hedge)

        ni, nj, nk = sample(collect(1:N), 3, replace=false, ordered=true)
        while A3[ni, nj, nk] == 1.0
            ni, nj, nk = sample(collect(1:N), 3, replace=false, ordered=true)
        end

        A3[i, j, k] = 0.0
        A3[i, k, j] = 0.0
        A3[j, i, k] = 0.0
        A3[j, k, i] = 0.0
        A3[k, i, j] = 0.0
        A3[k, j, i] = 0.0

        A3[ni, nj, nk] = 1.0
        A3[ni, nk, nj] = 1.0
        A3[nj, ni, nk] = 1.0
        A3[nj, nk, ni] = 1.0
        A3[nk, ni, nj] = 1.0
        A3[nk, nj, ni] = 1.0

        A3l[row:row+2, :] = [ni nj nk 1.0; nj ni nk 1.0; nk ni nj 1.0]
    end
end

"""
    shuffle_hyperedges(A3, A3l, p) -> (A3, A3l)

Non-mutating version of `shuffle_hyperedges!`.
"""
function shuffle_hyperedges(A3, A3l, p)
    cA3 = deepcopy(A3)
    cA3l = deepcopy(A3l)

    shuffle_hyperedges!(cA3, cA3l, p)

    return cA3, cA3l
end

# swap node labels `nid1` ↔ `nid2` in A3 and A3l in-place.
function _swap_nodes!(A3, A3l, nid1::Integer, nid2::Integer; id_temp=-1)
    while any(x -> x == id_temp, A3l)
        id_temp -= 1
    end

    for (i, row) in enumerate(eachrow(A3l))
        if in(row)(nid1)
            A3l[i, row .== nid1] .= id_temp # should only be one entry equal to nid1...
        end

        if in(row)(nid2)
            A3l[i, row .== nid2] .= nid1
        end

        if in(row)(id_temp)
            A3l[i, row .== id_temp] .= nid2
        end

        if A3l[i, 2] > A3l[i, 3]
            A3l[i, [2,3]] = A3l[i, [3,2]]
        end
    end

    A3l[:, 4] .= 1.0

    A3[[nid1, nid2], :, :] = A3[[nid2, nid1], :, :]
    A3[:, [nid1, nid2], :] = A3[:, [nid2, nid1], :]
    A3[:, :, [nid1, nid2]] = A3[:, :, [nid2, nid1]]
end

"""
    swap_nodes(A2, A3, A3l, n_swap) -> (A3, A3l)

Swap the `n_swap` highest-degree pairwise nodes with the `n_swap` lowest-degree
ones in the triadic structure, returning modified copies of `A3` and `A3l`.
"""
function swap_nodes(A2, A3, A3l, n_swap::Integer)
    n, _ = size(A2)

    # find ids for nodes with smallest and largest pairwise degree
    K2_dict = Dict(collect(1:n) .=> degrees(A2))

    nsmallest = sort(collect(K2_dict), by = x -> x.second)[1:n_swap]
    nid_small = [x.first for x in nsmallest]

    nlargest = sort(collect(K2_dict), by = x -> x.second, rev=true)[1:n_swap]
    nid_large = [x.first for x in nlargest]

    # perform n_swap node swaps
    cA3 = deepcopy(A3)
    cA3l = deepcopy(A3l)
    for (ns, nl) in zip(nid_small, nid_large)
        _swap_nodes!(cA3, cA3l, Int(ns), Int(nl))
    end

    return cA3, cA3l
end