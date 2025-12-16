using LinearAlgebra
import StatsBase: sample
import Statistics: cor, mean

function degrees(T::AbstractArray)
    s = ndims(T) # order of interaction
    n = size(T,1) # number of nodes
    @assert all(size(T,i) == n for i in 1:s) # adjacency tensor should be n x ⋯ x n

    if s == 2
        A = Array(T)
        K = sum(T, dims=2)[:, 1] # degree K_i = sum_j T[i,j]

        return K
    else
        K = vec(sum(T, dims=2:s)) ./ factorial(s-1)
        return K
    end
end

function degrees(E::AbstractMatrix, n::Integer)
    E = unique(sort(E, dims=2), dims=1)

    K = zeros(n) # degree

    # iterate hyperedges
    for row in eachrow(E)        
        for u in row
            @inbounds K[Int(u)] += 1
        end
    end

    return K
end

function degree_corr(A2, A3)
    K2 = degrees(A2)
    K3 = degrees(A3)

    return cor(K2, K3)
end

function degree_corr(A2l, A3l, n)
    K2 = degrees(A2l, n)
    K3 = degrees(A3l, n)

    return cor(K2, K3)
end

function degree_hetero_ratio(A2, A3)
    K2 = degrees(A2)
    K3 = degrees(A3)

    h2 = (maximum(K2) - mean(K2)) / mean(K2)
    h3 = (maximum(K3) - mean(K3)) / mean(K3)

    return h3 / h2
end

function degree_hetero_ratio(A2l, A3l, n)
    K2 = degrees(A2l, n)
    K3 = degrees(A3l, n)

    h2 = (maximum(K2) - mean(K2)) / mean(K2)
    h3 = (maximum(K3) - mean(K3)) / mean(K3)

    return h3 / h2
end

function _find_triangles(A2)
    N, _ = size(A2)
    
    ntri = tr(A2 * A2 * A2) / 6

    triangles = zeros(3, 0)
    tri_ct = 0

    for i in 1:N
        for j in i:N
            for k in j:N
                if all(x -> x .== 1.0, [A2[i, j], A2[i, k], A2[j, k]])
                    triangles = [triangles [i; j; k]]
                    tri_ct += 1
                end
            end
        end
    end

    @assert ntri == tri_ct

    return triangles
end

function _add_triangle!(A3, i, j, k)
    i = Int(i)
    j = Int(j)
    k = Int(k)

    A3[i, j, k] = 1.0
    A3[i, k, j] = 1.0
    A3[j, i, k] = 1.0
    A3[j, k, i] = 1.0
    A3[k, i, j] = 1.0
    A3[k, j, i] = 1.0
end

function flag_complex(A2, A2l; ptri = 1.0)
    N = size(A2, 1)

    all_triangles = _find_triangles(A2)
    ntri = size(all_triangles, 2)

    to_add = rand(ntri) .<= ptri
    nadd = sum(to_add)

    A3 = zeros(N, N, N)
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

function shuffle_hyperedges!(A3, A3l, p)
    N = size(A3, 1)

    sA3l = sort(A3l[:, 1:3], dims=2)
    unique_hedge_ids = unique(i -> sA3l[i, :], 1:size(sA3l, 1))
    unique_hedges = sA3l[unique_hedge_ids, :]

    for (row, hedge) in zip(unique_hedge_ids, eachrow(unique_hedges))
        if rand() <= p
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
end

function shuffle_hyperedges(A3, A3l, p)
    cA3 = deepcopy(A3)
    cA3l = deepcopy(A3l)

    shuffle_hyperedges!(cA3, cA3l, p)

    return cA3, cA3l
end

function node_swap!(A3, A3l, nid1, nid2; id_temp=-1)
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

function node_swap(A3, A3l, nid1, nid2; id_temp=-1)
    cA3 = deepcopy(A3)
    cA3l = deepcopy(A3l)

    node_swap!(cA3, cA3l, nid1, nid2; id_temp=id_temp)

    return cA3, cA3l
end