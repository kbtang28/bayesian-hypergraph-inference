using LinearAlgebra, SparseArrays
import Statistics: mean

function laplacian(T::AbstractArray; sparse_out=true, rescale_per_node=false)
    d = ndims(T)-1 # order d interaction involves d+1 nodes
    n = size(T,1) # number of nodes
    @assert all(size(T,i) == n for i in 1:d+1) # adjacency tensor should be n x ⋯ x n
    
    if d == 1 # pairwise case
        A = Array(T)
        K = sum(T, dims=2)[:, 1] # degree K_i = sum_j T[i,j]
        avgK = mean(K)
        L = Diagonal(K) - A

        return sparse_out ? sparse(L) : Matrix(L), K, A, avgK
    else # higher-order case
        T_flat = reshape(T, n, n, :)
        A = sum(T_flat, dims=3)[:, :, 1] ./ factorial(d-1)
        
        K = vec(sum(T, dims=2:(d+1))) ./ factorial(d)
        avgK = mean(K)
        
        if sparse_out
            spA = sparse(A)
            L = spdiagm(0 => d .* K) - spA
        else
            L = Diagonal(d .* K) - A
        end

        if rescale_per_node
            L = L ./ d
        end

        return L, K, A, avgK
    end
end

function multiorder_laplacian(Ts; gammas=Dict(), sparse_out=true, rescale_per_node=false)
    per = Dict{Int,Any}()
    Lmul = nothing

    # loop through each order
    for (k, T) in Ts
        L, K, A, avgK = laplacian(T; sparse_out=sparse_out, rescale_per_node=rescale_per_node)

        γ = get(gammas, k, 1.0) # get gamma or 1.0 (default)
        weight = γ / (avgK == 0 ? 1.0 : avgK)

        term = weight * L
        Lmul = Lmul === nothing ? term : Lmul + term

        per[k] = (L, K, A, avgK, γ, weight)
    end

    return Lmul, per
end

function laplacian(E::AbstractMatrix, n::Integer; sparse_out=true, rescale_per_node=false)
    # E is adjacency list; each row of E has indices of nodes involved.

    # make sure only unique hyperedges included
    E = unique(sort(E, dims=2), dims=1)

    M, s = size(E) # M interactions involving s = d + 1 nodes
    d = s - 1

    if M == 0
        return spzeros(n, n)
    end

    I = Vector{Int64}()
    J = Vector{Int64}()

    K = zeros(n) # degree

    # iterate hyperedges
    for row in eachrow(E)        
        for u in row
            @inbounds K[Int(u)] += 1
        end

        for i in 1:d
            ui = row[i]
            @inbounds for j in (i+1):s
                vj = row[j]
                push!(I, ui); push!(J, vj)
                push!(I, vj); push!(J, ui)
            end
        end
    end

    # average degree
    avgK = mean(K)

    # build sparse adjacency A
    A = sparse(I, J, ones(length(I)), n, n) ./ (factorial(d-1)) 

    # Laplacian
    L = spdiagm(0 => d .* K) - A

    if !sparse_out
        L = Array(L)
    end

    if rescale_per_node
        L = L ./ d
    end

    return L, K, A, avgK
end

function multiorder_laplacian(Es, n::Integer; gammas=Dict(), sparse_out=true, rescale_per_node=false)
    per = Dict{Int,Any}()
    Lmul = nothing

    # loop through each order
    for (k, E) in Es
        L, K, A, avgK = laplacian(E, n; sparse_out=sparse_out, rescale_per_node=rescale_per_node)

        γ = get(gammas, k, 1.0) # get gamma or 1.0 (default)
        weight = γ / (avgK == 0 ? 1.0 : avgK)

        term = weight * L
        Lmul = Lmul === nothing ? term : Lmul + term

        per[k] = (L, K, A, avgK, γ, weight)
    end

    return Lmul, per
end