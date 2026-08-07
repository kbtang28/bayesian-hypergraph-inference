using Random
using LinearAlgebra
import Statistics: cor, mean, std
import StatsBase: sample

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
    flag_complex_fixed(A2, A2l, k) -> (A2, A3, A2l, A3l)

Lift a pairwise graph with adjacency matrix `A2` and edge-list matrix `A2l` to a 
flag complex by promoting `k` triangles to a triadic hyperedge.
Returns nothing if fewer than `k` triangles available.
"""
function flag_complex_fixed(A2, A2l, k)
    tri = _find_triangles(A2)
    ntri = size(tri, 2)
    ntri >= k || return nothing
    keep = sample(1:ntri, k, replace = false)
    T = [Tuple(sort(Int.(tri[:, c]))) for c in keep]
    A3, A3l = triadic_from_list(T, size(A2, 1))
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

"""
    triads(A3l) -> Vector{NTuple{3,Int}}

Canonical list of unique triadic hyperedges, each sorted ascending, extracted
from an edge-list matrix `A3l` (three rows per hyperedge).
"""
function triads(A3l)
    T    = NTuple{3,Int}[]
    seen = Set{NTuple{3,Int}}()
    for row in eachrow(A3l)
        t = Tuple(sort(Int.(row[1:3])))
        if !(t in seen)
            push!(seen, t)
            push!(T, t)
        end
    end
    return T
end

"""
    triadic_from_list(T, n) -> (A3, A3l)

Rebuild the adjacency tensor and edge-list matrix from a canonical triad list,
matching the row convention produced by `flag_complex` / `gnm_random_hyperg`.
"""
function triadic_from_list(T::Vector{NTuple{3,Int}}, n::Integer)
    A3  = zeros(n, n, n)
    A3l = zeros(3 * length(T), 4)

    for (idx, (i, j, k)) in enumerate(T)
        _add_triangle!(A3, i, j, k)
        r = 3 * (idx - 1)
        A3l[r+1, :] = [i j k 1.0]
        A3l[r+2, :] = [j i k 1.0]
        A3l[r+3, :] = [k i j 1.0]
    end

    return A3, A3l
end

"""
    triadic_degrees(T, n) -> Vector{Float64}

Number of triads incident to each node.
"""
function triadic_degrees(T::Vector{NTuple{3,Int}}, n::Integer)
    K3 = zeros(n)
    for t in T, u in t
        @inbounds K3[u] += 1.0
    end
    return K3
end

"""
    permute_triads(T, p) -> Vector{NTuple{3,Int}}

Relabel the triadic structure by permutation `p`, where old node `v` receives
new label `p[v]`.
"""
permute_triads(T::Vector{NTuple{3,Int}}, p::AbstractVector{<:Integer}) =
    [Tuple(sort([p[i], p[j], p[k]])) for (i, j, k) in T]

# DC of a relabelling without rebuilding the tensor: correlation is invariant
# under applying the same permutation to both vectors, so
#   cor(K2, K3[invperm(p)]) == cor(K2[p], K3).
@inline dc_from_perm(K2, K3, p) = cor(view(K2, p), K3)

"""
    dc_bounds(K2, K3) -> (lo, hi)

Extremal cross-order DC attainable by relabelling, from the rearrangement
inequality: sorting both degree sequences the same way maximises the
correlation, opposite ways minimises it.  Gives a realisation-specific feasible
interval, which is a more meaningful axis than a swap count.
"""
function dc_bounds(K2, K3)
    a, b = sort(K2), sort(K3)
    return cor(a, reverse(b)), cor(a, b)
end

"""
    dc_null(K2, K3; nsamples=10_000, rng=Random.default_rng()) -> Vector{Float64}

Distribution of cross-order DC under uniform random relabelling.  Use as the
reference band: for n = 15 this is centred at 0 with sd ≈ 0.27, so |DC| ≳ 0.55
is not reachable by chance and must be annealed to.
"""
function dc_null(K2, K3; nsamples::Integer = 10_000, rng = Random.default_rng())
    n = length(K2)
    return [dc_from_perm(K2, K3, randperm(rng, n)) for _ in 1:nsamples]
end

"""
    anneal_to_dc(K2, K3, target; kwargs...) -> (p, achieved, accept_rate)

Simulated annealing over relabellings `p` with energy `(DC(p) - target)^2`,
followed by a constrained random walk that accepts any transposition keeping
`|DC - target| <= tol`.

Keyword arguments:
  `tol`       = half-width of the tolerance band for the decorrelation phase
  `nsteps`    = annealing steps;  `ndecorr` = constrained-walk steps
  `T0`,`Tend` = geometric temperature schedule
"""
function anneal_to_dc(K2, K3, target;
                      tol      = 0.02,
                      nsteps   = 4000,
                      ndecorr  = 2000,
                      T0       = 0.2,
                      Tend     = 1e-4,
                      rng      = Random.default_rng())

    n = length(K2)
    @assert std(K3) > 0 "triadic degree sequence is constant; DC undefined"
    @assert std(K2) > 0 "pairwise degree sequence is constant; DC undefined"

    p = randperm(rng, n)
    E = (dc_from_perm(K2, K3, p) - target)^2

    for s in 1:nsteps
        T = T0 * (Tend / T0)^((s - 1) / nsteps)

        i, j = sample(rng, 1:n, 2, replace = false)
        p[i], p[j] = p[j], p[i]

        Enew = (dc_from_perm(K2, K3, p) - target)^2

        if Enew <= E || rand(rng) < exp(-(Enew - E) / T)
            E = Enew    # accept
        else
            p[i], p[j] = p[j], p[i]   # reject
        end
    end

    # constrained random walk: uniform over relabellings inside the band
    naccept = 0
    for _ in 1:ndecorr
        i, j = sample(rng, 1:n, 2, replace = false)
        p[i], p[j] = p[j], p[i]

        if abs(dc_from_perm(K2, K3, p) - target) <= tol
            naccept += 1
        else
            p[i], p[j] = p[j], p[i]
        end
    end

    return p, dc_from_perm(K2, K3, p), naccept / max(ndecorr, 1)
end

# draw a uniformly random triple not already in `present`
function _random_absent_triad(n, present, rng)
    while true
        t = Tuple(sample(rng, 1:n, 3, replace = false, ordered = true))
        t in present || return t
    end
end

"""
    anneal_triads(A2, T0, target; kwargs...) -> (T, achieved, accept_rate, turnovers)

Simulated annealing over triad sets of fixed cardinality: each move replaces a
randomly chosen triad with a uniformly random absent triple, with energy
`(absent-pair fraction - target)^2`.
"""
function anneal_triads(A2, T0::Vector{NTuple{3,Int}}, target;
                       tol       = 0.02,
                       nsteps    = 6000,
                       turnovers = 8,
                       maxdecorr = 200_000,
                       Tstart    = 0.05,
                       Tend      = 1e-5,
                       rng       = Random.default_rng())

    n       = size(A2, 1)
    T       = copy(T0)
    present = Set(T)

    mult = Dict{Tuple{Int,Int},Int}()
    ncov = Ref(0)
    nabs = Ref(0)

    _faces(t) = ((t[1], t[2]), (t[1], t[3]), (t[2], t[3]))

    function _add!(t)
        for f in _faces(t)
            c = get(mult, f, 0)
            mult[f] = c + 1
            if c == 0
                ncov[] += 1
                A2[f[1], f[2]] == 0.0 && (nabs[] += 1)
            end
        end
    end

    function _del!(t)
        for f in _faces(t)
            c = mult[f]
            if c == 1
                delete!(mult, f)
                ncov[] -= 1
                A2[f[1], f[2]] == 0.0 && (nabs[] -= 1)
            else
                mult[f] = c - 1
            end
        end
    end

    function _swap!(idx, told, tnew)
        _del!(told); _add!(tnew)
        delete!(present, told); push!(present, tnew)
        T[idx] = tnew
    end

    for t in T; _add!(t); end

    _fa() = ncov[] > 0 ? nabs[] / ncov[] : NaN
    _energy() = (f = _fa(); isnan(f) ? Inf : (f - target)^2)

    E = _energy()

    for s in 1:nsteps
        Temp = Tstart * (Tend / Tstart)^((s - 1) / nsteps)

        idx  = rand(rng, 1:length(T))
        told = T[idx]
        tnew = _random_absent_triad(n, present, rng)
        _swap!(idx, told, tnew)

        Enew = _energy()

        if Enew <= E || rand(rng) < exp(-(Enew - E) / Temp)
            E = Enew
        else
            _swap!(idx, tnew, told)
        end
    end

    nacc_target = turnovers * length(T)
    naccept = 0
    nprop   = 0

    while naccept < nacc_target && nprop < maxdecorr
        nprop += 1

        idx  = rand(rng, 1:length(T))
        told = T[idx]
        tnew = _random_absent_triad(n, present, rng)
        _swap!(idx, told, tnew)

        if abs(_fa() - target) <= tol
            naccept += 1
        else
            _swap!(idx, tnew, told)
        end
    end

    return T, _fa(), naccept / max(nprop, 1), naccept / length(T)
end

"""
    pair_alignment(A2, T) -> NamedTuple

Statistics of the node pairs covered by triadic hyperedges. Returns:
- `n_covered`, `n_absent`, 
- `frac_absent` (unweighted, over distinct pairs)
- `w_total`, `w_absent`, `w_frac_absent` (weighted by number of triads covering each pair)
"""
function pair_alignment(A2, T::Vector{NTuple{3,Int}})
    mult = Dict{Tuple{Int,Int},Int}()

    for (i, j, k) in T, (a, b) in ((i, j), (i, k), (j, k))
        key = a < b ? (a, b) : (b, a)
        mult[key] = get(mult, key, 0) + 1
    end

    n_covered = length(mult)
    n_absent  = count(pq -> A2[pq[1], pq[2]] == 0.0, keys(mult))
    w_total   = sum(values(mult); init = 0)
    w_absent  = sum((v for (pq, v) in mult if A2[pq[1], pq[2]] == 0.0); init = 0)

    return (n_covered      = n_covered,
            n_absent       = n_absent,
            frac_absent    = n_covered == 0 ? NaN : n_absent / n_covered,
            w_total        = w_total,
            w_absent       = w_absent,
            w_frac_absent  = w_total  == 0 ? NaN : w_absent / w_total)
end
