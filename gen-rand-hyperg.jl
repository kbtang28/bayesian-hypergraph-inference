using Random, Combinatorics

function gnm_random_hyperg(n, t2, t3)
    # t2 = proportion of PAIRWISE interactions that exist
    # t3 = proportion of TRIADIC interactions that exist

    A2 = zeros(n,n) # adjacency matrix for pairwise interactions
    A2l = zeros(0,3)
    n2 = round(Int64, t2*binomial(n,2)) # number of pairwise interactions to add
    i2 = shuffle(1:binomial(n,2))[1:n2]
    E2 = Vector{Vector{Int64}}()
    count = 0
    for c in combinations(1:n,2)
        count += 1
        if count in i2
            push!(E2,c)
            A2[c[1], c[2]] = 1.0
            A2[c[2], c[1]] = 1.0
            A2l = [A2l; [c[1] c[2] 1.0; c[2] c[1] 1.0]]
        end
    end

    A3 = zeros(n,n,n) # adjacency tensor for triadic interactions
    A3l = zeros(0,4)
    n3 = round(Int64,t3*binomial(n,3)) # number of triadic interactions to add
    i3 = shuffle(1:binomial(n,3))[1:n3]
    E3 = Vector{Vector{Int64}}()
    count = 0
    for c in combinations(1:n,3)
        count += 1
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