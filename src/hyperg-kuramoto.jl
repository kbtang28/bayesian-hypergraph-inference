using LinearAlgebra, DelimitedFiles, Distributions, Statistics

"""
	f_kuramoto_3rd(θ, A2l, A3l, P, ϕ2, ϕ3) -> Vector

Evaluate the RHS of the triadic Kuramoto ODE using edge-list representations 
`A2l` (pairwise) and `A3l` (triadic). `P` is the vector of natural frequencies; 
`ϕ2` and `ϕ3` are phase lags.
"""
function f_kuramoto_3rd(θ::Vector{Float64}, A2l::Matrix{Float64}, A3l::Matrix{Float64}, 
					    P::Vector{Float64}, ϕ2::Float64=0., ϕ3::Float64=0.)
	fθ = copy(P)
	
	for l in axes(A2l, 1)
		i, j = Int.(A2l[l,1:2])
		a = A2l[l,3]
		fθ[i] -= a * (sin(θ[i] - θ[j] - ϕ2) + sin(ϕ2))
	end
	for l in axes(A3l, 1)
		i,j,k = Int64.(A3l[l,1:3])
		a = A3l[l,4]
		fθ[i] -= a * (sin(2*θ[i] - θ[j] - θ[k] - ϕ3) + sin(ϕ3))
	end

	return fθ
end

"""
	f_kuramoto_3rd(θ, A2, A3, P, ϕ2, ϕ3) -> similar(θ)

Evaluate the RHS of the triadic Kuramoto ODE using dense adjacency tensors `A2` and `A3`.
`P` is the vector of natural frequencies; `ϕ2` and `ϕ3` are phase lags.
"""
function f_kuramoto_3rd(θ, A2::Array{Float64,2}, A3::Array{Float64,3}, 
						P::Vector{Float64}, ϕ2::Float64=0., ϕ3::Float64=0.)
	n = length(θ)
	
	fθ = similar(θ, n)
	for i in 1:n
		x = P[i]
		for j in 1:n
			x -= A2[i,j] * (sin(θ[i] - θ[j] - ϕ2) + sin(ϕ2))
			for k in 1:n
				x -= A3[i,j,k] * (sin(2*θ[i] - θ[j] - θ[k] - ϕ3) + sin(ϕ3))
			end
		end
		fθ[i] = x
	end
	return fθ
end

"""
    f_kuramoto_3rd(Θ, A2, A3, P, ϕ2, ϕ3) -> Matrix

Batch evaluation of RHS of the triadic Kuramoto ODE across a trajectory 
matrix `Θ` (rows = time steps).
"""
function f_kuramoto_3rd(Θ::Matrix{Float64}, A2::Array{Float64,2}, A3::Array{Float64,3},
                        P::Vector{Float64}, ϕ2::Float64=0., ϕ3::Float64=0.)
    T, n = size(Θ)

    fΘ = Matrix{Float64}(undef, T, n)
    for t in 1:T
        fΘ[t, :] = f_kuramoto_3rd(Θ[t, :], A2, A3, P, ϕ2, ϕ3)
    end
    return fΘ
end

"""
    f_kuramoto_3rd!(dθ, θ, p, t)

In-place evaluation of RHS of the triadic Kuramoto ODE compatible with 
DifferentialEquations.jl ODE solvers. Takes `p = (A2, A3, ωs, ϕ2, ϕ3)`.
"""
function f_kuramoto_3rd!(dθ, θ, p, t)
    A2, A3, ωs, ϕ2, ϕ3 = p

    n = length(θ)
    for i in 1:n
        dθ[i] = ωs[i]
        for j in 1:n
            dθ[i] -= A2[i, j] * (sin(θ[i] - θ[j] - ϕ2) + sin(ϕ2))
            for k in 1:n
                dθ[i] -= A3[i, j, k] * (sin(2θ[i] - θ[j] - θ[k] - ϕ3) + sin(ϕ3))
            end
        end
    end
end