# Parameter generating function

using Distributions


def_ρ(M, kw) = rand(Normal(20, 4), M)
def_ω(M, kw) = rand(Uniform(1, 10), M)

function def_m(N, kw)
    m = zeros(N)
    for i in 1:N
        m[i] = kw[:m_0]*(0.1*4^(i-1))^(1-kw[:β])
    end
    return m
end

function def_r(N, kw)
    r = zeros(N)
    for i in 1:N
        r[i] = kw[:r_0]*(0.1*4^(i-1))^(1-kw[:β])
    end
    return r
end

function def_k(N, M, kw)

    dU = Uniform(0, 1)

    k = zeros(N, M)

    if !haskey(kw, :θ)
        θ = zeros(N, M)

        for i in 1:N
            θ[i, :] = rand(dU, M)
        end
    else
        θ = kw[:θ]
    end


    if !haskey(kw, :Ω)
        ## Sample specialisation parameter for each consumer
        Ω = fill(1.0, N)
    else
        Ω = kw[:Ω]
    end

    ## Sample total uptake capacity per consumer
    if !haskey(kw, :k_0)
        ## Sample specialisation parameter for each consumer
        k_0 = 1.0
    else
        k_0 = kw[:k_0]
    end

    ## Generate uptake matrix from a dirichlet distribution
    for i = 1:N
        dD = Dirichlet(Ω[i] * θ[i, :])
        k[i, :] = rand(dD) * k_0
    end
    return k

end

function def_η(N, M, kw)

    dU = Uniform(0, 1)

    η = zeros(N, M)

    if !haskey(kw, :θ)
        θ = zeros(N, M)

        for i in 1:N
            θ[i, :] = rand(dU, M)
        end
    else
        θ = kw[:θ]
    end


    if !haskey(kw, :Ω)
        ## Sample specialisation parameter for each consumer
        Ω = fill(1.0, N)
    else
        Ω = kw[:Ω]
    end

    ## Sample total uptake capacity per consumer
    if !haskey(kw, :η_0)
        ## Sample specialisation parameter for each consumer
        η_0 = rand(Normal(5,1), N)
    else
        η_0 = kw[:η_0]
    end

    ## Generate uptake matrix from a dirichlet distribution
    for i = 1:N
        dD = Dirichlet(Ω[i] * θ[i, :])
        η[i, :] = rand(dD) * η_0[i]
    end
    return η

end

function generate_params(N, M; f_m=def_m, f_ρ=def_ρ, f_η=def_η, f_r=def_r, f_k=def_k, f_ω=def_ω, kwargs...)
    kw = Dict{Symbol, Any}(kwargs)

    # consumers
     m = f_m(N, kw)
     r = f_r(N, kw)
     η = f_η(N, M, kw)
     k = f_k(N, M, kw)



     # resources
     ρ = f_ρ(M, kw)
     ω = f_ω(M, kw)

     kw_nt = (; kwargs...)
     p_nt = (N=N, M=M, m=m, r=r, ρ=ρ, η=η, k=k, ω=ω)

     out = Base.merge(p_nt, kw_nt)

     return out #(N=N, M=M, u=u, m=m, l=l, ρ=ρ, ω=ω, λ=λ)
 end