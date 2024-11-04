# This code randomly samples communities and simulates them

cd("D:/SFI")

using DataFrames
using DifferentialEquations
using LinearAlgebra
using Plots
using Sundials
using JLD2
using CSV

include("./equations.jl")
include("./params.jl")
include("./simulate.jl")





itr = 0
community_df = nothing

N=10
M=10

noise = Normal(1.0, 0.10)
NoiseM = rand(noise, N, M)

Nsim_draw = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
k_var = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5]
for k_m in k_var
    for N_sim in Nsim_draw

        θ = 1.0*I(N)

        θ_het = fill(1.0, N, M)

        θ = 100*N_sim*θ + broadcast(abs, NoiseM) + 30*(1.0 - N_sim)*θ_het

        Ω_mod = 10*(1/N_sim) + 100*N_sim

        Ω = fill(Ω_mod, N)

        m_0 = 0.138
        r_0 = 1
        β = 0.75
        k_0 = rand(Normal(k_m,k_m/3), N)
        η_0 = zeros(N, M)
        for i in 1:N
            η_0[i, :] = rand(Normal(2,0.3), M)
        end
        Ω = fill(10.0, N)

        p = generate_params(N, M; m_0=m_0, r_0=r_0, β=β, k_0=k_0, θ=θ, Ω=Ω, η_0=η_0)

        if itr == 0
            community_df = DataFrame([p])
        else
            temp = DataFrame([p])
            community_df = vcat(community_df, temp)
        end
    end

    itr += 1

end

save("./community03_291024.jld", "community_df", community_df)

itr = 0
results = Matrix{Float64}[]
for row in eachrow(community_df)
    p = NamedTuple(row)
    SAD = MiC_test(p=p, t_span = 200000)

    if itr == 0
        results = push!(results, SAD)
    else 
        temp =  SAD
        results = push!(results, SAD)
    end
    itr += 1

end

binr = range(log(results[1][1,1]/0.1), log(results[1][10,1]/0.1), length=10)

for r in 1:length(results)
    display(histogram(log.(results[r][:,1]/0.1), weights=results[r][:,2], bins=binr))
end

condition(du, t, integrator) = norm(integrator(t, Val{1})) <= 1e-5

affect!(integrator) = terminate!(integrator)

cb = DiscreteCallback(condition, affect!)

p = NamedTuple(community_df[44, :])

x0 = fill(0.0, (N+M))
for i in 1:N
    x0[i] = 2
end
 
# Set resource initial conditions
for α in (N+1):(N+M)
    x0[α] = 10
end
 
# Define integration timespan
tspan = (0.0, 100000.0)
 
# Generate DiffEq problem
prob = ODEProblem(dx!, x0, tspan, p)
 
#Solve DiffEq problem with callback and CVODE_BDF integratioon method
sol =solve(prob, CVODE_BDF(), callback=cb)
plot(sol)

C = sol[1:N, length(sol)]
R = sol[(N+1):(N+M), length(sol)]

sizes = zeros(N)
for i in 1:p.N
    sizes[i] = (0.1*4^(i-1))
end
SAD = ones(N,2)

SAD[:, 1] = sizes
SAD[:, 2] = C

binr = range(log(SAD[1,1]/0.1), log(SAD[10,1]/0.1), length=10)

histogram(log.(SAD[:,1]/0.1), weights=SAD[:,2], bins=binr)

SAD[:,2]

dist = DataFrame(BodyWeight=SAD[:,1], Abundance=SAD[:,2])
sol_data = DataFrame(sol)

CSV.write("dist_05.csv", dist)
CSV.write("sol_05.csv", sol_data)

############################################################################### Q
x0 = fill(0.0, (N+M))
for i in 1:N
    x0[i] = C[i]*1.5
end
 
# Set resource initial conditions
for α in (N+1):(N+M)
    x0[α] = R[α-p.N]
end
 
# Define integration timespan
tspan = (0.0, 100.0)
 
# Generate DiffEq problem
prob = ODEProblem(dx!, x0, tspan, p)
 
#Solve DiffEq problem with callback and CVODE_BDF integratioon method
sol =solve(prob, CVODE_BDF(), callback=cb)

plot(sol)

sol_data = DataFrame(sol)

CSV.write("pulse_05.csv", sol_data)

##############################################################################

x0 = vcat(C, R)

function σ_dx!(dx, x, p, t)
    for α in 1:N
        dx[α] = 0.01*x[α]
    end
end 

prob_sde = SDEProblem(dx!, σ_dx!, x0, tspan, p)
sol_sde =solve(prob_sde, SRIW1())

plot(sol_sde)

sol_sde_dat = DataFrame(sol_sde)

CSV.write("stochastic_05.csv", sol_sde_dat)

##########################################################

# Define integration timespan
tspan = (0.0, 100.0)
x0 = vcat(C, R)
 
# Generate DiffEq problem
prob = ODEProblem(dx!, x0, tspan, p)

condition(u, t, integrator) = t == 20
function affect!(integrator)
    for i in 1:p.N
        integrator.u[i+p.N] += 100
    end
end
cb = DiscreteCallback(condition, affect!)
 
#Solve DiffEq problem with callback and CVODE_BDF integratioon method
sol =solve(prob, CVODE_BDF(), callback=cb, tstops=20)

plot(sol, idxs=[1,2,3,4,5,6,7,8,9,10])
plot(sol, idxs=[11,12,13,14,15,16,17,18,19,20])

sol_data = DataFrame(sol)

CSV.write("pulseR_04.csv", sol_data)
