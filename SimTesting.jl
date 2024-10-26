cd("D:/SFI")

include("./equations.jl")
include("./params.jl")

using DifferentialEquations
using LinearAlgebra
using Plots
using Sundials

condition(du, t, integrator) = norm(integrator(t, Val{1})) <= 1e-5

affect!(integrator) = terminate!(integrator)

cb = DiscreteCallback(condition, affect!)

#
N = 7
M = 7

noise = Normal(1.0, 1.0)
NoiseM = rand(noise, N, M)

θ = 10.0*I(N) + broadcast(abs, NoiseM)
m_0 = 0.138
r_0 = 1
β = 0.75
k_0 = 0.2
η_0 = zeros(N, M)
for i in 1:N
    η_0[i, :] = rand(Normal(2,0.3), M)
end
Ω = fill(10.0, N)

# Generate MiCRM parameters
p = generate_params(N, M; m_0=m_0, r_0=r_0, β=β, k_0=k_0, θ=θ, Ω=Ω, η_0=η_0)
# Declare consumer initial conditions
x0 = fill(0.0, (N+M))
for i in 1:N
    x0[i] = 1
end

# Set resource initial conditions
for α in (N+1):(N+M)
    x0[α] = 10
end

# Define integration timespan
tspan = (0.0, 10000.0)

# Generate DiffEq problem
prob = ODEProblem(dx!, x0, tspan, p)

#Solve DiffEq problem with callback and CVODE_BDF integratioon method
sol =solve(prob, CVODE_BDF(), callback=cb)

plot(sol, idxs=[1,2,3,4,5,6,7])
plot(sol, idxs=[6,7,8,9,10])
plot(sol, idxs=[1])

C = sol[1:N, length(sol)]
R = sol[(N+1):(N+M), length(sol)]


