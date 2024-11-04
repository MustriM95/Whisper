# This code takes parameters, simulates a community and returns
# variables of interest.

condition(du, t, integrator) = norm(integrator(t, Val{1})) <= 1e-5

affect!(integrator) = terminate!(integrator)

cb = DiscreteCallback(condition, affect!)

function MiC_test(; p, t_span = 10000)
    ## Building the ODEs and initializing

    x0 = fill(0.0, (N+M))
    for i in 1:N
        x0[i] = 2
    end

    # Set resource initial conditions
    for α in (N+1):(N+M)
        x0[α] = 10
    end

    # Define integration timespan
    tspan = (0.0, t_span)

    # Generate DiffEq problem
    prob = ODEProblem(dx!, x0, tspan, p)

    #Solve DiffEq problem with callback and CVODE_BDF integratioon method
    sol = solve(prob, CVODE_BDF(), callback=cb)

    dtmin_err = SciMLBase.successful_retcode(sol)
    SAD = ones(N,2)

    if dtmin_err == true

        C = sol[1:N, length(sol)]
        R = sol[(N+1):(N+M), length(sol)]

        sizes = zeros(N)
        for i in 1:p.N
            sizes[i] = (0.1*4^(i-1))
        end

        SAD[:, 1] = sizes
        SAD[:, 2] = C
    end

    return SAD
 end