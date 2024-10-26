function Mod_growth!(dx, x, p, t, i)

    # maintenance
    dx[i] += -x[i]*p.m[i]

    # resoruce uptake and leakage
    G = zeros(p.M)
    for α = 1:p.M
        G[α] += x[α + p.N]*p.r[i]/(p.k[i, α] + x[α + p.N])
    end

    dx[i] += x[i]*minimum(G)
end

function Mod_supply!(dx, x, p, t, α)

    # inflow
    dx[α + p.N] += p.ρ[α] 


end

function Mod_depletion!(dx, x, p, t, α, i)
    G = zeros(p.M)
    for β = 1:p.M
        G[β] = x[β + p.N]*p.r[i]/(p.k[i, β] + x[β + p.N])
    end
    dx[α + p.N] += -x[i]*p.r[i]*p.η[i, α]*minimum(G) - p.ω[α]*x[α+ p.N]
end



function dx!(dx, x, p, t;
    growth!::Function = Mod_growth!,
    supply!::Function = Mod_supply!,
    depletion!::Function = Mod_depletion!)

    for i in 1:p.N
        # reset derivatives
        dx[i] = 0.0

        # Allee effect
        if x[i] > 1e-1
            # update dx of ith consumer
            growth!(dx, x, p, t, i)
        end
    end

    for α in 1:p.M
        # reset derivatives
        dx[p.N + α] = 0.0

        #supply term
        supply!(dx, x, p, t, α)

        # loop over consumers
        for i in 1:p.N
            if x[i] > 1e-5
                depletion!(dx, x, p, t, α, i)
            end
        end
    end

end