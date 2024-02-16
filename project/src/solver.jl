# Diffusion equation solver
using LinearAlgebra
using SparseArrays
using BenchmarkTools

function solve_diffusion_equation(n, m, Δt, Δx, D, u0, T)
    # Create the grid
    x = range(0, stop=Δx*(n-1), length=n)
    t = range(0, stop=Δt*(m-1), length=m)
    u = zeros(n, m)
    u[:,1] .= u0.(x)
    
    # Create the matrix
    A = spdiagm(-1 => -D*Δt/Δx^2 * ones(n-1),
                0 => (1 + 2*D*Δt/Δx^2) * ones(n),
                1 => -D*Δt/Δx^2 * ones(n-1))
    
    # Time-stepping loop
    for j in 2:m
        u[2:end,j] .= A * u[2:end,j-1]
    end
    return u
end

"""
    solve_diffusion_equation!(u, Δt, Δx, D, u0, T)

Solve the diffusion equation u_t = D u_xx on the interval [0, T] with Dirichlet boundary conditions
u(0, t) = u0(0), u(T, t) = u0(T) and initial condition u(x, 0) = u0(x).

# Arguments
- `u::Array{Float64,2}`: The solution array
- `Δt::Float64`: The time step
- `Δx::Float64`: The spatial step
- `D::Float64`: The diffusion coefficient
- `u0::Function`: The initial condition
- `T::Float64`: The final time

# Example
```julia
n, m = 100, 100
Δt, Δx = 0.01, 0.01
D = 0.1
u0(x) = exp(-x^2)
T = 1.0
u = zeros(n, m)
solve_diffusion_equation!(u, Δt, Δx, D, u0, T)
```

# References
- https://en.wikipedia.org/wiki/Finite_difference_method

# See also
- `solve_diffusion_equation`
"""
function solve_diffusion_equation!(u, Δt, Δx, D, u0, T)
    # Create the grid
    n, m = size(u)
    x = range(0, stop=Δx*(n-1), length=n)
    t = range(0, stop=Δt*(m-1), length=m)
    u[:,1] .= u0.(x)
    
    # Create the matrix
    A = spdiagm(-1 => -D*Δt/Δx^2 * ones(n-1),
                0 => (1 + 2*D*Δt/Δx^2) * ones(n),
                1 => -D*Δt/Δx^2 * ones(n-1))
    
    # Time-stepping loop
    for j in 2:m
        u[2:end,j] .= A * u[2:end,j-1]
    end
    return u
end

# Magneto-hydrodynamics solver
function LarmorRadius(mass, charge, velocity, B)
    return mass * velocity / (charge * B)
end

function solve_mhd_equation(n, m, Δt, Δx, ρ, μ, B0, u0, T)
    # Create the grid
    x = range(0, stop=Δx*(n-1), length=n)
    t = range(0, stop=Δt*(m-1), length=m)
    u = zeros(n, m)
    u[:,1] .= u0.(x)
    
    # Time-stepping loop
    for j in 2:m
        u[2:end,j] .= u[2:end,j-1]
    end
    return u
end