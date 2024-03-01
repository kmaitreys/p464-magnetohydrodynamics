# Solving the galactic dynamo using finite difference Runge-Kutta methods
using Random
using Plots
## Task 1
## 1.1
# Solving the diffusion equation in `r` (under the no-Z) approximation
function gradient(arr::Vector{Float64}, dr)
    return (arr[2:end] .- arr[1:end-1]) / dr
end


function get_laplacian(y::Vector{Float64}, dr)
    return gradient(gradient(y, dr), dr)
end

function runge_kutta_fourth_order!(y::Vector{Float64}, dr, η_t)
    # println("y: ", y)
    k1 = dt * η_t * get_laplacian(y, dr)
    println(size(k1))
    k2 = dt * η_t * get_laplacian(y + 0.5 * k1, dr)
    k3 = dt * η_t * get_laplacian(y + 0.5 * k2, dr)
    k4 = dt * η_t * get_laplacian(y + k3, dr)
    y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y
end

function evolve_magnetic_field(R, T, Nr, Nt, η_t)
    dr = R / (Nr - 1)
    R_vals = LinRange(0, R, Nr)
    B_r = @. sin(4 * π * R_vals / R)
    evolution_record = zeros(Nr, Nt)
    for i in 1:Nt
        runge_kutta_fourth_order!(B_r, dr, η_t)
        evolution_record[:, i] .= B_r
    end
    return evolution_record
end

# Parameters
η_t = 0.5 # Scaled turbulent magnetic diffusivity

# Radial grid
R = 10.0
Nr = 200
# Time
T = 5.0
Nt = 500
# Run simulation
evolution_record = evolve_magnetic_field(R, T, Nr, Nt, η_t)

# Plots 

# Time evolution of the magnetic field

T = LinRange(0, T, Nt)
plot(T, evolution_record, label="Magnetic field evolution", xlabel="Time", ylabel="Magnetic field")









# function init_magnetic_field(R)
#     seed = rand
# end


# Importing the Random module
# using Random

# # Array of strings containing 'sin' and 'cos'
# functions = ["sin", "cos"]

# # Assuming R is an array or sequence
# R = [1, 2, 3, 4, 5]  # Example array, replace this with your actual data

# # Generating random choices
# random_choices = rand(functions, length(R))

# # Printing the result
# println(random_choices)


