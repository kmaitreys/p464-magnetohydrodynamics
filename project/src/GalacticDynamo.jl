# Solving the galactic dynamo using finite difference Runge-Kutta methods
# module GalacticDynamo

using Random
using Plots
## Task 1
## 1.1
# Solving the diffusion equation in `r` (under the no-Z) approximation
function gradient(arr, dr)
    n = length(arr)
    grad = similar(arr)
    grad[1] = (arr[2] - arr[1]) / dr
    grad[end] = (arr[end] - arr[end-1]) / dr
    for i in 2:length(arr)-1
        grad[i] = (arr[i+1] - arr[i-1]) / (2 * dr)
    end
    return grad
end


function get_laplacian(y::Vector{Float64}, dr)
    return gradient(gradient(y, dr), dr)
end

function runge_kutta_fourth_order!(y::Vector{Float64}, dr, dt, η_t)
    k1 = dt * η_t * get_laplacian(y, dr)
    k2 = dt * η_t * get_laplacian(y + 0.5 * k1, dr)
    k3 = dt * η_t * get_laplacian(y + 0.5 * k2, dr)
    k4 = dt * η_t * get_laplacian(y + k3, dr)
    y .+= (k1 + 2 * k2 + 2 * k3 + k4) / 6
end

function evolve_magnetic_field(R, T, Nr, Nt, η_t)
    dr = R / (Nr - 1)
    dt = T / Nt
    R_vals = LinRange(0, R, Nr)
    B_r = @. sin(4 * π * R_vals / R)
    evolution_record = zeros(Nr, Nt)
    for i in 1:Nt
        runge_kutta_fourth_order!(B_r, dr, dt, η_t)
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
function plot_every_10th_row(data::Matrix)
    num_rows = size(data, 1)
    p = plot(legend=:topleft)  # Initialize plot object
    for i in 1:20:num_rows
        plot!(p, abs.(data[i, :]), label="Row $i")
    end
    xlabel!("Index")
    ylabel!("Value")
    title!("Plot of Every 10th Row")
    return p
end

plot_every_10th_row(evolution_record)


# function decay(t, A, τ)
#     return A * exp(-t / τ)
# end

# function get_pitch_angle(B_r, B_φ)
#     return atan(B_φ, B_r)
# end

# @. decay(t, A, τ) = A * exp(-t / τ)
# time_points = LinRange(0, 5, 100)
# magnetic_field_magnitude = decay






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


# end