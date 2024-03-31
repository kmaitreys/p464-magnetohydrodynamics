# Solving the galactic dynamo using finite difference Runge-Kutta methods
module GalacticDynamo

using Random
using Plots

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



"""
    magnetic_field_radial(B_r, B_ϕ, V_r, V_z, scale_height, α, dr, dt, η)

Solve the radial component of the magnetic field using the following equation:

# Arguments
- `B_r::Vector{Float64}`: The radial component of the magnetic field
- `B_ϕ::Vector{Float64}`: The azimuthal component of the magnetic field
- `V_r::Vector{Float64}`: The radial velocity of the gas
- `V_z::Vector{Float64}`: The vertical velocity of the gas
- `scale_height::Float64`: The scale height of the gas
- `α::Float64`: The alpha parameter
- `dr::Float64`: The radial step size
- `dt::Float64`: The time step size
- `η::Float64`: The magnetic diffusivity

"""
function magnetic_field_radial(B_r, B_ϕ, V_r, V_z, scale_height, α, dr, dt, η)
    evolution_record = zeros(length(B_r) + 2)



    evolution_record[2:end-1] = dt
end



end