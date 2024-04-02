# Solving the galactic dynamo using finite difference Runge-Kutta methods
module GalacticDynamo

using Random
using Plots
using LaTeXStrings
using ProgressMeter
using Statistics

Plots.default(titlefont=("computer modern"), legendfont=("computer modern"),
    guidefont=("computer modern"), tickfont=("computer modern"))

# function gradient(arr, dr)
#     n = length(arr)
#     grad = similar(arr)
#     grad[1] = (arr[2] - arr[1]) / dr
#     grad[end] = (arr[end] - arr[end-1]) / dr
#     for i in 2:length(arr)-1
#         grad[i] = (arr[i+1] - arr[i-1]) / (2 * dr)
#     end
#     return grad
# end
function gradient(arr, r)
    n = length(arr)
    grad = similar(arr)
    grad[1] = (arr[2] - arr[1]) / (r[2] - r[1])
    grad[end] = (arr[end] - arr[end-1]) / (r[end] - r[end-1])
    for i in 2:n-1
        grad[i] = (arr[i+1] - arr[i-1]) / (r[i+1] - r[i-1])
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

# function evolve_magnetic_field(R, T, Nr, Nt, η_t)
#     dr = R / (Nr - 1)
#     dt = T / Nt
#     R_vals = LinRange(0, R, Nr)
#     B_r = @. sin(4 * π * R_vals / R)
#     evolution_record = zeros(Nr, Nt)
#     for i in 1:Nt
#         runge_kutta_fourth_order!(B_r, dr, dt, η_t)
#         evolution_record[:, i] .= B_r
#     end
#     return evolution_record
# end



"""
    magnetic_field_radial(B_r, B_ϕ, V_r, V_z, scale_height, α, dr, dt, η)

Solve the radial component of the magnetic field using the following equation:

# Arguments
- `B_r::Vector{Float64}`: The radial component of the magnetic field
- `B_ϕ::Vector{Float64}`: The azimuthal component of the magnetic field
- `V_r::Vector{Float64}`: The radial velocity of the gas
- `V_z::Vector{Float64}`: The vertical velocity of the gas
- `H::Float64`: The scale height of the gas
- `η::Float64`: The magnetic diffusivity
- `α::Float64`: The alpha parameter
- `Ω::Vector{Float64}`: The angular velocity of the gas
- `dr::Float64`: The radial step size
- `dt::Float64`: The time step size

# Returns
- `evolution_record_radial::Vector{Float64}`: The radial component of the magnetic field at the next time step
"""
function magnetic_field_radial(B_r, B_ϕ, V_r, V_z, r, H, η, α, Ω, dr, dt)
    radial_field = zeros(length(B_r))

    for i = 2:(lastindex(B_r)-1)
        tmp_1 = V_r[i] / (2 * dr[i]) + η / dr[i]^2 - η / (2 * r[i] * dr[i])
        tmp_2 = (
            (V_r[i-1] - V_r[i+1]) / (2 * dr[i])
            -
            V_z[i] / (4 * H[i])
            -
            (2 * η) / dr[i]^2
            -
            η / (r[i]^2)
            -
            (η * π^2) / (4 * H[i]^2)
        )
        tmp_3 = (
            (-V_r[i] / (2 * dr[i]))
            + η / dr[i]^2
            + η / (2 * r[i] * dr[i])
        )

        tmp_4 = (
            -r[i] * (Ω[i+1] - Ω[i-1]) / (2 * dr[i])
            +
            (2 * α[i]) / (π * H[i])
        )

        radial_field[i] = dt * (
            tmp_1 * B_ϕ[i-1]
            + tmp_2 * B_ϕ[i]
            + tmp_3 * B_ϕ[i+1]
            + tmp_4 * B_r[i]
        )
    end
    return radial_field
end

"""
    magnetic_field_azimuthal(B_r, B_ϕ, V_r, V_z, scale_height, α, dr, dt, η)

Solve the azimuthal component of the magnetic field using the following equation:

# Arguments
- `B_r::Vector{Float64}`: The radial component of the magnetic field
- `B_ϕ::Vector{Float64}`: The azimuthal component of the magnetic field
- `V_r::Vector{Float64}`: The radial velocity of the gas
- `V_z::Vector{Float64}`: The vertical velocity of the gas
- `H::Float64`: The scale height of the gas
- `η::Float64`: The magnetic diffusivity
- `α::Float64`: The alpha parameter
- `Ω::Vector{Float64}`: The angular velocity of the gas
- `dr::Float64`: The radial step size
- `dt::Float64`: The time step size

# Returns
- `evolution_record_azimuthal::Vector{Float64}`: The azimuthal component of the magnetic field at the next time step
"""
function magnetic_field_azimuthal(B_r, B_ϕ, V_r, V_z, r, H, η, α, Ω, dr, dt)
    azimuthal_field = zeros(length(B_ϕ))

    for i = 2:(lastindex(B_r)-1)
        tmp_1 = V_r[i] / (2 * dr[i]) + η / dr[i]^2 - η / (2 * r[i] * dr[i])
        tmp_2 = (
            (V_r[i-1] - V_r[i+1]) / (2 * dr[i])
            -
            V_z[i] / (4 * H[i])
            -
            (2 * η) / dr[i]^2
            -
            η / (r[i]^2)
            -
            (η * π^2) / (4 * H[i]^2)
        )
        tmp_3 = (
            (-V_r[i] / (2 * dr[i]))
            + η / dr[i]^2
            + η / (2 * r[i] * dr[i])
        )
        tmp_4 = (
            -r[i] * (Ω[i+1] - Ω[i-1]) / (2 * dr[i])
            +
            (2 * α[i]) / (π * H[i])
        )

        azimuthal_field[i] = dt * (
            tmp_1 * B_ϕ[i-1]
            + tmp_2 * B_ϕ[i]
            + tmp_3 * B_ϕ[i+1]
            + tmp_4 * B_r[i]
        )
    end
    return azimuthal_field
end

function get_pitch_angle(B_r, B_ϕ)
    return atan.(B_ϕ ./ B_r)
end


function runge_kutta_fourth_order_coupled(B_r, B_ϕ, V_r, V_z, r, H, η, α, Ω, dr, dt)
    k1_r = dt * magnetic_field_radial(B_r, B_ϕ, V_r, V_z, r, H, η, α, Ω, dr, dt)
    k1_ϕ = dt * magnetic_field_azimuthal(B_r, B_ϕ, V_r, V_z, r, H, η, α, Ω, dr, dt)

    k2_r = dt * magnetic_field_radial(B_r + 0.5 * k1_r, B_ϕ + 0.5 * k1_ϕ, V_r, V_z, r, H, η, α, Ω, dr, dt)
    k2_ϕ = dt * magnetic_field_azimuthal(B_r + 0.5 * k1_r, B_ϕ + 0.5 * k1_ϕ, V_r, V_z, r, H, η, α, Ω, dr, dt)

    k3_r = dt * magnetic_field_radial(B_r + 0.5 * k2_r, B_ϕ + 0.5 * k2_ϕ, V_r, V_z, r, H, η, α, Ω, dr, dt)
    k3_ϕ = dt * magnetic_field_azimuthal(B_r + 0.5 * k2_r, B_ϕ + 0.5 * k2_ϕ, V_r, V_z, r, H, η, α, Ω, dr, dt)

    k4_r = dt * magnetic_field_radial(B_r + k3_r, B_ϕ + k3_ϕ, V_r, V_z, r, H, η, α, Ω, dr, dt)
    k4_ϕ = dt * magnetic_field_azimuthal(B_r + k3_r, B_ϕ + k3_ϕ, V_r, V_z, r, H, η, α, Ω, dr, dt)

    B_r .+= (k1_r + 2 * k2_r + 2 * k3_r + k4_r) / 6
    B_ϕ .+= (k1_ϕ + 2 * k2_ϕ + 2 * k3_ϕ + k4_ϕ) / 6

    return B_r, B_ϕ
end

function evolve_magnetic_field(B_r, B_ϕ, V_r, V_z, r, H, η, α, Ω, dr, dt)
    Nr = length(B_r)
    Nt = length(dt)
    evolution_B_r = zeros(Nr, Nt)
    evolution_B_ϕ = zeros(Nr, Nt)
    @showprogress for i in eachindex(dt)
        (
            B_r, B_ϕ
        ) = runge_kutta_fourth_order_coupled(B_r, B_ϕ, V_r, V_z, r, H, η, α, Ω, dr, dt[i])
        evolution_B_r[:, i] .= B_r
        evolution_B_ϕ[:, i] .= B_ϕ
    end
    return (
        evolution_B_r,
        evolution_B_ϕ
    )
end


function solve_diffusion()
    step_size_r = 0.01
    r_min = 0.0
    r_max = 10.0

    step_size_t = 0.01
    t_min = 0.0
    t_max = 10.0



    # Create the radial grid
    r = r_min:step_size_r:r_max
    r = collect(r)


    H = 0.1 * ones(size(r))

    t = t_min:step_size_t:t_max
    t = collect(t)
    # Initial conditions
    B_r = @. sin(0.4 * sin(10π * r / (r_max - step_size_r)) + π * r / (r_max - step_size_r))
    B_ϕ = @. -sin(0.4 * sin(4π * r / (r_max - step_size_r)) + π * r / (r_max - step_size_r))

    Ω = zeros(size(r))
    α = zeros(size(r))

    V_r = zeros(size(r))
    V_z = zeros(size(r))

    η_m = 0.02
    η_t = 0.2
    η = η_m + η_t

    p1 = plot(r, B_r, label="B_r", title="Initial conditions")
    p2 = plot(r, B_ϕ, label="B_ϕ", title="Initial conditions")

    plot(p1, p2, layout=(2, 1), legend=:bottomright)

    # Solve the diffusion equation
    dr = diff(r)
    dt = diff(t)

    (
        evolution_B_r,
        evolution_B_ϕ,
    ) = GalacticDynamo.evolve_magnetic_field(B_r, B_ϕ, V_r, V_z, r, H, η, α, Ω, dr, dt)

    # Plot heatmaps for the time evolution of B_r and B_ϕ
    heatmap(
        t, r[1:end-1], evolution_B_r',
        xlabel="Radial Position", ylabel="Time",
        title="Time Evolution of \$B_r\$"
    )


    savefig("B_r.png")
    heatmap(
        t, r[1:end-1], evolution_B_ϕ',
        xlabel="Radial Position", ylabel="Time",
        title="Time Evolution of \$B_\\phi\$"
    )
    savefig("B_ϕ.png")

    return evolution_B_r, evolution_B_ϕ

end


function evolve_pitch_angle(evolution_B_r, evolution_B_ϕ)
    @showprogress for i in eachindex(evolution_B_r[1, :])
        B_r = evolution_B_r[:, i]
        B_ϕ = evolution_B_ϕ[:, i]
        pitch_angle = get_pitch_angle(B_r, B_ϕ)

        # Animation
        anim = @animate for i in 1:10:length(pitch_angle)
            plot(pitch_angle[1:i], label="Pitch Angle", xlabel="Radial Position", ylabel="Pitch Angle")
        end
        # Save the animation
        gif(anim, "pitch_angle.gif", fps=15)
    end
end

function solve_mean_field(
    fB_r::Function,
    fB_ϕ::Function,
    fΩ::Function,
    fα::Function,
    fV_r::Function,
    fV_z::Function,
    rmax::Float64,
    tmax::Float64,
    H::Float64,
    η_m::Float64,
    η_t::Float64,
    step_size_r::Float64,
    step_size_t::Float64,
)
    rmin = 0.0
    tmin = 0.0

    r = rmin:step_size_r:rmax
    r = collect(r)
    dr = diff(r)

    H = H * ones(size(r))

    t = tmin:step_size_t:tmax
    t = collect(t)
    dt = diff(t)

    η = η_m + η_t

    Ω = fΩ(r)
    α = fα(r)
    V_r = fV_r(r)
    V_z = fV_z(r)

    q_Ω = -r .* gradient(Ω, r)
    D_c = -α[1] * q_Ω * H[3]^3 / η^2
    # Take average of D_c
    D_c = mean(D_c)

    B_r = fB_r(r)
    B_ϕ = fB_ϕ(r)

    (
        evolution_B_r,
        evolution_B_ϕ,
    ) = GalacticDynamo.evolve_magnetic_field(B_r, B_ϕ, V_r, V_z, r, H, η, α, Ω, dr, dt)

    # Plot heatmaps for the time evolution of B_r and B_ϕ
    heatmap(
        t, r[1:end-1], evolution_B_r',
        xlabel="Radial Position", ylabel="Time",
        title="Time Evolution of \$B_r\$"
    )

    savefig("B_r_dynamo.png")

    heatmap(
        t, r[1:end-1], evolution_B_ϕ',
        xlabel="Radial Position", ylabel="Time",
        title="Time Evolution of \$B_\\phi\$"
    )

    savefig("B_ϕ_dynamo.png")

    return evolution_B_r, evolution_B_ϕ
end

function evolve_magnetic_energy(evolution_B_r, evolution_B_ϕ)
    magnetic_energy = zeros(size(evolution_B_r)[2])
    for i in 1:size(evolution_B_r)[2]
        B_r = evolution_B_r[:, i]
        B_ϕ = evolution_B_ϕ[:, i]
        magnetic_energy[i] = @. sqrt(sum(B_r^2 + B_ϕ^2))
        println(magnetic_energy[i])
    end
    plot(magnetic_energy, xlabel="Time", ylabel="Magnetic Energy", title="Magnetic Energy Evolution")
    savefig("magnetic_energy.png")
    return magnetic_energy
end


end

(
    evolution_B_r,
    evolution_B_ϕ,
) = GalacticDynamo.solve_diffusion()


# GalacticDynamo.evolve_pitch_angle(evolution_B_r, evolution_B_ϕ)
function fB_r(r)
    return @. -(r - 5)^2 + 25
end

function fB_ϕ(r)
    return @. (r - 5)^2 - 25
end

function fΩ(r)
    return @. 10 / sqrt(1 + (r / 4)^2)
end

function fα(r)
    return ones(size(r))
end

function fV_r(r)
    return zeros(size(r))
end

function fV_z(r)
    return zeros(size(r))
end

(
    evolution_B_r,
    evolution_B_ϕ,
) = GalacticDynamo.solve_mean_field(
    fB_r,
    fB_ϕ,
    fΩ,
    fα,
    fV_r,
    fV_z,
    10.0,
    10.0,
    0.1,
    0.02,
    0.2,
    0.01,
    0.01,
)

magnetic_energy = GalacticDynamo.evolve_magnetic_energy(evolution_B_r, evolution_B_ϕ)