# Libraries
using CUDA
using Plots
using LinearAlgebra
using Random
using FFTW
using JLD
using Roots
using NumericalIntegration
using Dates

# All quantities are expressed in non-dimensional units
# Initialize a square spatial grid [-L/2 ; L/2] x [-L/2 ; L/2]
L  = 10*π                                            # System size
N  = 512                                             # Number of grid nodes
Δx = L / (N-1)                                       # Grid spacing
Δy = L / (N-1)
x  = CuArray{Float64}([-L/2 + i*Δx for i = 0:N-1])   # Space vector
y  = CuArray{Float64}([-L/2 + i*Δy for i = 0:N-1])   # Space vector
# Initialize reciprocal space grid
kx = CuArray{Float64}(2*pi*rfftfreq(Nx, 1/Δx)); Lkx = length(kx)
ky = CuArray{Float64}(2*pi*fftfreq(Ny, 1/Δy))
kx2 = kx.*kx
ky2 = ky.*ky
# FFT tools to calculate derivatives
FT  = plan_rfft(CUDA.ones(Float64, Nx, Ny))    # Operator that Fourier transforms a matrix
IFT = inv(FT)                                  # like FT, but Inverse Fourier transforms

# GPU parameters -- real space mapping
WrapsT = 16                                   # N. of wraps per block
Bx = ceil(Int, N/WrapsT)                      # N. of blocks along x
By = ceil(Int, N/WrapsT)                      # N. of blocks along y
Nx = WrapsT * Bx
Ny = WrapsT * Bx
block_dim = (WrapsT, WrapsT)                  # Dimension of each block
grid_dim = (Bx, By)                           # Dimension of grid
# GPU parameter -- reciprocal space mapping
# Note that we first do rfft along x, then fft along y, hence the asymmetric grid
# The +1 is for the k=0 mode
gridFFT_dim = (div(Bx,2)+1, By)

# Integration time step
Δt = 1e-4

# Model parameters
Dc  = 1e-2
Da  = 1e-1
Di  = 1
Dap = Da + Di
Dam = Da - Di
A   = 1
Kd  = 1
Ω0  = 0.5
Ω   = 15
Ωd  = 10
Z   = 15
B   = Pars["Z"]

# Homogeneous Steady State (used in initial conditions below)
# Solving EqnHSS = 0 for Na gives Na at HSS
EqnHSS(Na) = - (Pars["Ωd"] * Pars["A"] / Pars["Kd"]) * Na^2 + Pars["Ω0"] * (1 - Na) * (1 + Pars["Ω"] * Na^2)
NaHSS = find_zero(EqnHSS, (0, 1))
# Ni and C at HSS are calculated from NaHSS
NiHSS = 1 - NaHSS
CHSS = (Pars["A"] / Pars["Kd"]) * (1 + NmHSS)/2

# Initial Conditions (either HSS + weak noise or HSS + localized bump)
# Parameters of noisy IC
seedd = 123                                            # Seed of rand. number generator
Random.seed!(seedd)                                    # Seeds random number generator
ε   = 0.01                                             # Noise amplitude, small number
Noise = CuArray{Float64}(ε .* (-1 .+ 2 .* rand(Float64, Nx, Ny)))        # Matrix of weak noise, made of random numbers between ε*[-1, 1]
ZeroMeanNoise = CuArray{Float64}(Noise .- integrate((x,y), Noise)/(L^2)) # Noisy vector with zero average, s.t. HSS + Noise conserves tot. number of molecules
# Parameters of localized square bump IC
Bump = CuArray{Float64}(zeros(Nx, Ny))
Bump_center_x = 0                                      # Bump will be centered at (X,Y) = Bump_center_(x,y)
Bump_center_y = 0
for i in 1:Nx, j in 1:Ny                               # Bump has diameter 10 and height 1
    if sqrt(x[i]^2 + y[j]^2) <= 5
        Bump[i, j] = 1
    end
end
ZeroMeanBump = CuArray{Float64}(Bump .- integrate((x,y), Bump)/(L^2))    # Remove average from bump, same reason as for noisy IC above
# Initialize IC
# Uncomment if desired IC is HSS + localized bump
Perturbation = CuArray{Float64}(exp.(-(x./5).^2) + exp.(-(y./5).^2))
C_IC  .= CuArray{Float64}(CHSS        .* (1 .+ ZeroMeanBump))
Na_IC .= CuArray{Float64}(NaHSS       .* (1 .+ ZeroMeanBump))
Ni_IC .= CuArray{Float64}((1 - NaHSS) .* (1 .+ ZeroMeanBump))
# Uncomment if desired IC is HSS + weak noise
# C_IC  .= CuArray{Float64}(CHSS        .* (1 .+ ZeroMeanNoise))
# Na_IC .= CuArray{Float64}(NaHSS       .* (1 .+ ZeroMeanNoise))
# Ni_IC .= CuArray{Float64}((1 - NaHSS) .* (1 .+ ZeroMeanNoise))
# Replace (Na, Ni) → (Nm, Np)
@. Nm_IC = Na_IC - Ni_IC
@. Np_IC = Na_IC + Ni_IC

# Final time reached by simulation
FinalTime = 100

#= # Adaptive timestep parameters
MinTimeStep = 1e-15         # Simulation stops if time step drops below MinTimeStep
ErrorTolerance = 1e-10      # Error tolerance between Δt and 0.5*Δt steps =#

# Saving parameters
# Output file parameters
Nt = FinalTime / Δt
PrintEvery = 10                                   # Save state of system with time intervals = FrameTimeStep (in non-dim. units)
TotPrints = floor(Int64, FinalTime / PrintEvery)  # Total number of system states saved during the simulation
dt = Dates.format(now(), "yyyymmdd")              # String of today's date
DataFileName = "Data/"*dt*"-Data.jld"             # Name of output file