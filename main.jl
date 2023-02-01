# Set current directory as working directory
cd(@__DIR__)
include("kernels.jl")

# If absent, create Data/ folder
if isdir("Data") == false
    mkdir("Data")
end

function main()
    # Initialize dynamical fields
    C      = CuArray{Float64}(zeros(Float64, Nx, Ny))         # Actomyosin
    Np     = CuArray{Float64}(zeros(Float64, Nx, Ny))         # Sum of active and inactive nucleators
    Nm     = CuArray{Float64}(zeros(Float64, Nx, Ny))         # Difference btw active and inactive nucleators
    V      = CuArray{Float64}(zeros(Float64, Nx, Ny, 2))      # Velocity (1 = x, 2 = y)
    Π      = CuArray{Float64}(zeros(Float64, Nx, Ny))         # Non-viscous stress field

    #= # Initialize auxiliary fields used in AdaptiveTimeStep!() of kernels.jl
    C_Aux   = CuArray{Float64}(zeros(Float64, Nx, Ny))
    Np_Aux  = CuArray{Float64}(zeros(Float64, Nx, Ny))
    Nm_Aux  = CuArray{Float64}(zeros(Float64, Nx, Ny))
    # Δt-propagated fields
    C_LongStep      = CuArray{Float64}(zeros(Float64, Nx, Ny))
    Np_LongStep     = CuArray{Float64}(zeros(Float64, Nx, Ny))
    Nm_LongStep     = CuArray{Float64}(zeros(Float64, Nx, Ny))
    V_LongStep      = CuArray{Float64}(zeros(Float64, Nx, Ny, 2))
    # 1 x 0.5*Δt-propagated fields
    C_ShortStep_Aux   = CuArray{Float64}(zeros(Float64, Nx, Ny))
    Np_ShortStep_Aux  = CuArray{Float64}(zeros(Float64, Nx, Ny))
    Nm_ShortStep_Aux  = CuArray{Float64}(zeros(Float64, Nx, Ny))
    V_ShortStep_Aux   = CuArray{Float64}(zeros(Float64, Nx, Ny, 2))
    # 2 x 0.5*Δt-propagated fields
    C_ShortStep   = CuArray{Float64}(zeros(Float64, Nx, Ny))
    Np_ShortStep  = CuArray{Float64}(zeros(Float64, Nx, Ny))
    Nm_ShortStep  = CuArray{Float64}(zeros(Float64, Nx, Ny))
    V_ShortStep   = CuArray{Float64}(zeros(Float64, Nx, Ny, 2)) =#

    # Initialize RHS (s.t. ∂t A = RHS_A)
    RHS_C  = CuArray{Float64}(zeros(Float64, Nx, Ny)) 
    RHS_Np = CuArray{Float64}(zeros(Float64, Nx, Ny))
    RHS_Nm = CuArray{Float64}(zeros(Float64, Nx, Ny))
    # Initialize divergences ∂(AV)
    ∂CV    = CuArray{Float64}(zeros(Float64, Nx, Ny, 2))
    ∂NpV   = CuArray{Float64}(zeros(Float64, Nx, Ny, 2))
    ∂NmV   = CuArray{Float64}(zeros(Float64, Nx, Ny, 2))
    # Initialize divergences div_AV = ∇⋅(A𝐕)
    div_CV    = CuArray{Float64}(zeros(Float64, Nx, Ny))
    div_NpV   = CuArray{Float64}(zeros(Float64, Nx, Ny))
    div_NmV   = CuArray{Float64}(zeros(Float64, Nx, Ny))
    # Initialize laplacians ΔA = ∇⋅∇A
    ΔC    = CuArray{Float64}(zeros(Float64, Nx, Ny))
    ΔNp   = CuArray{Float64}(zeros(Float64, Nx, Ny))
    ΔNm   = CuArray{Float64}(zeros(Float64, Nx, Ny))

    # Factors for FFT derivatives
    factor_∂x = CUDA.zeros(ComplexF64, length(kx), Nx)
    factor_∂y = CUDA.zeros(ComplexF64, length(kx), Ny)
    factor_Δ = CUDA.zeros(Float64, length(kx), Nx)

    @cuda threads = block_dim blocks = gridFFT_dim kernel_compute_FFTderivative_factors!(factor_∂x, factor_∂y, factor_Δ, kx, ky, kx2, ky2)

    # FFT
    fV = CUDA.zeros(ComplexF64, Lkx, Ny, 2)
    fΠ = CUDA.zeros(ComplexF64, Lkx, Ny)

    #= # Initialize vectors where fields are saved for plots
    SavedFields = CuArray{Float64}(zeros(3, Nx, Ny, TotPrints))
    Savedv      = CuArray{Float64}(zeros(2, Nx, Ny, TotPrints)) =#

#=     # Initialize time evolution parameters
    ThisFrame    = 1             # Used to save system state in SavedFields and Savedv matrices, ∈ [1, TotPrints]
    LastSaveTime = 0             # Last time system state was saved, in non-dim. units
    CurrentTime  = 0             # Real time at current time frame
    Δt_old       = 0.01 =#

    # Parameters of localized square bump IC
    ZeroMeanBump = CuArray{Float64}(zeros(Nx, Ny))
    get_ZeroMeanBump!(ZeroMeanBump, L, 2^0.5, Δx, Δy)
    # Initial Conditions
    # Uncomment if desired IC is HSS + localized bump
    @. C[:,:]  = CHSS          * (1 + ZeroMeanBump)
    @. Nm[:,:] = (2*NaHSS - 1) * (1 + ZeroMeanBump)
    @. Np[:,:] =                  1 + ZeroMeanBump

    # 🚧 COPY LUDO'S FILE HANDLING 🚧
    #mkpath(string(file,"Data/")); global nm = ""

    println("Simulation starts...")
    # Simulation runs either until FinalTime reached, or until Δt drops below tolerance
    CUDA.@time for t in 0:Nt
        if (t%PrintEvery==0)
             any(isnan, C) && return 1
             numm = @sprintf("%08d", t)  #print time iteration step, number format
             #global nm = string(file,"Data/data",numm,".jld")
             global nm = string("Data/data",numm,".jld")
             save(nm, "actin", Array(C), "active_nucleator", Array((Np .+ Nm)/2), "inactive_nucleator", Array((Np .- Nm)/2), "velocity", Array(V))
             #println(idx, " ", t)
             println(t)
        end
        EulerForward!(Δt, V, Π, fV, fΠ, FT, IFT, Lkx, factor_∂x, factor_∂y, factor_Δ, Z, B, C, Np, Nm, ΔC, ΔNp, ΔNm, ∂CV, ∂NpV, ∂NmV, div_CV, div_NpV, div_NmV, RHS_C, RHS_Np, RHS_Nm, Dc, Dap, Dam, A, Kd, Ω0, Ω, Ωd)
    end

#=     while (CurrentTime < FinalTime && Δt_old > MinTimeStep)
        Δt = 2*Δt_old       # Double latest time step not to end up with very small Δt
        # Propagate fields
        AdaptiveTimeStep!(ErrorTolerance, Δt_old, Δt, Pars, v, Fields, Fields_Aux, v_LongStep, Fields_LongStep, v_ShortStepAux, Fields_ShortStepAux, v_ShortStep, Fields_ShortStep, ∂xx, ∂xFieldsv, σ, RHS, k, P, Pinv)
        CurrentTime += Δt_old  # Update current time (Δt_old is the adapted time step)
        # 🚧 THE SAVING PART NEEDS TO BE OPTIMIZED, LIKE LUDO'S 🚧
        # Save system state if a time interval ≃ FrameTimeStep has passed since last save
        if CurrentTime - LastSaveTime >= FrameTimeStep
            LastSaveTime = CurrentTime
            PrintCurrentTime = round(CurrentTime, digits = 3) # This is to limit the digits of CurrentTime when printing
            println("Reached time $PrintCurrentTime of $FinalTime")
            ThisFrame += 1
            Savedv[:, ThisFrame]  = v # 🚧 HOW TO DO THIS IN GPU? 🚧
            for i in 1:3
                SavedFields[i, :, ThisFrame] .= Fields[i, :] # 🚧 HOW TO DO THIS IN GPU? 🚧
            end
        end
        # Export saved data if either FinalTime reached, or if Δt drops below MinTimeStep
        if (CurrentTime >= FinalTime || Δt_old <= MinTimeStep)
            # Print message if Δt drops below tolerance
            if Δt_old <= MinTimeStep
                println("Simulation stops at frame $ThisFrame/$TotPrints. Time step dropped below $MinTimeStep")
            end
            save(DataFileName, "actomyosin", SavedFields[1, :, 1:ThisFrame], "active_nucleator", SavedFields[2, :, 1:ThisFrame], "inactive_nucleator", SavedFields[3, :, 1:ThisFrame], "velocity", Savedv[:, 1:ThisFrame])
            println("Data saved, simulation is over.")
        end
    end =#

    return nothing
end

# Launch simulation
main()

c = load("Data/data00000000.jld", "actin")

surface(x, y, c)