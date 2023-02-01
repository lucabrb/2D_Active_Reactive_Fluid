include("InputParameters.jl")

function kernel_bump!(bump, L, VarBump, Δx, Δy)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    @inbounds bump[i,j] = exp(-((i*Δx-L/2)^2 + (j*Δy-L/2)^2)/(2*(VarBump^2)))
    return nothing
end

@inline function get_ZeroMeanBump!(bump, L, VarBump, Δx, Δy)
    @cuda threads = block_dim blocks = grid_dim kernel_bump!(bump, L, VarBump, Δx, Δy)
    bump .-= integrate((x,y), bump)/(L^2)
end

function kernel_compute_FFTderivative_factors!(factor_∂x, factor_∂y, factor_Δ, kx, ky, kx2, ky2) # Indexing (i,j)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i>length(kx)
        return nothing
    end
    factor_∂x[i,j] = im*kx[i]
    factor_∂y[i,j] = im*ky[j]
    factor_Δ[i,j] = -kx2[i]-ky2[j]
    return nothing
end 

function kernel_compute_FFT_v!(fV, fΠ_, kx_, ky_, Lkx)  # Indexing (i,j)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i>Lkx
        return nothing
    end
    kx = kx_[i];     kx2 = kx*kx
    ky = ky_[j];     ky2 = ky*ky

    fΠ = fΠ_[i, j]
    
    fV[i,j,1] = im * kx * fΠ / (1 + 2*(kx2 + ky2))
    fV[i,j,2] = im * ky * fΠ / (1 + 2*(kx2 + ky2))

    return nothing
end

function kernel_compute_RHS!(C_, Np_, Nm_, ΔC_, ΔNp_, ΔNm_, ∂CV_, ∂NpV_, ∂NmV_, div_CV, div_NpV, div_NmV, RHS_C, RHS_Np, RHS_Nm, Dc, Dap, Dam, A, Kd, Ω0, Ω, Ωd)  # Indexing (i,j)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    C  = C_[i,j]
    Np = Np_[i,j]
    Nm = Nm_[i,j]

    ΔC = ΔC_[i, j]
    ΔNp = ΔNp_[i, j]
    ΔNm = ΔNm_[i, j]

    div_CV =  ∂CV_[i,j,1]  + ∂CV_[i,j,2]
    div_NpV = ∂NpV_[i,j,1] + ∂NpV_[i,j,2]
    div_NmV = ∂NmV_[i,j,1] + ∂NmV_[i,j,2]
    
    RHS_C[i,j]  = - div_CV  + Dc * ΔC + A * 0.5 * (Np + Nm) - Kd * C
    RHS_Np[i,j] = - div_NpV + 0.5 * (Dap * ΔNp + Dam * ΔNm)
    RHS_Nm[i,j] = - div_NmV + 0.5 * (Dam * ΔNp + Dap * ΔNm) + Ω0 * ( 1 + Ω * 0.25 * (Np + Nm)^2) * (Np - Nm) - Ωd * C * (Np + Nm)

    return nothing
end

function compute_RHS!(V, Π, fV, fΠ, FT, IFT, Lkx, factor_∂x, factor_∂y, factor_Δ, Z, B, C, Np, Nm, ΔC, ΔNp, ΔNm, ∂CV, ∂NpV, ∂NmV, div_CV, div_NpV, div_NmV, RHS_C, RHS_Np, RHS_Nm, Dc, Dap, Dam, A, Kd, Ω0, Ω, Ωd)
    
    Π[:,:] .= Z*(C[:,:]^2) - B*(C[:,:]^3)
    # 🚧 Might as well use the relation in Bois et al, PRL (2011)? 🚧
    # Π[:,:] .= Z*C[:,:]/(1+C[:,:])

    @inbounds @views begin
        fΠ[:,:] .= FT * Π[:,:]
    end

    @cuda threads = block_dim blocks = gridFFT_dim kernel_compute_FFT_v!(fV, fΠ, kx, ky, Lkx)

    @inbounds @views begin
        V[:,:,1]  .= IFT * fV[:,:,1]
        V[:,:,2]  .= IFT * fV[:,:,2]
    end

    @inbounds @views begin
        ∂CV[:,:,1]  .= IFT * (factor_∂x .* (FT * (C[:,:] .* V[:,:,1])))
        ∂CV[:,:,2]  .= IFT * (factor_∂y .* (FT * (C[:,:] .* V[:,:,2])))

        ∂NpV[:,:,1] .= IFT * (factor_∂x .* (FT * (Np[:,:] .* V[:,:,1])))
        ∂NpV[:,:,2] .= IFT * (factor_∂y .* (FT * (Np[:,:] .* V[:,:,2])))

        ∂NmV[:,:,1] .= IFT * (factor_∂x .* (FT * (Nm[:,:] .* V[:,:,1])))
        ∂NmV[:,:,2] .= IFT * (factor_∂y .* (FT * (Nm[:,:] .* V[:,:,2])))
    end

    @inbounds @views begin
        ΔC[:,:,1]  .= IFT * (factor_Δ .* (FT * C[:,:]))
        ΔNp[:,:,1] .= IFT * (factor_Δ .* (FT * Np[:,:]))
        ΔNm[:,:,1] .= IFT * (factor_Δ .* (FT * Nm[:,:]))
    end

    @cuda threads = block_dim blocks = grid_dim kernel_compute_RHS!(C, Np, Nm, ΔC, ΔNp, ΔNm, ∂CV, ∂NpV, ∂NmV, div_CV, div_NpV, div_NmV, RHS_C, RHS_Np, RHS_Nm, Dc, Dap, Dam, A, Kd, Ω0, Ω, Ωd)
end

#= function kernel_EulerForward!(Δt, C_new, Np_new, Nm_new, C_old, Np_old, Nm_old)  # Indexing (i,j)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    C_new[i,j]  = C_old[i,j]  + Δt * RHS[i,j,1]
    Np_new[i,j] = Np_old[i,j] + Δt * RHS[i,j,2]
    Nm_new[i,j] = Nm_old[i,j] + Δt * RHS[i,j,3]

    return nothing
end =#

function kernel_EulerForward!(Δt, C, Np, Nm, RHS_C, RHS_Np, RHS_Nm)  # Indexing (i,j)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    C[i,j]  += Δt * RHS_C[i,j]
    Np[i,j] += Δt * RHS_Np[i,j]
    Nm[i,j] += Δt * RHS_Nm[i,j]

    return nothing
end

function EulerForward!(Δt, V, Π, fV, fΠ, FT, IFT, Lkx, factor_∂x, factor_∂y, factor_Δ, Z, B, C, Np, Nm, ΔC, ΔNp, ΔNm, ∂CV, ∂NpV, ∂NmV, div_CV, div_NpV, div_NmV, RHS_C, RHS_Np, RHS_Nm, Dc, Dap, Dam, A, Kd, Ω0, Ω, Ωd)
    # This funct. propagates dynamical fields in time using Euler forward
    # Calculate right hand sides
    compute_RHS!(V, Π, fV, fΠ, FT, IFT, Lkx, factor_∂x, factor_∂y, factor_Δ, Z, B, C, Np, Nm, ΔC, ΔNp, ΔNm, ∂CV, ∂NpV, ∂NmV, div_CV, div_NpV, div_NmV, RHS_C, RHS_Np, RHS_Nm, Dc, Dap, Dam, A, Kd, Ω0, Ω, Ωd)
    # Update fields with their values at t + Δt
    @cuda threads = block_dim blocks = grid_dim kernel_EulerForward!(Δt, C, Np, Nm, RHS_C, RHS_Np, RHS_Nm)
    return nothing
end

# 🚧 SKETCH OF ADAPTIVE TIMESTEPPING BELOW, NOT YET DEBUGGED 🚧

#= function kernel_AssignFields!(C, Np, Nm, C_, Np_, Nm_)  # Indexing (i,j)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    C[i,j]  = C_[i,j]
    Np[i,j] = Np_[i,j]
    Nm[i,j] = Nm_[i,j]

    return nothing
end

function kernel_AssignFields!(C, Np, Nm, V, C_, Np_, Nm_, V_)  # Indexing (i,j)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    C[i,j]   = C_[i,j]
    Np[i,j]  = Np_[i,j]
    Nm[i,j]  = Nm_[i,j]
    V[i,j,1] = V_[i,j,1]
    V[i,j,2] = V_[i,j,2]

    return nothing
end

function MidpointMethod!(Δt, Pars, v, Fields, Fields_Aux, ∂xx, ∂xFieldsv, σ, RHS, k, P, Pinv)
    # This funct. propagates dynamical fields in time using the midpoint method
    # Save fields at step t
    @cuda threads = block_dim blocks = grid_dim kernel_AssignFields!(C_Aux, Np_Aux, Nm_Aux, C, Np, Nm)
    # Update fields with their values at midpoint (i.e., at t + Δt/2)
    EulerForward!(0.5*Δt, Pars, v, Fields, ∂xx, ∂xFieldsv, σ, RHS, k, P, Pinv)
    # Calculate RHS at midpoint by using updated fields
    CalculateRHS!(Pars, v, Fields, ∂xx, ∂xFieldsv, σ, RHS, k, P, Pinv)
    # Update fields with their values at t + Δt
    @cuda threads = block_dim blocks = grid_dim kernel_EulerForward!(Δt, C, Np, Nm, C_Aux, Np_Aux, Nm_Aux)
    return nothing
end

function AdaptiveTimeStep!(ErrorTolerance, Δt_old, Δt, Pars, v, Fields, Fields_Aux, v_LongStep, Fields_LongStep, v_ShortStepAux, Fields_ShortStepAux, v_ShortStep, Fields_ShortStep, ∂xx, ∂xFieldsv, σ, RHS, k, P, Pinv)
    # First run
    # Propagate fields once by Δt
    @cuda threads = block_dim blocks = grid_dim kernel_AssignFields!(C_LongStep, Np_LongStep, Nm_LongStep, C, Np, Nm)
    MidpointMethod!(Δt, Pars, v_LongStep, Fields_LongStep, Fields_Aux, ∂xx, ∂xFieldsv, σ, RHS, k, P, Pinv)
    # Propagate fields twice by 0.5*Δt, first integration
    @cuda threads = block_dim blocks = grid_dim kernel_AssignFields!(C_ShortStep, Np_ShortStep, Nm_ShortStep, C, Np, Nm)
    MidpointMethod!(0.5*Δt, Pars, v_ShortStep, Fields_ShortStep, Fields_Aux, ∂xx, ∂xFieldsv, σ, RHS, k, P, Pinv)
    # Save current half step fields for next run
    @cuda threads = block_dim blocks = grid_dim kernel_AssignFields!(C_ShortStep_Aux, Np_ShortStep_Aux, Nm_ShortStep_Aux, V_ShortStep_Aux, C_ShortStep, Np_ShortStep, Nm_ShortStep, V_ShortStep)
    # Propagate fields twice by 0.5*Δt, second integration
    MidpointMethod!(0.5*Δt, Pars, v_ShortStep, Fields_ShortStep, Fields_Aux, ∂xx, ∂xFieldsv, σ, RHS, k, P, Pinv)
    # Calculate maximum relative error
    MaxError = findmax(abs.((vcat(C_LongStep, Np_LongStep, Nm_LongStep) .- vcat(C_ShortStep, Np_ShortStep, Nm_ShortStep)) ./ vcat(C_LongStep, Np_LongStep, Nm_LongStep)))[1]
    # Proceed by halving Δt until MaxError < ErrorTolerance
    while MaxError >= ErrorTolerance
        # Halven time step
        Δt = 0.5*Δt
        # LongStep is one half step in previous run (i.e. "Aux" fields)
        @cuda threads = block_dim blocks = grid_dim kernel_AssignFields!(C_LongStep, Np_LongStep, Nm_LongStep, V_LongStep, C_ShortStep_Aux, Np_ShortStep_Aux, Nm_ShortStep_Aux, V_ShortStep_Aux)
        # Propagate fields twice by 0.5*Δt, first integration
        @cuda threads = block_dim blocks = grid_dim kernel_AssignFields!(C_ShortStep, Np_ShortStep, Nm_ShortStep, C, Np, Nm)
        MidpointMethod!(0.5*Δt, Pars, v_ShortStep, Fields_ShortStep, Fields_Aux, ∂xx, ∂xFieldsv, σ, RHS, k, P, Pinv)
        # Save current half step fields for next run
        @cuda threads = block_dim blocks = grid_dim kernel_AssignFields!(C_ShortStep_Aux, Np_ShortStep_Aux, Nm_ShortStep_Aux, V_ShortStep_Aux, C_ShortStep, Np_ShortStep, Nm_ShortStep, V_ShortStep)
        # Propagate fields twice by 0.5*Δt, second integration
        MidpointMethod!(0.5*Δt, Pars, v_ShortStep, Fields_ShortStep, Fields_Aux, ∂xx, ∂xFieldsv, σ, RHS, k, P, Pinv)
        # Calculate maximum relative error
        MaxError = findmax(abs.((vcat(C_LongStep, Np_LongStep, Nm_LongStep) .- vcat(C_ShortStep, Np_ShortStep, Nm_ShortStep)) ./ vcat(C_LongStep, Np_LongStep, Nm_LongStep)))[1]
    end
    # Update Δt_old
    Δt_old = Δt
    # Update fields
    @cuda threads = block_dim blocks = grid_dim kernel_AssignFields!(C, Np, Nm, V, C_LongStep, Np_LongStep, Nm_LongStep, V_LongStep)
    return nothing
end =#
