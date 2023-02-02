# Packages
using LinearAlgebra
using Roots
using Plots
using JLD
cd(@__DIR__)

# Model parameters
Pars = Dict{String,Float64}()
# Fixed ones
Pars["A"] = 1
Pars["Kd"] = 1
Pars["Dc"] = 1e-2
Pars["Da+"] = 1e-1 + 1
Pars["Da-"] = 1e-1 - 1
Pars["Ω0"] = 0.5
Pars["Ωd"] = 10
# Varying ones (here just initialized)
Pars["Z"] = 1000
Pars["B"] = Pars["Z"]
Pars["Ω"] = 1000
Pars["∂Π_HSS"] = 1000 # This is Π'_HSS in the Supplemental Material

# We produce a cut of the stability diagram in the plane spanned by two control parameters, Par1 Par2
# Par1 = Z ∈ [MinPar1, MaxPar1]
MinPar1 = 0
MaxPar1 = 100
# Par2 = Ω ∈ [MinPar2, MaxPar2]
MinPar2 = 0
MaxPar2 = 100
NPar1 = 500 # n. of points in the grid of Par1
NPar2 = 500 # n. of points in the grid of Par2
ΔPar1 = (MaxPar1 - MinPar1) / NPar1 # grid spacing for Par1
ΔPar2 = (MaxPar2 - MinPar2) / NPar2 # grid spacing for Par2

# Stability Diagram is a (NPar1 + 1) times (NPar2 + 1) matrix, whose entries will be:
# 0, if HSS is stable
# 1, if HSS is unstable and fastest growing eigenvalue is real
# 2, if HSS is unstable and fastest growing eigenvalue is complex
StabilityDiagram = zeros(Int64, NPar1 + 1, NPar2 + 1)

# Initialize real axis (prerequisite to initialize reciprocal axis)
L = 10*π
Nx = 512
Δx = L / (Nx - 1)
# Initialize reciprocal axis
Kmax = π / Δx
Kmin = 2*π / L
Nk = Int64(floor((Kmax - Kmin) / Kmin))

StabilityMatrix = zeros(Float64, 3, 3) # This is the matrix M mentioned in the Supplemental Material
EigenvStabilityMatrix = zeros(Complex{Float64}, 3, Nk + 1) # Matrix containing the three eigenvalues of M (lines), as functions of K (columns)

# This function produces the (i,j)-th element of the StabilityMatrix for given parameters and mode K
function StabilityMatrixElement(i, j, Pars, C_HSS, Nm_HSS, K) # (i,j) = index of matrix element
    if (i, j) == (1, 1)
        return (K^2 / (1 + K^2)) * Pars["∂Π_HSS"] * C_HSS -
        Pars["Dc"] * K^2 -
        Pars["Kd"]
    elseif (i, j) == (1, 2)
        return Pars["A"]/2
    elseif (i, j) == (1, 3)
        return Pars["A"]/2
    elseif (i, j) == (2, 1)
        return (K^2 / (1 + K^2)) * Pars["∂Π_HSS"]
    elseif (i, j) == (2, 2)
        return - (Pars["Da+"]/2) * K^2
    elseif (i, j) == (2, 3)
        return - (Pars["Da-"]/2) * K^2
    elseif (i, j) == (3, 1)
        return (K^2 / (1 + K^2)) * Pars["∂Π_HSS"] * Nm_HSS -
                (1 + Nm_HSS) * Pars["Ωd"]
    elseif (i, j) == (3, 2)
        return - (Pars["Da-"]/2) * K^2 +
                Pars["Ω0"] -
                (1/4) * (Nm_HSS - 3) * (Nm_HSS + 1) * Pars["Ω"] * Pars["Ω0"] -
                C_HSS * Pars["Ωd"]
    elseif (i, j) == (3, 3)
        return - (Pars["Da+"]/2) * K^2 -
                (1/4) * (4 + (1 + Nm_HSS)*(3*Nm_HSS - 1)*Pars["Ω"])*Pars["Ω0"] -
                C_HSS*Pars["Ωd"]
    end
end

# This function produces StabilityDiagram
function StabilityMatrix!(Pars, NPar1, MinPar1, ΔPar1, NPar2, MinPar2, ΔPar2, Nk, EigenvStabilityMatrix, StabilityDiagram)
    for (i, j) in collect(Iterators.product(1:NPar1+1, 1:NPar2+1))
        Pars["Z"] = MinPar1 + (i - 1) * ΔPar1
        Pars["Ω"] = MinPar2 + (j - 1) * ΔPar2
        Pars["B"] = Pars["Z"]

        # Homogeneous Steady State
        # Solving EqnHSS = 0 for Na gives Na at HSS
        EqnHSS(Na) = - (Pars["Ωd"] * Pars["A"] / Pars["Kd"]) * Na^2 + Pars["Ω0"] * (1 - Na) * (1 + Pars["Ω"] * Na^2)
        Na_HSS = find_zero(EqnHSS, (0, 1))
        # Ni, C, Nm (i.e., $N^-$ in the Supplemental Material) and ∂Π at HSS are calculated from Na_HSS
        Ni_HSS = 1 - Na_HSS
        C_HSS = (Pars["A"] / Pars["Kd"]) * Na_HSS
        Nm_HSS = Na_HSS - Ni_HSS
        Pars["∂Π_HSS"] = 2*Pars["Z"]*C_HSS - 3*Pars["B"]*C_HSS^2

        # Find eigenvalues of StabilityMatrix at current parameters
        for p = 1:Nk
            K = p * Kmin
            StabilityMatrix = [
                StabilityMatrixElement(l, m, Pars, C_HSS, Nm_HSS, K)
                for l = 1:3, m = 1:3
            ]
            EigenvStabilityMatrix[:, p] = eigvals(StabilityMatrix)
        end

        # Find eigenvalue with largest real part MaxEigen = MaxEigenvRe + im * MaxEigenvIm
        MaxEigenvIndex = findmax(real.(EigenvStabilityMatrix))[2] # MaxEigen is the element of StabilityMatrix indexed MaxEigenvIndex[1],MaxEigenvIndex[2]
        MaxEigenvRe = findmax(real.(EigenvStabilityMatrix))[1]
        MaxEigenvIm = imag.(EigenvStabilityMatrix[MaxEigenvIndex[1], MaxEigenvIndex[2]])

        # Entries of StabilityDiagram
        if MaxEigenvRe > 0
            if MaxEigenvIm == 0
                StabilityDiagram[i, j] = 1 # Signals Turing bifurcation
            else
                StabilityDiagram[i, j] = 2 # Signals Hopf bifurcation
            end
        else
            StabilityDiagram[i, j] = 0 # Homogeneous steady state is stable
        end
    end
end

# This function calculates the Turing- and Hopf- stability lines in parameter space (TuringLine and HopfLine, respectively)
function BifurcationLines(NPar1, MinPar1, ΔPar1, NPar2, MinPar2, ΔPar2)
    TuringLine = [0,0]
    HopfLine = [0,0]
    for (i, j) in collect(Iterators.product(1:NPar1+1, 1:NPar2+1))
        Z = MinPar1 + (i - 1) * ΔPar1
        Ω = MinPar2 + (j - 1) * ΔPar2
        # The condition below locates a Turing bifurcation in parameter space, by looking for transitions from 0 to 1 in the entries of StabilityDiagram
        if (i>1 && StabilityDiagram[i,j] == 1 && StabilityDiagram[i-1,j] == 0) || (j>1 && StabilityDiagram[i,j] == 0 && StabilityDiagram[i,j-1] == 1)
            TuringLine = hcat(TuringLine, [Z, Ω])
        end
        # The condition below locates a Hopf bifurcation in parameter space, by looking for transitions from 0 to 2 in the entries of StabilityDiagram
        if (i>1 && StabilityDiagram[i,j] == 2 && StabilityDiagram[i-1,j] == 0) || (j>1 && StabilityDiagram[i,j] == 0 && StabilityDiagram[i,j-1] == 2)
            HopfLine = hcat(HopfLine, [Z, Ω])
        end
    end
    TuringLine = transpose(TuringLine)
    HopfLine = transpose(HopfLine)
    return TuringLine[2:end,:], HopfLine[2:end,:]
end

# Calculate instability lines and plot stability diagram
StabilityMatrix!(Pars, NPar1, MinPar1, ΔPar1, NPar2, MinPar2, ΔPar2, Nk, EigenvStabilityMatrix, StabilityDiagram)
TuringLine, HopfLine = BifurcationLines(NPar1, MinPar1, ΔPar1, NPar2, MinPar2, ΔPar2)
p = scatter(TuringLine[:,1], TuringLine[:,2],
    label = "Turing")
p = scatter!(HopfLine[:,1], HopfLine[:,2],
    label = "Hopf")