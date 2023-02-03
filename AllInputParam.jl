include("../Utilities/using.jl")
using_pkg("DelimitedFiles, CSV, DataFrames, JLD")
using_mod(".JulUtils")
dir = @__DIR__

tΩ = 3:3:30
tZ = 3:3:30

listname = ["Omega", "Z"]
listtab = [tΩ, tZ]

generate_csv(dir, listname, listtab)