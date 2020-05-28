#export PATH="$PATH:/home/rima/julia-1.4.1/bin"
using Combinatorics
using DataStructures
using FileIO
using JLD2
using PyPlot

dataset = "pred_on_yakkety10"
p = 0.9
include("learn_CRU_model.jl")
learn(dataset, p)
