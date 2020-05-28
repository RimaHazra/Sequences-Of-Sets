using FileIO
#Change the dataset
f = open("pred_on_yakkety_weights.txt","w")
w = load("models/pred_on_yakkety10-CRU-0.9.jld2","w")
println(typeof(w))
for l in w
    write(f, "$l\n")
end
println(w)

println(length(w))
