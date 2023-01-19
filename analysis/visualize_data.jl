using NetCDF, GLMakie, LinearAlgebra, FFTW
filename = "tos_300yrs_Pacific_a_14_days.nc"
ncinfo(filename)
temperature = ncread(filename, "tos")

##
function missing_value_mask(temperature; fill_value=temperature[1, 1, 1])
    return temperature .!= fill_value
end

reduced_field = temperature[1:3:end, 1:3:end, :]
mask = missing_value_mask(reduced_field[:, :, 1]) # assume continents don't change in time
# full_mask = missing_value_mask(temperature)

heatmap(reduced_field[:, :, 40], colormap=:balance, colorrange=(-2, 2))

data_matrix = zeros(sum(mask), size(reduced_field, 3))
for i in 1:size(reduced_field, 3)
    data_matrix[:, i] .= reduced_field[mask, i]
end
##
U, Σ, Vᵀ = svd(data_matrix)
##
modes = 1
reduced = U[:, 1:modes] * Diagonal(Σ[1:modes]) * transpose(Vᵀ[:, 1:modes])
reduced_field2 = copy(reduced_field)

for i in 1:size(reduced_field, 3)
    reduced_field2[mask, i] .= reduced[:, i]
end
##
fig = Figure()
sl = Slider(fig[2, 1:2], range=1:size(reduced_field, 3), startvalue=1)
iobv = sl.value

title1 = @lift "Field at week " * string($iobv)
title2 = @lift "Mode 1 at week " * string($iobv)
ax1 = Axis(fig[1, 1]; title=title1)
ax2 = Axis(fig[1, 2]; title=title2)

field1 = @lift reduced_field[:, :, $iobv]
field2 = @lift reduced_field2[:, :, $iobv]
heatmap!(ax1, field1, colormap=:balance, colorrange=(-2, 2), interpolate=true)
heatmap!(ax2, field2, colormap=:balance, colorrange=(-2, 2), interpolate=true)
display(fig)

itr = 1:200
record(fig, "enso.mp4", itr; framerate=10) do i
    iobv[] = i # or some other manipulation of the Scene
end

##
timeseries = Vᵀ[:, 1]
##
using MarkovChainHammer, Statistics
import MarkovChainHammer.TransitionMatrix: generator, perron_frobenius, holding_times
qu_timeseries = quantile.(Ref(timeseries), (0.2, 0.4, 0.6, 0.8))

function embedding(x, qu_timeseries)
    return sum(qu_timeseries .> x) + 1 # +1 because we start at index 1 in Julia
end

function markov_embedding(timeseries, qu_timeseries)
    markov_chain = Int64[]
    for amplitude in timeseries
        push!(markov_chain, embedding(amplitude, qu_timeseries))
    end
    return markov_chain
end

markov_chain = markov_embedding(timeseries, qu_timeseries)

fig = Figure(resolution = (1269, 780))
ax = Axis(fig[1, 1])
# rescale timeseries to correspond to markov states for plotting purposes
rescale_timeseries = timeseries .- minimum(timeseries)
rescale_timeseries ./= -maximum(rescale_timeseries) # accidently made it upside down
rescale_timeseries = (rescale_timeseries .* (length(union(markov_chain))-1)) .+ (length(union(markov_chain)))
lines!(ax, markov_chain[1:1000], color=:red, label = "markov")
lines!(ax, rescale_timeseries[1:1000], color=:blue, label = "rescaled timeseries")
axislegend(ax, position=:rb, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=40)

##
Q = generator(markov_chain)
P = perron_frobenius(markov_chain)
ht = holding_times(markov_chain)
Λ, V = eigen(Q)
##
using ProgressBars
# from timeseries
function autocovariance(x; timesteps=length(x))
    μ = mean(x)
    autocor = zeros(timesteps)
    for i in ProgressBar(1:timesteps)
        autocor[i] = mean(x[i:end] .* x[1:end-i+1]) - μ^2
    end
    return autocor
end
# for generator 
function autocovariance(g⃗, Q::Eigen, timelist)
    autocov = zeros(length(timelist))
    # Q  = V Λ V⁻¹
    Λ, V = Q
    p = real.(V[:, end] ./ sum(V[:, end]))
    v1 = V \ (p .* g⃗)
    w1 = g⃗' * V
    μ = sum(p .* g⃗)
    for i in eachindex(timelist)
        autocov[i] = real(w1 * (exp.(Λ .* timelist[i]) .* v1)) - μ^2
    end
    return autocov
end
##
steps = 200
tlist = collect(0:steps-1)
autocor_m = autocovariance(markov_chain; timesteps =  steps)
autocor_t = autocovariance(timeseries; timesteps=steps)
##
fig = Figure()
ax = Axis(fig[1, 1]; title = "Autocorrelation of Markov Chain")
lines!(ax, autocor_m /autocor_m[1], color = :red, linewidth = 10, label = "Markov Chain")
scatter!(ax, autocor_t / autocor_t[1], color = (:blue, 0.5), label = "Timeseries")
scatter!(ax, exp.(tlist .* Λ[end-1]), color = :purple, label = "Slowest Exponential Decay from Generator")
axislegend(ax, position=:rt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=40)
display(fig)