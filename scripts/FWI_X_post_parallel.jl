using DrWatson
@quickactivate "INN_Velocity-Migration"

using JLD, Statistics, PyPlot, ImageFiltering, InvertibleNetworks, LinearAlgebra, Distributed, SharedArrays
addprocs(20)
@everywhere include("../../../src/FWI_64x200.jl")
@everywhere using .FWI_64x200, Statistics, ImageFiltering, Dates

X_orig = load("../../../data/vel_4k_samples_64x200_lin_vel.jld")["m_all"];
n1, n2, nc, nsamples = size(X_orig)
ntrain = Int(nsamples*.9)
X_train_orig=X_orig[:, :, :, 1:ntrain]

AN = ActNorm(ntrain)
X_train_orig = AN.forward(X_train_orig) # zero mean and unit std of the training data X

# define fig path
datapath = "../figs/test3_6000_16_16/"
figfolder = "test3_6000_16_16/posterior_samples2_p"
mkpath(joinpath(pwd(), "figs", figfolder))

X_fixed = load(joinpath(datapath, "posterior_samples2.jld"), "X_fixed")
X_post  = load(joinpath(datapath, "posterior_samples2.jld"), "X_post")
X_fixed = AN.inverse(X_fixed) # unnormarlize based on the same parameter of the training data X
X_post = AN.inverse(X_post)   # unnormarlize based on the same parameter of the training data X
X_post_mean = mean(X_post; dims=4)

m = X_fixed[:, :, 1, 1] # true velocity in squared slowness
idx_wb = 9 # index of the water bottom
niterations = 15


####################################################################################################
## Scenario 1: initial velocity m0 = all posterior samples
println(string("Scenario 1: initial velocity m0 = posterior samples"))

test_size = size(X_post)[4] #100
m0_all = SharedArray{Float32,3}(n1, n2, test_size)
fwi_all = SharedArray{Float32,3}(n1, n2, test_size)
obj_all = SharedArray{Float32,2}(niterations, test_size)

t1 = now()
println(string("FWI parallel computation starts at ", t1))

@sync @distributed for i = 1:test_size #collect(5:5:100)

    println(string("FWI using the ", i, "th sample of X posterior starts at ", now() ))

    m0 = X_post[:, :, 1, i]
    m0[1:idx_wb, :] .= m[1,1]
    m0[idx_wb+1:end, :] = imfilter(m0[idx_wb+1:end, :], Kernel.gaussian(3)) # smoothed velocity
    fwi_all[:,:,i], obj_all[:,i] = fwi(m, m0, niterations)

    m0_all[:,:,i] = m0

end

t2 = now()
println(string("FWI parallel computation finishes after ", Dates.value.(t2-t1)/3600000, " hours"))


m0_all = Array(m0_all)
fwi_all = Array(fwi_all)
obj_all = Array(obj_all)
save(string("./figs/", figfolder, "/fwi_post_samples.jld"), "m0_all", m0_all, "fwi_all", fwi_all, "obj_all", obj_all)

i=1
v = sqrt.(1f0 ./ m)
v0 = sqrt.(1f0 ./ m0_all[:,:,i])
fwi_v = sqrt.(1f0 ./ fwi_all[:,:,i])
obj = obj_all[:,i]

figure(figsize=[16,8])
ax1 = subplot(2, 2, 1); imshow(v, cmap="jet", aspect = "auto"); plt.colorbar(); title("True velocity (km/s)")
ax2 = subplot(2, 2, 2); imshow(v0, cmap="jet", aspect = "auto"); plt.colorbar(); title(string("Initial model, L2=", round(norm(v-v0,2), sigdigits=4)))
ax3 = subplot(2, 2, 3); imshow(fwi_v, cmap="jet", aspect = "auto"); plt.colorbar(); title(string("FWI result, L2=", round(norm(v-fwi_v,2), sigdigits=4))) 
ax4 = subplot(2, 2, 4); plot(obj); title("Objective function value")
savefig(string("./figs/", figfolder, "/fwi_post_samples.png"))

####################################################################################################
## Scenario 2: initial velocity m0 = Mean of the FWI results in Scenario 1
println(string("Scenario 2: initial velocity m0 = fwi_mean"))

fwi_mean =  dropdims( mean(fwi_all, dims=3), dims=3 )
fwi_std =  dropdims( std(fwi_all, dims=3), dims=3 )
figure(figsize=[16,8])
ax1 = subplot(1, 2, 1); imshow(fwi_mean, cmap="jet", aspect = "auto"); plt.colorbar(); title("Mean of FWI")
ax2 = subplot(1, 2, 2); imshow(fwi_std, cmap="jet", aspect = "auto"); plt.colorbar(); title("Standard deviation of FWI")
savefig(string("./figs/", figfolder, "/fwi_mean&std.png"))

fwi_mean[1:idx_wb, :] .= m[1,1]
fwi1, obj1 = fwi(m, fwi_mean, niterations)
save(string("./figs/", figfolder, "/fwi_fwi_mean.jld"), "m", m, "m0", fwi_mean, "fwi", fwi1, "obj", obj1)

v = sqrt.(1f0 ./ m)
v0 = sqrt.(1f0 ./ fwi_mean)
fwi_v = sqrt.(1f0 ./ fwi3)
obj = obj3

figure(figsize=[16,8])
ax1 = subplot(2, 2, 1); imshow(v, cmap="jet", aspect = "auto"); plt.colorbar(); title("True velocity (km/s)")
ax2 = subplot(2, 2, 2); imshow(v0, cmap="jet", aspect = "auto"); plt.colorbar(); title(string("Initial model, L2=", round(norm(v-v0,2), sigdigits=4)))
ax3 = subplot(2, 2, 3); imshow(fwi_v, cmap="jet", aspect = "auto"); plt.colorbar(); title(string("FWI result, L2=", round(norm(v-fwi_v,2), sigdigits=4))) 
ax4 = subplot(2, 2, 4); plot(obj); title("Objective function value")
savefig(string("./figs/", figfolder, "/fwi_fwi_mean.png"))


####################################################################################################
## Scenario 3: initial velocity m0 = posterior mean
println(string("Scenario 3: initial velocity m0 = posterior mean"))

m0 = X_post_mean[:, :, 1, 1]
m0[1:idx_wb, :] .= m[1,1]
m0[idx_wb+1:end, :] = imfilter(m0[idx_wb+1:end, :], Kernel.gaussian(3)) # smoothed velocity

fwi2, obj2 = fwi(m, m0, niterations)
save(string("./figs/", figfolder, "/fwi_post_mean.jld"), "m", m, "m0", m0, "fwi", fwi2, "obj", obj2)

v = sqrt.(1f0 ./ m)
v0 = sqrt.(1f0 ./ m0)
fwi_v = sqrt.(1f0 ./ fwi2)
obj = obj2

figure(figsize=[16,8])
ax1 = subplot(2, 2, 1); imshow(v, cmap="jet", aspect = "auto"); plt.colorbar(); title("True velocity (km/s)")
ax2 = subplot(2, 2, 2); imshow(v0, cmap="jet", aspect = "auto"); plt.colorbar(); title(string("Initial model, L2=", round(norm(v-v0,2), sigdigits=4)))
ax3 = subplot(2, 2, 3); imshow(fwi_v, cmap="jet", aspect = "auto"); plt.colorbar(); title(string("FWI result, L2=", round(norm(v-fwi_v,2), sigdigits=4))) 
ax4 = subplot(2, 2, 4); plot(obj); title("Objective function value")
savefig(string("./figs/", figfolder, "/fwi_post_mean.png"))


####################################################################################################
## Scenario 4: initial velocity m0 = migration velocity used in rtm
println(string("Scenario 4: initial velocity m0 = migration velocity used in rtm"))

# build a migration velocity m0 whose value increases linearly with depth
v0 = ones(n1, 1)
v0[idx_wb+1:end] = collect(range(3.5, 5.5, length = n1-idx_wb))
v0 = (1f0 ./ v0).^2 # convert to slowness squared
v0[1:idx_wb] .= m[1,1] # water layer velocity
m_lin = repeat(v0, 1, n2)

fwi3, obj3 = fwi(m, m_lin, niterations)
save(string("./figs/", figfolder, "/fwi_lin_vel.jld"), "m", m, "m0", m_lin, "fwi", fwi3, "obj", obj3)

v = sqrt.(1f0 ./ m)
v0 = sqrt.(1f0 ./ m_lin)
fwi_v = sqrt.(1f0 ./ fwi3)
obj = obj3

figure(figsize=[16,8])
ax1 = subplot(2, 2, 1); imshow(v, cmap="jet", aspect = "auto"); plt.colorbar(); title("True velocity (km/s)")
ax2 = subplot(2, 2, 2); imshow(v0, cmap="jet", aspect = "auto"); plt.colorbar(); title(string("Initial model, L2=", round(norm(v-v0,2), sigdigits=4)))
ax3 = subplot(2, 2, 3); imshow(fwi_v, cmap="jet", aspect = "auto"); plt.colorbar(); title(string("FWI result, L2=", round(norm(v-fwi_v,2), sigdigits=4))) 
ax4 = subplot(2, 2, 4); plot(obj); title("Objective function value")
savefig(string("./figs/", figfolder, "/fwi_lin_vel.png"))


####################################################################################################
## Scenario 5: initial velocity m0 = smoothed true velocity
println(string("Scenario 5: initial velocity m0 = smoothed true velocity"))

m_smooth = m[1,1] .* ones(Float32, size(m)) # smoothed velocity
m_smooth[idx_wb+1:end, :] = imfilter(m[idx_wb+1:end, :], Kernel.gaussian(10)) # smoothed velocity

fwi4, obj4 = fwi(m, m_smooth, niterations)
save(string("./figs/", figfolder, "/fwi_smooth_vel.jld"), "m", m, "m0", m_smooth, "fwi", fwi4, "obj", obj4)

v = sqrt.(1f0 ./ m)
v0 = sqrt.(1f0 ./ m_smooth)
fwi_v = sqrt.(1f0 ./ fwi4)
obj = obj4

figure(figsize=[16,8])
ax1 = subplot(2, 2, 1); imshow(v, cmap="jet", aspect = "auto"); plt.colorbar(); title("True velocity (km/s)")
ax2 = subplot(2, 2, 2); imshow(v0, cmap="jet", aspect = "auto"); plt.colorbar(); title(string("Initial model, L2=", round(norm(v-v0,2), sigdigits=4)))
ax3 = subplot(2, 2, 3); imshow(fwi_v, cmap="jet", aspect = "auto"); plt.colorbar(); title(string("FWI result, L2=", round(norm(v-fwi_v,2), sigdigits=4))) 
ax4 = subplot(2, 2, 4); plot(obj); title("Objective function value")
savefig(string("./figs/", figfolder, "/fwi_smooth_vel.png"))