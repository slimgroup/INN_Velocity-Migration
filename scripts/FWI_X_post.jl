using DrWatson
@quickactivate "INN_Velocity-Migration"

using JLD, Statistics, PyPlot, ImageFiltering, InvertibleNetworks, LinearAlgebra, Dates
include("../../../src/FWI_64x200.jl")
using .FWI_64x200

X_orig = load("../../../data/vel_4k_samples_64x200_lin_vel.jld")["m_all"];
n1, n2, nc, nsamples = size(X_orig)
ntrain = Int(nsamples*.9)
X_train_orig=X_orig[:, :, :, 1:ntrain]

AN = ActNorm(ntrain)
X_train_orig = AN.forward(X_train_orig) # zero mean and unit std of the training data X

# define fig path
datapath = "../figs/test3_6000_16_16/"
figfolder = "test3_6000_16_16/posterior_samples2_GN_test"
mkpath(joinpath(pwd(), "figs", figfolder))

X_fixed = load(joinpath(datapath, "posterior_samples2.jld"), "X_fixed")
X_post  = load(joinpath(datapath, "posterior_samples2.jld"), "X_post")
X_fixed = AN.inverse(X_fixed) # unnormarlize based on the same parameter of the training data X
X_post = AN.inverse(X_post)   # unnormarlize based on the same parameter of the training data X
X_post_mean = mean(X_post; dims=4)

m = X_fixed[:, :, 1, 1] # true velocity in squared slowness
idx_wb = 9 # index of the water bottom
niterations = 15

t1 = now()
println(string("FWI computation starts at ", t1))
####################################################################################################
## Scenario 1: initial velocity m0 = all posterior samples
println(string("Scenario 1: initial velocity m0 = posterior samples"))

test_size = size(X_post)[4] #100
m0_all = zeros(Float32, n1, n2, test_size)
fwi_all = zeros(Float32, n1, n2, test_size)
obj_all = zeros(Float32, niterations, test_size)

for i = 1:3

    m0 = X_post[:, :, 1, i]
    m0[1:idx_wb, :] .= m[1,1]
    m0[idx_wb+1:end, :] = imfilter(m0[idx_wb+1:end, :], Kernel.gaussian(3)) # smoothed velocity
    fwi_all[:,:,i], obj_all[:,i] = fwi(m, m0, niterations)

    m0_all[:,:,i] = m0

    println(string("FWI using the ", i, "th sample of X posterior finished"))
end

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
## Scenario 2: initial velocity m0 = posterior mean
println(string("Scenario 2: initial velocity m0 = posterior mean"))

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
## Scenario 3: initial velocity m0 = migration velocity used in rtm
println(string("Scenario 3: initial velocity m0 = migration velocity used in rtm"))

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
## Scenario 4: initial velocity m0 = smoothed true velocity
println(string("Scenario 4: initial velocity m0 = smoothed true velocity"))

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


t2 = now()
println(string("FWI computation finishes after ", Dates.value.(t2-t1)/3600000, " hours"))
####################################################################################################
# Load data for Scenario 1-4
m0_all, fwi_all, obj_all = load(string("./figs/", figfolder, "/fwi_post_samples.jld"), "m0_all", "fwi_all", "obj_all")
m, m0, fwi2, obj2 = load(string("./figs/", figfolder, "/fwi_post_mean.jld"), "m", "m0", "fwi", "obj")
m, m_lin, fwi3, obj3 = load(string("./figs/", figfolder, "/fwi_lin_vel.jld"), "m", "m0", "fwi", "obj")
m, m_smooth, fwi4, obj4 = load(string("./figs/", figfolder, "/fwi_smooth_vel.jld"), "m", "m0", "fwi", "obj")

# Figures in slowness squared (s^2/km^2)
figure(); imshow(m, cmap="jet", aspect = "auto"); plt.colorbar(); title(L"True velocity ($s^2/km^2$)")
savefig(string("./figs/", figfolder, "/vel_m.png"))

figure(figsize=[20,12])
ax1 = subplot(3, 4, 1); imshow(m0_all[idx_wb+1:end,:,1], cmap="jet", aspect = "auto"); plt.colorbar(); title("Smoothed posterior sample1")
ax2 = subplot(3, 4, 2); imshow(m0_all[idx_wb+1:end,:,2], cmap="jet", aspect = "auto"); plt.colorbar(); title("Smoothed posterior sample2")
ax3 = subplot(3, 4, 3); imshow(m0_all[idx_wb+1:end,:,3], cmap="jet", aspect = "auto"); plt.colorbar(); title("Smoothed posterior sample3")
ax4 = subplot(3, 4, 4); imshow(m0[idx_wb+1:end, :], cmap="jet", aspect = "auto"); plt.colorbar(); title("Smoothed posterior mean")
ax5 = subplot(3, 4, 5); imshow(fwi_all[idx_wb+1:end,:,1], cmap="jet", aspect = "auto"); plt.colorbar(); title(string("FWI result, L2=", round(norm(m-fwi_all[:,:,1],2), sigdigits=4)))
ax6 = subplot(3, 4, 6); imshow(fwi_all[idx_wb+1:end,:,2], cmap="jet", aspect = "auto"); plt.colorbar(); title(string("FWI result, L2=", round(norm(m-fwi_all[:,:,2],2), sigdigits=4)))
ax7 = subplot(3, 4, 7); imshow(fwi_all[idx_wb+1:end,:,3], cmap="jet", aspect = "auto"); plt.colorbar(); title(string("FWI result, L2=", round(norm(m-fwi_all[:,:,3],2), sigdigits=4)))
ax8 = subplot(3, 4, 8); imshow(fwi2[idx_wb+1:end, :], cmap="jet", aspect = "auto"); plt.colorbar(); title(string("FWI result, L2=", round(norm(m-fwi2,2), sigdigits=4)))
ax9 = subplot(3, 4, 9); plot(obj_all[:,1]); title("Objective function value")
ax10 = subplot(3, 4, 10); plot(obj_all[:,2]); title("Objective function value")
ax11 = subplot(3, 4, 11); plot(obj_all[:,3]); title("Objective function value")
ax12 = subplot(3, 4, 12); plot(obj2); title("Objective function value")
savefig(string("./figs/", figfolder, "/fwi_m_post.png"))

figure(figsize=[20,12])
ax1 = subplot(3, 4, 1); imshow(m0_all[idx_wb+1:end,:,1], cmap="jet", aspect = "auto"); plt.colorbar(); title("Smoothed posterior sample")
ax2 = subplot(3, 4, 2); imshow(m0[idx_wb+1:end, :], cmap="jet", aspect = "auto"); plt.colorbar(); title("Smoothed posterior mean")
ax3 = subplot(3, 4, 3); imshow(m_lin[idx_wb+1:end, :], cmap="jet", aspect = "auto"); plt.colorbar(); title("Migration velocity in RTM")
ax4 = subplot(3, 4, 4); imshow(m_smooth[idx_wb+1:end, :], cmap="jet", aspect = "auto"); plt.colorbar(); title("Smoothed true velocity")
ax5 = subplot(3, 4, 5); imshow(fwi_all[idx_wb+1:end,:,1], cmap="jet", aspect = "auto"); plt.colorbar(); title(string("FWI result, L2=", round(norm(m-fwi_all[:,:,1],2), sigdigits=4)))
ax6 = subplot(3, 4, 6); imshow(fwi2[idx_wb+1:end, :], cmap="jet", aspect = "auto"); plt.colorbar(); title(string("FWI result, L2=", round(norm(m-fwi2,2), sigdigits=4))) 
ax7 = subplot(3, 4, 7); imshow(fwi3[idx_wb+1:end, :], cmap="jet", aspect = "auto"); plt.colorbar(); title(string("FWI result, L2=", round(norm(m-fwi3,2), sigdigits=4)))
ax8 = subplot(3, 4, 8); imshow(fwi4[idx_wb+1:end, :], cmap="jet", aspect = "auto"); plt.colorbar(); title(string("FWI result, L2=", round(norm(m-fwi4,2), sigdigits=4)))
ax9 = subplot(3, 4, 9); plot(obj_all[:,1]); title("Objective function value")
ax10 = subplot(3, 4, 10); plot(obj2); title("Objective function value")
ax11 = subplot(3, 4, 11); plot(obj3); title("Objective function value")
ax12 = subplot(3, 4, 12); plot(obj4); title("Objective function value")
savefig(string("./figs/", figfolder, "/fwi_m.png"))

# Figures in velocity (km/s)
v = sqrt.(1f0 ./ m)

v0_all = sqrt.(1f0 ./ m0_all[:,:,1:3])
v0 = sqrt.(1f0 ./ m0)
v_lin = sqrt.(1f0 ./ m_lin)
v_smooth = sqrt.(1f0 ./ m_smooth)

fwi_v_all = sqrt.(1f0 ./ fwi_all[:,:,1:3])
fwi2_v = sqrt.(1f0 ./ fwi2)
fwi3_v = sqrt.(1f0 ./ fwi3)
fwi4_v = sqrt.(1f0 ./ fwi4)

figure(); imshow(v, vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); plt.colorbar(); title(L"True velocity ($km/s$)")
savefig(string("./figs/", figfolder, "/vel.png"))

figure(figsize=[20,12])
ax1 = subplot(3, 4, 1); imshow(v0_all[:,:,1], vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); plt.colorbar(); title("Smoothed posterior sample1")
ax2 = subplot(3, 4, 2); imshow(v0_all[:,:,2], vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); plt.colorbar(); title("Smoothed posterior sample2")
ax3 = subplot(3, 4, 3); imshow(v0_all[:,:,3], vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); plt.colorbar(); title("Smoothed posterior sample3")
ax4 = subplot(3, 4, 4); imshow(v0, vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); plt.colorbar(); title("Smoothed posterior mean")
ax5 = subplot(3, 4, 5); imshow(fwi_v_all[:,:,1], vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); plt.colorbar(); title(string("FWI result, L2=", round(norm(v-fwi_v_all[:,:,1],2), sigdigits=4)))
ax6 = subplot(3, 4, 6); imshow(fwi_v_all[:,:,2], vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); plt.colorbar(); title(string("FWI result, L2=", round(norm(v-fwi_v_all[:,:,2],2), sigdigits=4)))
ax7 = subplot(3, 4, 7); imshow(fwi_v_all[:,:,3], vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); plt.colorbar(); title(string("FWI result, L2=", round(norm(v-fwi_v_all[:,:,3],2), sigdigits=4)))
ax8 = subplot(3, 4, 8); imshow(fwi2_v, vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); plt.colorbar(); title(string("FWI result, L2=", round(norm(v-fwi2_v,2), sigdigits=4)))
ax9 = subplot(3, 4, 9); plot(obj_all[:,1]); title("Objective function value")
ax10 = subplot(3, 4, 10); plot(obj_all[:,2]); title("Objective function value")
ax11 = subplot(3, 4, 11); plot(obj_all[:,3]); title("Objective function value")
ax12 = subplot(3, 4, 12); plot(obj2); title("Objective function value")
savefig(string("./figs/", figfolder, "/fwi_v_post.png"))

figure(figsize=[20,12])
ax1 = subplot(3, 4, 1); imshow(v0_all[:,:,1], vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); plt.colorbar(); title("Smoothed posterior sample")
ax2 = subplot(3, 4, 2); imshow(v0, vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); plt.colorbar(); title("Smoothed posterior mean")
ax3 = subplot(3, 4, 3); imshow(v_lin, vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); plt.colorbar(); title("Migration velocity in RTM")
ax4 = subplot(3, 4, 4); imshow(v_smooth, vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); plt.colorbar(); title("Smoothed true velocity")
ax5 = subplot(3, 4, 5); imshow(fwi_v_all[:,:,1], vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); plt.colorbar(); title(string("FWI result, L2=", round(norm(v-fwi_v_all[:,:,1],2), sigdigits=4)))
ax6 = subplot(3, 4, 6); imshow(fwi2_v, vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); plt.colorbar(); title(string("FWI result, L2=", round(norm(v-fwi2_v,2), sigdigits=4))) 
ax7 = subplot(3, 4, 7); imshow(fwi3_v, vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); plt.colorbar(); title(string("FWI result, L2=", round(norm(v-fwi3_v,2), sigdigits=4)))
ax8 = subplot(3, 4, 8); imshow(fwi4_v, vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); plt.colorbar(); title(string("FWI result, L2=", round(norm(v-fwi4_v,2), sigdigits=4)))
ax9 = subplot(3, 4, 9); plot(obj_all[:,1]); title("Objective function value")
ax10 = subplot(3, 4, 10); plot(obj2); title("Objective function value")
ax11 = subplot(3, 4, 11); plot(obj3); title("Objective function value")
ax12 = subplot(3, 4, 12); plot(obj4); title("Objective function value")
savefig(string("./figs/", figfolder, "/fwi_v.png"))
