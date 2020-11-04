using DrWatson
@quickactivate "INN_Velocity-Migration"

using JLD, Statistics, PyPlot, ImageFiltering, InvertibleNetworks, LinearAlgebra
include(srcdir("RTM_64x200.jl"))
using .RTM_64x200

X_orig = load(datadir("vel_4k_samples_64x200_lin_vel.jld"), "m_all")
X_orig = Float32.(X_orig)
n1, n2, nc, nsamples = size(X_orig)
ntrain = Int(nsamples*.9)
X_train_orig=X_orig[:, :, :, 1:ntrain]

AN = ActNorm(ntrain)
X_train_orig = AN.forward(X_train_orig) # zero mean and unit std of the training data X

# define data and figure path
datapath = plotsdir("chint/test3_6000_16_16")
figfolder = "rtm/test3_6000_16_16/posterior_samples2"
mkpath(plotsdir(figfolder))

X_fixed = load(joinpath(datapath, "posterior_samples2.jld"), "X_fixed")
X_post  = load(joinpath(datapath, "posterior_samples2.jld"), "X_post")
Y_fixed = load(joinpath(datapath, "posterior_samples2.jld"), "Y_fixed")
X_fixed = AN.inverse(X_fixed) # unnormarlize based on the same parameter of the training data X
X_post = AN.inverse(X_post)   # unnormarlize based on the same parameter of the training data X
X_post_mean = mean(X_post; dims=4)
Y_fixed = AN.inverse(Y_fixed) # unnormarlize based on the same parameter of the training data X

m = X_fixed[:, :, 1, 1] # true velocity in squared slowness
rtm0 = Y_fixed[:, :, 1, 1] # rtm images used in training INN
d = (25., 25.)
idx_wb = 9 # index of the water bottom

figure(); plot_rtm(adjoint(rtm0), d; new_fig=false, aspect = "auto"); title("RTM used in INN training")
savefig(plotsdir(figfolder, "rtm_INN.png"))


####################################################################################################
## Scenario 1: migration velocity m0 = all posterior samples
println(string("Scenario 1: migration velocity m0 = posterior samples"))

test_size = size(X_post)[4] #100
m0_all = zeros(Float32, n1, n2, test_size)
rtm_all = zeros(Float32, n1, n2, test_size)

for i = 1:3 #test_size

    m0 = X_post[:, :, 1, i]
    m0[1:idx_wb, :] .= m[1,1]
    m0[idx_wb+1:end, :] = imfilter(m0[idx_wb+1:end, :], Kernel.gaussian(3)) # smoothed velocity
    rtm = rtm_isic(m, m0)

    m0_all[:,:,i] = m0
    rtm_all[:,:,i] = adjoint(reshape(rtm, (n2, n1) )) 

    println(string("RTM using the ", i, "th sample of X posterior finished"))
end

save(plotsdir(figfolder, "rtm_post_samples.jld"), "m0_all", m0_all, "rtm_all", rtm_all)


####################################################################################################
## Scenario 2: migration velocity m0 = posterior mean
println(string("Scenario 2: migration velocity m0 = posterior mean"))

m0 = X_post_mean[:, :, 1, 1]
m0[1:idx_wb, :] .= m[1,1]
m0[idx_wb+1:end, :] = imfilter(m0[idx_wb+1:end, :], Kernel.gaussian(3)) # smoothed velocity

rtm2 = rtm_isic(m, m0)
rtm2 = adjoint(reshape(rtm2, (n2, n1) ))
save(plotsdir(figfolder, "rtm_post_mean.jld"), "m", m, "m0", m0, "rtm", rtm2)


####################################################################################################
## Scenario 3: migration velocity m0 = migration velocity used in rtm
println(string("Scenario 3: migration velocity m0 = migration velocity used in rtm"))

# build a migration velocity m0 whose value increases linearly with depth
v0 = ones(n1, 1)
v0[idx_wb+1:end] = collect(range(3.5, 5.5, length = n1-idx_wb))
v0 = (1f0 ./ v0).^2 # convert to slowness squared
v0[1:idx_wb] .= m[1,1] # water layer velocity
m_lin = repeat(v0, 1, n2)

rtm3 = rtm_isic(m, m_lin)
rtm3 = adjoint(reshape(rtm3, (n2, n1) )) 
save(plotsdir(figfolder, "rtm_lin_vel.jld"), "m", m, "m0", m_lin, "rtm", rtm3)


####################################################################################################
## Scenario 4: migration velocity m0 = smoothed true velocity
println(string("Scenario 4: migration velocity m0 = smoothed true velocity"))

m_smooth = m[1,1] .* ones(Float32, size(m)) # smoothed velocity
m_smooth[idx_wb+1:end, :] = imfilter(m[idx_wb+1:end, :], Kernel.gaussian(3)) # smoothed velocity

rtm4 = rtm_isic(m, m_smooth)
rtm4 = adjoint(reshape(rtm4, (n2, n1) )) 
save(plotsdir(figfolder, "rtm_smooth_vel.jld"), "m", m, "m0", m_smooth, "rtm", rtm4)


####################################################################################################
## Scenario 5: migration velocity m0 = true velocity
println(string("Scenario 5: migration velocity m0 = true velocity"))

rtm5 = rtm_isic(m, m)
rtm5 = adjoint(reshape(rtm5, (n2, n1) )) 
save(plotsdir(figfolder, "rtm_true_vel.jld"), "m", m, "m0", m, "rtm", rtm5)


####################################################################################################
# Load data for Scenario 1-5
m0_all, rtm_all = load(plotsdir(figfolder, "rtm_post_samples.jld"), "m0_all", "rtm_all")
m, m0, rtm2 = load(plotsdir(figfolder, "rtm_post_mean.jld"), "m", "m0", "rtm")
m, m_lin, rtm3 = load(plotsdir(figfolder, "rtm_lin_vel.jld"), "m", "m0", "rtm")
m, m_smooth, rtm4 = load(plotsdir(figfolder, "rtm_smooth_vel.jld"), "m", "m0", "rtm")
m, m_true, rtm5 = load(plotsdir(figfolder, "rtm_true_vel.jld"), "m", "m0", "rtm")


# Figures in slowness squared (s^2/km^2)
figure(); imshow(m, cmap="jet", aspect = "auto"); plt.colorbar(); title(L"True velocity ($s^2/km^2$)")
savefig(plotsdir(figfolder, "vel_m.png"))

figure(figsize=[20,8])
ax1 = subplot(2, 4, 1); imshow(m0_all[:,:,1], cmap="jet", aspect = "auto"); title("Smoothed posterior sample1")
ax2 = subplot(2, 4, 2); imshow(m0_all[:,:,2], cmap="jet", aspect = "auto"); title("Smoothed posterior sample2")
ax3 = subplot(2, 4, 3); imshow(m0_all[:,:,3], cmap="jet", aspect = "auto"); title("Smoothed posterior sample3")
ax4 = subplot(2, 4, 4); imshow(m0, cmap="jet", aspect = "auto"); title("Smoothed posterior mean")
ax5 = subplot(2, 4, 5); plot_rtm(adjoint(rtm_all[:,:,1]), d; new_fig=false, aspect = "auto"); title("RTM using posterior sample1")
ax6 = subplot(2, 4, 6); plot_rtm(adjoint(rtm_all[:,:,2]), d; new_fig=false, aspect = "auto"); title("RTM using posterior sample2")
ax7 = subplot(2, 4, 7); plot_rtm(adjoint(rtm_all[:,:,3]), d; new_fig=false, aspect = "auto"); title("RTM using posterior sample3")
ax8 = subplot(2, 4, 8); plot_rtm(adjoint(rtm2), d; new_fig=false, aspect = "auto"); title("RTM using posterior mean")
savefig(plotsdir(figfolder, "rtm_m_post.png"))

figure(figsize=[20,8])
ax1 = subplot(2, 4, 1); imshow(m0_all[:,:,1], cmap="jet", aspect = "auto"); title("Smoothed posterior sample")
ax2 = subplot(2, 4, 2); imshow(m0, cmap="jet", aspect = "auto"); title("Smoothed posterior mean")
ax3 = subplot(2, 4, 3); imshow(m_lin, cmap="jet", aspect = "auto"); title("Migration velocity in RTM")
# ax4 = subplot(2, 4, 4); imshow(m_smooth, cmap="jet", aspect = "auto"); title("Smoothed true velocity")
ax4 = subplot(2, 4, 4); imshow(m_true, cmap="jet", aspect = "auto"); title("Smoothed true velocity")
ax5 = subplot(2, 4, 5); plot_rtm(adjoint(rtm_all[:,:,1]), d; new_fig=false, aspect = "auto"); title("RTM using posterior sample")
ax6 = subplot(2, 4, 6); plot_rtm(adjoint(rtm2), d; new_fig=false, aspect = "auto"); title("RTM using posterior mean")
ax7 = subplot(2, 4, 7); plot_rtm(adjoint(rtm3), d; new_fig=false, aspect = "auto"); title("RTM using linearly varying velocity")
# ax8 = subplot(2, 4, 8); plot_rtm(adjoint(rtm4), d; new_fig=false, aspect = "auto"); title("RTM using smootehd true velocity")
ax8 = subplot(2, 4, 8); plot_rtm(adjoint(rtm5), d; new_fig=false, aspect = "auto"); title("RTM using true velocity")
savefig(plotsdir(figfolder, "rtm_m.png"))


# Figures in velocity (km/s)
v = sqrt.(1f0 ./ m)
v0_all = sqrt.(1f0 ./ m0_all[:,:,1:3])
v0 = sqrt.(1f0 ./ m0)
v_lin = sqrt.(1f0 ./ m_lin)
v_smooth = sqrt.(1f0 ./ m_smooth)
v_true = sqrt.(1f0 ./ m_true)


figure(); imshow(v, vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); plt.colorbar(); title("True velocity (km/s)")
savefig(plotsdir(figfolder, "vel.png"))

figure(figsize=[20,8])
ax1 = subplot(2, 4, 1); imshow(v0_all[:,:,1], vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); title("Smoothed posterior sample1")
ax2 = subplot(2, 4, 2); imshow(v0_all[:,:,2], vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); title("Smoothed posterior sample2")
ax3 = subplot(2, 4, 3); imshow(v0_all[:,:,3], vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); title("Smoothed posterior sample3")
ax4 = subplot(2, 4, 4); imshow(v0, vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); title("Smoothed posterior mean")
ax5 = subplot(2, 4, 5); plot_rtm(adjoint(rtm_all[:,:,1]), d; new_fig=false, aspect = "auto"); title("RTM using posterior sample1")
ax6 = subplot(2, 4, 6); plot_rtm(adjoint(rtm_all[:,:,2]), d; new_fig=false, aspect = "auto"); title("RTM using posterior sample2")
ax7 = subplot(2, 4, 7); plot_rtm(adjoint(rtm_all[:,:,3]), d; new_fig=false, aspect = "auto"); title("RTM using posterior sample3")
ax8 = subplot(2, 4, 8); plot_rtm(adjoint(rtm2), d; new_fig=false, aspect = "auto"); title("RTM using posterior mean")
savefig(plotsdir(figfolder, "rtm_v_post.png"))

figure(figsize=[20,8])
ax1 = subplot(2, 4, 1); imshow(v0_all[:,:,1], vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); title("Smoothed posterior sample")
ax2 = subplot(2, 4, 2); imshow(v0, vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); title("Smoothed posterior mean")
ax3 = subplot(2, 4, 3); imshow(v_lin, vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); title("Migration velocity in RTM")
# ax4 = subplot(2, 4, 4); imshow(v_smooth, vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); title("Smoothed true velocity")
ax4 = subplot(2, 4, 4); imshow(v_true, vmin=1.8, vmax=5.0, cmap="jet", aspect = "auto"); title("True velocity")
ax5 = subplot(2, 4, 5); plot_rtm(adjoint(rtm_all[:,:,1]), d; new_fig=false, aspect = "auto"); title("RTM using posterior sample")
ax6 = subplot(2, 4, 6); plot_rtm(adjoint(rtm2), d; new_fig=false, aspect = "auto"); title("RTM using posterior mean")
ax7 = subplot(2, 4, 7); plot_rtm(adjoint(rtm3), d; new_fig=false, aspect = "auto"); title("RTM using used for INN training")
# ax8 = subplot(2, 4, 8); plot_rtm(adjoint(rtm4), d; new_fig=false, aspect = "auto"); title("RTM using smootehd true velocity")
ax8 = subplot(2, 4, 8); plot_rtm(adjoint(rtm5), d; new_fig=false, aspect = "auto"); title("RTM using true velocity")
savefig(plotsdir(figfolder, "rtm_v.png"))


figure(figsize=[20,4])
ax1 = subplot(1, 3, 1); imshow(v, cmap="jet", aspect = "auto"); plt.colorbar(); title("True velocity (km/s)")
ax2 = subplot(1, 3, 2); imshow(v0_all[:,:,1], cmap="jet", aspect = "auto"); plt.colorbar(); title("Migration velocity (km/s)")
ax3 = subplot(1, 3, 3); plot_rtm(adjoint(rtm_all[:,:,1]), d; new_fig=false, aspect = "auto"); title("RTM using posterior samples")
savefig(plotsdir(figfolder, "rtm_post_samples.png"))

figure(figsize=[20,4])
ax1 = subplot(1, 3, 1); imshow(v, cmap="jet", aspect = "auto"); plt.colorbar(); title("True velocity (km/s)")
ax2 = subplot(1, 3, 2); imshow(v0, cmap="jet", aspect = "auto"); plt.colorbar(); title("Migration velocity (km/s)")
ax3 = subplot(1, 3, 3); plot_rtm(adjoint(rtm2), d; new_fig=false, aspect = "auto"); title("RTM using posterior mean")
savefig(plotsdir(figfolder, "rtm_post_mean.png"))

figure(figsize=[20,4])
ax1 = subplot(1, 3, 1); imshow(v, cmap="jet", aspect = "auto"); plt.colorbar(); title("True velocity (km/s)")
ax2 = subplot(1, 3, 2); imshow(v_lin, cmap="jet", aspect = "auto"); plt.colorbar(); title("Migration velocity (km/s)")
ax3 = subplot(1, 3, 3); plot_rtm(adjoint(rtm3), d; new_fig=false, aspect = "auto"); title("RTM using linearly varying velocity")
savefig(plotsdir(figfolder, "rtm_lin_vel.png"))

figure(figsize=[20,4])
ax1 = subplot(1, 3, 1); imshow(v, cmap="jet", aspect = "auto"); plt.colorbar(); title("True velocity (km/s)")
ax2 = subplot(1, 3, 2); imshow(v_smooth, cmap="jet", aspect = "auto"); plt.colorbar(); title("Migration velocity (km/s)")
ax3 = subplot(1, 3, 3); plot_rtm(adjoint(rtm4), d; new_fig=false, aspect = "auto"); title("RTM using smoothed true velocity")
savefig(plotsdir(figfolder, "rtm_smooth_vel.png"))

figure(figsize=[20,4])
ax1 = subplot(1, 3, 1); imshow(v, cmap="jet", aspect = "auto"); plt.colorbar(); title("True velocity (km/s)")
ax2 = subplot(1, 3, 2); imshow(v_true, cmap="jet", aspect = "auto"); plt.colorbar(); title("Migration velocity (km/s)")
ax3 = subplot(1, 3, 3); plot_rtm(adjoint(rtm5), d; new_fig=false, aspect = "auto"); title("RTM using true velocity")
savefig(plotsdir(figfolder, "rtm_true_vel.png"))

figure()
plot( hcat(rtm_all[idx_wb+1:end,100,1], rtm2[idx_wb+1:end,100], rtm3[idx_wb+1:end,100], rtm5[idx_wb+1:end,100]) );
legend(["posterior samples", "posterior mean", "linearly varying velocity", "true velocity"])
title("Vertical profiles of RTM images")
savefig(plotsdir(figfolder, "rtm_vertical_profiles.png"))
