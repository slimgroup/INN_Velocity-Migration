using DrWatson
@quickactivate "INN_Velocity-Migration"

using JLD, PyPlot, LinearAlgebra, Statistics, JUDI.TimeModeling
include(srcdir("plot_rtm.jl"))

####################################################################################################
D = load(datadir("overthrust_images_train.jld"))
num = length(D["m"])

# Crop images and models to n2 x n1
n1 = 60
n2 = 200
m_all = zeros(Float32, n2, n1, num*4)
m0_all = zeros(Float32, n2, n1, num*4)

for i = 1:num

    m_all[:,:,4*i-3] = D["m"][i][1:n2, 6:n1+5] # keep 5 water layers out of 10 water layers
    m_all[:,:,4*i-2] = D["m"][i][n2+1:n2*2, 6:n1+5]
    m_all[:,:,4*i-1] = D["m"][i][1:n2, n1+1:n1*2]
    m_all[:,:,4*i] = D["m"][i][n2+1:n2*2, n1+1:n1*2]

    m0_all[:,:,4*i-3] = D["m0"][i][1:n2, 6:n1+5] # keep 5 water layers out of 10 water layers
    m0_all[:,:,4*i-2] = D["m0"][i][n2+1:n2*2, 6:n1+5]
    m0_all[:,:,4*i-1] = D["m0"][i][1:n2, n1+1:n1*2]
    m0_all[:,:,4*i] = D["m0"][i][n2+1:n2*2, n1+1:n1*2]

end

m_all[:,1:5,:] .= m_all[1,1,1]
m0_all[:,1:5,:] .= m0_all[1,1,1]

# add 4 water layers, i.e., totally 9 water layers
n2, n1, num = size(m_all)
m_temp = m_all[1,1,1] .* ones(n2, 4, num) 
m_all = cat(m_temp, m_all; dims = 2)
m0_all = cat(m_temp, m0_all; dims = 2)

save(datadir("overthrust_8k_models_200x64.jld"), "m_all", m_all, "m0_all", m0_all)

# seperate the models w.r.t velocity
idx = sort([ collect(range(1,num, step=4)); collect(range(2,num, step=4))])
m_all1 = m_all[:,:,idx]
m_all2 = m_all[:,:,setdiff(1:end, idx)]
m0_all1 = m0_all[:,:,idx]
m0_all2 = m0_all[:,:,setdiff(1:end, idx)]

m_mean1 = mean(mean(m_all1, dims=3), dims=1)
m_mean2 = mean(mean(m_all2, dims=3), dims=1)
v_mean1 = sqrt.(1f0 ./ m_mean1) # velocity(km/s): 1.8, 3.5->4.3
v_mean2 = sqrt.(1f0 ./ m_mean2) # velocity(km/s): 1.8, 4.3->5.5

save(datadir("overthrust_4k_models_200x64.jld"), "m_all1", m_all1, "m0_all1", m0_all1, "m_all2", m_all2, "m0_all2", m0_all2)


####################################################################################################
m_all, m0_all = load(datadir("overthrust_4k_models_200x64.jld"), "m_all1", "m0_all1")

figfolder = "rtm_64x200_lin_vel"
mkpath(datadir("figs", figfolder))

## Computing RTM images
i = 1

m = m_all[:,:,i]
n = size(m)
# m0 = m0_all[:,:,i] # migration velocity
# m0 = ones(Float32, n) .* m[1,1] # constant velocity

# build a migration velocity m0 whose value increases linearly with depth
idx_wb = 9 # index of the water bottom
v0 = ones(1, n[2])
v0[idx_wb+1:end] = collect(range(3.5, 4.3, length = n[2]-idx_wb))
v0 = (1f0 ./ v0).^2
v0[1:idx_wb] .= m[1,1]
m0 = repeat(v0, n[1], 1)

dm = m - m0

figure(figsize=[8,8])
ax1 = subplot(3, 1, 1); imshow(adjoint(m), cmap="jet"); plt.colorbar(); title(L"True velocity ($s^2/km^2$)")
ax2 = subplot(3, 1, 2); imshow(adjoint(m0), cmap="jet"); plt.colorbar(); title(L"Migration velocity ($s^2/km^2$)")
ax3 = subplot(3, 1, 3); imshow(adjoint(dm), cmap="jet"); plt.colorbar(); title(L"Perturbation ($s^2/km^2$)")
savefig(datadir("figs", figfolder, "vel_example.png"))

# Computing RTM images
n = size(m)
dm = vec(m - m0)
                
# Setup info and model structure
d = (25., 25.)
o = (0., 0.)
nsrc = 21	# number of sources
                
## Set up receiver geometry in meters
nxrec = 201
xrec = range(0f0, stop=5000f0, length=nxrec)
yrec = 0f0
zrec = range(0f0, stop=0f0, length=nxrec)
                
# receiver sampling and recording time
timeR = 2000f0   # receiver recording time [ms]
dtR = 1f0    # receiver sampling interval [ms]
                
# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)
                
## Set up source geometry (cell array with source locations for each shot)
xsrc = convertToCell(range(0f0, stop=5000f0, length=nsrc))
ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc))
zsrc = convertToCell(range(0f0, stop=0f0, length=nsrc))
                
# source sampling and number of time steps
timeS = timeR  # ms
dtS = dtR   # ms
                
# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)
                
# setup wavelet
f0 = 0.006f0     # KHz
wavelet = ricker_wavelet(timeS, dtS, f0) 
q = judiVector(srcGeometry, wavelet);

# Set up model and info structure for linear operators
model = Model(n, d, o, m; nb=200)
model0 = Model(n, d, o, m0; nb=200)
ntComp = get_computational_nt(srcGeometry, recGeometry, model)
info = Info(prod(n), nsrc, ntComp)

opt = Options(space_order=16, isic=true) # Use linearized inverse scattering imaging condition for the Jacobian

# Setup operators
Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model; options=opt)
F0 = judiModeling(info, model0; options=opt)
Ps = judiProjection(info, srcGeometry)
J = judiJacobian(Pr * F0 * adjoint(Ps), q) # Born forward modeling operator

# Forward modeling and RTM
d_obs = Pr * F * adjoint(Ps) * q  # Practical observation data generated on m
d_syn = Pr * F0 * adjoint(Ps) * q # Synthetic observation data generated on m0

rtm = adjoint(J) * (d_obs - d_syn)

# rtm = rtm_isic(m, m0)
rtm1 = adjoint(reshape(rtm, n)) 
figure(figsize=[8,2]); imshow(rtm1, cmap="gray", vmin=-2e-2, vmax=2e-2); plt.colorbar(); title("RTM_isic result")
savefig(datadir("figs", figfolder, "rtm.png"))

plot_rtm(reshape(rtm, n), d)
savefig(datadir("figs", figfolder, "rtm1.png"))

figure(figsize=[12,8])
ax1 = subplot(1,3,1); imshow(d_obs.data[1], vmin=-1, vmax=1, cmap="gray", aspect="auto"); title("d_obs from src1")
ax2 = subplot(1,3,2); imshow(d_obs.data[11], vmin=-1, vmax=1, cmap="gray", aspect="auto"); title("d_obs from src11")
ax3 = subplot(1,3,3); imshow(d_obs.data[21], vmin=-1, vmax=1, cmap="gray", aspect="auto"); title("d_obs from src21")
savefig(datadir("figs", figfolder, "d_obs.png"))

figure(figsize=[12,8])
ax1 = subplot(1,3,1); imshow(d_syn.data[1], vmin=-1, vmax=1, cmap="gray", aspect="auto"); title("d_syn from src1")
ax2 = subplot(1,3,2); imshow(d_syn.data[11], vmin=-1, vmax=1, cmap="gray", aspect="auto"); title("d_syn from src11")
ax3 = subplot(1,3,3); imshow(d_syn.data[21], vmin=-1, vmax=1, cmap="gray", aspect="auto"); title("d_syn from src21")
savefig(datadir("figs", figfolder, "d_syn.png"))

figure(figsize=[12,8])
ax1 = subplot(1,3,1); imshow(d_obs.data[1]-d_syn.data[1], vmin=-1, vmax=1, cmap="gray", aspect="auto"); title("d_diff from src1")
ax2 = subplot(1,3,2); imshow(d_obs.data[11]-d_syn.data[11], vmin=-1, vmax=1, cmap="gray", aspect="auto"); title("d_diff from src11")
ax3 = subplot(1,3,3); imshow(d_obs.data[21]-d_syn.data[21], vmin=-1, vmax=1, cmap="gray", aspect="auto"); title("d_diff from src21")
savefig(datadir("figs", figfolder, "d_diff.png"))


# Right-hand preconditioners (model topmute)
# assign a mask where depth 0 to idx_wb-2 should be exactly 0, 
# idx_wb-2 to idx_wb go from 0 to 1 smoothly,
# after idx_wb is exactly 1
d = (25., 25.)
o = (0., 0.)
model0 = Model(n, d, o, m0)
idx_wb = find_water_bottom(reshape(dm, n)) # find the index of the water bottom (you could also manually add this)
Tm = judiTopmute(model0.n, idx_wb, 2)  # Mute water column
S = judiDepthScaling(model0) # design a depth scaling operator
Mr = S * Tm
rtm = Mr' * rtm # mute water layers
rtm2 = adjoint(reshape(rtm, n)) 
figure(figsize=[8,2]); imshow(rtm2, cmap="gray", vmin=-2e-2, vmax=2e-2); plt.colorbar(); title("RTM_isic result with topmute")
savefig(datadir("figs", figfolder, "rtm_TopMute.png"))

plot_rtm(reshape(rtm, n), d)
savefig(datadir("figs", figfolder, "rtm2.png"))

figure(figsize=[8,2]); imshow(rtm1 - rtm2, cmap="gray", vmin=-2e-2, vmax=2e-2); plt.colorbar(); title("Difference of RTM_isic results")
savefig(datadir("figs", figfolder, "rtm_difference.png"))
