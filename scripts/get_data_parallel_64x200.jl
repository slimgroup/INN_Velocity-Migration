using DrWatson
@quickactivate "INN_Velocity-Migration"

using JLD, PyPlot, LinearAlgebra, Distributed, SharedArrays
addprocs(20)
@everywhere include("../src/RTM_64x200.jl")
@everywhere using .RTM_64x200, JUDI.TimeModeling, Dates

m_all = load(datadir("overthrust_4k_models_200x64.jld"), "m_all1")
n2, n1, num = size(m_all)

# build a migration velocity m0 whose value increases linearly with depth
idx_wb = 9 # index of the water bottom
v0 = ones(1, n1)
v0[idx_wb+1:end] = collect(range(3.5, 4.3, length = n1-idx_wb))
v0 = (1f0 ./ v0).^2
v0[1:idx_wb] .= m_all[1,1,1]
m0 = repeat(v0, n2, 1)

# Right-hand preconditioners (model topmute)
# assign a mask where depth 0 to idx_wb-2 should be exactly 0, 
# idx_wb-2 to idx_wb go from 0 to 1 smoothly,
# after idx_wb is exactly 1
n = size(m0)
d = (25., 25.)
o = (0., 0.)
model0 = Model(n, d, o, m0)
# dm = m - m0
# idx_wb = find_water_bottom(dm) # find the index of the water bottom (you could also manually add this)
Tm = judiTopmute(model0.n, idx_wb, 2)  # Mute water column
S = judiDepthScaling(model0) # design a depth scaling operator
Mr = S * Tm

## Computing RTM images
rtm1_all = SharedArray{Float32,3}(n1, n2, num)
rtm2_all = SharedArray{Float32,3}(n1, n2, num)

@sync @distributed for i = 1:num

    println(string("for the ", i, "th model at ", now()))

    m = m_all[:,:,i]
    # m0 = m0_all[:,:,i] # Migration velocity
    # m0 = ones(Float32, n) .* m[1,1] # constant velocity

    # Computing RTM images
    rtm = rtm_isic(adjoint(m), adjoint(m0) )
    rtm1_all[:,:,i] = adjoint(reshape(rtm, n)) 

    rtm = Mr' * rtm # mute water layers
    rtm2_all[:,:,i] = adjoint(reshape(rtm, n)) 

end

rtm1_all = Array(rtm1_all)
rtm2_all = Array(rtm2_all)

println("Parallel computation finished")
save(datadir("overthrust_4k_rtm_64x200_lin_vel.jld"), "rtm1_all", rtm1_all, "rtm2_all", rtm2_all)

# figures of rtm examples
figure(figsize=[20,8])
ax1 = subplot(2, 3, 1); imshow(adjoint(m_all[:,:,1]), cmap="jet"); title("True velocity model1")
ax2 = subplot(2, 3, 2); imshow(adjoint(m_all[:,:,2]), cmap="jet"); title("True velocity model2")
ax3 = subplot(2, 3, 3); imshow(adjoint(m_all[:,:,3]), cmap="jet"); title("True velocity model3")
ax4 = subplot(2, 3, 4); plot_rtm(adjoint(rtm1_all[:,:,1]), d; new_fig=false); title("rtm1")
ax5 = subplot(2, 3, 5); plot_rtm(adjoint(rtm1_all[:,:,2]), d; new_fig=false); title("rtm2")
ax6 = subplot(2, 3, 6); plot_rtm(adjoint(rtm1_all[:,:,3]), d; new_fig=false); title("rtm3")
savefig(datadir("figs", "examples_rtm_64x200_1.png"))

figure(figsize=[20,8])
ax1 = subplot(2, 3, 1); imshow(adjoint(m_all[:,:,4]), cmap="jet"); title("True velocity model4")
ax2 = subplot(2, 3, 2); imshow(adjoint(m_all[:,:,5]), cmap="jet"); title("True velocity model5")
ax3 = subplot(2, 3, 3); imshow(adjoint(m_all[:,:,6]), cmap="jet"); title("True velocity model6")
ax4 = subplot(2, 3, 4); plot_rtm(adjoint(rtm1_all[:,:,4]), d; new_fig=false); title("rtm4")
ax5 = subplot(2, 3, 5); plot_rtm(adjoint(rtm1_all[:,:,5]), d; new_fig=false); title("rtm5")
ax6 = subplot(2, 3, 6); plot_rtm(adjoint(rtm1_all[:,:,6]), d; new_fig=false); title("rtm6")
savefig(datadir("figs", "examples_rtm_64x200_2.png"))


####################################################################################################
# generate INN dataset of size n1 x n2 x nc x num
m_all = permutedims(m_all, [2,1,3])
m_all = reshape(m_all, (n1, n2, 1, num))
rtm1_all = reshape(rtm1_all, (n1, n2, 1, num))
rtm2_all = reshape(rtm2_all, (n1, n2, 1, num))

save(datadir("vel_4k_samples_64x200_lin_vel.jld"), "m_all", m_all)
save(datadir("rtm_4k_samples_64x200_lin_vel.jld"), "rtm1_all", rtm1_all, "rtm2_all", rtm2_all)
