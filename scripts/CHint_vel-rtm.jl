# Generative model using the change of variables formula
# Author: Yuxiao Ren (ryxchina@gmail.com) and Philipp Witte (pwitte3@gatech.edu)
# Date: October 2020

using DrWatson
@quickactivate "INN_Velocity-Migration"

using LinearAlgebra, InvertibleNetworks, PyPlot, Flux, Random, Test, JLD, Statistics, Dates
import Flux.Optimise.update!

# Random seed
Random.seed!(66)


####################################################################################################
# Load original data X (size of n1 x n2 x nc x ntrain)
X_orig = load(datadir("vel_4k_samples_64x200_lin_vel.jld"), "m_all")
Y_orig = load(datadir("rtm_4k_samples_64x200_lin_vel.jld"), "rtm2_all")
X_orig = Float32.(X_orig)
Y_orig = Float32.(Y_orig)
n1, n2, nc, nsamples = size(X_orig)

# Split in training - testing
ntrain = Int(nsamples*.9)
ntest = nsamples - ntrain
X_train_orig=X_orig[:, :, :, 1:ntrain]
Y_train_orig=Y_orig[:, :, :, 1:ntrain]

X_test_orig=X_orig[:, :, :, ntrain+1:end]
Y_test_orig=Y_orig[:, :, :, ntrain+1:end]

AN = ActNorm(ntrain)
X_train_orig = AN.forward(X_train_orig) # zero mean and unit std
X_test_orig = AN.forward(X_test_orig) # zero mean and unit std using the same parameters of training dataset

AN = ActNorm(ntrain) # reinitialize ActNorm for Y
Y_train_orig = AN.forward(Y_train_orig) # zero mean and unit std
Y_test_orig = AN.forward(Y_test_orig) # zero mean and unit std using the same parameters of training dataset

# Dimensions after wavelet squeeze to increase no. of channels
nx = Int(n1/2)
ny = Int(n2/2)
n_in = Int(nc*4)

# Apply wavelet squeeze (change dimensions to -> n1/2 x n2/2 x nc*4 x ntrain)
X_train = zeros(Float32, nx, ny, n_in, ntrain)
Y_train = zeros(Float32, nx, ny, n_in, ntrain)
for j=1:ntrain
    X_train[:, :, :, j:j] = wavelet_squeeze(X_train_orig[:, :, :, j:j])
    Y_train[:, :, :, j:j] = wavelet_squeeze(Y_train_orig[:, :, :, j:j])
end

X_test = zeros(Float32, nx, ny, n_in, ntest)
Y_test = zeros(Float32, nx, ny, n_in, ntest)
for j=1:ntest
    X_test[:, :, :, j:j] = wavelet_squeeze(X_test_orig[:, :, :, j:j])
    Y_test[:, :, :, j:j] = wavelet_squeeze(Y_test_orig[:, :, :, j:j])
end

# Create network
n_hidden = 64
batchsize = 16 #4
depth = 16
CH = NetworkConditionalHINT(nx, ny, n_in, batchsize, n_hidden, depth) |>gpu
Params = get_params(CH)

####################################################################################################

# Loss
function loss(CH, X, Y)
    Zx, Zy, logdet = CH.forward(X, Y)
    f = -log_likelihood(tensor_cat(Zx, Zy)) - logdet
    ΔZ = -∇log_likelihood(tensor_cat(Zx, Zy))
    ΔZx, ΔZy = tensor_split(ΔZ)
    ΔX, ΔY = CH.backward(ΔZx, ΔZy, Zx, Zy)[1:2]
    return f, ΔX, ΔY
end

# Training
max_epoch = 200
max_iter = floor(Int, ntrain/batchsize)
opt = Flux.ADAM(1f-3)
lr_step = 100
lr_decay_fn = Flux.ExpDecay(1f-3, .9, lr_step, 0.)
losses = zeros(Float32, max_epoch*max_iter)
fval = zeros(Float32, max_epoch)

t1 = now()
println(string("Training starts at ", t1))
for i=1:max_epoch

    index = randperm(ntrain)

    for j=1:max_iter

        # Evaluate objective and gradients
        idx = randperm(ntrain)[1:batchsize]
        X = X_train[:, :, :, idx] |>gpu
        Y = Y_train[:, :, :, idx] |>gpu
        # Y = X + .5f0*randn(Float32, nx, ny, n_in, batchsize)
        
        losses[(i-1)*max_iter + j] = loss(CH, X, Y)[1]
        GC.gc()

        # Update params
        for p in Params
            update!(opt, p.data, p.grad)
            update!(lr_decay_fn, p.data, p.grad)
        end
        clear_grad!(CH)
    end

    fval[i] = losses[i*max_iter]
    mod(i, 10) == 0 && (print("Epoch: ", i, "; f = ", fval[i], "; at ", now(), "\n"))

end
CH = CH |>cpu

t2 = now()
println(string("Training finishes after ", Dates.value.(t2-t1)/3600000, " hours"))

####################################################################################################
## Plotting
# test1 for vel + rtm_mig_vel
# test2 for vel + rtm_const_vel
# test3 for vel + rtm_lin_vel
figfolder = string("chint/test3_", max_epoch, "_", depth, "_", batchsize)
mkpath(plotsdir(figfolder))

save(plotsdir(figfolder, "chint.jld"), "CH", CH, "losses", losses, "fval", fval)

# Testing
test_size = 100
idx = randperm(ntest)[1:test_size]  # draw random samples from testing data
X = X_test[:, :, :, idx]
Y = Y_test[:, :, :, idx]
# Y = X + .5f0*randn(Float32, nx, ny, n_in, test_size)
Zx_, Zy_ = CH.forward(X, Y)[1:2]

Zx = randn(Float32, nx, ny, n_in, test_size)
Zy = randn(Float32, nx, ny, n_in, test_size)
X_, Y_ = CH.inverse(Zx, Zy)

# Now select single fixed sample from all Ys
for idx = 1:10

    # idx = 10
    X_fixed = X[:, :, :, idx:idx]
    Y_fixed = Y[:, :, :, idx:idx]
    Zy_fixed = CH.forward_Y(Y_fixed)
    
    # Draw new Zx, while keeping Zy fixed
    X_post = CH.inverse(Zx, Zy_fixed.*ones(Float32, nx, ny, n_in, test_size))[1]

    X_fixed = wavelet_unsqueeze(X_fixed)
    Y_fixed = wavelet_unsqueeze(Y_fixed)
    Zy_fixed = wavelet_unsqueeze(Zy_fixed)
    X_post = wavelet_unsqueeze(X_post)

    save(plotsdir(figfolder, string("posterior_samples", idx, ".jld")), "X_fixed", X_fixed, "Y_fixed", Y_fixed, "X_post", X_post, "Zy_fixed", Zy_fixed)

    # Plot posterior samples, mean and standard deviation
    figure(figsize=[20,8])
    X_post_mean = mean(X_post; dims=4)
    X_post_std = std(X_post; dims=4)
    ax1 = subplot(2,4,1); imshow(X_fixed[:, :, 1, 1], cmap="jet", aspect="auto"); colorbar(); title("True velocity x")
    ax2 = subplot(2,4,2); imshow(X_post[:, :, 1, 1], cmap="jet", aspect="auto"); colorbar(); title(L"Post. sample: $x = f(zx|zy_{fix})^{-1}$")
    ax3 = subplot(2,4,3); imshow(X_post[:, :, 1, 2], cmap="jet", aspect="auto"); colorbar(); title(L"Post. sample: $x = f(zx|zy_{fix})^{-1}$")
    ax4 = subplot(2,4,4); imshow(X_post_mean[:, :, 1, 1], cmap="jet", aspect="auto"); colorbar(); title("Posterior mean")
    ax5 = subplot(2,4,5); imshow(Y_fixed[:, :, 1, 1], cmap="jet", aspect="auto"); colorbar(); title(L"RTM: $y_i=RTM(x_i)$ ")
    ax6 = subplot(2,4,6); imshow(X_post[:, :, 1, 4], cmap="jet", aspect="auto"); colorbar(); title(L"Post. sample: $x = f(zx|zy_{fix})^{-1}$")
    ax7 = subplot(2,4,7); imshow(X_post[:, :, 1, 5], cmap="jet", aspect="auto"); colorbar(); title(L"Post. sample: $x = f(zx|zy_{fix})^{-1}$")
    ax8 = subplot(2,4,8); imshow(X_post_std[:, :, 1,1], cmap="binary", aspect="auto", vmin=0, vmax=0.9*maximum(X_post_std)); 
    colorbar(); title("Posterior std");

    savefig(plotsdir(figfolder, string("posterior_samples", idx, ".png") ))

end

# Unsqueeze all tensors
X = wavelet_unsqueeze(X)
Y = wavelet_unsqueeze(Y)
Zx_ = wavelet_unsqueeze(Zx_)
Zy_ = wavelet_unsqueeze(Zy_)

X_ = wavelet_unsqueeze(X_)
Y_ = wavelet_unsqueeze(Y_)
Zx = wavelet_unsqueeze(Zx)
Zy = wavelet_unsqueeze(Zy)

# X_fixed = wavelet_unsqueeze(X_fixed)
# Y_fixed = wavelet_unsqueeze(Y_fixed)
# Zy_fixed = wavelet_unsqueeze(Zy_fixed)
# X_post = wavelet_unsqueeze(X_post)

# Plot one sample from X and Y and their latent versions
figure(figsize=[20,8])
ax1 = subplot(2,4,1); imshow(X[:, :, 1, 1], cmap="jet", aspect="auto"); colorbar(); title(L"Model space: $x \sim \hat{p}_x$")
ax2 = subplot(2,4,2); imshow(Y[:, :, 1, 1], cmap="jet", aspect="auto"); colorbar(); title(L"RTM: $y=RTM(x)$ ")
ax3 = subplot(2,4,3); imshow(X_[:, :, 1, 1], cmap="jet", aspect="auto"); colorbar(); title(L"Model space: $x = f(zx|zy)^{-1}$")
ax4 = subplot(2,4,4); imshow(Y_[:, :, 1, 1], cmap="jet", aspect="auto"); colorbar(); title(L"Data space: $y = f(zx|zy)^{-1}$")
ax5 = subplot(2,4,5); imshow(Zx_[:, :, 1, 1], cmap="jet", aspect="auto"); colorbar(); title(L"Latent space: $zx = f(x|y)$")
ax6 = subplot(2,4,6); imshow(Zy_[:, :, 1, 1], cmap="jet", aspect="auto"); colorbar(); title(L"Latent space: $zy = f(x|y)$")
ax7 = subplot(2,4,7); imshow(Zx[:, :, 1, 1], cmap="jet", aspect="auto"); colorbar(); title(L"Latent space: $zx \sim \hat{p}_{zx}$")
ax8 = subplot(2,4,8); imshow(Zy[:, :, 1, 1], cmap="jet", aspect="auto"); colorbar(); title(L"Latent space: $zy \sim \hat{p}_{zy}$")
savefig(plotsdir(figfolder, "general.png"))

# Plot various samples from X and Y
figure(figsize=[20,8])
i = randperm(test_size)[1:4]
ax1 = subplot(2,4,1); imshow(X_[:, :, 1, i[1]], cmap="jet", aspect="auto"); colorbar(); title(L"Model space: $x = f(zx|zy)^{-1}$")
ax2 = subplot(2,4,2); imshow(X_[:, :, 1, i[2]], cmap="jet", aspect="auto"); colorbar(); title(L"Model space: $x = f(zx|zy)^{-1}$")
ax3 = subplot(2,4,3); imshow(X_[:, :, 1, i[3]], cmap="jet", aspect="auto"); colorbar(); title(L"Model space: $x = f(zx|zy)^{-1}$")
ax4 = subplot(2,4,4); imshow(X_[:, :, 1, i[4]], cmap="jet", aspect="auto"); colorbar(); title(L"Model space: $x = f(zx|zy)^{-1}$")
ax5 = subplot(2,4,5); imshow(X[:, :, 1, i[1]], cmap="jet", aspect="auto"); colorbar(); title(L"Model space: $x \sim \hat{p}_x$")
ax6 = subplot(2,4,6); imshow(X[:, :, 1, i[2]], cmap="jet", aspect="auto"); colorbar(); title(L"Model space: $x \sim \hat{p}_x$")
ax7 = subplot(2,4,7); imshow(X[:, :, 1, i[3]], cmap="jet", aspect="auto"); colorbar(); title(L"Model space: $x \sim \hat{p}_x$")
ax8 = subplot(2,4,8); imshow(X[:, :, 1, i[4]], cmap="jet", aspect="auto"); colorbar(); title(L"Model space: $x \sim \hat{p}_x$")
savefig(plotsdir(figfolder, "model_space_samples.png"))

# Plot loss values
figure(); plot(1:max_epoch, fval[1:max_epoch]); title("loss values")
savefig(plotsdir(figfolder, "loss_curve.png"))
