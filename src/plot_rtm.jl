
"""
plot_rtm(image, spacing)

Plots an rtm image. Image needs to be (x, z).

# Arguments

- `image`: RTM image shape (nx, nz)
- `spacing`: Grid spacing (dx, dz)
- `perc`: perc for vmin/vmax, default at 98
- `cmap`: Colormap, default is "PuOr"
- `o`: Origin of image, default is (0, 0)
- `interp`: matplotlib interplolation, default is "hanning"
- `aspect`: Aspect ratio of plot, defaults to "auto"
- `d_scale`: Depth scaling power, depth^d_scale, default is 1.5
- `name`: Plot title, default is "RTM"
- `units`: Unit of spacing, default is "m"
- `new_fig`: Whether to create a new figure, default is true
- `save`: Whether to save the figure, default name is "name" if just true, or save if save is a string
"""

function plot_rtm(image, spacing; perc=98, cmap="PuOr", o=(0, 0),
                interp="hanning", aspect=nothing, d_scale=1.5,
                name="RTM", units="m", new_fig=true, save=nothing)
nx, nz = size(image)
dx, dz = spacing
ox, oz = o
depth = range(oz, oz + (nz - 1)*spacing[2], length=nz).^d_scale

scaled = image .* depth'

a = quantile(abs.(vec(scaled)), perc/100)
extent = [ox, ox+ (nx-1)*dx, oz+(nz-1)*dz, oz]
isnothing(aspect) && (aspect = .5 * nx/nz)

if new_fig
    figure()
end
imshow(scaled', vmin=-a, vmax=a, cmap=cmap, aspect=aspect,
       interpolation=interp, extent=extent)
xlabel("X ($units)")
ylabel("Depth ($units)")
title("$name")

if ~isnothing(save)
    save == True ? filename=name : filename=save
    savefig(filename, bbox_inches="tight", dpi=150)
end
end
