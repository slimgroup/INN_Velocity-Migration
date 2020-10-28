module RTM_64x200

    using JUDI.TimeModeling, SegyIO, LinearAlgebra, PyPlot, JUDI.SLIM_optim, Statistics, Random
    using ImageFiltering, Random, DelimitedFiles
    
    export rtm_isic, plot_rtm
    
    function rtm_isic(m, m0)

        m = adjoint(m)
        m0 = adjoint(m0)

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
        J = judiJacobian(Pr*F0*adjoint(Ps), q) # Born forward modeling operator

        # Forward modeling and RTM
        d_obs = Pr*F*adjoint(Ps)*q  # Practical observation data generated on m
        d_syn = Pr*F0*adjoint(Ps)*q # Synthetic observation data generated on m0

        rtm = adjoint(J) * (d_obs - d_syn)

        return rtm
    end

    
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



end