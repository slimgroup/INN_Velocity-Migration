module FWI_64x200
    
    using LinearAlgebra, PyPlot, JLD, Statistics, Random, Dates
    using JUDI.TimeModeling, SegyIO, JUDI.SLIM_optim, ImageFiltering, IterativeSolvers

    export fwi

    function fwi(m, m0, niterations)
        
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

        opt = Options(space_order=16) # Use linearized inverse scattering imaging condition for the Jacobian

        # Setup operators
        Pr = judiProjection(info, recGeometry)
        F = judiModeling(info, model; options=opt)
        F0 = judiModeling(info, model0; options=opt)
        Ps = judiProjection(info, srcGeometry)
        J = judiJacobian(Pr * F0 * adjoint(Ps), q)
        
        # Nonlinear modeling
        d_obs = Pr * F * adjoint(Ps) * q # Practical observation data generated on m
        # d_syn = Pr * F0* adjoint(Ps) * q # Synthetic observation data generated on m0
        # qad = Ps * adjoint(F) * adjoint(Pr) * d_obs

        # # Linearized modeling
        # # J.options.file_name = "linearized_shot"
        # dD = J * dm
        # rtm = adjoint(J) * dD # data from linearized modeling (Born modeling)
        # grad = adjoint(J) * (d_syn - d_obs) # data from Nonlinear modeling, which gives the RTM in practice

        # rtm = adjoint(reshape(rtm, n))
        # grad = adjoint(reshape(grad, n))

        # Right-hand preconditioners (model topmute)
        idx_wb = find_water_bottom(reshape(dm, n)) # find the index of the water bottom (you could also manually add this)
        Tm = judiTopmute(model0.n, idx_wb, 2)  # Mute water column
        S = judiDepthScaling(model0) # design a depth scaling operator
        Mr = S * Tm


        # Bound constraints
        vmin = ones(Float32, model0.n) .+ 0.3f0
        vmax = ones(Float32, model0.n) .+ 5.5f0

        mmin = vec((1f0 ./ vmax).^2)	# convert to slowness squared [s^2/km^2]
        mmax = vec((1f0 ./ vmin).^2)


        ####################################################################################################
        # FWI using Stochastic gradient decent method
        # Optimization parameters
        # batchsize = 10
        # fhistory_SGD = zeros(Float32, niterations)

        # # Projection operator for bound constraints
        # proj(x) = reshape(median([vec(mmin) vec(x) vec(mmax)]; dims=2), model0.n)

        # # Main loop
        # for j = 1:niterations

        #     # get fwi objective function value and gradient
        #     i = randperm(d_obs.nsrc)[1:batchsize]
        #     fval, gradient = fwi_objective(model0, q[i], d_obs[i])
        #     println("FWI iteration no: ", j, "; function value: ", fval)
        #     fhistory_SGD[j] = fval

        #     # linesearch
        #     step = backtracking_linesearch(model0, q[i], d_obs[i], fval, gradient, proj; alpha=1f0)

        #     # Update model and bound projection
        #     model0.m = proj(model0.m + reshape(step, model0.n))
        # end
        # obj = fhistory_SGD


        ####################################################################################################
        # FWI using Gauss-Newton method
        # Optimization parameters
        maxiter_GN = 1
        fhistory_GN = zeros(Float32, niterations)
        
        # Projection operator for bound constraints
        proj(x) = reshape(median([vec(mmin) vec(x) vec(mmax)]; dims=2), model0.n)
        
        # Gauss-Newton method
        for j=1:niterations
            
            d_syn = Pr * F0* adjoint(Ps) * q # Synthetic observation data generated on m0
            fval = .5f0*norm(d_syn - d_obs)^2

            println("FWI iteration no: ", j, "; function value: ", fval, "; at ", now() )
            fhistory_GN[j] = fval

            # GN update steps
            J1 = J * Mr
            step =  zeros(Float32, n[1]*n[2])
            lsqr!(step, J1, d_syn - d_obs; maxiter=maxiter_GN, verbose=true)

            # update model and bound constraints
            model0.m = proj(model0.m - reshape(step, model0.n))  # alpha=1
        end
        obj = fhistory_GN


        ####################################################################################################
        fwi = adjoint(model0.m) # inversion result in slowness squared [s^2/km^2]
        # fwi = sqrt.(1f0 ./ adjoint(model0.m)) # inversion result in km/s

        return fwi, obj

    end
end