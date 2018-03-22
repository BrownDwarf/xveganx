def user_defined_lnprior(p):
    '''
    Takes a vector of stellar parameters and returns the ln prior.
    '''
    if not ( (p[11] > 0) and
             (p[6] < p[0]) and
             (p[3] < 100) and (p[3] > -100) and
             (p[4] < 60) and (p[4] > 10)  ):
    	return -np.inf

    # Solar metalicity to within +/- 0.05 dex
    lnp_FeH = -p[2]**2/(2.0*0.05**2)
    # Logg 3.8 to within +/- 0.1 dex
    #lnp_logg = -(p[1] - 3.8)**2/(2.0*0.1**2)
    # vsini of 29.0 +/- 5 from previous experiments
    lnp_vsini = -(p[4] - 29.0)**2/(2.0*10**2)
    # No prior on v_z.

    ln_prior_out = lnp_FeH

    return ln_prior_out
