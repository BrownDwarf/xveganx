def user_defined_lnprior(p):
    '''
    Takes a vector of stellar parameters and returns the ln prior.
    '''
    if not ( (p[0] > 3600) and (p[0] < 4500) and
             (p[1] < 4.0) and (p[1] > 3.5) and
             (p[2] < 0.5) and (p[2] > -0.5) and
             (p[3] < 30.0) and (p[3] > 0.0) and
             (p[4] < 30.0) and (p[4] > 5.0) and
             (p[5] > -10.0) and (p[5] < 0.0) and
             (p[6] > 2700.0) and (p[6] < 3600.0) and
             (p[7] > -10.0) and (p[7] < 0.0) and
             (p[11] > 0.05) and (p[11] < 1.95) and
             (p[12] > -2.0) and (p[12] < -1.4) and
             (p[13] > 1.0) and (p[13] < 30.0)):
      return -np.inf

    # Solar metalicity to within +/- 0.05 dex
    lnp_FeH = -p[2]**2/(2.0*0.05**2)
    # Logg 3.8 to within +/- 0.1 dex
    lnp_logg = -(p[1] - 3.8)**2/(2.0*0.1**2)
    # vsini of 20.0 +/- 5 from previous experiments
    lnp_vsini = -(p[4] - 20.0)**2/(2.0*10**2)
    # No prior on v_z.

    ln_prior_out = lnp_FeH + lnp_logg + lnp_vsini

    return ln_prior_out
