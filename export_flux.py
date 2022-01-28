# compute the vector field
pp.fvutils.compute_darcy_flux(
    gb,
    keyword=key,
    d_name=flux,
    p_name=variable,
    lam_name=mortar,
)

# to export the flux
discr_P0_flux = pp.MVEM(key)
for g, d in gb:
    discr_P0_flux.discretize(g, d)
    d[pp.STATE][flux] = d[pp.PARAMETERS][key][flux]

# construct the P0 flux reconstruction
pp.project_flux(gb, discr_P0_flux, flux, flux_P0, mortar)

