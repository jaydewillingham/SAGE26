#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    /* Calculate the Eddington accretion rate for a black hole */
    double eddington_accretion_rate(const double black_hole_mass, const struct params *run_params);

    /* Limit accretion rate by Eddington limit if flag is set.
    * Returns the final accretion rate (either limited or unlimited).
    * Stores the pre-limited accretion rate in BHMaxaccretionMass. */
    double eddington_limited_accretion_rate(double accretion_mass, int eddington_flag, double black_hole_mass, 
                                            const struct params *run_params, float *BHMaxaccretionMass);

#ifdef __cplusplus
}
#endif
