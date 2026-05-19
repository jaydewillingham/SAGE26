#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    /* Calculate the Eddington accretion rate for a black hole */
    double eddington_accretion_rate(const double black_hole_mass, const struct params *run_params);

    /* Limit accretion rate by Eddington limit if flag is set.
     * Returns the final accretion rate (either limited or unlimited).
     * Stores the pre-limited accretion rate in BHMaxaccretionMass[snapnum] and the Eddington rate in BHEddingtonRateLimit[snapnum]. */
    double eddington_limited_accretion_rate(double accretion_rate, int eddington_flag, double black_hole_mass,
                                            int snapnum, const struct params *run_params,
                                            float BHMaxaccretionMass[ABSOLUTEMAXSNAPS], float BHEddingtonRateLimit[ABSOLUTEMAXSNAPS]);

#ifdef __cplusplus
}
#endif
