#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "core_allvars.h"

#include "model_mergers.h"
#include "model_misc.h"
#include "model_starformation_and_feedback.h"
#include "model_disk_instability.h"


// Seeding Function for Black Holes to go here
double seed_black_hole(const int p, const struct GALAXY *galaxies, const struct params *run_params)
{
    if(run_params->BlackHoleSeedingOn == 0) {
        return 0.0; // No seeding
    }

    if(run_params->BlackHoleSeedingOn == 1) {
        if(galaxies[p].Mvir > run_params->BHSeedMinHaloMass && galaxies[p].BlackHoleMass <= 0.0) {
            return 0.0; // Light BH Seeds
        } else {
            return 0.0;
        }    
    }

    if(run_params->BlackHoleSeedingOn == 2) {
       return 0.0; //heavy seeds
    }

    return 0.0; // Default fallback
}
// -------------------------------------------------------------------
// Eddington Accretion Rate and Limiter Functions
// -------------------------------------------------------------------

double eddington_accretion_rate(const double black_hole_mass, const struct params *run_params)
{
    // Eddington luminosity: L_Edd = 1.3e38 * M_BH (in Msun) erg/s
    // Convert to code units: divide by UnitEnergy_in_cgs and UnitTime_in_s

    if(black_hole_mass <= 0.0) {
        return 0.0; // No accretion for non-positive mass
    }


    return (1.3e38 * black_hole_mass * 1e10 / run_params->Hubble_h) / (run_params->UnitEnergy_in_cgs / run_params->UnitTime_in_s) / (0.1 * 9e10);
}

// Accretion rate limiter by Eddington limit
double eddington_limited_accretion_rate(double accretion_rate, int eddington_flag, double black_hole_mass,
                                        int snapnum, const struct params *run_params,
                                        float BHMaxaccretionRate[ABSOLUTEMAXSNAPS], float BHEddingtonRateLimit[ABSOLUTEMAXSNAPS])
{
    double edd_rate = 0.0;
    double return_rate = accretion_rate;
    const int valid_snap = (snapnum >= 0 && snapnum < ABSOLUTEMAXSNAPS);
    const int is_seed_bh = (black_hole_mass <= 0.0);

    if (accretion_rate > 0.0) {
        // Store the unlimited rate for diagnostics before any limit is applied.
        if(valid_snap) {
            BHMaxaccretionRate[snapnum] = (float)accretion_rate;
        }

        if(is_seed_bh) {
            // Seed black holes accrete without Eddington limiting.
            if(valid_snap) {
                BHEddingtonRateLimit[snapnum] = 0.0f;
            }
            return accretion_rate;
        }

        // Calculate Eddington accretion rate 
        edd_rate = eddington_accretion_rate(black_hole_mass, run_params);
        if(valid_snap) {
            BHEddingtonRateLimit[snapnum] = (float)edd_rate;
        }

        // If accretion exceeds Eddington limit and flag is set, apply the limit
        if (accretion_rate > edd_rate && eddington_flag == 1) {
            return_rate = edd_rate;
        }
    }

    return return_rate;
}