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



// -------------------------------------------------------------------
// Eddington Accretion Rate and Limiter Functions
// -------------------------------------------------------------------

double eddington_accretion_rate(const double black_hole_mass, const struct params *run_params)
{
    // Eddington luminosity: L_Edd = 1.3e38 * M_BH (in Msun) erg/s
    // Convert to code units: divide by UnitEnergy_in_cgs and UnitTime_in_s
    return (1.3e38 * black_hole_mass * 1e10 / run_params->Hubble_h) / (run_params->UnitEnergy_in_cgs / run_params->UnitTime_in_s) / (0.1 * 9e10);
}

// Accretion rate limiter by Eddington limit
double eddington_limited_accretion_rate(double accretion_rate, int eddington_flag, double black_hole_mass,
                                        int snapnum, const struct params *run_params,
                                        float BHMaxaccretionMass[ABSOLUTEMAXSNAPS], float BHEddingtonRateLimit[ABSOLUTEMAXSNAPS])
{
    double edd_rate = 0.0;
    double return_rate = accretion_rate;

    if(snapnum < 0 || snapnum >= ABSOLUTEMAXSNAPS) {
        // Don't print here as it's just an invalid call
        return accretion_rate;
    }
    
    if (accretion_rate > 0.0) {
        // Calculate Eddington accretion rate only if there's accretion
        edd_rate = eddington_accretion_rate(black_hole_mass, run_params);
        BHEddingtonRateLimit[snapnum] = (float)edd_rate;  // Store the Eddington limit for this snapshot
    
        // If accretion exceeds Eddington limit, store the original value
        if (accretion_rate > edd_rate)
        {
            BHMaxaccretionMass[snapnum] = (float)accretion_rate;  // Store the unlimited accretion rate
            
            // If flag is set, prepare to return the limited rate
            if (eddington_flag) {
                return_rate = edd_rate;
            }
        }
    }

    // Always print the state before returning
   //printf("snapnum: %d, accretion_rate: %e, edd_rate: %e, black_hole_mass: %e, eddington_flag: %d, return_rate: %e\n",
     //           snapnum, accretion_rate, edd_rate, black_hole_mass, eddington_flag, return_rate);

    return return_rate;
}