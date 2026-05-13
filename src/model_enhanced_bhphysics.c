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
double eddington_limited_accretion_rate(double accretion_mass, int eddington_flag, double black_hole_mass,
                                        int snapnum, const struct params *run_params,
                                        float BHMaxaccretionMass[ABSOLUTEMAXSNAPS])
{
    double edd_rate;

    if(snapnum < 0 || snapnum >= ABSOLUTEMAXSNAPS) {
        return accretion_mass;
    }
    
    if (accretion_mass <= 0.0) {
        return accretion_mass;
    }
    
    // Calculate Eddington accretion rate
    edd_rate = eddington_accretion_rate(black_hole_mass, run_params);
    
    // If accretion exceeds Eddington limit, store the original value
    if (accretion_mass > edd_rate)
    {
        BHMaxaccretionMass[snapnum] = (float)accretion_mass;  // Store the unlimited accretion rate
        
        // If flag is set, return the limited rate; otherwise return unlimited
        if (eddington_flag) {
            return edd_rate;  // Return the Eddington-limited rate
        }
    }
    
    // Return unlimited accretion rate
    return accretion_mass;
}