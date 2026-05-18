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

//double eddington_accretion_rate(const double black_hole_mass, const struct params *run_params)
//{
    // Eddington luminosity: L_Edd = 1.3e38 * M_BH (in Msun) erg/s
    // Convert to code units: divide by UnitEnergy_in_cgs and UnitTime_in_s
//    return (1.3e38 * black_hole_mass * 1e10 / run_params->Hubble_h) / (run_params->UnitEnergy_in_cgs / run_params->UnitTime_in_s) / (0.1 * 9e10);
//}

double eddington_accretion_rate(const double black_hole_mass, const struct params *run_params)
{
    // Physical constants (CGS)
    const double L_Edd_coeff = 1.3e38;  // erg/s per solar mass
    const double c_cm_per_s = 3.0e10;   // Speed of light
    const double eta = 0.1;              // Radiative efficiency
        
    // BH mass in solar masses
    double BH_mass_solar = black_hole_mass * 1e10 / run_params->Hubble_h;
    
    // Eddington luminosity in erg/s
    double L_Edd = L_Edd_coeff * BH_mass_solar;
    
    // Eddington accretion rate in CGS (g/s)
    // M_dot_Edd = L_Edd / (eta * c^2)
    double M_dot_Edd_cgs = L_Edd / (eta * c_cm_per_s * c_cm_per_s);
    
    // Convert to code units
    // M_dot_code = (M_dot_cgs / UnitMass_in_g) * UnitTime_in_s
    // This converts g/s → code_mass/code_time
    double M_dot_Edd_code = (M_dot_Edd_cgs / run_params->UnitMass_in_g) * run_params->UnitTime_in_s;
    
    return M_dot_Edd_code;
}

// Accretion rate limiter by Eddington limit
double eddington_limited_accretion_rate(double accretion_rate, int eddington_flag, double black_hole_mass,
                                        int snapnum, const struct params *run_params,
                                        float BHMaxaccretionMass[ABSOLUTEMAXSNAPS])
{
    double edd_rate;

    if(snapnum < 0 || snapnum >= ABSOLUTEMAXSNAPS) {
        return accretion_rate;
    }
    
    if (accretion_rate <= 0.0) {
        return accretion_rate;
    }
    
    // Calculate Eddington accretion rate
    edd_rate = eddington_accretion_rate(black_hole_mass, run_params);
    
    // If accretion exceeds Eddington limit, store the original value
    if (accretion_rate > edd_rate)
    {
        BHMaxaccretionMass[snapnum] = (float)accretion_rate;  // Store the unlimited accretion rate
        
        // If flag is set, return the limited rate; otherwise return unlimited
        if (eddington_flag) {
            return edd_rate;  // Return the Eddington-limited rate
        }
    }
    
    // Return unlimited accretion rate
    return accretion_rate;
}