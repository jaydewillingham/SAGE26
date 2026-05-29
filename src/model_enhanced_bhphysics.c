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

#include <math.h>
#include <stdlib.h>


double seed_black_hole(const int p, const struct GALAXY *galaxies, const struct params *run_params)
{
    if(run_params->BlackHoleSeedingOn == 0) {
        return 0.0; // No seeding
    }

    if(run_params->BlackHoleSeedingOn == 1) {
        if(galaxies[p].Mvir > run_params->BHSeedMinHaloMass && galaxies[p].BlackHoleMass <= 0.0) {
            
            double seed_mass = 0.0;

            // Draw from a power law with bounds 30 M_sun < M_seed < 100 M_sun and slope -0.3
            // Following Ricarte & Natarajan 2018
            // Power law: p(M) ∝ M^α, where α = -0.3
            // We use inverse transform sampling to draw from this distribution
            
            double M_min = 30.0;   // Lower bound in solar masses
            double M_max = 100.0;  // Upper bound in solar masses
            double alpha = -0.3;   // Power law slope
            
            // Generate uniform random number in [0, 1)
            double u = drand48(); // or use your preferred RNG
            
            // Inverse transform for power law sampling
            // For α ≠ -1: M = M_min * (1 + u * (M_max^(α+1) / M_min^(α+1) - 1))^(1/(α+1))
            // Simplified form:
            // M = (M_min^(α+1) + u * (M_max^(α+1) - M_min^(α+1)))^(1/(α+1))
            
            double exp = 1.0 / (alpha + 1.0);  // exponent = 1 / (α + 1) = 1 / 0.7 ≈ 1.4286
            double M_min_pow = pow(M_min, alpha + 1.0);
            double M_max_pow = pow(M_max, alpha + 1.0);
            
            seed_mass = pow(M_min_pow + u * (M_max_pow - M_min_pow), exp);

            return seed_mass / (1.0e10 / run_params->Hubble_h); // Convert to code units
        } 
    }

    if(run_params->BlackHoleSeedingOn == 2) {
        if(galaxies[p].Mvir > run_params->BHSeedMinHaloMass && galaxies[p].BlackHoleMass <= 0.0) {
            return (1.0e5) / (1.0e10 / run_params->Hubble_h); // Heavy BH Seeds: constant 10^5 solar masses in code units
        }
    }

    return 0.0; // Default fallback
}



// -------------------------------------------------------------------
// Eddington Accretion Rate and Limiter Functions
// -------------------------------------------------------------------

double dynamical_time(const double r_bulge, const double M_bulge_encl, const struct params *run_params)
{
    // Dynamical time calculation: t_dyn ~ r / v_circular
    // where v_circular = sqrt(GM/r)
    
    // All inputs in code units:
    // r_bulge: kpc/h
    // M_bulge_encl: 10^10 M_sun/h
    // run_params->G: code units
    
    double rscale = r_bulge / 1.67;
    double vbulge = sqrt(run_params->G * M_bulge_encl / rscale);  // velocity in code units
                                   // length in code units
    double t_dyn = rscale / vbulge;                               // time in code units
    
    // Convert to Megayears for output
    double t_dyn_myr = t_dyn * run_params->UnitTime_in_Megayears;
    //printf("t_dyn = %g Myr (code units: %g)\n", t_dyn_myr, t_dyn);

    return t_dyn;  // Return in code units for internal use
}

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