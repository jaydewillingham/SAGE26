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
#include "model_enhanced_bhphysics.h"

double estimate_merging_time(const int sat_halo, const int mother_halo, const int ngal, struct halo_data *halos, struct GALAXY *galaxies, const struct params *run_params)
{
    double mergtime;
    const int MinNumPartSatHalo = 10;

    if(sat_halo == mother_halo) {
        fprintf(stderr, "Error: \t\tSnapNum, Type, IDs, sat radius:\t%i\t%i\t%i\t%i\t--- sat/cent have the same ID\n",
               galaxies[ngal].SnapNum, galaxies[ngal].Type, sat_halo, mother_halo);
        return -1.0;
    }

    const double coulomb = log1p(halos[mother_halo].Len / ((double) halos[sat_halo].Len) );//MS: 12/9/2019. As pointed out by codacy -> log1p(x) is better than log(1 + x)

    const double SatelliteMass = get_virial_mass(sat_halo, halos, run_params) + galaxies[ngal].StellarMass + galaxies[ngal].ColdGas;
    const double SatelliteRadius = get_virial_radius(mother_halo, halos, run_params);

    if(SatelliteMass > 0.0 && coulomb > 0.0 && halos[sat_halo].Len >= MinNumPartSatHalo) {
        mergtime = 2.0 *
            1.17 * SatelliteRadius * SatelliteRadius * get_virial_velocity(mother_halo, halos, run_params) / (coulomb * run_params->G * SatelliteMass);
    } else {
        mergtime = -1.0;
    }

    if (mergtime >= 999.0)
    {
        mergtime = 998.0;
        // implementing time ceiling since some objects have merge times longer than universe age when using
        // TNG50 merger trees because of lower simulation particle mass 
    }

    return mergtime;

}

// ------ NEW ------- //
static double merger_feedback_factor(const int merger_centralgal,
                       struct GALAXY *galaxies,
                       const struct params *run_params)
{
    if(run_params->SupernovaRecipeOn != 1) return 0.0;
 
    if(run_params->FIREmodeOn == 1) {
        const double z  = run_params->ZZ[galaxies[merger_centralgal].SnapNum];
        const double vc = galaxies[merger_centralgal].Vvir;
        const double V_CRIT = 60.0;
 
        if(vc <= 0.0 || z < 0.0) return 0.0;
 
        double z_term = pow(1.0 + z, run_params->RedshiftPowerLawExponent);
        double v_term;
        const double vc_floored = (vc < 1.0) ? 1.0 : vc;
        if(vc_floored < V_CRIT)
            v_term = pow(vc_floored / V_CRIT, -3.2);
        else
            v_term = pow(vc_floored / V_CRIT, -1.0);
 
        return run_params->FeedbackReheatingEpsilon * z_term * v_term;
    } else {
        return run_params->FeedbackReheatingEpsilon;
    }
}

double calculate_merger_remnant_radius(const struct GALAXY *g1, const struct GALAXY *g2)
{
    // 1. Calculate Total Baryonic Mass (Stars + Gas) for both progenitors
    double M1 = g1->StellarMass + g1->ColdGas;
    double M2 = g2->StellarMass + g2->ColdGas;
    double M_tot = M1 + M2;

    if (M_tot <= 0.0) return 0.0;

    // 2. Calculate Half-Mass Radius for both progenitors
    // For Discs: R_half ~ 1.68 * R_scale (Exponential profile)
    // For Bulges: We assume the stored radius is the half-mass radius
    
    // Progenitor 1 (Central)
    double R1_disk_half = 1.68 * g1->DiskScaleRadius;
    double R1_bulge_half = g1->BulgeRadius;
    double R1;

    if (g1->StellarMass + g1->ColdGas > 0) {
        // Mass-weighted average radius of the whole galaxy
        // Note: For pure discs, BulgeMass is 0, so this works naturally
        double M1_disk = g1->ColdGas + (g1->StellarMass - g1->BulgeMass);
        double M1_bulge = g1->BulgeMass;
        R1 = (M1_disk * R1_disk_half + M1_bulge * R1_bulge_half) / M1;
    } else {
        R1 = 0.0;
    }

    // Progenitor 2 (Satellite)
    double R2_disk_half = 1.68 * g2->DiskScaleRadius;
    double R2_bulge_half = g2->BulgeRadius;
    double R2;

    if (g2->StellarMass + g2->ColdGas > 0) {
        double M2_disk = g2->ColdGas + (g2->StellarMass - g2->BulgeMass);
        double M2_bulge = g2->BulgeMass;
        R2 = (M2_disk * R2_disk_half + M2_bulge * R2_bulge_half) / M2;
    } else {
        R2 = 0.0;
    }

    // Safeguard against zero radius (e.g., pure gas cloud with no set radius yet)
    if (R1 <= 0.0) R1 = R2; 
    if (R2 <= 0.0) R2 = R1;
    if (R1 <= 0.0) return 0.0; // Both zero

    // 3. Calculate Energy Terms (ignoring G, as it cancels out)
    // We use "Potential" units: P = M^2 / R
    
    // E_initial (Eq 21): Self-binding energy of progenitors
    double E_init = (M1 * M1) / R1 + (M2 * M2) / R2;

    // E_orbital (Eq 22): Interaction energy at merger
    // Approximated as circular orbit energy at separation R1 + R2
    double E_orb = (M1 * M2) / (R1 + R2);

    // E_rad (Eq 23): Radiative losses due to gas
    // C_rad = 2.75 (from Covington et al. 2011, cited in Tonini 2016)
    double C_rad = 2.75;
    double f_gas = (g1->ColdGas + g2->ColdGas) / M_tot;
    double E_rad = C_rad * E_init * f_gas;

    // 4. Total Final Energy (Eq 20)
    // E_final = E_init + E_orb + E_rad
    double E_final = E_init + E_orb + E_rad;

    // BUG FIX: Check E_final > 0 to avoid division by zero or negative
    // This can happen with high gas fractions where E_rad dominates
    if(E_final <= 0.0) {
        // Fallback: use mass-weighted average of progenitor radii
        return (M1 * R1 + M2 * R2) / M_tot;
    }

    // 5. Final Radius (Eq 17 rearranged)
    // R_final = M_tot^2 / E_final
    double R_final = (M_tot * M_tot) / E_final;

    return R_final;
}

/* ============================================================
 * deal_with_galaxy_merger  (REFACTORED)
 *
 * Key change: joint cold-gas budget computed before any
 * starburst or BH accretion call.
 * ============================================================ */
void deal_with_galaxy_merger(const int p,
                              const int merger_centralgal,
                              const int centralgal,
                              const double time,
                              const double dt,
                              const int halonr,
                              const int step,
                              struct GALAXY *galaxies,
                              const struct params *run_params)
{
    double mi, ma, mass_ratio;
 
    /* ---- MASS RATIO ---- */
    if(galaxies[p].StellarMass + galaxies[p].ColdGas <
       galaxies[merger_centralgal].StellarMass + galaxies[merger_centralgal].ColdGas) {
        mi = galaxies[p].StellarMass + galaxies[p].ColdGas;
        ma = galaxies[merger_centralgal].StellarMass + galaxies[merger_centralgal].ColdGas;
    } else {
        mi = galaxies[merger_centralgal].StellarMass + galaxies[merger_centralgal].ColdGas;
        ma = galaxies[p].StellarMass + galaxies[p].ColdGas;
    }
 
    if(ma > 0)       mass_ratio = mi / ma;
    else if(mi > 0)  mass_ratio = 1.0;
    else             mass_ratio = 0.0;
 
    /* ---- PRE-MERGER MORPHOLOGY FLAGS ---- */
    double central_disk_mass = galaxies[merger_centralgal].StellarMass
                             - galaxies[merger_centralgal].BulgeMass;
    int is_disk_dominated    = (central_disk_mass >
                                0.5 * galaxies[merger_centralgal].StellarMass);
 
    const double old_disk_radius = galaxies[merger_centralgal].DiskScaleRadius;
 
    /* ---- COMBINE GALAXIES ---- */
    add_galaxies_together(merger_centralgal, p, galaxies, run_params);
 
    /* ---- DETERMINE BURST DESTINATION ---- */
    int burst_to_merger_bulge;
    if(mass_ratio > run_params->ThreshMajorMerger) {
        burst_to_merger_bulge = 1;
    } else {
        burst_to_merger_bulge = is_disk_dominated ? 0 : 1;
    }
 
    /* ==============================================================
     * JOINT GAS BUDGET
     * ==============================================================
     * We compute the *uncapped* demand from each process, then scale
     * all three down by a single factor if their sum exceeds ColdGas.
     * ============================================================== */
 
    const double cold_gas = galaxies[merger_centralgal].ColdGas;
 
    /* -- (a) Starburst demand -- */
    double eburst;
    if(/* mode == */ 0 == 1)   /* mode is always 0 here */
        eburst = mass_ratio;
    else
        eburst = 0.56 * pow(mass_ratio, 0.7);
 
    double gas_for_starburst;
    if(run_params->SFprescription == 1 || run_params->SFprescription == 3 ||
       run_params->SFprescription == 4 || run_params->SFprescription == 5 ||
       run_params->SFprescription == 6 || run_params->SFprescription == 7)
        gas_for_starburst = galaxies[merger_centralgal].H2gas;
    else
        gas_for_starburst = cold_gas;
 
    double stars_demanded = eburst * gas_for_starburst;
    if(stars_demanded < 0.0) stars_demanded = 0.0;
 
    /* -- (b) SN feedback demand (proportional to stars) -- */
    double feedback_factor   = merger_feedback_factor(merger_centralgal, galaxies, run_params);
    double reheated_demanded = feedback_factor * stars_demanded;
 
    /* -- (c) BH accretion demand -- */
    double BHaccrete_demanded = 0.0;
    if(run_params->AGNrecipeOn && cold_gas > 0.0) {
        BHaccrete_demanded = run_params->BlackHoleGrowthRate * mass_ratio /
                             (1.0 + SQR(280.0 / galaxies[merger_centralgal].Vvir)) *
                             cold_gas;
        if(BHaccrete_demanded < 0.0) BHaccrete_demanded = 0.0;
    }
 
    /* -- Joint scaling -- */
    double total_demanded = stars_demanded + reheated_demanded + BHaccrete_demanded;
    double scale          = 1.0;
    if(total_demanded > cold_gas && total_demanded > 0.0)
        scale = cold_gas / total_demanded;
 
    double stars_scaled     = stars_demanded    * scale;
    double reheated_scaled  = reheated_demanded * scale;
    double BHaccrete_scaled = BHaccrete_demanded * scale;

    /* Guard against floating-point negatives */
    if(stars_scaled    < 0.0) stars_scaled    = 0.0;
    if(reheated_scaled < 0.0) reheated_scaled = 0.0;
    if(BHaccrete_scaled < 0.0) BHaccrete_scaled = 0.0;
    /* ==============================================================
     * END JOINT GAS BUDGET
     * ============================================================== */
 
    /* ---- BH ACCRETION (pre-scaled) ---- */
    if(run_params->AGNrecipeOn) {
        grow_black_hole(merger_centralgal, mass_ratio, 0, dt,
                        BHaccrete_scaled,          /* pre-scaled demand */
                        galaxies, run_params);
    }
 
    /* ---- STARBURST (pre-scaled) ---- */
    collisional_starburst_recipe(mass_ratio, merger_centralgal, centralgal,
                                  time, dt, halonr,
                                  0, step,
                                  burst_to_merger_bulge, old_disk_radius,
                                  stars_scaled, reheated_scaled, /* pre-scaled */
                                  galaxies, run_params);
 
    /* ---- MERGER REMNANT RADIUS ---- */
    double new_merger_radius =
        calculate_merger_remnant_radius(&galaxies[merger_centralgal], &galaxies[p]);
 
    /* ---- MORPHOLOGY UPDATE ---- */
    if(mass_ratio > run_params->ThreshMajorMerger) {
        make_bulge_from_burst(merger_centralgal, galaxies);
        galaxies[merger_centralgal].MergerBulgeRadius = new_merger_radius;
        galaxies[merger_centralgal].BulgeRadius       = new_merger_radius;
        galaxies[merger_centralgal].TimeOfLastMajorMerger = time;
        galaxies[p].mergeType = 2;
    } else {
        galaxies[p].mergeType = 1;
        galaxies[merger_centralgal].TimeOfLastMinorMerger = time;
        if(!is_disk_dominated) {
            galaxies[merger_centralgal].MergerBulgeRadius = new_merger_radius;
            get_bulge_radius(merger_centralgal, galaxies, run_params);
        }
    }
}



/* ============================================================
 * grow_black_hole  (REFACTORED)
 *
 * New parameter: double BHaccrete_in
 *   >= 0  → use this pre-scaled demand directly (joint-budget path)
 *   <  0  → compute demand internally as before (legacy path)
 *
 * Everything after the demand calculation is unchanged:
 *   accretiontime, Eddington limiting, ColdGas deduction,
 *   per-channel tracking, quasar_mode_wind.
 * ============================================================ */
void grow_black_hole(const int merger_centralgal,
                     const double mass_ratio,
                     const int from_instability,
                     const double dt,
                     const double BHaccrete_in,          /* NEW */
                     struct GALAXY *galaxies,
                     const struct params *run_params)
{
    double BHaccrete, metallicity;
    const int snap = galaxies[merger_centralgal].SnapNum;
 
    if(snap >= 0 && snap < ABSOLUTEMAXSNAPS) {
        galaxies[merger_centralgal].dt[snap] = (float)dt;
    }
 
    if(galaxies[merger_centralgal].ColdGas <= 0.0) return;
 
    /* ---- DEMAND CALCULATION (skipped when caller supplies value) ---- */
    if(BHaccrete_in >= 0.0) {
        /* Joint-budget path: use the pre-scaled value. */
        BHaccrete = BHaccrete_in;
    } else {
        /* Legacy path: compute demand and cap to ColdGas. */
        BHaccrete = run_params->BlackHoleGrowthRate * mass_ratio /
                    (1.0 + SQR(280.0 / galaxies[merger_centralgal].Vvir)) *
                    galaxies[merger_centralgal].ColdGas;
 
        if(BHaccrete > galaxies[merger_centralgal].ColdGas)
            BHaccrete = galaxies[merger_centralgal].ColdGas;
    }
 
    if(BHaccrete <= 0.0) return;
 
    /* ---- ACCRETION TIME ---- */
    double accretiontime;
    if(run_params->AGNDynamicAccretionOn) {
        double tdyn = dynamical_time(galaxies[merger_centralgal].BulgeRadius,
                                     galaxies[merger_centralgal].BulgeMass,
                                     run_params);
        accretiontime = tdyn;
    } else {
        accretiontime = dt;
    }
 
    /* Guard against zero or negative accretion time */
    if(accretiontime <= 0.0) accretiontime = dt;
 
    double BHaccreterate = BHaccrete / accretiontime;
 
    /* ---- EDDINGTON LIMITING ---- */
    int EddFlag = run_params->EddingtonLimitOn;
    int EddType  = from_instability ? 2 : 1;
 
    BHaccreterate = eddington_limited_accretion_rate(
                        BHaccreterate, EddFlag,
                        galaxies[merger_centralgal].BlackHoleMass,
                        galaxies[merger_centralgal].SnapNum,
                        EddType, run_params,
                        galaxies[merger_centralgal].BHAccretionType,
                        galaxies[merger_centralgal].BHMaxaccretionRate,
                        galaxies[merger_centralgal].BHEddingtonRateLimit);
 
    BHaccrete = BHaccreterate * accretiontime;
 
    /* Re-cap to ColdGas in case Eddington limiting didn't already do it
     * (should be rare after joint budget, but be defensive). */
    if(BHaccrete > galaxies[merger_centralgal].ColdGas)
        BHaccrete = galaxies[merger_centralgal].ColdGas;
 
    /* ---- SEED TRACKING ---- */
    if(galaxies[merger_centralgal].BlackHoleMass <= 0.0 && BHaccrete > 0.0)
        galaxies[merger_centralgal].BHSeedMass = BHaccrete;
 
    /* ---- APPLY TO GALAXY ---- */
    metallicity = get_metallicity(galaxies[merger_centralgal].ColdGas,
                                  galaxies[merger_centralgal].MetalsColdGas);
    galaxies[merger_centralgal].BHMassatAccretion[snap] = galaxies[merger_centralgal].BlackHoleMass;
    galaxies[merger_centralgal].BlackHoleMass     += BHaccrete;
    galaxies[merger_centralgal].ColdGas           -= BHaccrete;
    galaxies[merger_centralgal].MetalsColdGas     -= metallicity * BHaccrete;
 
    if(galaxies[merger_centralgal].MetalsColdGas < 0.0)
        galaxies[merger_centralgal].MetalsColdGas = 0.0;
 
    /* ---- PER-CHANNEL TRACKING ---- */
    if(from_instability)
        galaxies[merger_centralgal].InstabilityDrivenBHaccretionMass[snap] += BHaccrete;
    else
        galaxies[merger_centralgal].MergerDrivenBHaccretionMass[snap]      += BHaccrete;
 
    quasar_mode_wind(merger_centralgal, BHaccrete, galaxies, run_params);
 
    galaxies[merger_centralgal].QuasarModeBHaccretionMass += BHaccrete;
}



void quasar_mode_wind(const int gal, const double BHaccrete, struct GALAXY *galaxies, const struct params *run_params)
{
    // work out total energy in quasar wind (eta*m*c^2)
    const double quasar_energy = run_params->QuasarModeEfficiency * 0.1 * BHaccrete * (C / run_params->UnitVelocity_in_cm_per_s) * (C / run_params->UnitVelocity_in_cm_per_s);
    const double cold_gas_energy = 0.5 * galaxies[gal].ColdGas * galaxies[gal].Vvir * galaxies[gal].Vvir;

    // compare quasar wind and cold gas energies and eject cold
    if(quasar_energy > cold_gas_energy) {
        galaxies[gal].EjectedMass += galaxies[gal].ColdGas;
        galaxies[gal].MetalsEjectedMass += galaxies[gal].MetalsColdGas;

        galaxies[gal].ColdGas = 0.0;
        galaxies[gal].MetalsColdGas = 0.0;
    }

    // compare quasar wind and cold+hot/CGM gas energies and eject from appropriate reservoir
    if(run_params->CGMrecipeOn == 1) {
        if(galaxies[gal].Regime == 0) {
            // CGM-regime: check and eject from CGM
            const double cgm_gas_energy = 0.5 * galaxies[gal].CGMgas * galaxies[gal].Vvir * galaxies[gal].Vvir;
            
            if(quasar_energy > cold_gas_energy + cgm_gas_energy) {
                galaxies[gal].EjectedMass += galaxies[gal].CGMgas;
                galaxies[gal].MetalsEjectedMass += galaxies[gal].MetalsCGMgas;

                galaxies[gal].CGMgas = 0.0;
                galaxies[gal].MetalsCGMgas = 0.0;
            }
        } else {
            // Hot-ICM-regime: check and eject from HotGas
            const double hot_gas_energy = 0.5 * galaxies[gal].HotGas * galaxies[gal].Vvir * galaxies[gal].Vvir;
            
            if(quasar_energy > cold_gas_energy + hot_gas_energy) {
                galaxies[gal].EjectedMass += galaxies[gal].HotGas;
                galaxies[gal].MetalsEjectedMass += galaxies[gal].MetalsHotGas;

                galaxies[gal].HotGas = 0.0;
                galaxies[gal].MetalsHotGas = 0.0;
            }
        }
    } else {
        // Original SAGE behavior: check and eject from HotGas
        const double hot_gas_energy = 0.5 * galaxies[gal].HotGas * galaxies[gal].Vvir * galaxies[gal].Vvir;
        
        if(quasar_energy > cold_gas_energy + hot_gas_energy) {
            galaxies[gal].EjectedMass += galaxies[gal].HotGas;
            galaxies[gal].MetalsEjectedMass += galaxies[gal].MetalsHotGas;

            galaxies[gal].HotGas = 0.0;
            galaxies[gal].MetalsHotGas = 0.0;
        }
    }
}



void add_galaxies_together(const int t, const int p, struct GALAXY *galaxies, const struct params *run_params)
{
    galaxies[t].ColdGas += galaxies[p].ColdGas;
    galaxies[t].MetalsColdGas += galaxies[p].MetalsColdGas;

    galaxies[t].StellarMass += galaxies[p].StellarMass;
    galaxies[t].MetalsStellarMass += galaxies[p].MetalsStellarMass;

    galaxies[t].HotGas += galaxies[p].HotGas;
    galaxies[t].MetalsHotGas += galaxies[p].MetalsHotGas;

    galaxies[t].EjectedMass += galaxies[p].EjectedMass;
    galaxies[t].MetalsEjectedMass += galaxies[p].MetalsEjectedMass;

    galaxies[t].ICS += galaxies[p].ICS;
    galaxies[t].MetalsICS += galaxies[p].MetalsICS;

    galaxies[t].BlackHoleMass += galaxies[p].BlackHoleMass;
    galaxies[t].BHMergerMass[galaxies[t].SnapNum] += galaxies[p].BlackHoleMass; // jayde note Track BH mass growth from mergers separately

    //if BHExsituGrowthOn is enabled, we track the contributon to BH growth from satellites after merger.
    if(run_params->BHExsituGrowthOn) {
        
        galaxies[t].QuasarModeBHaccretionMass += galaxies[p].QuasarModeBHaccretionMass;

        for(int snap = 0; snap < ABSOLUTEMAXSNAPS; snap++) {

            galaxies[t].RadioModeBHaccretionMass[snap] += galaxies[p].RadioModeBHaccretionMass[snap];
            galaxies[t].InstabilityDrivenBHaccretionMass[snap] += galaxies[p].InstabilityDrivenBHaccretionMass[snap];
            galaxies[t].MergerDrivenBHaccretionMass[snap] += galaxies[p].MergerDrivenBHaccretionMass[snap];
        }
    }

    galaxies[t].CGMgas += galaxies[p].CGMgas;
    galaxies[t].MetalsCGMgas += galaxies[p].MetalsCGMgas;

    if (run_params->SFprescription == 1 || run_params->SFprescription == 3 ||
        run_params->SFprescription == 4 || run_params->SFprescription == 5 ||
        run_params->SFprescription == 6 || run_params->SFprescription == 7) {
        galaxies[t].H2gas += galaxies[p].H2gas;
        galaxies[t].H1gas += galaxies[p].H1gas;
    }

    // add merger to bulge
    galaxies[t].BulgeMass += galaxies[p].StellarMass;
    galaxies[t].MetalsBulgeMass += galaxies[p].MetalsStellarMass;

    // FIX 1.1: Preserve the satellite's existing bulge component breakdown
    // The satellite's bulge already has InstabilityBulgeMass and MergerBulgeMass components
    // These should be transferred to the central's corresponding components
    galaxies[t].InstabilityBulgeMass += galaxies[p].InstabilityBulgeMass;
    galaxies[t].MergerBulgeMass += galaxies[p].MergerBulgeMass;

    // The satellite's DISK mass (StellarMass - BulgeMass) becomes new bulge mass
    // Track this based on the central's current morphology (Tonini+2016 logic)
    const double satellite_disk_mass = galaxies[p].StellarMass - galaxies[p].BulgeMass;

    if(satellite_disk_mass > 0.0) {
        const double disk_mass = galaxies[t].StellarMass - galaxies[t].BulgeMass;
        const double disk_fraction = (galaxies[t].StellarMass > 0.0) ?
                                     disk_mass / galaxies[t].StellarMass : 0.0;

        if(disk_fraction > 0.5) {
            // Disc-dominated: minor merger triggers instability
            galaxies[t].InstabilityBulgeMass += satellite_disk_mass;
            const double old_disk_radius = galaxies[t].DiskScaleRadius;

            // UPDATE: Tonini incremental radius evolution (equation 16)
            update_instability_bulge_radius(t, satellite_disk_mass, old_disk_radius, galaxies, run_params);
        } else {
            // Spheroid-dominated: grows merger bulge
            galaxies[t].MergerBulgeMass += satellite_disk_mass;
        }
    }

    for(int step = 0; step < STEPS; step++) {
        galaxies[t].SfrBulge[step] += galaxies[p].SfrDisk[step] + galaxies[p].SfrBulge[step];
        galaxies[t].SfrBulgeColdGas[step] += galaxies[p].SfrDiskColdGas[step] + galaxies[p].SfrBulgeColdGas[step];
        galaxies[t].SfrBulgeColdGasMetals[step] += galaxies[p].SfrDiskColdGasMetals[step] + galaxies[p].SfrBulgeColdGasMetals[step];
    }

    // Transfer star formation history from satellite to central
    // During a merger, the central inherits all star formation history from the satellite
    if(run_params->SaveFullSFH) {
        for(int snap = 0; snap < ABSOLUTEMAXSNAPS; snap++) {
            galaxies[t].SFHMassDisk[snap] += galaxies[p].SFHMassDisk[snap];
            galaxies[t].SFHMassBulge[snap] += galaxies[p].SFHMassBulge[snap];
        }
    }
}



void make_bulge_from_burst(const int p, struct GALAXY *galaxies)
{
    // generate bulge
    galaxies[p].BulgeMass = galaxies[p].StellarMass;
    galaxies[p].MergerBulgeMass = galaxies[p].StellarMass;      // All merger-driven
    galaxies[p].InstabilityBulgeMass = 0.0;                      // Destroyed
    galaxies[p].MetalsBulgeMass = galaxies[p].MetalsStellarMass;

    // galaxies[p].BulgeRadius = get_bulge_radius(p, galaxies, run_params);

    // update the star formation rate
    for(int step = 0; step < STEPS; step++) {
        galaxies[p].SfrBulge[step] += galaxies[p].SfrDisk[step];
        galaxies[p].SfrBulgeColdGas[step] += galaxies[p].SfrDiskColdGas[step];
        galaxies[p].SfrBulgeColdGasMetals[step] += galaxies[p].SfrDiskColdGasMetals[step];
        galaxies[p].SfrDisk[step] = 0.0;
        galaxies[p].SfrDiskColdGas[step] = 0.0;
        galaxies[p].SfrDiskColdGasMetals[step] = 0.0;
    }
}

/* ============================================================
 * collisional_starburst_recipe  (REFACTORED)
 *
 * New parameters: double stars_in, double reheated_in
 *   Both >= 0  → use pre-scaled demand directly (joint-budget path)
 *   Both <  0  → compute demand internally as before (legacy path)
 *
 * Everything from update_from_star_formation onward is unchanged.
 * ============================================================ */
void collisional_starburst_recipe(const double mass_ratio,
                                  const int merger_centralgal,
                                  const int centralgal,
                                  const double time,
                                  const double dt,
                                  const int halonr,
                                  const int mode,
                                  const int step,
                                  const int burst_to_merger_bulge,
                                  const double old_disk_radius,
                                  const double stars_in,       /* NEW */
                                  const double reheated_in,    /* NEW */
                                  struct GALAXY *galaxies,
                                  const struct params *run_params)
{
    XASSERT(step >= 0 && step < STEPS, -1,
            "Error: step = %d is out of bounds [0, %d)\n", step, STEPS);
    XASSERT(dt > 0.0, -1,
            "Error: dt = %g must be > 0 for SFR calculation\n", dt);
 
    double stars, reheated_mass, ejected_mass, fac, metallicity, eburst, gas_for_starburst;
 
    /* ---- DEMAND CALCULATION (skipped when caller supplies values) ---- */
    if(stars_in >= 0.0 && reheated_in >= 0.0) {
        /* Joint-budget path: use pre-scaled values directly. */
        stars         = stars_in;
        reheated_mass = reheated_in;
        if(stars < 0.0)         stars         = 0.0;
        if(reheated_mass < 0.0) reheated_mass = 0.0;    
    } else {
        /* Legacy path: compute demand, cap to ColdGas. */
        if(mode == 1)
            eburst = mass_ratio;
        else
            eburst = 0.56 * pow(mass_ratio, 0.7);
 
        if(run_params->SFprescription == 1 || run_params->SFprescription == 3 ||
           run_params->SFprescription == 4 || run_params->SFprescription == 5 ||
           run_params->SFprescription == 6 || run_params->SFprescription == 7)
            gas_for_starburst = galaxies[merger_centralgal].H2gas;
        else
            gas_for_starburst = galaxies[merger_centralgal].ColdGas;
 
        stars = eburst * gas_for_starburst;
        if(stars < 0.0) stars = 0.0;
 
        if(run_params->SupernovaRecipeOn == 1) {
            if(run_params->FIREmodeOn == 1) {
                const double z  = run_params->ZZ[galaxies[merger_centralgal].SnapNum];
                const double vc = galaxies[merger_centralgal].Vvir;
                const double V_CRIT = 60.0;
                if(vc <= 0.0 || z < 0.0) {
                    reheated_mass = 0.0;
                } else {
                    double z_term = pow(1.0 + z, run_params->RedshiftPowerLawExponent);
                    double v_term;
                    const double vc_floored = (vc < 1.0) ? 1.0 : vc;
                    if(vc_floored < V_CRIT)
                        v_term = pow(vc_floored / V_CRIT, -3.2);
                    else
                        v_term = pow(vc_floored / V_CRIT, -1.0);
                    double eta_reheat = run_params->FeedbackReheatingEpsilon * z_term * v_term;
                    reheated_mass    = eta_reheat * stars;
                }
            } else {
                reheated_mass = run_params->FeedbackReheatingEpsilon * stars;
            }
        } else {
            reheated_mass = 0.0;
        }
 
        XASSERT(reheated_mass >= 0.0, -1,
                "Error: Reheated mass = %g should be >= 0.0", reheated_mass);
 
        if((stars + reheated_mass) > galaxies[merger_centralgal].ColdGas) {
            fac           = galaxies[merger_centralgal].ColdGas / (stars + reheated_mass);
            stars        *= fac;
            reheated_mass *= fac;
        }
    }
 
    /* ---- EJECTED MASS (always computed fresh from the final stars value) ---- */
    if(run_params->SupernovaRecipeOn == 1) {
        if(galaxies[merger_centralgal].Vvir > 0.0) {
            if(run_params->FIREmodeOn == 1) {
                const double z  = run_params->ZZ[galaxies[merger_centralgal].SnapNum];
                const double vc = galaxies[merger_centralgal].Vvir;
                const double V_CRIT = 60.0;
                if(vc <= 0.0 || z < 0.0) {
                    ejected_mass = 0.0;
                } else {
                    double z_term = pow(1.0 + z, run_params->RedshiftPowerLawExponent);
                    double v_term;
                    const double vc_floored = (vc < 1.0) ? 1.0 : vc;
                    if(vc_floored < V_CRIT)
                        v_term = pow(vc_floored / V_CRIT, -3.2);
                    else
                        v_term = pow(vc_floored / V_CRIT, -1.0);
                    double scaling_factor = z_term * v_term;
                    double E_FB  = run_params->FeedbackEjectionEfficiency * scaling_factor *
                                   0.5 * stars * (run_params->EtaSNcode * run_params->EnergySNcode);
                    double E_lift = 0.5 * reheated_mass * vc * vc;
                    ejected_mass  = (E_FB > E_lift) ? (E_FB - E_lift) / (0.5 * vc * vc) : 0.0;
                }
            } else {
                ejected_mass =
                    (run_params->FeedbackEjectionEfficiency *
                     (run_params->EtaSNcode * run_params->EnergySNcode) /
                     (galaxies[merger_centralgal].Vvir * galaxies[merger_centralgal].Vvir) -
                     run_params->FeedbackReheatingEpsilon) * stars;
            }
        } else {
            ejected_mass = 0.0;
        }
        if(ejected_mass < 0.0) ejected_mass = 0.0;
    } else {
        ejected_mass = 0.0;
    }
 
    /* ---- EVERYTHING FROM HERE IS UNCHANGED ---- */
 
    galaxies[merger_centralgal].SfrBulge[step]              += stars / dt;
    galaxies[merger_centralgal].SfrBulgeColdGas[step]       += galaxies[merger_centralgal].ColdGas;
    galaxies[merger_centralgal].SfrBulgeColdGasMetals[step] += galaxies[merger_centralgal].MetalsColdGas;
 
    metallicity = get_metallicity(galaxies[merger_centralgal].ColdGas,
                                  galaxies[merger_centralgal].MetalsColdGas);
    update_from_star_formation(merger_centralgal, stars, metallicity, galaxies, run_params);
 
    if(run_params->SaveFullSFH) {
        const int snapnum = galaxies[merger_centralgal].SnapNum;
        if(snapnum >= 0 && snapnum < ABSOLUTEMAXSNAPS)
            galaxies[merger_centralgal].SFHMassBulge[snapnum] +=
                (1.0 - run_params->RecycleFraction) * stars;
    }
 
    const double recycled_stars = (1 - run_params->RecycleFraction) * stars;
 
    galaxies[merger_centralgal].BulgeMass       += recycled_stars;
    galaxies[merger_centralgal].MetalsBulgeMass += metallicity * recycled_stars;
 
    if(burst_to_merger_bulge) {
        galaxies[merger_centralgal].MergerBulgeMass += recycled_stars;
    } else {
        galaxies[merger_centralgal].InstabilityBulgeMass += recycled_stars;
        update_instability_bulge_radius(merger_centralgal, recycled_stars,
                                        old_disk_radius, galaxies, run_params);
    }
 
    metallicity = get_metallicity(galaxies[merger_centralgal].ColdGas,
                                  galaxies[merger_centralgal].MetalsColdGas);
 
    /* BUG FIX: guard against ColdGas going negative after update_from_star_formation */
    if(galaxies[merger_centralgal].ColdGas < 0.0) galaxies[merger_centralgal].ColdGas = 0.0;
    if(reheated_mass > galaxies[merger_centralgal].ColdGas) reheated_mass = galaxies[merger_centralgal].ColdGas;
    if(reheated_mass < 0.0) reheated_mass = 0.0;

    update_from_feedback(merger_centralgal, centralgal,
                         reheated_mass, ejected_mass, metallicity,
                         galaxies, run_params);
 
    /* Clamp H2/H1 after gas has been consumed and ejected */
    if(run_params->SFprescription == 1 || run_params->SFprescription == 3 ||
       run_params->SFprescription == 4 || run_params->SFprescription == 5 ||
       run_params->SFprescription == 6 || run_params->SFprescription == 7) {
        if(galaxies[merger_centralgal].H2gas > galaxies[merger_centralgal].ColdGas)
            galaxies[merger_centralgal].H2gas = galaxies[merger_centralgal].ColdGas;
        galaxies[merger_centralgal].H1gas =
            (galaxies[merger_centralgal].ColdGas * 0.74) - galaxies[merger_centralgal].H2gas;
        if(galaxies[merger_centralgal].H1gas < 0.0)
            galaxies[merger_centralgal].H1gas = 0.0;
    }
 
    if(run_params->DiskInstabilityOn && mode == 0) {
        if(mass_ratio < run_params->ThreshMajorMerger) {
            check_disk_instability(merger_centralgal, centralgal, halonr, time, dt,
                                   step, galaxies, (struct params *) run_params);
        }
    }
 
    if(galaxies[merger_centralgal].ColdGas > 1e-8 && mass_ratio < run_params->ThreshMajorMerger) {
        const double FracZleaveDiskVal =
            run_params->FracZleaveDisk * exp(-1.0 * galaxies[centralgal].Mvir / 30.0);
        galaxies[merger_centralgal].MetalsColdGas += run_params->Yield * (1.0 - FracZleaveDiskVal) * stars;
        const double metals_leaving_disk = run_params->Yield * FracZleaveDiskVal * stars;
        if(run_params->CGMrecipeOn == 1) {
            if(galaxies[centralgal].Regime == 0)
                galaxies[centralgal].MetalsCGMgas  += metals_leaving_disk;
            else
                galaxies[centralgal].MetalsHotGas  += metals_leaving_disk;
        } else {
            galaxies[centralgal].MetalsHotGas += metals_leaving_disk;
        }
    } else {
        const double all_metals = run_params->Yield * stars;
        if(run_params->CGMrecipeOn == 1) {
            if(galaxies[centralgal].Regime == 0)
                galaxies[centralgal].MetalsCGMgas += all_metals;
            else
                galaxies[centralgal].MetalsHotGas += all_metals;
        } else {
            galaxies[centralgal].MetalsHotGas += all_metals;
        }
    }
}


void disrupt_satellite_to_ICS(const int centralgal, const int gal, const double time, struct GALAXY *galaxies, const struct params *run_params)
{
    // Transfer satellite's gas to central's hot/CGM reservoir (regime-dependent)
    const double total_gas = galaxies[gal].ColdGas + galaxies[gal].HotGas + galaxies[gal].CGMgas;
    const double total_metals_gas = galaxies[gal].MetalsColdGas + galaxies[gal].MetalsHotGas + galaxies[gal].MetalsCGMgas;
    
    if(run_params->CGMrecipeOn == 1) {
        if(galaxies[centralgal].Regime == 0) {
            // CGM-regime: disrupted gas goes to CGM
            galaxies[centralgal].CGMgas += total_gas;
            galaxies[centralgal].MetalsCGMgas += total_metals_gas;
        } else {
            // Hot-ICM-regime: disrupted gas goes to HotGas
            galaxies[centralgal].HotGas += total_gas;
            galaxies[centralgal].MetalsHotGas += total_metals_gas;
        }
    } else {
        // Original SAGE behavior: disrupted gas goes to HotGas
        galaxies[centralgal].HotGas += total_gas;
        galaxies[centralgal].MetalsHotGas += total_metals_gas;
    }

    // Transfer ejected mass (same for all regimes)
    galaxies[centralgal].EjectedMass += galaxies[gal].EjectedMass;
    galaxies[centralgal].MetalsEjectedMass += galaxies[gal].MetalsEjectedMass;

    // Transfer ICS (same for all regimes)
    galaxies[centralgal].ICS += galaxies[gal].ICS;
    galaxies[centralgal].MetalsICS += galaxies[gal].MetalsICS;

    // Track ICS assembly: pre-existing satellite ICS goes to ICS_accrete
    // This ICS was formed elsewhere (in the satellite's halo) and is being brought in
    if(run_params->TrackICSAssembly && galaxies[gal].ICS > 0.0) {
        galaxies[centralgal].ICS_accrete += galaxies[gal].ICS;
        // Inherit satellite's mass-weighted deposit-time accumulator so the
        // mean ICS-assembly time reflects when the stars were *originally* stripped,
        // not when this packet transferred into the central's reservoir.
        galaxies[centralgal].ICS_sum_mt += galaxies[gal].ICS_sum_mt;
    }

    // Disrupt stellar mass: split between ICS and BCG
    double frac_to_ICS;
    if(run_params->DynamicDisruptionSplit >= 1) {
        // Dynamic split based on halo mass ratio: f_ICL = 1 - (Msub/Mhost)^alpha_eff
        // Low mass-ratio satellites -> mostly ICL (disrupted on wide orbits)
        // High mass-ratio satellites -> more to BCG (deposited near centre)
        const double Msub = (double)galaxies[gal].infallMvir;
        const double Mhost = (double)galaxies[centralgal].Mvir;
        if(Msub > 0.0 && Mhost > 0.0) {
            double mass_ratio = Msub / Mhost;
            if(mass_ratio > 1.0) mass_ratio = 1.0;

            double alpha_eff = run_params->DisruptionSplitAlpha;
            if(run_params->DynamicDisruptionSplit == 2) {
                // Concentration-weighted: concentrated satellites resist stripping
                // alpha_eff = alpha_0 * (c_ref / c_sat)
                // High c_sat -> small alpha -> f_ICL closer to 0 -> more to BCG
                // Low c_sat  -> large alpha -> f_ICL closer to 1 -> more to ICL
                const double c_sat = (double)galaxies[gal].Concentration;
                if(c_sat > 0.0) {
                    alpha_eff *= run_params->DisruptionSplitCref / c_sat;
                }
            }

            frac_to_ICS = 1.0 - pow(mass_ratio, alpha_eff);
        } else {
            frac_to_ICS = run_params->FractionDisruptedToICS;  // fallback
        }
    } else {
        // Fixed fraction mode (original behavior)
        frac_to_ICS = run_params->FractionDisruptedToICS;
    }
    const double frac_to_BCG = 1.0 - frac_to_ICS;
    const double new_ICS_from_stripping = frac_to_ICS * galaxies[gal].StellarMass;

    galaxies[centralgal].ICS += new_ICS_from_stripping;
    galaxies[centralgal].MetalsICS += frac_to_ICS * galaxies[gal].MetalsStellarMass;
    
    // Track ICS assembly: newly disrupted stellar mass goes to ICS_disrupt
    if(run_params->TrackICSAssembly) {
        galaxies[centralgal].ICS_disrupt += new_ICS_from_stripping;
        // Record deposition time for the mass-weighted assembly-time accumulator
        galaxies[centralgal].ICS_sum_mt += new_ICS_from_stripping * time;
    }

    // Add remainder to BCG bulge (accreted onto outer envelope)
    galaxies[centralgal].StellarMass += frac_to_BCG * galaxies[gal].StellarMass;
    galaxies[centralgal].MetalsStellarMass += frac_to_BCG * galaxies[gal].MetalsStellarMass;
    galaxies[centralgal].BulgeMass += frac_to_BCG * galaxies[gal].StellarMass;
    galaxies[centralgal].MetalsBulgeMass += frac_to_BCG * galaxies[gal].MetalsStellarMass;
    galaxies[centralgal].MergerBulgeMass += frac_to_BCG * galaxies[gal].StellarMass;  // Track as merger-driven
    get_bulge_radius(centralgal, galaxies, run_params);

    // Transfer star formation history from disrupted satellite to central
    // - Fraction going to BCG bulge: track in SFHMassBulge (stellar ages)
    // Note: For ICS stellar ages, we would need SFHMassICS, but that's been replaced
    // by ICS_disrupt/ICS_accrete which track assembly times, not stellar ages
    if(run_params->SaveFullSFH) {
        for(int snap = 0; snap < ABSOLUTEMAXSNAPS; snap++) {
            const double sat_sfh = galaxies[gal].SFHMassDisk[snap] + galaxies[gal].SFHMassBulge[snap];
            galaxies[centralgal].SFHMassBulge[snap] += frac_to_BCG * sat_sfh;
        }
    }

    // what should we do with the disrupted satellite BH?
    galaxies[gal].mergeType = 4;  // mark as disruption to the ICS
}