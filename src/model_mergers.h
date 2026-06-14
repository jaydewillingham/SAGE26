#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    /* functions in model_mergers.c*/
    extern void disrupt_satellite_to_ICS(const int centralgal, const int gal, const double time, struct GALAXY *galaxies, const struct params *run_params);
    extern double estimate_merging_time(const int sat_halo, const int mother_halo, const int ngal, struct halo_data *halos, struct GALAXY *galaxies, const struct params *run_params);
    extern void deal_with_galaxy_merger(const int p, int merger_centralgal, const int centralgal, const double time,
                                        const double dt, const int halonr, const int step, struct GALAXY *galaxies, const struct params *run_params);
    extern void quasar_mode_wind(const int gal, const double BHaccrete, struct GALAXY *galaxies, const struct params *run_params);
    extern void add_galaxies_together(const int t, const int p, struct GALAXY *galaxies, const struct params *run_params);
    extern void make_bulge_from_burst(const int p, struct GALAXY *galaxies);
    extern  void grow_black_hole(int merger_centralgal, double mass_ratio, int from_instability, double dt, double BHaccrete_in, struct GALAXY *galaxies, const struct params *run_params);    
    extern  void collisional_starburst_recipe(double mass_ratio,
                        int merger_centralgal, int centralgal,
                         double time, double dt, int halonr,
                        int mode, int step,
                         int burst_to_merger_bulge,
                         double old_disk_radius,
                         double stars_in,               // NEW: <0 = legacy
                         double reheated_in,            // NEW: <0 = legacy
                         struct GALAXY *galaxies,
                         const struct params *run_params);

#ifdef __cplusplus
}
#endif