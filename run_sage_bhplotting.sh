#!/bin/bash

#SBATCH --job-name=sage_bhs
#SBATCH --output=slurm-bhs-%A_%a.out
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=70G

#for a third size 50G and 0:30:00

ml h5py/3.8.0
ml numpy/1.24.2-scipy-bundle-2023.02
ml matplotlib/3.7.0

#python plotting/bh_mass_func_only.py
#python plotting/bh_mass_func_withcuts.py
#python plotting/bhmf_diva.py
#python3 plotting/bh_mass_func_lit.py
#python3 plotting/bh_mass_func_sep.py
#python3 plotting/bh_growth_tracking_median.py -i /fred/oz004/jwillingham/millennium_full/model_*.hdf5
#python3 plotting/bh_growth_median_halos.py -i /fred/oz004/jwillingham/millennium_full/model_*.hdf5
#python3 plotting/bh_growth_median_halos_MstarCut.py -i /fred/oz004/jwillingham/millennium_full_bhsage_insitu/model_*.hdf5
#python3 plotting/bh_growth_history_opt.py -i /fred/oz004/jwillingham/millennium_full_bhsage_insitu/model_*.hdf5 #70G
python3 plotting/bh_growth_median_halos_ID.py -i /fred/oz004/jwillingham/millennium_full_bhsage/model_*.hdf5 #70G