CONFETTI - CONvolutional Fiber Tract Inference

Analysis toolbox for the convolutional/deep learning based analysis of white matter fiber tracts properties - with application mainly to diffusion MRI data


Notes:

* Re Imputation experiments: In average 2.76 (+/- 2.15) fibers have missing data - set holdout tracts to 7 (mean + 2 * stdev)

* run: python evaluate_imputation.py --holdout-tracts 7 --siren-epochs 500 FiberAxisProfiles_merged.vtk