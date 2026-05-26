# CONFETTI - CONvolutional Fiber Tract Inference

Analysis toolbox for the convolutional/deep learning based analysis of white matter fiber tracts properties - with application mainly to diffusion MRI data



 
## Re Imputation experiments
* In average 2.76 (+/- 2.15) fibers have missing data - set holdout tracts to 7 (mean + 2 * stdev)
* run: python evaluate_imputation.py --holdout-tracts 7 --siren-epochs 500 FiberAxisProfiles_merged.vtk
* best performance with 200 epochs and omage 0 = 10 , over 200 epochs the INR starts overfitting (which is an issue as many missing tract information need extrapolation)

## ToDo
* use site information for harmonization -> do it during training as additional information, as well as during statistical analysis as a linear covariate (given the small intersite differences)
* Create Neighborhood Graph (radius = median nearest distance * 2, if null then find nearest neighbor ) at multiple resolutions, but halfing samples along the tract. Given the sparse nature of the axis points, the distances are computed on the full fibers, not just the axes. Visualize as VTK.
* Use GraphConvolution - UNet style to predict group, Vineland
