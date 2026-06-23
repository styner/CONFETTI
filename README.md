# CONFETTI - CONvolutional Fiber Tract Inference

Analysis toolbox for the convolutional/deep learning based analysis of white matter fiber tracts properties - with application mainly to diffusion MRI data (can also work with structural intensity information)

Background/Information: Per-tract-bundle mean summaries are commonly used for
diffusion MRI based white-matter (WM) microstructure analyses to
reduce the dimensionality of the data, but collapsing each tract to a
set of scalars (one per diffusion metric) discards the within-tract spatial
structure and vastly reduces spatial localization of effects. 

CONFETTI (CONvolutional FibEr TracT Inference) is a WM analysis
pipeline built around three components: (i) a hierarchical multi-resolution
neighborhood graph over parametrized fiber axes that retains within-
tract spatial structure and between-tract proximity; (ii) an implicit-
neural-representation-based imputation for missing diffusion tract profiles;
and (iii) a Graph Convolutional Network (GCN) analysis including per-
tract/node/covariate interpretable attribution. We implemented CONFETTI
with three structurally distinct architectures: 1) single-level GCN, 2)
multi-scale parallel-branch concatenation, 3) hierarchical Graph U-Net.
