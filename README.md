# Photometric Stereo
Estimating albedo, surface normal, depth from multiple illuminated images.

# About
This repo implements some photometric stereo algorithms. The algorithms can be found the following papers:
- Shape and Albedo from MUltiple Images using Integrability (Yuille and Snow)
- Resolving the Generalized Bas-Relief Ambiguity by Entropy Minimization (Alldrin et al.)

# Results
## Albedo, normals, and depth estimation
![est](./results/result.gif)

## Resolving GBR ambiguity
![gbr](./results/gbr_comparison_cat.gif)

![gbr](./results/gbr_comparison_women.gif)

## GPU speedup
An implementation in `pytorch` is also provided.
|Dataset|Numpy|Pytorch|
|--|--|--|
|cat|47.672s||
|women|45.056s||

# Todo
- Coarse to fine refinement
- Paper: Reflections on the Generalized Bas-Relief Ambiguity