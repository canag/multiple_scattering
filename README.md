# multiple_scattering

This repository contains the codes developed in the article "Scattering properties of collective dipolar systems" by A. Canaguier-Durand, A. Lambrecht and S. Reynaud, to model optical properties of collective dipolar systems. 


# Organisation of the repository

- `collective_Smatrix.py` contains all tool functions to compute the collective scattering matrix of a collective dipolar ensemble. It is organized in successive levels (*aka lasagna code$), starting from the global and user-friendly functions, each layer using subroutines from the lower layer.
- `imag_axis.py` contains tools functions to compute collective radiation corrections in such dipolar ensembles, with additional functions that evaluate the structure X matrix at imaginary frequencies.
- `platonic_solid.py` give the positions of the vertices for Platonic solids.
- `collective_dipolar_figures_part1.ipynb` is a notebook that replicates the figure 2 of the paper, computing the relative variation of the absorption of a single nanoparticle when adding an external shell of non-absorbing nanoparticles.
- `collective_dipolar_figures_part2.ipynb` is a notebook that replicates the figure 3 of the paper, computing the collective shift of an array of metallic nanoparticles in 1D, 2D and 3D and its relative difference to the pairwise quantity.

# Prerequisites

The code only requires standard python packages `numpy` and `scipy`, its present practical implementation uses packages `pandas` and `matplolib`. 
