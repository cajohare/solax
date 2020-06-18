[![DOI](https://zenodo.org/badge/156694427.svg)](https://zenodo.org/badge/latestdoi/156694427)
[![arXiv](https://img.shields.io/badge/arXiv-1909.04684-B31B1B.svg)](https://arxiv.org/abs/1909.04684)
[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)


# solax
Python-3 Code to reproduce the results from our paper arXiv:[2006.XXXXX] "Axion helioscopes as solar magnetometers"

If all you need is the data for the longitudinal plasmon flux for our benchmark seismic solar magnetic field model then [click here](https://github.com/cajohare/solax/raw/master/data/solar/LPlasmonFlux_SeismicB.txt). If you want the refined Primakoff flux data then [click here](https://github.com/cajohare/solax/raw/master/data/solar/PrimakoffFlux_PlasmonCorrected.txt)

If you need any assistance or have any questions contact me at ciaran.aj.ohare@gmail.com

# Less common requirements:
* [`iminuit`](https://iminuit.readthedocs.io/en/latest/)
* [`astropy`](https://www.astropy.org/)
* [`cmocean`](https://matplotlib.org/cmocean/)
* [`numba`](http://numba.pydata.org/)

# Contents
* `data/` - Contains various data files, fluxes, solar models and axion limit data
* `src/` - Main python functions for doing the meat of the analysis
* `notebooks/` - for plotting and doing some extra analysis not found in the main paper
* `plots/` - Plots in pdf or png format


# Examples:
Click to go to the notebook used to make the plot

[<img src="plots/plots_png/Bfield_sensitivity.png" width="1000">](https://github.com/cajohare/AxionLimits/blob/master/Plot_Bfield.ipynb)

[<img src="plots/plots_png/AxionPhoton.png" width="1000">](https://github.com/cajohare/AxionLimits/blob/master/AxionPhoton_Constraint_Plot.ipynb)

[<img src="plots/plots_png/IAXO_Sensitivity_Asimov.png" width="1000">](https://github.com/cajohare/AxionLimits/blob/master/Test_IAXO_SensitivityAsimov.ipynb)

[<img src="plots/plots_png/IAXO_SeismicBfield_sensitivity_post_discovery.png" width="1000">](https://github.com/cajohare/AxionLimits/blob/master/Plot_IAXO_Bconstraint.ipynb)


---
