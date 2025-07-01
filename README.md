# uphummel_MA4

## Packages needed
- scipy
- pandas
- plotly
- os
- openpyxl
- matplotlib
- sklearn
- seaborn

## Project description:

This project aims at exploring the striatum's functional connectivity with other motor structures after a stroke. It is composed of three main goals: a longitudinal analysis with statistical tests between FC values at T1 vs T3 and T4, correlations between FC at T1 vs motor scores at T3 and T4, and finally regressions to try and predict the motor scores from T1 FC.

## Project files:


### data/ folder:

- TiMeS_regression_info_processed.xlsx : 
All the subject information and motor test results, except the MRIs

- TiMeS_rsfMRI_full_info:
Has only subject and their stroke information, not the motor tests' scores

- hcp_mmp10_yeo7_modes_indices.csv:
Yeo network split

- HCP-MMP1_RegionsCorticesList_379.csv:
Glasser atlas

- Raw_MissingDataImputed/ folder:
Has all the motor tests' scores at different timepoints (needed for untransformed FM)

### viz.ipynb:

A file where you can look at all the FC matrices

### longitudinal_anal:

A file where the longitudinal analysis is applied, for all the different cases

### correlations.ipynb:

Here you can find the correlations

### see_corr.ipynb:

Sanity check on the correlations: look in details at FC matrices correlations, and try with and without outliers

### regression.ipynb:

All regression models

### main.py:

Original trial to make an interface where the user could enter the project parameters and get all the results, ended up running way too long and I decided to split in different notebooks.

### test