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

This EPFL semester project in Pr. Hummel's lab (UPHummel) aims at exploring the striatum's functional connectivity with other motor structures after a stroke. It is composed of three main goals: a longitudinal analysis with statistical tests between FC values at T1 vs T3 and T4, correlations between FC at T1 vs motor scores at T3 and T4, and finally regressions to try and predict the motor scores from T1 FC.

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

A file where you can look at all the FC matrices.

### longitudinal_anal:

A file where the longitudinal analysis is applied, for all the different cases.

### correlations.ipynb:

Here you can find the correlations.

### see_corr.ipynb:

Sanity check on the correlations: look in details at FC matrices correlations, and try with and without outliers.

### regression.ipynb:

All regression models.

### main.py:

Original trial to make an interface where the user could enter the project parameters and get all the results, ended up running way too long and I decided to split in different notebooks.

### functions.py:

All the functions used: they are grouped together depending on their use cases, and you can find the use case by typing cntrl F to find them, such as searching for "correlations", "regression", etc.

## How to use the files:

Just run the notebooks in order ! or the data loading and then by section, but be careful to run the whole section, especially the specific data laoding for each case.

## Acknowledgements:

AI tools were used to help with the code. Otherwise this is the work of Maylis Müller, EPFL SV MA4 student, under the supervision of Camille Proulx, post-doc at UPHUMMEL's lab.