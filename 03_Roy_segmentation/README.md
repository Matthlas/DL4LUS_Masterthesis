# Segmentation Analysis

This folder contains the code to replicate the analysis of the Segmentation model by Roy et al. on the Maastricht data set. It assumes that their code and model has been used to predict all frames of the maastricht data set. Start by running `post_processing.py` and modify the `segmentation_path` variable to point to the result location.

Afterwards you can reproduce the analysis by running `segmentation_output_analysis_maastricht.ipynb` assuming that the `data_utils.py` file has been modified to point to the correct data locations. 

The scripts `caculate_persistence.py` and `calculate_cubicial_persistence.py` are used to calculate the persistence diagrams for an output video of the segmentation model based on the Vietoris-Rips complex and the cubical complex respectively. The Notebooks `analyze_persistence.ipynb` and `analyze_cubical_persistence.ipynb` are used to calculate the persistence features and train the ml pipeline for COVID-19 diagnosis prediction based on the persistence diagrams. The Notebooks `tda.ipynb`and `tda_cubicial.ipynb` contain initial experiments with the TDA approaches and produce plots visualizing the process which are used in the thesis.
