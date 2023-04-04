# covEcho Analysis

This folder contains the code to replicate the analysis of the covEcho model by Joseph et al. on the Maastricht data set. It assumes that their code and model has been used to predict all frames of the maastricht data set. Start by running `yolo_output_postprocessing.ipynb` and modify the `results_path` variable at the top to point to the result location.

Afterwards you can reproduce the analysis by running `yolo_output_analysis_maastricht.ipynb` and `yolo_output_regression_analysis_maastricht.ipynb` assuming that the `data_utils.py` file has been modified to point to the correct data locations.
