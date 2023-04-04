# VGG Training & Analysis

This folder contains the code to replicate the results of the VGG16 model by Born et al. on the Maastricht data set. It assumes that the data set is located at a disc location outside this repository and the `load_datasets.py` file in the `data` folder has been modified to point to the correct data locations. The data loader has been written to load and combine multiple data sets and can be used for other data sets like the pocovid dataset. It also assumes that the `pocoivdnet` package has been installed. 

## Training
To train the VGG16 model, run the `train_simple.py -ds maastricht`. It trains the VGG16 model on the Maastricht data set. The script only trains on one fold at a time so you have to run it multiple times to train on all folds.

To pre-train the VGG16 model on the pocovid data set, run the `train_simple.py -ds pocovid`. You can then use the `train_continued.py` script to continue training on the Maastricht data set.

## Evaluation
First run `predict_all_frames_all_trained_models.py` to predict all frames for all trained models. Then run `evaluate_aggregation_strategies.py` to evaluate the models on the frame, video and patient level. The results are saved in the folder the trained model resides in. The `Model_evaluation_analysis.ipynb` notebook reads and displays the tables with results used in the thesis.
