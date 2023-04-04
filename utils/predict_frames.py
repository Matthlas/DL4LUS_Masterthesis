"""
Function to predict all frames of a trained model on a given dataset.
"""
from data.load_datasets import get_dataset_loader
from utils.evaluator import Evaluator
import tensorflow as tf
import numpy as np
import pandas as pd


def predict_all_frames(weights_path, model_id, dataset, save_path, folds=5, test=False):
    print(f"Creating data loader for {dataset} dataset")

    IL = get_dataset_loader(dataset)
    
    if test:
        IL.stride(50)
        print("!!!TEST MODE - TEST MODE - TEST MODE!!!")
        
    predictions_df = pd.DataFrame()
    for fold in range(folds):
        tf.keras.backend.clear_session()
        print("------------- SPLIT ", fold, "-------------------")
        
        print(f'Loading test data for fold {fold}...')
        testX, testY, test_df = IL.get_test_data(fold)
        num_classes = len(np.unique(testY))

        CLASSES = list(IL.class_names)
        print(f"Class mappings: {CLASSES}")

        if model_id == 'dense':
            test_data = np.expand_dims(np.stack(testX), 1)
            preprocess = False
        else:
            preprocess = True

        print("Load model")
        model = None
        # load model
        model = Evaluator(
            weights_dir=weights_path,
            ensemble=False,
            split=fold,
            num_classes=len(CLASSES),
            model_id=model_id
        )

        print(f"Testing on {len(testX)} files.")
        print( "Feeding through model and compute logits...")

        # MAIN STEP: feed through model and compute logits
        logits = np.array(
            [model(img, preprocess=preprocess) for img in testX]
        )

        # build ground truth data
        gt_one_hot = testY
        gt_class_idx = np.argmax(testY, axis=1)
        gt_class_label = [CLASSES[x] for x in gt_class_idx]
        
        # save logits and gt in the test dataframe containing patient ID, bluepoint, etc.
        pd.options.mode.chained_assignment = None
        test_df["logits"] = list(logits)
        test_df["gt"] = gt_class_idx
        test_df["predIdxs"] = np.argmax(logits, axis=1)
        test_df["gt_label (sanity check)"] = gt_class_label
        pd.options.mode.chained_assignment = "warn"

        predictions_df = pd.concat([predictions_df, test_df], axis=0)

    predictions_df = predictions_df.reset_index(drop=True)

    if save_path is not None:
        predictions_df.to_csv(save_path + "_all_frames_predicted.csv")

    print("_" * 80)
    print("Done")
    print("_" * 80)
    print(f"Done predicting all frames for {dataset} dataset on model {model_id} with \n\
        weights from {weights_path}. \n\
        Saved results to {save_path}")
    return predictions_df
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, default=None, help="Path to weights")
    parser.add_argument("--m_id", type=str, required=True, default=None, help="Model ID")
    parser.add_argument("--dataset", type=str, required=True, default=None, help="Dataset")
    parser.add_argument("--save_path", type=str, required=True, default=None, help="Path to save predictions")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--test", help="Test mode")
    args = parser.parse_args()
    
    predict_all_frames(args.weights, args.m_id, args.dataset, args.save_path, args.folds, args.test)
    print("Predicted all frames.")
    print(f"Saved predictions to {args.save_path}")
    if args.test:
        print("RAN IN TEST MODE!!!")