import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, balanced_accuracy_score,
    accuracy_score, matthews_corrcoef
)

def mcc_multiclass(y_true, y_pred):
    """
    MCC score for multiclass problem
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mcc_out = []
    for classe in np.unique(y_true):
        y_true_binary = (y_true == classe).astype(int)
        y_pred_binary = (y_pred == classe).astype(int)
        mcc_out.append(matthews_corrcoef(y_true_binary, y_pred_binary))
    return mcc_out


def specificity(y_true, y_pred):
    """
    Compute specificity for multiclass predictions
    """
    # true negatives / negatives
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    spec_out = []
    for classe in np.unique(y_true):
        negatives = np.sum((y_true != classe).astype(int))
        tn = np.sum((y_pred[y_true != classe] != classe).astype(int))
        spec_out.append(tn / negatives)
    return spec_out


def evaluate_logits(gt, pred_idx, CLASSES):

    report = classification_report(
        gt, pred_idx, target_names=CLASSES, output_dict=True
    )
    mcc_scores = mcc_multiclass(gt, pred_idx)
    spec_scores = specificity(gt, pred_idx)
    for i, cl in enumerate(CLASSES):
        report[cl]["mcc"] = mcc_scores[i]
        report[cl]["specificity"] = spec_scores[i]
    df = pd.DataFrame(report).transpose()
    df = df.drop(columns="support")
    df["accuracy"] = [report["accuracy"] for _ in range(len(df))]
    bal = balanced_accuracy_score(gt, pred_idx)
    df["balanced"] = [bal for _ in range(len(df))]
    return df.iloc[:len(CLASSES)]