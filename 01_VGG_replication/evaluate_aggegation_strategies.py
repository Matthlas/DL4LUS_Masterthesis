import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "vscode"

import sys; sys.path.insert(0, "../data/"); sys.path.insert(0, "../utils/")
from data.load_datasets import get_dataset_loader
from statistics import evaluate_logits

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--models_path", type=str, default="/itet-stor/mrichte/covlus_bmicnas02/matthias/trained_models/")
args = parser.parse_args()
MODELS_PATH = args.models_path

# Create dataframe with all trained models
trained_models = os.listdir(MODELS_PATH)
trained_models = [m for m in trained_models if "vgg" in m]
model_df = pd.DataFrame(trained_models)
model_df_1 = model_df[0].str.replace("simple_", "").str.replace("pre_", "").str.split("_", n=2, expand=True)
model_df = pd.concat([model_df, model_df_1], axis=1)
model_df.columns = ["dir", "train_type", "model_id", "dataset"]
model_df.dataset = model_df.dataset.str.replace("^.*?maastricht", "maastricht", regex=True)


for idx, row in model_df.iterrows():
    args = {
    "save_path": os.path.join(MODELS_PATH, row.dir, row.dir + "_results")
    }
    # Check if file exists
    if os.path.exists(args["save_path"] + "_all_frames_predicted.csv"):
        print(f"Predictions exist for {row.dir}")
    else:
        print(f"Predictions do not exist for {row.dir}")

for idx, row in model_df.iterrows():

    args = {
    "dataset": row.dataset,
    "weights": os.path.join(MODELS_PATH, row.dir),
    "m_id": "vgg_base",
    "classes": 2,
    "folds": 5,
    "save_path": os.path.join(MODELS_PATH, row.dir, row.dir + "_results")
    }

    # Check if file exists
    if os.path.exists(args["save_path"] + "_all_frames_predicted.csv"):
        print(f"Predictions exist for {row.dir}")
        print("Starting evaluation")
    else:
        print(f"Predictions do not exist for {row.dir}. Continuing without evaluation.")
        continue

    print("Loading frame predictions")
    predict_df = pd.read_csv(args["save_path"] + "_all_frames_predicted.csv", index_col=0)

    def makeArray(text):
        return np.fromstring(text[1:-1], sep=' ')

    predict_df['logits'] = predict_df['logits'].apply(makeArray)
    predict_df['Label_one_hot'] = predict_df['Label_one_hot'].apply(makeArray)
    df = predict_df

    IL = get_dataset_loader(args["dataset"])
    CLASSES = list(IL.class_names)


    print("Evalutaing predictions on frame level")
    folds = df.groupby("Fold")

    eval_df = pd.DataFrame()
    for fold, fold_df in folds:
        eval = evaluate_logits(fold_df["gt"], fold_df["predIdxs"], CLASSES)
        eval = pd.concat([eval], keys=["None"], names=['Aggregation Strategy', 'Class'])
        eval = pd.concat([eval], keys=[fold], names=['Fold'])
        eval_df = pd.concat([eval_df, eval])
    

    mean = eval_df.groupby(["Aggregation Strategy", "Class"]).mean()
    std = eval_df.groupby(["Aggregation Strategy", "Class"]).std()

    # Set index
    mean.index = mean.index.set_levels(["Mean"], level=0)
    std.index = std.index.set_levels(["Standard deviation"], level=0)

    # Save basic evaluation summary
    base_summary = pd.concat([mean, std])
    base_summary.to_csv(args["save_path"] + "_base_evaluation.csv")
    print("Saved base evaluation summary to " + args["save_path"] + "_base_evaluation.csv")


    # ### Define patient level and bluepoint level dataframe
    patient_df = predict_df.groupby("Patient ID").first().drop(columns=["Frame", "logits", "predIdxs"])
    patient_df["logits"] = predict_df.groupby("Patient ID")["logits"].apply(lambda x: np.vstack(x.values))
    patient_df["predIdxs"] = predict_df.groupby("Patient ID")["predIdxs"].apply(lambda x: np.vstack(x.values))
    patient_df.reset_index(inplace=True)

    bluepoint_df = predict_df.groupby(["Patient ID", "Bluepoint"]).first().drop(columns=["Frame", "logits", "predIdxs"])
    bluepoint_df["logits"] = predict_df.groupby(["Patient ID", "Bluepoint"])["logits"].apply(lambda x: np.vstack(x.values))
    bluepoint_df["predIdxs"] = predict_df.groupby(["Patient ID", "Bluepoint"])["predIdxs"].apply(lambda x: np.vstack(x.values))
    bluepoint_df.reset_index(inplace=True)


    # ### Define different logit aggregation strategies
    pred_max = patient_df["predIdxs"].apply(lambda x: int(np.max(x, axis=0))).rename("pred_max_patient")
    pred_mean = patient_df["predIdxs"].apply(lambda x: int(np.mean(x, axis=0) > 0.5)).rename("pred_mean_patient")
    pred_median = patient_df["predIdxs"].apply(lambda x: int(np.median(x, axis=0) > 0.5)).rename("pred_median_patient")
    pred_majority = patient_df["predIdxs"].apply(lambda x: int(np.sum(x, axis=0) > len(x)/2)).rename("pred_majority_patient")

    logits_mean = patient_df["logits"].apply(lambda x: np.mean(x, axis=0)).apply(lambda x: np.argmax(x)).rename("logits_mean_patient")
    logits_median = patient_df["logits"].apply(lambda x: np.median(x, axis=0)).apply(lambda x: np.argmax(x)).rename("logits_median_patient")
    logits_max = patient_df["logits"].apply(lambda x: np.max(x, axis=0)).apply(lambda x: np.argmax(x)).rename("logits_max_patient")

    pred_df_patient = pd.DataFrame([pred_max, pred_mean, pred_median, pred_majority, logits_mean, logits_median, logits_max]).T

    pred_max = bluepoint_df["predIdxs"].apply(lambda x: int(np.max(x, axis=0))).rename("pred_max_bluepoint")
    pred_mean = bluepoint_df["predIdxs"].apply(lambda x: int(np.mean(x, axis=0) > 0.5)).rename("pred_mean_bluepoint")
    pred_median = bluepoint_df["predIdxs"].apply(lambda x: int(np.median(x, axis=0) > 0.5)).rename("pred_median_bluepoint")
    pred_majority = bluepoint_df["predIdxs"].apply(lambda x: int(np.sum(x, axis=0) > len(x)/2)).rename("pred_majority_bluepoint")

    logits_mean = bluepoint_df["logits"].apply(lambda x: np.mean(x, axis=0)).apply(lambda x: np.argmax(x)).rename("logits_mean_bluepoint")
    logits_median = bluepoint_df["logits"].apply(lambda x: np.median(x, axis=0)).apply(lambda x: np.argmax(x)).rename("logits_median_bluepoint")
    logits_max = bluepoint_df["logits"].apply(lambda x: np.max(x, axis=0)).apply(lambda x: np.argmax(x)).rename("logits_max_bluepoint")

    pred_df_bluepoint = pd.DataFrame([pred_max, pred_mean, pred_median, pred_majority, logits_mean, logits_median, logits_max]).T

    pred_df_patient = pd.concat([patient_df, pred_df_patient], axis=1)
    pred_df_bluepoint = pd.concat([bluepoint_df, pred_df_bluepoint], axis=1)

    strategies = ["pred_max",
        "pred_mean",
        "pred_median",
        "pred_majority",	
        "logits_mean",	
        "logits_median",
        "logits_max"]

    patient_strategies = [s + "_patient" for s in strategies]
    bluepoint_strategies = [s + "_bluepoint" for s in strategies]


    print("Evaluating predictions on patient level")

    folds = pred_df_patient.groupby("Fold")

    eval_df_patient = pd.DataFrame()
    for fold, fold_df in folds:
        for strategy in patient_strategies:
            eval = evaluate_logits(fold_df["gt"], fold_df[strategy], CLASSES)
            eval = pd.concat([eval], keys=[strategy], names=['Aggregation Strategy', 'Class'])
            eval = pd.concat([eval], keys=[fold], names=['Fold'])
            eval_df_patient = pd.concat([eval_df_patient, eval])

    print("Evaluating predictions on bluepoint level")
    folds = pred_df_bluepoint.groupby("Fold")

    eval_df_bluepoint = pd.DataFrame()
    for fold, fold_df in folds:
        for strategy in bluepoint_strategies:
            eval = evaluate_logits(fold_df["gt"], fold_df[strategy], CLASSES)
            eval = pd.concat([eval], keys=[strategy], names=['Aggregation Strategy', 'Class'])
            eval = pd.concat([eval], keys=[fold], names=['Fold'])
            eval_df_bluepoint = pd.concat([eval_df_bluepoint, eval])


    eval_strategies_df = pd.concat([eval_df, eval_df_patient, eval_df_bluepoint], axis=0)
    eval_strategies_df.to_csv(args["save_path"] + "_aggregation_strategy_evaluation.csv")
    print(f"Saved evaluation of all aggregation strategies to {args['save_path']}_aggregation_strategy_evaluation.csv")


    mean_df = eval_strategies_df.groupby(["Aggregation Strategy", "Class"]).mean()
    # eval_df_patient.groupby(["Strategy","Class"]).std()

    mean_df.reset_index().melt(id_vars=["Aggregation Strategy", "Class"],
            var_name="Score", 
            value_name="Score Value")



    # plot_df = mean_df.reset_index().melt(id_vars=["Aggregation Strategy", "Class"],
    #         var_name="Score", 
    #         value_name="Score Value")
    # fig = px.line(plot_df[plot_df["Class"] == 0], x="Score", y="Score Value", color='Aggregation Strategy', markers=True)
    # fig.add_traces(
    #     list(px.line(plot_df[plot_df.Class == 1], x="Score", y="Score Value", color='Aggregation Strategy', markers=True).select_traces())
    # )
    # fig.show()

    # Plotting
    class_mean_df = mean_df.groupby(["Aggregation Strategy"]).mean().rename(columns={"precision": "precision (mean)", "recall": "recall (mean)", "f1-score": "F1 (mean)", "specificity": "specificity (mean)"})

    plot_df = class_mean_df.reset_index().melt(id_vars=["Aggregation Strategy"],
            var_name="Score", 
            value_name="Score Value")
    fig = px.line(plot_df, x="Score", y="Score Value", color='Aggregation Strategy', markers=True, title= f"Aggregation strategy evaluation for {row.dir}")
    fig.write_html(args["save_path"] + "_aggregation_eval.html")
    print("Saved plot to " + args["save_path"] + "_aggregation_eval.html")



