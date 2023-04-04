import os
import argparse
import pandas as pd
from utils.predict_frames import predict_all_frames

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description='Evaluate models')
parser.add_argument('--model_path', type=str, default="/itet-stor/mrichte/covlus_bmicnas02/matthias/trained_models/")
args = parser.parse_args()
MODELS_PATH = args.model_path


# Create dataframe with all trained models
trained_models = os.listdir(MODELS_PATH)
trained_models = [m for m in trained_models if "vgg" in m]
model_df = pd.DataFrame(trained_models)
model_df_1 = model_df[0].str.replace("simple_", "").str.replace("pre_", "").str.split("_", n=2, expand=True)
model_df = pd.concat([model_df, model_df_1], axis=1)
model_df.columns = ["dir", "train_type", "model_id", "dataset"]
model_df.dataset = model_df.dataset.str.replace("^.*?maastricht", "maastricht", regex=True)
print("DF od models to evaluate:")
print(model_df)


for idx, row in model_df.iterrows():
    print("_" * 80)
    print("Predicting all frames for model", row.dir)
    args = {
    "dataset": row.dataset,
    "weights": os.path.join(MODELS_PATH, row.dir),
    "m_id": "vgg_base",
    "classes": 2,
    "folds": 5,
    "save_path": os.path.join(MODELS_PATH, row.dir, row.dir + "_results")
    }

    # Check if results already exist
    if os.path.exists(args["save_path"] + "_all_frames_predicted.csv"):
        print("Results already exist")
        continue

    predict_all_frames(args["weights"], args["m_id"], args["dataset"], args["save_path"], args["folds"])
    print(f"Predicted all frames for model in {row.dir}.")
    save_path = args["save_path"]
    print(f"Saved predictions to {save_path}")
