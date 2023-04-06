import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet, LinearRegression, Ridge
#from xgboost import XGBClassifier, XGBRegressor

# Pipeline
from sklearn.model_selection import cross_validate,StratifiedKFold, KFold, GroupKFold, StratifiedGroupKFold #,RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Scores
from sklearn.metrics import recall_score, f1_score, cohen_kappa_score, precision_score
from sklearn.metrics import make_scorer

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 



CLASSIFICATION_MODELS = { #"KNN" : KNeighborsClassifier(),
                        "LR" : LogisticRegression(),
                        "SVM" : LinearSVC(max_iter=10000),
                        "RF" : RandomForestClassifier(),
                        #"XGB" : XGBClassifier(),
                        }

CLASSIFICATION_METRICS = { "accuracy" : "accuracy",
                        #"recall" : make_scorer(recall_score),
                        #"precision" : make_scorer(precision_score),
                        "sensitivity" : make_scorer(recall_score, pos_label=1),
                        "specificity" : make_scorer(recall_score, pos_label=0),
                        "f1" : make_scorer(f1_score),
                        }  

CLASSIFICATION_METRICS_MULTILABEL = { "accuracy" : "accuracy",
                        "f1" : make_scorer(f1_score, average="weighted"),
                        "recall" : make_scorer(recall_score, average="weighted"),
                        "precision" : make_scorer(precision_score, average="weighted"),
                        # "sensitivity" : make_scorer(recall_score, average="weighted", pos_label=1),
                        #"cohen_kappa" : make_scorer(cohen_kappa_score),
                        }  

REGRESSION_MODELS = { "LR" : LinearRegression(),
                    "Ridge" : Ridge(),
                    "ElasticNet" : ElasticNet(),
                    "SVR" : LinearSVR(max_iter=10000),
                    "RF" : RandomForestRegressor(),
                    #"XGB" : XGBRegressor(),
                    }

REGRESSION_METRICS = { "r2" : "r2",
                    #"mse" : "neg_mean_squared_error",
                    }


class ModelEvaluation:
    def __init__(self, mode="classification"):
        # Set seed
        self.SEED = 42
        np.random.seed(self.SEED)

        # Check if mode is valid
        if mode not in ["classification", "classification_multi_label", "regression"]:
            raise ValueError("Mode must be either classification, classification_multi_label or regression.")
        # Set mode
        self.mode = mode

        # Define metrics
        if self.mode == "classification":
            self.eval_metrics = CLASSIFICATION_METRICS
        elif self.mode == "classification_multi_label":
            self.eval_metrics = CLASSIFICATION_METRICS_MULTILABEL
        else:
            self.eval_metrics = REGRESSION_METRICS

        # Define models
        self.models_dict = REGRESSION_MODELS if self.mode == "regression" else CLASSIFICATION_MODELS

        # Make pipeline adding standard scaler to all models
        self.models_dict = {model_name: make_pipeline(StandardScaler(), model) for model_name, model in self.models_dict.items()}

        # Define cross validation. One for ungroupped data and one for groupped data.
        if self.mode == "regression":
            self.cv = KFold(n_splits=5, shuffle=True, random_state=self.SEED)
            self.group_cv = GroupKFold(n_splits=5)
        else:
            self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.SEED)
            self.group_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.SEED)

    def train_models(self, X, y, groups=None, plot=True, verbose=False, return_train_scores=False, return_model_predictions=False):

        # If return_model_predictions is not false it must be a df with the same index as X containing the video_name variable
        if return_model_predictions is not False:
            assert isinstance(return_model_predictions, pd.DataFrame)
            assert return_model_predictions.index.equals(X.index)
            assert "video_name" in return_model_predictions.columns

        if groups is not None:
            splitter = self.group_cv
        else:
            splitter = self.cv


        model_perfomance = pd.DataFrame()
        for model_name, model in self.models_dict.items():

            if verbose:
                print(f"Training {model_name}...")
            scores = cross_validate(model, X, y, 
                                    scoring=self.eval_metrics, 
                                    cv=splitter, groups=groups, 
                                    return_train_score=return_train_scores, 
                                    return_estimator=return_model_predictions is not False,
                                    )
            temp = pd.DataFrame(scores)
            temp["model"] = model_name

            # If return_model_predictions is True, obtain X and y folds from the splitter and add them to the data frame
            if return_model_predictions is not False:
                X_fold = [X.iloc[test_index] for train_index, test_index in splitter.split(X, y, groups=groups)]
                y_fold = [list(y.iloc[test_index]) for train_index, test_index in splitter.split(X, y, groups=groups)]
                temp["X"] = X_fold
                temp["y"] = y_fold
                # Then add the video names of the predictions
                pred_idx_fold = [list(return_model_predictions.video_name.iloc[test_index]) for train_index, test_index in splitter.split(X, y, groups=groups)]
                temp["pred_idx"] = pred_idx_fold
            model_perfomance = pd.concat([model_perfomance, temp])


        model_performance_mean = model_perfomance.groupby("model").mean().drop(columns=["fit_time", "score_time"])

        if plot:
            self.plot_model_performance(model_perfomance, model_performance_mean, X, y)

        # Add predictions
        if return_model_predictions is not False:
            model_perfomance["prediction"] = model_perfomance.apply(self.get_model_prediction, axis=1)
            model_perfomance["predictions"] = model_perfomance.apply(self.combine_predictions, axis=1)
            predictions = self.pivot_predictions(model_perfomance)
            # Drop unnecessary columns from model_perfomance
            model_perfomance = model_perfomance.drop(columns=["estimator", "X", "y", "pred_idx", "prediction", "predictions"])
            return model_perfomance, model_performance_mean, predictions
        else:
            return model_perfomance, model_performance_mean

    def plot_model_performance(self, model_perfomance, model_performance_mean, X, y):
        # Plot model performance with error bars
        sns.set()
        
        model_performance_mean.plot.bar(rot=0, #figsize=(10, 5), 
                            yerr=model_perfomance.groupby("model").std().drop(columns=["fit_time", "score_time"]))
        plt.ylabel("Test Score")
        plt.xlabel("Model")
        plt.legend(loc='lower right')
        s, f = X.shape
        plt.title(f"Test scores. Using {s} samples and {f} features.")

        # Draw line at chance level if multiclass model
        if self.mode == "classification_multi_label":
            plt.axhline(y=1/len(np.unique(y)), color='r', linestyle='-')
            
        plt.show()

    # For each row in the dataframe, get the model prediction
    def get_model_prediction(self, row):
        model = row["estimator"]
        X = row["X"]
        predictions = model.predict(X)
        return predictions
    
    def combine_predictions(self, row):
        return pd.DataFrame({"video_name":row.pred_idx, "y":row.y, "y_pred":row.prediction})

    def pivot_predictions(self, model_perfomance):
        pred_df = model_perfomance[['predictions', 'model']].copy()
        # For each model combine all predictions by concatenating the dataframes
        grp = pred_df.groupby("model")

        pred_df = pd.DataFrame()
        for name, group in grp:
            for i, row in group.iterrows():
                if i == 0:
                    row_df = row["predictions"]
                else:
                    row_df = pd.concat([row_df, row["predictions"]])
            row_df["model"] = name
            pred_df = pd.concat([pred_df, row_df])

        pred_df = pred_df.reset_index(drop=True)

        #Pivot on model and y_pred keeping y as the same
        pred_df = pred_df.pivot(index=["video_name", "y"], columns="model", values="y_pred")

        # Add error 
        # Calculate the error of each model with respect to y
        models = pred_df.columns
        # Add y to the dataframe
        y = pred_df.index.get_level_values(1)
        for model in models:
            pred_df[f"{model}_error"] = pred_df[model] - y
        pred_df

        return pred_df


def highlight_max(s):
    is_max = s == s.max()
    return ['color: green' if cell else '' for cell in is_max]