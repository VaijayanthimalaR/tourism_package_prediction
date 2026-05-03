

# IMPORTS

import pandas as pd
import joblib
import os
import datetime
import mlflow
import mlflow.sklearn
import xgboost as xgb

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


# MLFLOW SETUP

mlflow.set_experiment("tourism-mlops-experiment")


# AUTH

HF_TOKEN = "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"
api = HfApi(token=HF_TOKEN)


# LOAD DATA

Xtrain = pd.read_csv("hf://datasets/vaijayanthimala07/tourism-package-prediction/processed/v1/Xtrain.csv")
Xtest  = pd.read_csv("hf://datasets/vaijayanthimala07/tourism-package-prediction/processed/v1/Xtest.csv")
ytrain = pd.read_csv("hf://datasets/vaijayanthimala07/tourism-package-prediction/processed/v1/ytrain.csv")
ytest  = pd.read_csv("hf://datasets/vaijayanthimala07/tourism-package-prediction/processed/v1/ytest.csv")

# Fix shape
ytrain = ytrain.values.ravel()
ytest = ytest.values.ravel()

print(" Data loaded")


# FEATURE GROUPS

numeric_features = [
    'Age','CityTier','DurationOfPitch','NumberOfPersonVisiting',
    'NumberOfFollowups','PreferredPropertyStar','NumberOfTrips',
    'Passport','PitchSatisfactionScore','OwnCar',
    'NumberOfChildrenVisiting','MonthlyIncome'
]

categorical_features = [
    'TypeofContact','Occupation','Gender',
    'ProductPitched','MaritalStatus','Designation'
]


# CLASS WEIGHT (SAFE)

counts = pd.Series(ytrain).value_counts()
class_weight = counts[0] / counts[1] if 1 in counts else 1


# PIPELINE

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

pipeline = make_pipeline(preprocessor, model)


# PARAM GRID

param_grid = {
    'xgbclassifier__n_estimators': [50, 75],
    'xgbclassifier__max_depth': [2, 3],
    'xgbclassifier__learning_rate': [0.01, 0.05],
    'xgbclassifier__colsample_bytree': [0.4, 0.5],
}


# TRAIN + TRACKING

with mlflow.start_run(run_name="xgboost_v1"):

    # Tags (important for production)
    mlflow.set_tags({
        "project": "tourism_prediction",
        "model": "XGBoost",
        "stage": "training"
    })

    # Dataset tracking
    mlflow.log_param("dataset_source", "huggingface")
    mlflow.log_param("dataset_version", "v1")

    # Train
    grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    grid.fit(Xtrain, ytrain)

    best_model = grid.best_estimator_

    # Log params
    mlflow.log_params(grid.best_params_)

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "test_accuracy": test_report['accuracy'],
        "test_recall_class1": test_report['1']['recall'],
        "test_f1_class1": test_report['1']['f1-score']
    })


    # MODEL VERSIONING

    version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"model_{version}.joblib"

    joblib.dump(best_model, model_path)

    # Log model
    mlflow.log_artifact(model_path, artifact_path="model")
    mlflow.sklearn.log_model(best_model, "model")

    print(f" Model saved: {model_path}")


# UPLOAD TO HUGGING FACE

repo_id = "vaijayanthimala07/tourism_package_prediction_model"

try:
    api.repo_info(repo_id=repo_id, repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=f"models/{model_path}",
    repo_id=repo_id,
    repo_type="model",
)

print(" Model uploaded to Hugging Face")
