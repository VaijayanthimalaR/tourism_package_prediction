

# IMPORTS

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi


HF_TOKEN = "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"
api = HfApi(token=HF_TOKEN)


# LOAD DATA

DATASET_PATH = "hf://datasets/vaijayanthimala07/tourism-package-prediction/tourism.csv"

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")


# CLEANING


# Remove unwanted columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)

# Fix Gender issue
def clean_gender(x):
    x = str(x).replace(' ', '').lower()
    if x in ['female', 'f']:
        return 'Female'
    elif x in ['male', 'm']:
        return 'Male'
    else:
        return 'Other'

df['Gender'] = df['Gender'].apply(clean_gender)

# Handle missing values
for col in df.select_dtypes(include='object'):
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in df.select_dtypes(include='number'):
    df[col].fillna(df[col].median(), inplace=True)

print(" Data cleaning completed.")


# ENCODING

categorical_cols = [
    'TypeofContact','Occupation','Gender',
    'ProductPitched','MaritalStatus','Designation'
]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save encoders
joblib.dump(encoders, "encoders.pkl")

print(" Encoding completed.")


# SPLIT

target_col = 'ProdTaken'

X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(" Train-test split completed.")


# SAVE FILES

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Files saved locally.")


# UPLOAD TO HUGGING FACE

repo_id = "vaijayanthimala07/tourism-package-prediction"

files = [
    "Xtrain.csv","Xtest.csv",
    "ytrain.csv","ytest.csv",
    "encoders.pkl"
]

for file_path in files:
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    repo_path = f"processed/v1/{file_path}"

    if repo_path in repo_files:
        print(f" {file_path} already exists, skipping upload")
        continue

    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Added {file_path}"
    )

print(" Upload completed successfully.")
