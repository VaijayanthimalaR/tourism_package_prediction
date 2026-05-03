

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

DATASET_PATH = "hf://datasets/vaijayanthimala07/tourism-package-predict/tourism.csv"

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
    df[col] = df[col].fillna(df[col].mode()[0])

for col in df.select_dtypes(include='number'):
    df[col] = df[col].fillna(df[col].median())

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

repo_id = "vaijayanthimala07/tourism-package-predict"

files = [
    "Xtrain.csv","Xtest.csv",
    "ytrain.csv","ytest.csv",
    "encoders.pkl"
]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename

        repo_id="vaijayanthimala07/tourism-package-predict",

        repo_type="dataset",
    )
print(" Upload completed successfully.")
