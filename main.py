import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score
import joblib

# load dataset
url = 'https://raw.githubusercontent.com/funnyPhani/HouseData/main/kc_house_data.csv'
df = pd.read_csv(url)
print(df.head())

# config 
TARGET_COL = 'price'
RANDOM_SEED = 40
np.random.seed(RANDOM_SEED)


# Basic sanity checks
assert TARGET_COL in df.columns, f"'{TARGET_COL}' not found in columns!"
print("Rows, Cols:", df.shape)
print("Head:\n", df.head(3))
print("\nDtypes:\n", df.dtypes)

# Missing values quick look
na_counts = df.isna().sum().sort_values(ascending=False)
print("\nTop missing columns:\n", na_counts.head(10))

# Parse date: format like '20141013T000000'
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S", errors="coerce")
    df["sale_year"]  = df["date"].dt.year
    df["sale_month"] = df["date"].dt.month
    df["sale_day"]   = df["date"].dt.day

df_fe = df.copy()

# Feature engineering (safe & useful)
if {"yr_built", "sale_year"}.issubset(df_fe.columns):
    df_fe["age_at_sale"] = df_fe["sale_year"] - df_fe["yr_built"]

if "yr_renovated" in df_fe.columns:
    df_fe["renovated_flag"] = (df_fe["yr_renovated"] > 0).astype(int)
    if "sale_year" in df_fe.columns:
        df_fe["years_since_renovation"] = np.where(
            df_fe["yr_renovated"] > 0,
            df_fe["sale_year"] - df_fe["yr_renovated"],
            0
        )

# Columns that usually add no signal / leak nothing but noise
drop_cols = [c for c in ["id", "date"] if c in df_fe.columns]
df_fe = df_fe.drop(columns=drop_cols)


# Decide categorical columns present in your data
categorical_candidates = [col for col in ["zipcode", "waterfront", "view"] if col in X_train.columns]
categorical_cols = categorical_candidates

numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# OneHotEncoder: keep dense output for Keras (we'll densify below if needed)
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False) if "sparse_output" in OneHotEncoder().get_params() \
      else OneHotEncoder(handle_unknown="ignore", sparse=False)

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", ohe)
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols)
    ]
)

# Fit on train only; transform all sets
X_train_proc = preprocessor.fit_transform(X_train)
X_valid_proc = preprocessor.transform(X_valid)
X_test_proc  = preprocessor.transform(X_test)

# Densify if sparse (older sklearn)
if hasattr(X_train_proc, "toarray"):
    X_train_proc = X_train_proc.toarray()
    X_valid_proc = X_valid_proc.toarray()
    X_test_proc  = X_test_proc.toarray()

print("Processed feature dims:", X_train_proc.shape[1])

# Save preprocessor for reuse
joblib.dump(preprocessor, "preprocessor.pkl")






