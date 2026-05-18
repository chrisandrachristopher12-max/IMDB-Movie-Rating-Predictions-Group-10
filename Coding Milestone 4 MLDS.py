import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load and prepare data

df = pd.read_csv("IMDB_Movie_Dataset.csv")

# Convert and clean data
df["Released_Year"] = pd.to_numeric(df["Released_Year"], errors="coerce")
df["Runtime_min"] = df["Runtime"].astype(str).str.extract(r"(\d+)").astype(float)
df["Gross"] = pd.to_numeric(df["Gross"].astype(str).str.replace(",", "", regex=False), errors="coerce")
df["No_of_Votes"] = pd.to_numeric(df["No_of_Votes"].astype(str).str.replace(",", "", regex=False), errors="coerce")
df["Meta_score"] = pd.to_numeric(df["Meta_score"], errors="coerce")
df["IMDB_Rating"] = pd.to_numeric(df["IMDB_Rating"], errors="coerce")

# Drop missing values
df = df.dropna(subset=["Released_Year", "Runtime_min", "Certificate", "Genre", 
                       "Meta_score", "No_of_Votes", "Gross", "IMDB_Rating"])

# 2. Define features and target

X = df[["Released_Year", "Runtime_min", "Certificate", "Genre", "Meta_score", "No_of_Votes", "Gross"]]
y = df["IMDB_Rating"]

# 3. Train-test split

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1765, random_state=42)

# 4. Preprocessing

numeric_features = ["Released_Year", "Runtime_min", "Meta_score", "No_of_Votes", "Gross"]
categorical_features = ["Certificate", "Genre"]

try:
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", one_hot_encoder, categorical_features)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# 5. GridSearchCV to find best hyperparameters

param_grid = {
    "n_estimators"    : [100, 200, 300, 500],
    "learning_rate"   : [0.01, 0.05, 0.1],
    "max_depth"       : [2, 3, 4],        # shallower trees = less overfit
    "min_samples_leaf": [5, 10, 20],      # each leaf must cover N movies
    "subsample"       : [0.7, 0.8, 1.0], # row sampling per tree
    "max_features"    : ["sqrt", "log2"], # feature sampling per tree
}

grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    refit=True,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_processed, y_train)

best_params = grid_search.best_params_
print("Best CV RMSE :", round(-grid_search.best_score_, 4))
print("Best params  :", best_params)

# 6. Validate best model on validation set

best_model = grid_search.best_estimator_  # already fitted on full X_train

val_pred = best_model.predict(X_val_processed)
val_rmse = mean_squared_error(y_val, val_pred) ** 0.5
val_mae  = mean_absolute_error(y_val, val_pred)
val_r2   = r2_score(y_val, val_pred)

print(f"\nValidation RMSE : {val_rmse:.4f}")
print(f"Validation MAE  : {val_mae:.4f}")
print(f"Validation R2   : {val_r2:.4f}")

# 7. Final evaluation on test set

test_pred = best_model.predict(X_test_processed)
test_rmse = mean_squared_error(y_test, test_pred) ** 0.5
test_mae  = mean_absolute_error(y_test, test_pred)
test_r2   = r2_score(y_test, test_pred)

print(f"\nTest RMSE : {test_rmse:.4f}")
print(f"Test MAE  : {test_mae:.4f}")
print(f"Test R2   : {test_r2:.4f}")

# 8. Print final summary

print("\n--- Optimized Model Summary ---")
print(f"Best n_estimators    : {best_params['n_estimators']}")
print(f"Best learning_rate   : {best_params['learning_rate']}")
print(f"Best max_depth       : {best_params['max_depth']}")
print(f"Best min_samples_leaf: {best_params['min_samples_leaf']}")
print(f"Best subsample       : {best_params['subsample']}")
print(f"Best max_features    : {best_params['max_features']}")
