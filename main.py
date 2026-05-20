# =============================================================================
# IMDB Movie Rating Predictor
#
# Phases:
#   Phase 1 : Data Pipeline       — load, EDA, visualise
#   Phase 2 : Architecture        — define features & pipeline structure
#   Phase 3 : Model Training      — boosting-stage sweep, pick best baseline
#   Phase 4 : Model Optimization  — GridSearchCV hyperparameter tuning
#   Phase 5 : Final Evaluation    — regression + classification metrics & plots
# =============================================================================

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use("Agg")            # remove this line if you want interactive pop-up windows
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection    import train_test_split, GridSearchCV
from sklearn.compose            import ColumnTransformer
from sklearn.preprocessing      import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble           import GradientBoostingRegressor
from sklearn.pipeline           import Pipeline
from sklearn.metrics            import (mean_squared_error, mean_absolute_error, r2_score,
                                        accuracy_score, precision_score, recall_score,
                                        f1_score, confusion_matrix)
from transformers import pipeline

warnings.filterwarnings("ignore")

# Create directory to save plotted graphs
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

print("===============================================")
print("  IMDB MOVIE RATING PREDICTOR  —  Full Pipeline")
print("===============================================")


# ── Locate CSV ────────────────────────────────────────────────────────────────
csv_path = None


print(f"\n  Dataset : {csv_path}")
print(f"  Plots   : {os.path.abspath(PLOTS_DIR)}/")


# =============================================================================
# PHASE 1 — DATA PIPELINE
# =============================================================================
print("===============================================")
print("PHASE 1 : DATA PREPROCESSING")
print("===============================================")


# 1.1  Load 

df = pd.read_csv("IMDB_Movie_Dataset.csv")
df.columns = df.columns.str.strip()

print("Load dataset")
print(f"  Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
print("\n  Preview (first 3 rows):")
print(df.head(3).to_string())


# 1.2  EDA 
print("\nExploratory Data Analysis")

print("\n  Data types:")                #check data type for each column
print(df.dtypes.to_string())

print("\n  Missing values per column:") #check missing values for each column
print(df.isnull().sum().to_string())

print("\n  Summary:")                   #print out summary 
print(df.describe().to_string())
print("\n")


# 1.3  Plot Rating distribution histogram 
print(" Plotting rating distribution ...")

plt.figure()
df["IMDB_Rating"].hist(bins=10, color="steelblue", edgecolor="white")
plt.title("Distribution of IMDB Ratings")
plt.xlabel("IMDB Ratings")
plt.ylabel("Count")
plt.savefig(os.path.join(PLOTS_DIR, "rating_distribution.png"), bbox_inches="tight", dpi=120)
plt.close()
print(f"  [saved, open in -> {PLOTS_DIR}/rating_distribution.png]")


# 1.4  Plot Correlation heatmap 
print("\nPlotting correlation heatmap ...")

df_eda = df.copy()
le = LabelEncoder()
for col in ["Genre", "Certificate"]:
    if col in df_eda.columns:
        df_eda[col] = le.fit_transform(df_eda[col].astype(str))

heat_cols = [c for c in ["IMDB_Rating", "Meta_score", "No_of_Votes", "Gross"] if c in df_eda.columns]
plt.figure(figsize=(6, 5))
sns.heatmap(df_eda[heat_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), bbox_inches="tight", dpi=120)
plt.close()
print(f"  [saved, open in -> {PLOTS_DIR}/correlation_heatmap.png]")
print("\n")
print("MILESTONE 1 done; dataset cleaned, loaded and explored.")


# ====================================================
# PHASE 2 — MODEL ARCHITECTURE
# ====================================================
print("===============================================")
print("PHASE 2 : MODEL ARCHITECTURE ")
print("===============================================")


# 2.1  define feature groups 
numeric_features     = ["Released_Year", "Runtime_min", "Meta_score", "No_of_Votes", "Gross"]
categorical_features = ["Certificate", "Genre"]
target_variable      = "IMDB_Rating"

print("\n")
print(f"  Numeric features    : {numeric_features}")
print(f"  Categorical features: {categorical_features}")
print(f"  Target variable     : {target_variable}")
print(f"  Total input features: {len(numeric_features) + len(categorical_features)}")
print(f"  Model               : GradientBoostingRegressor")
print(f"  Task type           : Regression — predict continuous rating 1.0 to 10.0")


# 2.2  preprocessing pipeline structure 
try:
    _ohe_preview = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    _ohe_preview = OneHotEncoder(handle_unknown="ignore", sparse=False)

_preprocessor_preview = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", _ohe_preview,categorical_features),
])

pipeline = Pipeline(steps=[
    ("preprocessor", _preprocessor_preview),
    ("model",        GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,max_depth=3,random_state=42)),
])

print("\n  Full ML Pipeline:")
print(f"    {pipeline}")
print("\n")
print("MILESTONE 2 done; model chosen : Gradient Boosting Regressor")


# ==================================================== 
# PHASE 3 — MODEL TRAINING
# ====================================================
print("\n")
print("===============================================")
print("PHASE 3 : MODEL TRAINING")
print("===============================================")


# 3.1  Clean & convert data type

print("Cleaning and converting data types...")

df["Released_Year"] = pd.to_numeric(df["Released_Year"], errors="coerce")
df["Runtime_min"]   = df["Runtime"].astype(str).str.extract(r"(\d+)").astype(float)
df["Gross"]         = pd.to_numeric(df["Gross"].astype(str).str.replace(",", "", regex=False),
                                    errors="coerce")
df["No_of_Votes"]   = pd.to_numeric(df["No_of_Votes"].astype(str).str.replace(",", "", regex=False),
                                    errors="coerce")
df["Meta_score"]    = pd.to_numeric(df["Meta_score"],  errors="coerce")
df["IMDB_Rating"]   = pd.to_numeric(df["IMDB_Rating"], errors="coerce")

required_cols = numeric_features + categorical_features + [target_variable]
df = df.dropna(subset=required_cols)

print(f"  Dataset shape after cleaning and converting: {df.shape}")


# 3.2  Define input features and target variables
X     = df[numeric_features + categorical_features]
y     = df[target_variable]

y_all = y.copy()   # this is for Phase 5 calculation later on


# 3.3  Train/Validation/Test split (70%/15%/15%)
print("\n Splitting dataset ...")

X_train_full, X_test,  y_train_full, y_test  = train_test_split( 
    X, y, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1765, random_state=42   
)

print("  Training set shape:", X_train.shape)
print("  Validation set shape:", X_val.shape)
print("  Testing set shape:", X_test.shape)

# 3.4  Preprocessing 
print("\n")
print("Preprocessing on training data ...")

# OneHotEncoder syntax based on sklearn data
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", ohe,              categorical_features),
])

X_train_processed = preprocessor.fit_transform(X_train) #preprocessing only fit on training data 
X_val_processed   = preprocessor.transform(X_val)       
X_test_processed  = preprocessor.transform(X_test)

print("  Processed training shape:", X_train_processed.shape)
print("  Processed validation shape:", X_val_processed.shape)
print("  Processed testing shape:", X_test_processed.shape)


# 3.5  Gradient Boosting Training 
print("\nTraining data with Gradient Boosting ...")

estimator_values = [10, 20, 50, 100, 150, 200, 300]
history = []

for n_estimators in estimator_values:
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=0.05, max_depth=3, random_state=42)
    
    # Train model
    model.fit(X_train_processed, y_train)   

    # Predict training and validation values
    train_pred  = model.predict(X_train_processed)
    val_pred = model.predict(X_val_processed)

 #!!!!!!!!!!!! CHECK HERE !!!!!!!!!!!!!!

    # Calculate training metrics
    train_rmse = mean_squared_error(y_train, train_pred) ** 0.5
    train_mae = mean_absolute_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)
    
    # Calculate validation metrics
    val_rmse = mean_squared_error(y_val, val_pred) ** 0.5
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    # Store result
    history.append({
        "Boosting Stages": n_estimators,
        "Train RMSE": train_rmse,
        "Train MAE": train_mae,
        "Train R2": train_r2,
        "Validation RMSE": val_rmse,
        "Validation MAE": val_mae,
        "Validation R2": val_r2
    })

    print(
        "Boosting stages:", n_estimators,
        "| Train RMSE:", round(train_rmse, 4),
        "| Validation RMSE:", round(val_rmse, 4),
        "| Validation MAE:", round(val_mae, 4),
        "| Validation R2:", round(val_r2, 4)
    )

history_df = pd.DataFrame(history)

# 3.6  Find best model based on lowest validation RMSE

best_row   = history_df.loc[history_df["Validation RMSE"].idxmin()] #idxmin
best_n_estimators     = int(best_row["Boosting Stages"])

print(f"\n  Best boosting stages (lowest Val RMSE): {best_n_estimators}")
print(f"  Best Validation RMSE : {best_row['Validation RMSE']:.4f}")
print(f"  Best Validation MAE  : {best_row['Validation MAE']:.4f}")
print(f"  Best Validation R2   : {best_row['Validation R2']:.4f}")


# 3.7  Plot training progress (RMSE, MAE, R2 vs boosting stages) 
print("\n Plotting training progress ...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history_df["Boosting Stages"], history_df["Train RMSE"],
             marker="o", label="Train")
axes[0].plot(history_df["Boosting Stages"], history_df["Validation RMSE"],
             marker="o", label="Validation")
axes[0].set_title("RMSE vs Boosting Stages")
axes[0].set_xlabel("Boosting Stages")
axes[0].set_ylabel("RMSE")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history_df["Boosting Stages"], history_df["Validation MAE"],
             marker="o", color="orange")
axes[1].set_title("Validation MAE vs Boosting Stages")
axes[1].set_xlabel("Boosting Stages")
axes[1].set_ylabel("MAE")
axes[1].grid(True)

axes[2].plot(history_df["Boosting Stages"], history_df["Validation R2"],
             marker="o", color="green")
axes[2].set_title("Validation R2 vs Boosting Stages")
axes[2].set_xlabel("Boosting Stages")
axes[2].set_ylabel("R2")
axes[2].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "3_training_progress.png"), bbox_inches="tight", dpi=120)
plt.close()
print(f"  [saved, open in -> {PLOTS_DIR}/3_training_progress.png]")


# 3.7  Baseline test evaluation usingv best number of boosting stage

print("\n")
print("Evaluating best baseline model on test set ...")

baseline_model = GradientBoostingRegressor(
    n_estimators=best_n_estimators, learning_rate=0.05, max_depth=3, random_state=42
)
baseline_model.fit(X_train_processed, y_train)
baseline_test_pred = baseline_model.predict(X_test_processed)

print(f"  Test RMSE : {(mean_squared_error(y_test, baseline_test_pred) ** 0.5):.4f}")
print(f"  Test MAE  : {mean_absolute_error(y_test, baseline_test_pred):.4f}")
print(f"  Test R2   : {r2_score(y_test, baseline_test_pred):.4f}")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, baseline_test_pred, alpha=0.5)
plt.xlabel("Actual IMDB Rating")
plt.ylabel("Predicted IMDB Rating")
plt.title("Phase 3 — Actual vs Predicted")
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, "3_actual_vs_predicted.png"), bbox_inches="tight", dpi=120)
plt.close()
print(f"  [saved, please  -> {PLOTS_DIR}/3_actual_vs_predicted.png]")

print("\n") 
print("MILESTONE 3 complete; best baseline model found.")


# =====================================================
# PHASE 4 — MODEL OPTIMIZATION  
# =====================================================
print("\n")
print("===============================================")
print("PHASE 4 : MODEL OPTIMIZATION  ")
print("===============================================")


# 4.1  Find best hyperparameters for Gradient Boosting Regressor
param_grid = {
    "n_estimators"    : [100, 200, 300, 500],
    "learning_rate"   : [0.01, 0.05, 0.1],
    "max_depth"       : [2, 3, 4],              # shallower trees = less overfitting
    "min_samples_leaf": [5, 10, 20],            # each leaf must cover N movies
    "subsample"       : [0.7, 0.8, 1.0],        # row sampling per tree
    "max_features"    : ["sqrt", "log2"],       # feature sampling per tree
}

# number of combinations searched by the GridSearchCV
combination_num = (len(param_grid["n_estimators"]) * len(param_grid["learning_rate"])  *
                len(param_grid["max_depth"]) * len(param_grid["min_samples_leaf"]) *
                len(param_grid["subsample"]) * len(param_grid["max_features"]))

print("\n")
print("  (this might take a few minutes, please wait :)")


# 4.2  Run GridSearchCV 
grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    refit=True,
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_train_processed, y_train)

best_params = grid_search.best_params_
best_model  = grid_search.best_estimator_

print(f"\n  Best CV RMSE : {-grid_search.best_score_:.4f}")
print(f"  Best params  : {best_params}")


# 4.3  Validation metrics 
val_pred = best_model.predict(X_val_processed)
val_rmse = mean_squared_error(y_val, val_pred) ** 0.5
val_mae  = mean_absolute_error(y_val, val_pred)
val_r2   = r2_score(y_val, val_pred)

print(f"\n  Validation RMSE : {val_rmse:.4f}")
print(f"  Validation MAE  : {val_mae:.4f}")
print(f"  Validation R2   : {val_r2:.4f}")


# 4.4  Test metrics 
opt_test_pred = best_model.predict(X_test_processed)
opt_test_rmse = mean_squared_error(y_test, opt_test_pred) ** 0.5
opt_test_mae  = mean_absolute_error(y_test, opt_test_pred)
opt_test_r2   = r2_score(y_test, opt_test_pred)

print(f"\n  Test RMSE : {opt_test_rmse:.4f}")
print(f"  Test MAE  : {opt_test_mae:.4f}")
print(f"  Test R2   : {opt_test_r2:.4f}")

print(f"\n  Optimized Model Summary")
print(f"  {'─' * 35}")
for k, v in best_params.items():
    print(f"  {k:<22}: {v}")

print("\n")
print("MILESTONE 4 complete; data trained based on optimized hyperparameters.")


# =============================================================================
# PHASE 5 — FINAL EVALUATION
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 5 : FINAL EVALUATION")
print("=" * 60)


# ── 5.1  Final predictions ────────────────────────────────────────────────────
y_pred_test = best_model.predict(X_test_processed)


# ── 5.2  Regression metrics ───────────────────────────────────────────────────
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae  = mean_absolute_error(y_test, y_pred_test)
r2   = r2_score(y_test, y_pred_test)

print(f"\n  REGRESSION METRICS")
print(f"  {'─' * 30}")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  R2   : {r2:.4f}")


# ── 5.3  Classification metrics (data-driven threshold) ──────────────────────
threshold  = y_pred_test.mean()
y_test_bin = (y_test >= y_all.mean()).astype(int)
y_pred_bin = (y_pred_test >= threshold).astype(int)

accuracy  = accuracy_score(y_test_bin,  y_pred_bin)
precision = precision_score(y_test_bin, y_pred_bin, zero_division=0)
recall    = recall_score(y_test_bin,    y_pred_bin, zero_division=0)
f1        = f1_score(y_test_bin,        y_pred_bin, zero_division=0)

print(f"\n  CLASSIFICATION METRICS  (threshold = {threshold:.2f})")
print(f"  {'─' * 30}")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")


# ── 5.4  4-panel evaluation plot ─────────────────────────────────────────────
print("\n  Generating evaluation plots ...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Phase 5 — Final Model Evaluation", fontsize=14, fontweight="bold")

# (a) Actual vs Predicted
sns.regplot(x=y_test, y=y_pred_test, ax=axes[0, 0], scatter_kws={"alpha": 0.4})
axes[0, 0].set_title("Actual vs Predicted IMDB Rating")
axes[0, 0].set_xlabel("Actual")
axes[0, 0].set_ylabel("Predicted")

# (b) Classification metrics bar chart
_metrics = ["Accuracy", "Precision", "Recall", "F1"]
_scores  = [accuracy, precision, recall, f1]
sns.barplot(x=_metrics, y=_scores, ax=axes[0, 1], palette="Blues_d")
for i, v in enumerate(_scores):
    axes[0, 1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
axes[0, 1].set_ylim(0, 1.1)
axes[0, 1].set_title("Classification Metrics")

# (c) Confusion matrix
cm = confusion_matrix(y_test_bin, y_pred_bin)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0],
            xticklabels=["Below avg", "Above avg"],
            yticklabels=["Below avg", "Above avg"])
axes[1, 0].set_title("Confusion Matrix")
axes[1, 0].set_xlabel("Predicted")
axes[1, 0].set_ylabel("Actual")

# (d) Residual distribution
errors = y_test.values - y_pred_test
sns.histplot(errors, bins=20, kde=True, ax=axes[1, 1], color="steelblue")
axes[1, 1].axvline(0, color="red", linestyle="--", linewidth=1.5)
axes[1, 1].set_title("Residual Distribution")
axes[1, 1].set_xlabel("Prediction Error")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "phase5_evaluation.png"), bbox_inches="tight", dpi=120)
plt.close()
print(f"  [saved -> {PLOTS_DIR}/phase5_evaluation.png]")


# ── 5.5  Sample predictions & error summary ──────────────────────────────────
sample_df = pd.DataFrame({
    "Actual"    : y_test.values[:10],
    "Predicted" : y_pred_test[:10].round(2),
    "Error"     : (y_test.values[:10] - y_pred_test[:10]).round(3),
})
print("\n  Sample Predictions (first 10):")
print(sample_df.to_string(index=False))

large_errors = np.sum(np.abs(errors) > 1.0)
print(f"\n  Large errors (|error| > 1.0) : {large_errors} / {len(errors)} samples")
print("\nPhase 5 complete — evaluation done.")


# =============================================================================
# PIPELINE COMPLETE
# =============================================================================
print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)
print("\n  Plots saved:")
for fname in sorted(os.listdir(PLOTS_DIR)):
    print(f"    {PLOTS_DIR}/{fname}")
print()
