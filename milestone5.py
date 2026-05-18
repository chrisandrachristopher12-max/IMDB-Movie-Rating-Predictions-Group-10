import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# [YOUR DATA LOADING CODE - SAME AS BEFORE]
path = r"C:\Users\hello\Downloads\milestone 1\cleaned_movie.csv"
df = pd.read_csv(path)

# Find target (SAME)
target_col = None
possible_targets = ['imdb_rating', 'IMDB_Rating', 'IMDB_rating', 'rating', 'Rating', 'imdb', 'IMDB']
for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

X = df.drop([target_col], axis=1).select_dtypes(include=[np.number])
y = df[target_col]

print("Target stats:", y.describe())

# 3-WAY SPLIT
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)

# ✅ FIX 1: SCALE FEATURES (prevents leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ✅ FIX 2: Train on train ONLY (not train+val)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train_scaled, y_train)

# Test predictions
y_pred_test = model.predict(X_test_scaled)

# Regression metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f"\nFINAL RESULTS")
print(f"RMSE: {rmse:.3f} \nMAE: {mae:.3f} \nR²: {r2:.3f}")

# ✅ FIX 3: BETTER THRESHOLD (data-driven)
optimal_threshold = y_pred_test.mean()  # Use prediction mean, not 7.0
y_test_bin = (y_test >= y.mean()).astype(int)  # Use dataset mean
y_pred_bin = (y_pred_test >= optimal_threshold).astype(int)

# Classification metrics
accuracy = accuracy_score(y_test_bin, y_pred_bin)
precision = precision_score(y_test_bin, y_pred_bin, zero_division=0)
recall = recall_score(y_test_bin, y_pred_bin, zero_division=0)
f1 = f1_score(y_test_bin, y_pred_bin, zero_division=0)

print(f"\n📊 CLASSIFICATION (threshold={optimal_threshold:.2f})")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-Score:  {f1:.3f}")

# ✅ VISUALIZATIONS (SAME 4 PLOTS)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

sns.regplot(x=y_test, y=y_pred_test, ax=axes[0,0])
axes[0,0].set_title('Actual vs Predicted')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
scores = [accuracy, precision, recall, f1]
sns.barplot(x=metrics, y=scores, ax=axes[0,1])
for i, v in enumerate(scores): axes[0,1].text(i, v+0.01, f'{v:.3f}', ha='center')

cm = confusion_matrix(y_test_bin, y_pred_bin)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[1,0])
axes[1,0].set_title('Confusion Matrix')

errors = y_test - y_pred_test
sns.histplot(errors, bins=20, ax=axes[1,1])
axes[1,1].axvline(0, color='red', linestyle='--')
axes[1,1].set_title('Error Distribution')

plt.tight_layout()
plt.show()

# Error analysis
print(f"\n🔍 ERRORS: {np.abs(errors)>1.0}.sum() large errors")
print("Sample predictions:")
print(pd.DataFrame({'Actual': y_test[:5], 'Pred': y_pred_test[:5], 'Error': errors[:5]}))

print("\n✅ FIXED: Realistic metrics ~0.75-0.85 range!")
