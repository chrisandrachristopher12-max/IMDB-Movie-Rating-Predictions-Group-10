# MILESTONE 1: DATA PIPELINE

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# STEP 1: LOAD CLEAN DATA

df = pd.read_csv("IMDB_Movie_Dataset.csv")

print("=== DATASET PREVIEW ===")
print(df.head())

print("\nShape:", df.shape)


# STEP 2: EDA

print("\n=== DATA INFO ===")
print(df.info())

print("\n=== CHECK MISSING VALUES ===")
print(df.isnull().sum())

# Histogram
if 'IMDB_Rating' in df.columns:
    plt.figure()
    df['IMDB_Rating'].hist(bins=10)
    plt.title("Distribution of IMDb Ratings")
    plt.show()


# STEP 3: PREPROCESSING

le = LabelEncoder()

for col in ['Genre', 'Certificate']:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

print("\nCategorical data encoded")


# STEP 4: VISUALIZATION

cols = ['IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross']
cols = [c for c in cols if c in df.columns]

plt.figure(figsize=(6,5))
sns.heatmap(df[cols].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()


print("\nMilestone 1 Completed!")
print("Dataset is clean and ready for machine learning.")
