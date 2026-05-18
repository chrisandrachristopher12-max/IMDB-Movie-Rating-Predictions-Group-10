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


# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
print("\n=== DATA INFO ===")
print(df.info())

print("\n=== CHECK MISSING VALUES ===")
print(df.isnull().sum())


# Distribution of IMDb Ratings
if 'IMDB_Rating' in df.columns:
    plt.figure(figsize=(8, 5))
    df['IMDB_Rating'].hist(
        bins=10,
        color='#C2185B',   
        edgecolor='black'  
    )
    plt.title("Distribution of IMDb Ratings")
    plt.xlabel("IMDb Rating")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.show()


# STEP 3: PREPROCESSING
le = LabelEncoder()

for col in ['Genre', 'Certificate']:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

print("\nCategorical data encoded")


# STEP 4: CORRELATION HEATMAP
cols = ['IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross']
cols = [c for c in cols if c in df.columns]

plt.figure(figsize=(6, 5))
sns.heatmap(
    df[cols].corr(),
    annot=True
)
plt.title("Correlation Heatmap")
plt.show()


print("\nMilestone 1 Completed!")
print("Dataset is clean and ready for machine learning.")

