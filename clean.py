import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Load Dataset
# -----------------------------
file_path = r"E:\New folder (3)\OneDrive\Desktop/Titanic-Dataset.csv"   # Keep CSV in same folder
df = pd.read_csv(r"E:\New folder (3)\OneDrive\Desktop/Titanic-Dataset.csv")

print("Initial Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# -----------------------------
# 2. Handle Missing Values
# -----------------------------
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin (too many nulls)
if 'Cabin' in df.columns:
    df.drop(columns=['Cabin'], inplace=True)

# -----------------------------
# 3. Drop Irrelevant Columns
# -----------------------------
drop_cols = ['PassengerId', 'Name', 'Ticket']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# -----------------------------
# 4. Encode Categorical Variables
# -----------------------------
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# -----------------------------
# 5. Remove Outliers (IQR Method)
# -----------------------------
# Select only numeric columns
numeric_df = df.select_dtypes(include=np.number)

Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1

df = df[~((numeric_df < (Q1 - 1.5 * IQR)) | 
          (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]

# -----------------------------
# 6. Feature Scaling
# -----------------------------
scaler = StandardScaler()

numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# -----------------------------
# 7. Save Cleaned Dataset
# -----------------------------
df.to_csv("Titanic_Cleaned.csv", index=False)

print("\nFinal Shape:", df.shape)
print("\nData Cleaning Completed Successfully ✅")
