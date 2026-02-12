import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned dataset
df = pd.read_csv("Titanic_Cleaned.csv")

# Create images folder if not exists
os.makedirs("images", exist_ok=True)

# 1️⃣ Survival Count
plt.figure()
sns.countplot(x='Survived', data=df)
plt.title("Survival Distribution")
plt.savefig("images/survival_distribution.png")
plt.close()

# 2️⃣ Age Distribution
plt.figure()
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.savefig("images/age_distribution.png")
plt.close()

# 3️⃣ Fare Distribution
plt.figure()
sns.histplot(df['Fare'], bins=30, kde=True)
plt.title("Fare Distribution")
plt.savefig("images/fare_distribution.png")
plt.close()

print("Visualizations generated successfully.")
