import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("AirQualityUCI.csv")

print("Dataset loaded successfully!")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumns in the dataset:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nFirst 5 rows:")
print(df.head())

# Missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Duplicate rows
print("\nNumber of duplicate rows:", df.duplicated().sum())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Numerical columns
numeric_columns = df.select_dtypes(include=np.number).columns
print("\nNumerical columns:")
print(numeric_columns.tolist())

# Replace -200 with NaN
for column in numeric_columns:
    df[column] = df[column].replace(-200, np.nan)

# Boxplots
for column in numeric_columns:
    if df[column].dropna().empty:
        print(f"Skipping boxplot for '{column}' as it contains only null values.")
        continue

    plt.figure(figsize=(7, 4))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column}")
    plt.show()

# Correlation heatmap
correlation = df[numeric_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap of Environmental Parameters")
plt.show()

# Temperature vs Relative Humidity
plt.figure(figsize=(7, 5))
plt.scatter(
    df["T"],
    df["RH"],
    alpha=0.5
)
plt.xlabel("Temperature")
plt.ylabel("Relative Humidity")
plt.title("Temperature vs Relative Humidity")
plt.show()

# EDA Summary
print("\nEDA SUMMARY")
print("-" * 40)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Duplicate rows:", df.duplicated().sum())
print("Total missing values:", df.isnull().sum().sum())
print("Numerical columns:", len(numeric_columns))
