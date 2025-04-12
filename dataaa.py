import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("air_quality_data.csv")

# Basic exploration
print("📄 Data Preview:", df.head())
print("\n📊 Null Values:", df.isnull().sum())
df = df.dropna()  # Remove rows with missing values

# Descriptive stats
print("\n📈 Description:", df.describe())

# Pollutant Frequency
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='pollutant_id', hue='pollutant_id', palette='Set2')
plt.title("Pollutant Frequency")
plt.tight_layout()
plt.show()


# Outlier detection using IQR
Q1, Q3 = df['pollutant_avg'].quantile([0.25, 0.75])
IQR = Q3 - Q1
outliers = df[(df['pollutant_avg'] < Q1 - 1.5 * IQR) | (df['pollutant_avg'] > Q3 + 1.5 * IQR)]
print("\n⚠️ Outliers in pollutant_avg:", outliers[['state', 'city', 'pollutant_id', 'pollutant_avg']])



# Histogram of Pollutant Average
plt.figure(figsize=(8, 5))
sns.histplot(df['pollutant_avg'], bins=15, kde=True, color='skyblue', edgecolor='black')
plt.title("Distribution of Pollutant Average")
plt.tight_layout()
plt.show()

# Lineplot of Pollutant Levels Over Time
df['last_update'] = pd.to_datetime(df['last_update'])
df = df.sort_values('last_update')
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='last_update', y='pollutant_avg', hue='pollutant_id', marker='o')
plt.title("Pollutant Trends Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scatterplot of Pollutant Average vs Latitude
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='latitude', y='pollutant_avg', hue='pollutant_id', palette='viridis')
plt.title("Pollutant Average vs Latitude")
plt.tight_layout()
plt.show()

# Correlation Heatmap
numerical_df = df[['pollutant_min', 'pollutant_max', 'pollutant_avg', 'latitude', 'longitude']]
corr_matrix = numerical_df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
