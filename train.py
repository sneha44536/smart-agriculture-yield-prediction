import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create plots folder automatically
os.makedirs("plots", exist_ok=True)

# Load Dataset
df = pd.read_csv("data/crop_yield.csv")

print("Dataset Loaded Successfully!")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Encode categorical columns
le_crop = LabelEncoder()
le_season = LabelEncoder()
le_state = LabelEncoder()

df["crop"] = le_crop.fit_transform(df["crop"])
df["season"] = le_season.fit_transform(df["season"])
df["state"] = le_state.fit_transform(df["state"])

# Features and Target
# Production removed to avoid data leakage
X = df.drop(["yield", "production"], axis=1)
y = df["yield"]

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Random Forest Model
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

print("\nModel Training Completed!")

# Predictions
y_pred = rf.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("-" * 40)
print("MAE :", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Feature Importance
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print("-" * 40)
print(importance_df)

# ==========================================
# VISUALIZATION 1 - Yield Distribution
# ==========================================
plt.figure(figsize=(8, 5))
plt.hist(df["yield"], bins=30)
plt.title("Distribution of Crop Yield")
plt.xlabel("Yield")
plt.ylabel("Frequency")
plt.savefig("plots/yield_distribution.png")
plt.close()

# ==========================================
# VISUALIZATION 2 - Top 10 Crops
# ==========================================
crop_yield = (
    df.groupby("crop")["yield"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 5))
crop_yield.plot(kind="bar")
plt.title("Top 10 Crops by Average Yield")
plt.xlabel("Crop")
plt.ylabel("Average Yield")
plt.tight_layout()
plt.savefig("plots/top_10_crops.png")
plt.close()

# ==========================================
# VISUALIZATION 3 - Yield Over Years
# ==========================================
yearly_yield = df.groupby("year")["yield"].mean()

plt.figure(figsize=(10, 5))
plt.plot(
    yearly_yield.index,
    yearly_yield.values,
    marker="o"
)
plt.title("Average Yield Over Years")
plt.xlabel("Year")
plt.ylabel("Average Yield")
plt.grid(True)
plt.savefig("plots/yield_over_years.png")
plt.close()

# ==========================================
# VISUALIZATION 4 - Fertilizer vs Yield
# ==========================================
plt.figure(figsize=(8, 5))
plt.scatter(
    df["fertilizer"],
    df["yield"]
)
plt.title("Fertilizer vs Yield")
plt.xlabel("Fertilizer")
plt.ylabel("Yield")
plt.savefig("plots/fertilizer_vs_yield.png")
plt.close()

# ==========================================
# VISUALIZATION 5 - Pesticide vs Yield
# ==========================================
plt.figure(figsize=(8, 5))
plt.scatter(
    df["pesticide"],
    df["yield"]
)
plt.title("Pesticide vs Yield")
plt.xlabel("Pesticide")
plt.ylabel("Yield")
plt.savefig("plots/pesticide_vs_yield.png")
plt.close()

# ==========================================
# VISUALIZATION 6 - Area vs Yield
# ==========================================
plt.figure(figsize=(8, 5))
plt.scatter(
    df["area"],
    df["yield"]
)
plt.title("Area vs Yield")
plt.xlabel("Area")
plt.ylabel("Yield")
plt.savefig("plots/area_vs_yield.png")
plt.close()

# ==========================================
# VISUALIZATION 7 - Correlation Heatmap
# ==========================================
plt.figure(figsize=(10, 6))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.savefig("plots/correlation_heatmap.png")
plt.close()

# ==========================================
# VISUALIZATION 8 - Feature Importance
# ==========================================
plt.figure(figsize=(8, 5))
plt.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/feature_importance.png")
plt.close()

# ==========================================
# VISUALIZATION 9 - Actual vs Predicted
# ==========================================
plt.figure(figsize=(8, 6))
plt.scatter(
    y_test,
    y_pred
)
plt.title("Actual vs Predicted Yield")
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.savefig("plots/actual_vs_predicted.png")
plt.close()

print("\nAll visualizations saved successfully in the 'plots' folder.")