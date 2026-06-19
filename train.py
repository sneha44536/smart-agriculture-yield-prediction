import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/crop_yield.csv")

print("Dataset Loaded Successfully!")

print("\nShape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

le_crop = LabelEncoder()
le_season = LabelEncoder()
le_state = LabelEncoder()

df["crop"] = le_crop.fit_transform(df["crop"])
df["season"] = le_season.fit_transform(df["season"])
df["state"] = le_state.fit_transform(df["state"])

# Remove production to avoid data leakage
X = df.drop(["yield", "production"], axis=1)
y = df["yield"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("-" * 40)
print("MAE :", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

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

# Yield Distribution
plt.figure(figsize=(8, 5))
plt.hist(df["yield"], bins=30)
plt.title("Yield Distribution")
plt.xlabel("Yield")
plt.ylabel("Frequency")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

# Feature Importance Plot
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
plt.show()