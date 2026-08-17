# ==========================================================
# CAR PREDICTION DATASET
# CRISP-DM FRAMEWORK IMPLEMENTATION
# ==========================================================

# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================================================
# 2. BUSINESS UNDERSTANDING
# ==========================================================

print("="*60)
print("BUSINESS UNDERSTANDING")
print("="*60)

print("""
Goal:
Predict the selling price of used cars based on vehicle attributes
such as present price, age, mileage, fuel type, and transmission.
""")

# ==========================================================
# 3. DATA UNDERSTANDING
# ==========================================================

df = pd.read_csv("car_prediction_data.csv")

print("\nFirst Five Records")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

# ==========================================================
# Visualizations
# ==========================================================

# Selling Price Distribution
plt.figure(figsize=(7,5))
sns.histplot(df["Selling_Price"], bins=15, kde=True)
plt.title("Selling Price Distribution")
plt.show()

# Fuel Type Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Fuel_Type", data=df)
plt.title("Fuel Type Distribution")
plt.show()

# Seller Type Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Seller_Type", data=df)
plt.title("Seller Type Distribution")
plt.show()

# Transmission Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Transmission", data=df)
plt.title("Transmission Distribution")
plt.show()

# ==========================================================
# 4. DATA PREPARATION
# ==========================================================

print("\n"+"="*60)
print("DATA PREPARATION")
print("="*60)

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Feature Engineering: Calculate Car Age from Manufacturing Year
df['Car_Age'] = 2026 - df['Year']

# Drop non-predictive columns
df_prep = df.drop(columns=['Year', 'Car_Name'])

# Handle missing values (if any)
for col in df_prep.columns:
    if df_prep[col].dtype in ['float64', 'int64']:
        df_prep[col] = df_prep[col].fillna(df_prep[col].median())

# One-Hot Encoding for Categorical Columns
df_prep = pd.get_dummies(df_prep, drop_first=True)

print("\nMissing Values After Cleaning")
print(df_prep.isnull().sum())

# Separate Features and Target
X = df_prep.drop("Selling_Price", axis=1)
y = df_prep["Selling_Price"]

print("\nFeature Shape :", X.shape)
print("Target Shape :", y.shape)

# ==========================================================
# 5. MODELING
# ==========================================================

print("\n"+"="*60)
print("MODELING")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = DecisionTreeRegressor(random_state=42)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

# ==========================================================
# 6. EVALUATION
# ==========================================================

print("\n"+"="*60)
print("EVALUATION")
print("="*60)

r2 = r2_score(y_test, prediction)
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)

print(f"R2 Score            : {r2:.4f}")
print(f"Mean Absolute Error : {mae:.4f}")
print(f"Mean Squared Error  : {mse:.4f}")

# ==========================================================
# Feature Importance
# ==========================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(importance)

plt.figure(figsize=(10,6))
sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)
plt.title("Feature Importance")
plt.show()

# ==========================================================
# Correlation Heatmap
# ==========================================================

plt.figure(figsize=(10,6))
sns.heatmap(
    df_prep.corr(),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Correlation Heatmap")
plt.show()

# ==========================================================
# 7. DEPLOYMENT
# ==========================================================

print("\n"+"="*60)
print("DEPLOYMENT")
print("="*60)

sample = X.iloc[[0]]

result = model.predict(sample)

print(f"Predicted Selling Price : {result[0]:.2f} Lakhs")

print("\nModel is ready for deployment.")
print("CRISP-DM Process Completed Successfully.")
