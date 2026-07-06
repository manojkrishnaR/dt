# ==========================================================
# HEART DISEASE DATASET
# CRISP-DM FRAMEWORK IMPLEMENTATION
# ==========================================================

# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================================
# 2. BUSINESS UNDERSTANDING
# ==========================================================

print("="*60)
print("BUSINESS UNDERSTANDING")
print("="*60)

print("""
Goal:
Predict whether a patient has heart disease
based on medical attributes.
""")

# ==========================================================
# 3. DATA UNDERSTANDING
# ==========================================================

df = pd.read_csv(r"D:\heart.csv")

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

plt.figure(figsize=(7,5))
sns.histplot(df["age"], bins=10)
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="sex", data=df)
plt.title("Gender Distribution")
plt.show()

plt.figure(figsize=(7,5))
sns.countplot(x="cp", data=df)
plt.title("Chest Pain Type")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="target", data=df)
plt.title("Heart Disease Distribution")
plt.show()

# ==========================================================
# 4. DATA PREPARATION
# ==========================================================

print("\n"+"="*60)
print("DATA PREPARATION")
print("="*60)

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Handle missing values
for col in df.columns:
    df[col] = df[col].fillna(df[col].median())

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# Separate Features and Target

X = df.drop("target", axis=1)

y = df["target"]

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

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

# ==========================================================
# 6. EVALUATION
# ==========================================================

print("\n"+"="*60)
print("EVALUATION")
print("="*60)

accuracy = accuracy_score(y_test, prediction)

print("Accuracy :", accuracy)

print("\nClassification Report")
print(classification_report(y_test, prediction))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, prediction))

# ==========================================================
# Feature Importance
# ==========================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(importance)

plt.figure(figsize=(10,6))
sns.barplot(data=importance,
            x="Importance",
            y="Feature")
plt.title("Feature Importance")
plt.show()

# ==========================================================
# Correlation Heatmap
# ==========================================================

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),
            annot=True,
            cmap="coolwarm")

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

if result[0] == 1:
    print("Prediction : Heart Disease Present")
else:
    print("Prediction : No Heart Disease")

print("\nModel is ready for deployment.")
print("CRISP-DM Process Completed Successfully.")
