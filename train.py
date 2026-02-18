import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load dataset
df = pd.read_csv("data/health_activity_data.csv")

# 2. Encode target
df["Heart_Disease"] = df["Heart_Disease"].map({"Yes": 1, "No": 0})

# 3. Encode categorical columns
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Smoker"] = df["Smoker"].map({"Yes": 1, "No": 0})
df["Diabetic"] = df["Diabetic"].map({"Yes": 1, "No": 0})

# 4. Feature engineering: split Blood Pressure
bp = df["Blood_Pressure"].str.split("/", expand=True)
df["BP_Systolic"] = bp[0].astype(int)
df["BP_Diastolic"] = bp[1].astype(int)
df.drop("Blood_Pressure", axis=1, inplace=True)

# 5. Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Drop BMI
df.drop("BMI", axis=1, inplace=True)
df = df.drop(columns=["ID"])


# 6. Split features & target
X = df.drop("Heart_Disease", axis=1)
print(X.shape)
print(X.columns)
y = df["Heart_Disease"]

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Train model (Classifier)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 9. Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 10. Save model
joblib.dump(model, "rf_heart_disease_model.pkl")
print("Model saved as heart_disease_model.pkl")

print(model.feature_names_in_)

import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "heart_disease_model.pkl")

model = joblib.load(model_path)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression


model = LogisticRegression(max_iter=1000, class_weight="balanced")

model.fit(X_train, y_train)

import joblib

# Save model
joblib.dump({
    "model": model,
    "columns": X.columns.tolist()
}, "lr_model.pkl")

print("✅ Model saved!")


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))







