"""
SafeSphere AI - Flood Risk Model Training
Loads dataset -> preprocess -> train/test split -> train RandomForest ->
evaluate -> save model. This is the real ML pipeline judges can inspect.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# 1. Load dataset
df = pd.read_csv("flood_dataset.csv")

FEATURES = ["rainfall", "water_level", "flow_rate", "temperature",
            "humidity", "elevation", "historical_flood_flag"]
TARGET = "risk_level"

X = df[FEATURES]
y = df[TARGET]

# 2. Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # HIGH/LOW/MODERATE -> 0/1/2 alphabetically

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 4. Train model
model = RandomForestClassifier(
    n_estimators=150, max_depth=8, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

# 5. Predict + evaluate
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted")
rec = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")
cm = confusion_matrix(y_test, y_pred)

print("=== SafeSphere AI - Flood Risk Model (prototype) ===")
print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1-score:  {f1:.3f}")
print("Confusion Matrix (rows=actual, cols=predicted):")
print(cm)
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nFeature importances:")
for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:25s} {imp:.3f}")

# 6. Save model + label encoder + feature list
joblib.dump({
    "model": model,
    "label_encoder": le,
    "features": FEATURES,
    "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1},
}, "flood_model.pkl")

print("\nSaved -> flood_model.pkl")
print("NOTE: Trained on synthetic prototype data. Not a production-accuracy model.")
