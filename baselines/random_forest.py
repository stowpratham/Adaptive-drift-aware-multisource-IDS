import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score,
    ConfusionMatrixDisplay,
)

os.makedirs("results/baselines", exist_ok=True)

print("Loading datasets...")

X_train = joblib.load("baselines/data/X_train_bal.pkl")
y_train = joblib.load("baselines/data/y_train_bal.pkl")

X_test = joblib.load("baselines/data/X_drift_stream.pkl")
y_test = joblib.load("baselines/data/y_drift_stream.pkl")

print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)

rf.fit(X_train, y_train)

print("Evaluating...")

y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

far = fp / (fp + tn)
dr = tp / (tp + fn)

results = {
    "Model": "Random Forest",
    "Accuracy": round(acc, 4),
    "Precision": round(prec, 4),
    "Recall": round(rec, 4),
    "F1": round(f1, 4),
    "Kappa": round(kappa, 4),
    "FAR": round(far, 4),
    "Detection Rate": round(dr, 4),
}

print(results)

with open(
    "results/baselines/random_forest_results.json",
    "w",
) as f:
    json.dump(results, f, indent=4)

disp = ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred)
)

disp.plot()

plt.title("Random Forest Confusion Matrix")

plt.savefig(
    "results/baselines/rf_confusion_matrix.png"
)

plt.close()

print("✓ Random Forest baseline complete")