import os
import json
import joblib
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
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

print("Training XGBoost...")

xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric="logloss",
)

xgb.fit(X_train, y_train)

print("Evaluating...")

y_pred = xgb.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

far = fp / (fp + tn)
dr = tp / (tp + fn)

results = {
    "Model": "XGBoost",
    "Accuracy": round(acc, 4),
    "Precision": round(prec, 4),
    "Recall": round(rec, 4),
    "F1": round(f1, 4),
    "Kappa": round(kappa, 4),
    "FAR": round(far, 4),
    "Detection Rate": round(dr, 4),
}

print("\n===== XGBOOST RESULTS =====")

for key, value in results.items():
    print(f"{key}: {value}")

with open(
    "results/baselines/xgboost_results.json",
    "w",
) as f:
    json.dump(results, f, indent=4)

disp = ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred)
)

disp.plot()

plt.title("XGBoost Confusion Matrix")

plt.savefig(
    "results/baselines/xgb_confusion_matrix.png"
)

plt.close()

print("\n✓ Results saved")
print("✓ Confusion matrix saved")
print("✓ XGBoost baseline complete")