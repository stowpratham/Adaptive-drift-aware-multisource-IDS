import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

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

print("Reshaping for LSTM...")

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print("Building LSTM...")

model = Sequential([
    LSTM(64, input_shape=(32, 1)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("Training LSTM...")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=256,
    callbacks=[early_stop],
    verbose=1,
)

print("Evaluating...")

y_prob = model.predict(X_test)

y_pred = (y_prob > 0.5).astype(int).flatten()

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

far = fp / (fp + tn)
dr = tp / (tp + fn)

results = {
    "Model": "LSTM",
    "Accuracy": round(acc, 4),
    "Precision": round(prec, 4),
    "Recall": round(rec, 4),
    "F1": round(f1, 4),
    "Kappa": round(kappa, 4),
    "FAR": round(far, 4),
    "Detection Rate": round(dr, 4),
}

print("\n===== LSTM RESULTS =====")

for key, value in results.items():
    print(f"{key}: {value}")

with open(
    "results/baselines/lstm_results.json",
    "w",
) as f:
    json.dump(results, f, indent=4)

disp = ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred)
)

disp.plot()

plt.title("LSTM Confusion Matrix")

plt.savefig(
    "results/baselines/lstm_confusion_matrix.png"
)

plt.close()

print("\n✓ Results saved")
print("✓ Confusion matrix saved")
print("✓ LSTM baseline complete")