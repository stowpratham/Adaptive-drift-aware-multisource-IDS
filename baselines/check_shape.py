import joblib

X_train = joblib.load("baselines/data/X_train_bal.pkl")
X_test = joblib.load("baselines/data/X_drift_stream.pkl")

print("Train:", X_train.shape)
print("Test :", X_test.shape)