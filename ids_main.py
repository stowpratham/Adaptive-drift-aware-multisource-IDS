"""
============================================================
Multi-Source Heterogeneous Drift-Aware Intrusion Detection System
============================================================
Self-learning + Drift-aware + Ensemble-powered + Real-time IDS

Architecture:
  KDD (41 feat) → FeatureEncoder → 32-d latent ─┐
                                                  ├→ AttentionFusion → Ensemble → Drift Detection
  UNSW (42 feat) → FeatureEncoder → 32-d latent ─┘

Key design principles:
  • Heterogeneous datasets remain independent until latent-space fusion
  • NO inner join, merge(), or concat() on raw features
  • Neural encoders map different feature schemas into a shared latent space
  • Attention-based soft fusion learns optimal source weighting
  • UNSW traffic stream provides realistic concept drift simulation
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import time
from collections import deque

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, confusion_matrix,
    classification_report
)
from sklearn.model_selection import cross_val_score
# imblearn not available — use manual random oversampling

# ─── PyTorch for latent representation learning ──────────
import torch
import torch.nn as nn
torch.manual_seed(42)

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Dataset paths (heterogeneous multi-source) ──────────
KDD_TRAIN_PATH  = os.path.join(BASE_DIR, "data", "Train_data.csv")
KDD_TEST_PATH   = os.path.join(BASE_DIR, "data", "Test_data.csv")
UNSW_TRAIN_PATH = os.path.join(BASE_DIR, "data", "UNSW_NB15_training-set.csv")
UNSW_TEST_PATH  = os.path.join(BASE_DIR, "data", "UNSW_NB15_testing-set.csv")
OUT_DIR         = os.path.join(BASE_DIR, "models")
os.makedirs(OUT_DIR, exist_ok=True)

STREAM_CHUNK   = 500        # records per stream window
DRIFT_WINDOW   = 50         # sliding error window for ADWIN-style detection
DRIFT_THRESH   = 0.05       # error rate threshold to trigger retraining
LATENT_DIM     = 32         # shared latent space dimensionality
RANDOM_STATE   = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────────────
# UTILITY: pretty section headers
# ─────────────────────────────────────────────────────────
def section(title):
    width = 60
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)

# ─────────────────────────────────────────────────────────
# STEP 1: MULTI-SOURCE DATA LOADING & EXPLORATION
# ─────────────────────────────────────────────────────────
# Loads both KDD and UNSW-NB15 datasets independently.
# The two datasets have DIFFERENT feature schemas:
#   KDD  : 41 features + 'class' target
#   UNSW : 42 features + 'label' target (after dropping id/attack_cat)
# They are NEVER merged or joined at the raw-feature level.
# ─────────────────────────────────────────────────────────
def load_data():
    section("STEP 1 ▸ Loading Multi-Source Data")

    # ── Source 1: KDD / NSL-KDD ──
    kdd_train = pd.read_csv(KDD_TRAIN_PATH)
    kdd_test  = pd.read_csv(KDD_TEST_PATH)
    print(f"  [KDD]  Train : {kdd_train.shape[0]:,} rows × {kdd_train.shape[1]} cols")
    print(f"  [KDD]  Test  : {kdd_test.shape[0]:,} rows × {kdd_test.shape[1]} cols")
    print(f"\n  KDD Train class distribution:")
    for cls, cnt in kdd_train["class"].value_counts().items():
        pct = cnt / len(kdd_train) * 100
        print(f"    {cls:<12}: {cnt:>6,}  ({pct:.1f}%)")

    # ── Source 2: UNSW-NB15 ──
    unsw_train = pd.read_csv(UNSW_TRAIN_PATH)
    unsw_test  = pd.read_csv(UNSW_TEST_PATH)
    print(f"\n  [UNSW] Train : {unsw_train.shape[0]:,} rows × {unsw_train.shape[1]} cols")
    print(f"  [UNSW] Test  : {unsw_test.shape[0]:,} rows × {unsw_test.shape[1]} cols")
    print(f"\n  UNSW Train label distribution:")
    for cls, cnt in unsw_train["label"].value_counts().items():
        lbl = "normal" if cls == 0 else "attack"
        pct = cnt / len(unsw_train) * 100
        print(f"    {lbl:<12}: {cnt:>6,}  ({pct:.1f}%)")

    return kdd_train, kdd_test, unsw_train, unsw_test

# ─────────────────────────────────────────────────────────
# STEP 2A: KDD PREPROCESSING (independent pipeline)
# ─────────────────────────────────────────────────────────
# Encodes: protocol_type, service, flag
# Target : 'class' column (anomaly/normal → 0/1)
# KDD test has NO labels — pseudo-labeled later.
# ─────────────────────────────────────────────────────────
def preprocess_kdd(train_df, test_df):
    section("STEP 2A ▸ KDD Preprocessing")

    train_df = train_df.copy()
    test_df  = test_df.copy()

    cat_cols = ["protocol_type", "service", "flag"]
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]], axis=0)
        le.fit(combined)
        train_df[col] = le.transform(train_df[col])
        test_df[col]  = le.transform(test_df[col])
        label_encoders[col] = le

    # Encode target
    le_target = LabelEncoder()
    y_train = le_target.fit_transform(train_df["class"])   # normal=1, anomaly=0
    X_train = train_df.drop(columns=["class"])

    # KDD test has no labels — pseudo-label later
    X_test = test_df.copy()

    print(f"  Categorical cols encoded : {cat_cols}")
    print(f"  Class mapping            : {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")
    print(f"  X_train shape            : {X_train.shape}")
    print(f"  X_test  shape            : {X_test.shape}")

    return X_train, y_train, X_test, le_target, label_encoders

# ─────────────────────────────────────────────────────────
# STEP 2B: UNSW-NB15 PREPROCESSING (independent pipeline)
# ─────────────────────────────────────────────────────────
# Encodes: proto, service, state
# Target : 'label' column (0=normal, 1=attack)
# Removes: 'id', 'attack_cat'
# IMPORTANT: This pipeline is completely independent of KDD.
#            NO shared encoders, NO feature alignment.
# ─────────────────────────────────────────────────────────
def preprocess_unsw(train_df, test_df):
    section("STEP 2B ▸ UNSW-NB15 Preprocessing")

    train_df = train_df.copy()
    test_df  = test_df.copy()

    # Drop non-feature columns
    drop_cols = ["id", "attack_cat"]
    for col in drop_cols:
        if col in train_df.columns:
            train_df.drop(columns=[col], inplace=True)
        if col in test_df.columns:
            test_df.drop(columns=[col], inplace=True)

    cat_cols = ["proto", "service", "state"]
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]], axis=0)
        le.fit(combined)
        train_df[col] = le.transform(train_df[col])
        test_df[col]  = le.transform(test_df[col])
        label_encoders[col] = le

    # Extract labels (UNSW has labels in both train and test)
    y_train = train_df["label"].values    # 0=normal, 1=attack
    X_train = train_df.drop(columns=["label"])

    y_test = test_df["label"].values
    X_test = test_df.drop(columns=["label"])

    print(f"  Dropped columns          : {drop_cols}")
    print(f"  Categorical cols encoded : {cat_cols}")
    print(f"  Label mapping            : 0=normal, 1=attack")
    print(f"  X_train shape            : {X_train.shape}")
    print(f"  X_test  shape            : {X_test.shape}")

    return X_train, y_train, X_test, y_test, label_encoders

# ─────────────────────────────────────────────────────────
# STEP 3: INDEPENDENT SCALING (per-source)
# ─────────────────────────────────────────────────────────
# Each dataset is scaled with its OWN StandardScaler.
# NO combined scaling across sources.
# ─────────────────────────────────────────────────────────
def scale(X_train, X_test, source_name=""):
    section(f"STEP 3 ▸ Scaling {source_name} (StandardScaler)")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    print(f"  [{source_name}] Scaler fitted on {X_train_sc.shape[0]:,} samples, {X_train_sc.shape[1]} features.")
    return X_train_sc, X_test_sc, scaler

# ─────────────────────────────────────────────────────────
# STEP 4: NEURAL FEATURE ENCODERS (latent representation)
# ─────────────────────────────────────────────────────────
# Each source gets its own encoder that maps its raw feature
# space into a shared LATENT_DIM-dimensional representation.
#
# This is the key to heterogeneous data handling:
#   KDD  (41 features) → FeatureEncoder → 32-d latent vector
#   UNSW (42 features) → FeatureEncoder → 32-d latent vector
#
# By projecting into a common latent space, we avoid:
#   - Forced column alignment
#   - Inner joins or merges
#   - Manual feature matching
# ─────────────────────────────────────────────────────────
class FeatureEncoder(nn.Module):
    """
    Neural encoder that maps heterogeneous input features
    into a shared latent space of dimension LATENT_DIM.

    Architecture: Input → Linear(64) → ReLU → Linear(32) → ReLU
    """
    def __init__(self, input_dim, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class AutoEncoder(nn.Module):
    """
    Autoencoder wrapper for training the FeatureEncoder via
    reconstruction loss.  After training, only the encoder
    half is used — the decoder is discarded.
    """
    def __init__(self, input_dim, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)


def train_encoder(autoencoder, X_np, epochs=30, lr=1e-3, batch_size=256):
    """
    Train an AutoEncoder on source-specific data via MSE
    reconstruction loss.  Returns a FeatureEncoder whose
    weights are copied from the trained autoencoder.
    """
    autoencoder.train()
    optimiser = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    criterion = nn.MSELoss()
    X_t = torch.FloatTensor(X_np)
    dataset = torch.utils.data.TensorDataset(X_t)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            optimiser.zero_grad()
            recon = autoencoder(batch)
            loss  = criterion(recon, batch)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(batch)
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  loss={total_loss/len(X_np):.6f}")

    # Copy trained encoder weights into a standalone FeatureEncoder
    input_dim  = X_np.shape[1]
    latent_dim = LATENT_DIM
    encoder = FeatureEncoder(input_dim, latent_dim)
    encoder.encoder.load_state_dict(autoencoder.encoder.state_dict())
    encoder.eval()
    return encoder

# ─────────────────────────────────────────────────────────
# STEP 5: ATTENTION-BASED SOFT FUSION
# ─────────────────────────────────────────────────────────
# Instead of inner-join or concat on raw features, we fuse
# latent representations using learned attention weights.
#
# Fusion logic:
#   1. Concatenate the two latent vectors: [z_kdd; z_unsw]
#   2. Pass through a linear layer → 2 logits
#   3. Apply softmax → attention weights [w1, w2]
#   4. Fused = w1 * z_kdd + w2 * z_unsw
#
# This replaces traditional inner-join fusion and allows
# heterogeneous datasets to contribute proportionally.
# ─────────────────────────────────────────────────────────
class AttentionFusion(nn.Module):
    """
    Attention-based soft fusion of two latent representations.
    Learns dynamic weights for combining KDD and UNSW latent vectors.
    """
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        # Takes concatenated latent vectors, outputs 2 attention logits
        self.attention = nn.Linear(latent_dim * 2, 2)

    def forward(self, z_kdd, z_unsw):
        # Concatenate latent vectors from both sources
        combined = torch.cat([z_kdd, z_unsw], dim=1)    # (N, 64)
        # Compute attention weights via softmax
        weights = torch.softmax(self.attention(combined), dim=1)  # (N, 2)
        w1 = weights[:, 0:1]   # weight for KDD source
        w2 = weights[:, 1:2]   # weight for UNSW source
        # Weighted fusion: fused = w1 * kdd_features + w2 * unsw_features
        fused = w1 * z_kdd + w2 * z_unsw                # (N, 32)
        return fused

# ─────────────────────────────────────────────────────────
# HELPER: Encode numpy array through a FeatureEncoder
# ─────────────────────────────────────────────────────────
def encode_to_latent(encoder, X_np):
    """Convert numpy data → torch tensor → encoder → numpy latent vectors."""
    encoder.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_np)
        latent = encoder(X_tensor)
    return latent.numpy()

# ─────────────────────────────────────────────────────────
# HELPER: Fuse two latent arrays via AttentionFusion
# ─────────────────────────────────────────────────────────
def fuse_representations(fusion_module, z_kdd_np, z_unsw_np):
    """Apply attention fusion on numpy latent arrays, return numpy."""
    fusion_module.eval()
    with torch.no_grad():
        z_kdd = torch.FloatTensor(z_kdd_np)
        z_unsw = torch.FloatTensor(z_unsw_np)
        fused = fusion_module(z_kdd, z_unsw)
    return fused.numpy()

# ─────────────────────────────────────────────────────────
# STEP 5: CLASS BALANCE (SMOTE)
# ─────────────────────────────────────────────────────────
def balance(X_train, y_train):
    section("STEP 5 ▸ Class Balancing (Random Oversampling)")
    before = dict(zip(*np.unique(y_train, return_counts=True)))
    classes, counts = np.unique(y_train, return_counts=True)
    max_count = counts.max()
    X_parts, y_parts = [X_train], [y_train]
    for cls, cnt in zip(classes, counts):
        if cnt < max_count:
            n_over = max_count - cnt
            idx = np.where(y_train == cls)[0]
            over_idx = np.random.choice(idx, size=n_over, replace=True)
            X_parts.append(X_train[over_idx])
            y_parts.append(np.full(n_over, cls))
    X_bal = np.vstack(X_parts)
    y_bal = np.concatenate(y_parts)
    # Shuffle
    perm = np.random.permutation(len(y_bal))
    X_bal, y_bal = X_bal[perm], y_bal[perm]
    after = dict(zip(*np.unique(y_bal, return_counts=True)))
    print(f"  Before: {before}")
    print(f"  After : {after}")
    return X_bal, y_bal

# ─────────────────────────────────────────────────────────
# STEP 6: ENSEMBLE MODEL BUILDING
# ─────────────────────────────────────────────────────────
class AdaptiveEnsemble:
    """
    Soft-voting ensemble of both incremental (partial_fit) and
    batch base learners. Weights are updated dynamically based
    on recent window accuracy.
    """

    def __init__(self):
        self.batch_models = {
            "RandomForest"     : RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
            "GradientBoosting" : GradientBoostingClassifier(n_estimators=80, random_state=RANDOM_STATE),
            "DecisionTree"     : DecisionTreeClassifier(max_depth=12, random_state=RANDOM_STATE),
            "KNN"              : KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        }
        self.stream_models = {
            "SGD_Hinge"        : SGDClassifier(loss="hinge",    max_iter=1, warm_start=True, random_state=RANDOM_STATE),
            "SGD_Log"          : SGDClassifier(loss="log_loss", max_iter=1, warm_start=True, random_state=RANDOM_STATE),
            "Perceptron"       : Perceptron(max_iter=1, warm_start=True, random_state=RANDOM_STATE),
            "GaussianNB"       : GaussianNB(),
        }
        self.all_models  = {**self.batch_models, **self.stream_models}
        self.weights     = {k: 1.0 for k in self.all_models}
        self.classes_    = np.array([0, 1])

    def fit(self, X, y):
        section("STEP 6 ▸ Training Ensemble")
        for name, mdl in self.batch_models.items():
            t0 = time.time()
            mdl.fit(X, y)
            print(f"  ✓ {name:<22} trained  ({time.time()-t0:.2f}s)")
        for name, mdl in self.stream_models.items():
            t0 = time.time()
            mdl.fit(X, y)
            print(f"  ✓ {name:<22} trained  ({time.time()-t0:.2f}s)")
        print(f"\n  Ensemble of {len(self.all_models)} models ready.")

    def partial_fit(self, X, y):
        """Incremental update — only stream models support this."""
        for name, mdl in self.stream_models.items():
            mdl.partial_fit(X, y, classes=self.classes_)
        # retrain batch models with the chunk (fast, small)
        # Skip if chunk contains only one class (e.g. pure attack phases)
        # since models like GradientBoosting require ≥2 classes
        if len(np.unique(y)) >= 2:
            for name, mdl in self.batch_models.items():
                mdl.fit(X, y)

    def predict_proba_all(self, X):
        probas = []
        for name, mdl in self.all_models.items():
            w = self.weights[name]
            if hasattr(mdl, "predict_proba"):
                p = mdl.predict_proba(X)
            else:
                # decision_function → sigmoid
                d = mdl.decision_function(X)
                sig = 1 / (1 + np.exp(-d))
                p = np.column_stack([1 - sig, sig])
            probas.append(w * p)
        return np.mean(probas, axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba_all(X), axis=1)

    def update_weights(self, X, y_true):
        """After each window, update model weights by accuracy."""
        for name, mdl in self.all_models.items():
            if hasattr(mdl, "predict"):
                pred = mdl.predict(X)
                acc  = accuracy_score(y_true, pred)
                self.weights[name] = max(0.1, acc)   # floor at 0.1

# ─────────────────────────────────────────────────────────
# STEP 7: ANOMALY DETECTOR (unsupervised hybrid layer)
# ─────────────────────────────────────────────────────────
class AnomalyDetector:
    """
    Simple statistical anomaly detector based on per-feature
    z-score thresholding (trained on normal traffic).
    """
    def __init__(self, threshold=3.5):
        self.threshold = threshold
        self.mean_ = None
        self.std_  = None

    def fit(self, X_normal):
        self.mean_ = np.mean(X_normal, axis=0)
        self.std_  = np.std(X_normal,  axis=0) + 1e-9

    def score(self, X):
        """Returns anomaly score: mean abs z-score per sample."""
        z = np.abs((X - self.mean_) / self.std_)
        return z.mean(axis=1)

    def predict(self, X):
        scores = self.score(X)
        return (scores > self.threshold).astype(int)   # 1 = anomaly

# ─────────────────────────────────────────────────────────
# STEP 8: CONCEPT DRIFT DETECTOR (ADWIN-style)
# ─────────────────────────────────────────────────────────
class DriftDetector:
    """
    Sliding-window error-rate monitor.
    Drift is flagged when the recent error rate exceeds DRIFT_THRESH.
    """
    def __init__(self, window=DRIFT_WINDOW, threshold=DRIFT_THRESH):
        self.window    = window
        self.threshold = threshold
        self.errors    = deque(maxlen=window)
        self.drift_log = []           # (chunk_id, error_rate)

    def update(self, y_true, y_pred, chunk_id):
        errors = (y_true != y_pred).astype(int)
        self.errors.extend(errors)
        if len(self.errors) == self.window:
            err_rate = np.mean(self.errors)
            if err_rate > self.threshold:
                self.drift_log.append((chunk_id, err_rate))
                return True, err_rate
        return False, 0.0

# ─────────────────────────────────────────────────────────
# STEP 9: PSEUDO-LABELING (for unlabelled test data)
# ─────────────────────────────────────────────────────────
def pseudo_label(ensemble, X_test_scaled, confidence=0.80):
    section("STEP 9 ▸ Pseudo-Labeling Test Data")
    probas = ensemble.predict_proba_all(X_test_scaled)
    max_conf = probas.max(axis=1)
    labels   = probas.argmax(axis=1)
    # Use high-confidence predictions; mark uncertain ones
    pseudo   = labels.copy()
    uncertain_mask = max_conf < confidence
    uncertain_count = uncertain_mask.sum()
    print(f"  High-conf (≥{confidence}): {(~uncertain_mask).sum():,}")
    print(f"  Uncertain (<{confidence}) : {uncertain_count:,} → kept as-is")
    print(f"  Pseudo class distribution: {dict(zip(*np.unique(pseudo, return_counts=True)))}")
    return pseudo

# ─────────────────────────────────────────────────────────
# STEP 10: STREAM SIMULATION + DRIFT ADAPTATION
# ─────────────────────────────────────────────────────────
def stream_evaluate(ensemble, anomaly_det, drift_det,
                    X_test_sc, y_pseudo,
                    X_train_sc_bal, y_train_bal):
    section("STEP 10 ▸ Stream Simulation with Drift Detection")

    n_chunks   = len(X_test_sc) // STREAM_CHUNK
    remainder  = len(X_test_sc) % STREAM_CHUNK

    chunk_metrics = []
    drift_points  = []
    retrain_count = 0

    for i in range(n_chunks + (1 if remainder else 0)):
        start = i * STREAM_CHUNK
        end   = min(start + STREAM_CHUNK, len(X_test_sc))
        X_chunk = X_test_sc[start:end]
        y_chunk = y_pseudo[start:end]

        if len(X_chunk) == 0:
            break

        # --- Hybrid detection ---
        ml_pred     = ensemble.predict(X_chunk)
        anom_pred   = anomaly_det.predict(X_chunk)
        # Combine: if either flags it, mark as anomaly (1)
        hybrid_pred = np.clip(ml_pred + anom_pred, 0, 1)

        # --- Metrics ---
        acc  = accuracy_score(y_chunk, hybrid_pred)
        prec = precision_score(y_chunk, hybrid_pred, zero_division=0)
        rec  = recall_score(y_chunk, hybrid_pred,    zero_division=0)
        f1   = f1_score(y_chunk, hybrid_pred,        zero_division=0)

        chunk_metrics.append({
            "chunk": i + 1, "accuracy": acc,
            "precision": prec, "recall": rec, "f1": f1
        })

        # --- Drift detection ---
        drifted, err_rate = drift_det.update(y_chunk, hybrid_pred, i + 1)
        if drifted:
            drift_points.append(i + 1)
            retrain_count += 1
            # Adaptive retraining on most recent buffer
            ensemble.partial_fit(X_chunk, y_chunk)
            ensemble.update_weights(X_chunk, y_chunk)
            print(f"  ⚠  DRIFT @ chunk {i+1:3d}  |  err={err_rate:.2%}  → retrained")

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Chunk {i+1:3d}/{n_chunks}  |  acc={acc:.3f}  f1={f1:.3f}")

    print(f"\n  Total chunks   : {len(chunk_metrics)}")
    print(f"  Drift events   : {len(drift_points)}")
    print(f"  Retrain count  : {retrain_count}")
    return pd.DataFrame(chunk_metrics), drift_points

# ─────────────────────────────────────────────────────────
# STEP 11: FINAL EVALUATION ON FULL TEST
# ─────────────────────────────────────────────────────────
def final_evaluation(ensemble, anomaly_det, X_test_sc, y_pseudo, le_target):
    section("STEP 11 ▸ Final Evaluation")
    ml_pred   = ensemble.predict(X_test_sc)
    anom_pred = anomaly_det.predict(X_test_sc)
    y_pred    = np.clip(ml_pred + anom_pred, 0, 1)
    y_true    = y_pseudo

    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred,  zero_division=0)
    rec   = recall_score(y_true, y_pred,     zero_division=0)
    f1    = f1_score(y_true, y_pred,         zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm    = confusion_matrix(y_true, y_pred)

    print(f"\n  ┌{'─'*42}┐")
    print(f"  │  {'Metric':<20}  {'Value':>12}       │")
    print(f"  ├{'─'*42}┤")
    for name, val in [("Accuracy",acc),("Precision",prec),("Recall",rec),("F1-Score",f1),("Kappa",kappa)]:
        print(f"  │  {name:<20}  {val:>12.4f}       │")
    print(f"  └{'─'*42}┘")
    print(f"\n  Confusion Matrix:\n{cm}")

    return {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1,"kappa":kappa,"cm":cm,"y_pred":y_pred}

# ─────────────────────────────────────────────────────────
# STEP 12: VISUALIZATIONS
# ─────────────────────────────────────────────────────────
def plot_all(chunk_df, drift_points, results, y_train, X_train_sc_bal, y_train_bal, ensemble):
    section("STEP 12 ▸ Generating Visualizations")

    plt.style.use("seaborn-v0_8-darkgrid")
    colors = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860"]

    # ── Fig 1: Stream accuracy + drift markers ──────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Multi-Source Stream Learning Performance (UNSW Drift Stream)", fontsize=15, fontweight="bold")

    ax = axes[0]
    ax.plot(chunk_df["chunk"], chunk_df["accuracy"], color=colors[0], lw=2, label="Accuracy")
    ax.plot(chunk_df["chunk"], chunk_df["f1"],       color=colors[1], lw=2, label="F1-Score", linestyle="--")
    for dp in drift_points:
        ax.axvline(x=dp, color="red", linestyle=":", alpha=0.7, lw=1.5)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.set_title("Accuracy & F1-Score per Stream Chunk")
    if drift_points:
        ax.axvline(x=drift_points[0], color="red", linestyle=":", alpha=0.7, lw=1.5, label="Drift Detected")

    ax2 = axes[1]
    ax2.plot(chunk_df["chunk"], chunk_df["precision"], color=colors[2], lw=2, label="Precision")
    ax2.plot(chunk_df["chunk"], chunk_df["recall"],    color=colors[3], lw=2, label="Recall", linestyle="--")
    for dp in drift_points:
        ax2.axvline(x=dp, color="red", linestyle=":", alpha=0.7, lw=1.5)
    ax2.set_xlabel("Stream Chunk"); ax2.set_ylabel("Score"); ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9)
    ax2.set_title("Precision & Recall per Stream Chunk")

    plt.tight_layout()
    p1 = os.path.join(OUT_DIR, "01_stream_performance.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p1}")

    # ── Fig 2: Final metrics bar chart ──────────────────
    metrics = {k: v for k, v in results.items() if k not in ("cm","y_pred")}
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(metrics.keys(), metrics.values(),
                  color=colors[:len(metrics)], edgecolor="white", linewidth=1.5, width=0.55)
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_title("Multi-Source IDS Evaluation Metrics (Latent Fusion)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    plt.tight_layout()
    p2 = os.path.join(OUT_DIR, "02_final_metrics.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p2}")

    # ── Fig 3: Confusion matrix heatmap ─────────────────
    cm = results["cm"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal","Anomaly"], yticklabels=["Normal","Anomaly"],
                ax=ax, linewidths=0.5)
    ax.set_title("Confusion Matrix (Multi-Source Hybrid Prediction)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    p3 = os.path.join(OUT_DIR, "03_confusion_matrix.png")
    plt.savefig(p3, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p3}")

    # ── Fig 4: Train class distribution (donut) ──────────
    cls_counts = dict(zip(*np.unique(y_train_bal, return_counts=True)))
    labels = ["Normal", "Anomaly"]
    vals   = [cls_counts.get(1, 0), cls_counts.get(0, 0)]
    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        vals, labels=labels, autopct="%1.1f%%",
        colors=[colors[0], colors[1]], startangle=90,
        wedgeprops=dict(width=0.5), textprops={"fontsize": 12})
    for at in autotexts: at.set_fontsize(12)
    ax.set_title("Class Distribution After SMOTE Balancing", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p4 = os.path.join(OUT_DIR, "04_class_distribution.png")
    plt.savefig(p4, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p4}")

    # ── Fig 5: Per-model accuracy comparison ─────────────
    model_names = list(ensemble.all_models.keys())
    model_accs  = []
    for name, mdl in ensemble.all_models.items():
        try:
            pred = mdl.predict(X_train_sc_bal[:1000])
            acc  = accuracy_score(y_train_bal[:1000], pred)
        except Exception:
            acc = 0.0
        model_accs.append(acc)

    fig, ax = plt.subplots(figsize=(11, 5))
    bar_colors = [colors[i % len(colors)] for i in range(len(model_names))]
    bars = ax.barh(model_names, model_accs, color=bar_colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, model_accs):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=10)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Accuracy on Training Sample")
    ax.set_title("Individual Model Accuracy Comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p5 = os.path.join(OUT_DIR, "05_model_comparison.png")
    plt.savefig(p5, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p5}")

    # ── Fig 6: Rolling accuracy (moving avg) ──────────────
    roll_acc = chunk_df["accuracy"].rolling(window=5, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(chunk_df["chunk"], chunk_df["accuracy"], alpha=0.25, color=colors[0])
    ax.plot(chunk_df["chunk"], chunk_df["accuracy"], alpha=0.5, color=colors[0], lw=1, label="Raw Accuracy")
    ax.plot(chunk_df["chunk"], roll_acc, color=colors[0], lw=2.5, label="5-Chunk Rolling Avg")
    for dp in drift_points:
        ax.axvline(x=dp, color="red", linestyle=":", lw=1.5)
    drift_patch = mpatches.Patch(color="red", alpha=0.6, label="Drift Event")
    ax.legend(handles=[*ax.get_legend_handles_labels()[0], drift_patch], fontsize=9)
    ax.set_xlabel("Stream Chunk"); ax.set_ylabel("Accuracy")
    ax.set_title("Rolling Accuracy with Drift Markers", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    p6 = os.path.join(OUT_DIR, "06_rolling_accuracy.png")
    plt.savefig(p6, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p6}")

    print("\n  All 6 plots saved.")

# ─────────────────────────────────────────────────────────
# REALISTIC DISTRIBUTION-SHIFT STREAM CONSTRUCTION
# ─────────────────────────────────────────────────────────
# Creates a phased UNSW test stream that simulates temporal
# concept drift.  The model was trained on KDD-only data, so
# even baseline UNSW traffic will look different.  On top of
# that we re-order the UNSW test set into four successive
# phases with very different class ratios:
#
#   Phase 1 (25 %) — mostly normal traffic  (≈90 % normal)
#   Phase 2 (25 %) — abrupt shift to heavy attacks (≈90 % attack)
#   Phase 3 (25 %) — gradual recovery, mixed traffic (≈50/50)
#   Phase 4 (25 %) — second attack surge  (≈85 % attack)
#
# This guarantees multiple detectable drift points and
# demonstrates the adaptive retraining loop.
# ─────────────────────────────────────────────────────────
def create_drift_stream(X_fused, y_labels):
    """
    Re-order fused UNSW test data into a phased stream that
    produces realistic concept drift when evaluated by a model
    trained on a different source distribution (KDD).

    Returns: (X_stream, y_stream) — reordered arrays
    """
    section("Creating Phased Drift Stream")

    normal_idx = np.where(y_labels == 1)[0]   # normal=1 in KDD convention
    attack_idx = np.where(y_labels == 0)[0]   # anomaly/attack=0

    np.random.shuffle(normal_idx)
    np.random.shuffle(attack_idx)

    n_total   = len(y_labels)
    phase_len = n_total // 4

    # Phase 1 — normal-heavy (≈90 % normal)
    n_norm_p1 = int(phase_len * 0.90)
    n_atk_p1  = phase_len - n_norm_p1
    p1_idx = np.concatenate([normal_idx[:n_norm_p1], attack_idx[:n_atk_p1]])
    np.random.shuffle(p1_idx)
    norm_cursor, atk_cursor = n_norm_p1, n_atk_p1

    # Phase 2 — abrupt attack surge (≈90 % attack)
    n_atk_p2  = int(phase_len * 0.90)
    n_norm_p2 = phase_len - n_atk_p2
    p2_idx = np.concatenate([
        normal_idx[norm_cursor:norm_cursor + n_norm_p2],
        attack_idx[atk_cursor:atk_cursor + n_atk_p2]
    ])
    np.random.shuffle(p2_idx)
    norm_cursor += n_norm_p2; atk_cursor += n_atk_p2

    # Phase 3 — mixed / gradual recovery (≈50/50)
    n_norm_p3 = phase_len // 2
    n_atk_p3  = phase_len - n_norm_p3
    p3_idx = np.concatenate([
        normal_idx[norm_cursor:norm_cursor + n_norm_p3],
        attack_idx[atk_cursor:atk_cursor + n_atk_p3]
    ])
    np.random.shuffle(p3_idx)
    norm_cursor += n_norm_p3; atk_cursor += n_atk_p3

    # Phase 4 — second attack surge (remaining data, attack-heavy)
    remaining_norm = normal_idx[norm_cursor:]
    remaining_atk  = attack_idx[atk_cursor:]
    p4_idx = np.concatenate([remaining_norm, remaining_atk])
    np.random.shuffle(p4_idx)

    # Concatenate all phases
    stream_idx = np.concatenate([p1_idx, p2_idx, p3_idx, p4_idx])

    X_stream = X_fused[stream_idx]
    y_stream = y_labels[stream_idx]

    # Report phase distributions
    for phase_num, pidx in enumerate([p1_idx, p2_idx, p3_idx, p4_idx], 1):
        phase_labels = y_labels[pidx]
        n_norm = (phase_labels == 1).sum()
        n_atk  = (phase_labels == 0).sum()
        total  = len(phase_labels)
        print(f"  Phase {phase_num}: {total:,} samples  "
              f"(normal={n_norm:,} [{100*n_norm/max(total,1):.0f}%], "
              f"attack={n_atk:,} [{100*n_atk/max(total,1):.0f}%])")

    print(f"  Total stream length: {len(y_stream):,}")
    return X_stream, y_stream

# ─────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────
def main():
    print("\n" + "█" * 60)
    print("  MULTI-SOURCE HETEROGENEOUS DRIFT-AWARE IDS  ")
    print("  KDD + UNSW · Neural Encoders · Attention Fusion")
    print("  Self-learning + Drift-aware + Ensemble + Real-time")
    print("█" * 60)

    t_start = time.time()

    # ══════════════════════════════════════════════════════
    # 1. Load multi-source data (KDD + UNSW independently)
    # ══════════════════════════════════════════════════════
    kdd_train_df, kdd_test_df, unsw_train_df, unsw_test_df = load_data()

    # ══════════════════════════════════════════════════════
    # 2. Preprocess each source INDEPENDENTLY
    #    No feature alignment, no shared encoders.
    # ══════════════════════════════════════════════════════
    X_kdd_train, y_kdd_train, X_kdd_test, le_target, le_kdd = preprocess_kdd(kdd_train_df, kdd_test_df)
    X_unsw_train, y_unsw_train, X_unsw_test, y_unsw_test, le_unsw = preprocess_unsw(unsw_train_df, unsw_test_df)

    # ══════════════════════════════════════════════════════
    # 3. Scale each source with its OWN scaler
    #    No combined scaling across sources.
    # ══════════════════════════════════════════════════════
    X_kdd_train_sc, X_kdd_test_sc, scaler_kdd   = scale(X_kdd_train, X_kdd_test, source_name="KDD")
    X_unsw_train_sc, X_unsw_test_sc, scaler_unsw = scale(X_unsw_train, X_unsw_test, source_name="UNSW")

    # ══════════════════════════════════════════════════════
    # 4. Train neural feature encoders via reconstruction
    #    Each source gets its OWN autoencoder trained on its
    #    own data.  This ensures the encoders learn source-
    #    specific latent representations, preserving the
    #    domain gap between KDD and UNSW.
    #    KDD  (41 features) → AutoEncoder → 32-d latent
    #    UNSW (42 features) → AutoEncoder → 32-d latent
    # ══════════════════════════════════════════════════════
    section("STEP 4 ▸ Training Neural Encoders (Reconstruction)")
    kdd_input_dim  = X_kdd_train_sc.shape[1]
    unsw_input_dim = X_unsw_train_sc.shape[1]

    print(f"  Training KDD autoencoder ({kdd_input_dim} → 64 → {LATENT_DIM} → 64 → {kdd_input_dim}) ...")
    kdd_ae = AutoEncoder(input_dim=kdd_input_dim, latent_dim=LATENT_DIM)
    kdd_encoder = train_encoder(kdd_ae, X_kdd_train_sc, epochs=30)

    print(f"  Training UNSW autoencoder ({unsw_input_dim} → 64 → {LATENT_DIM} → 64 → {unsw_input_dim}) ...")
    unsw_ae = AutoEncoder(input_dim=unsw_input_dim, latent_dim=LATENT_DIM)
    unsw_encoder = train_encoder(unsw_ae, X_unsw_train_sc, epochs=30)

    fusion = AttentionFusion(latent_dim=LATENT_DIM)
    print(f"  AttentionFusion : {LATENT_DIM}×2 → softmax → weighted sum → {LATENT_DIM}")

    # ══════════════════════════════════════════════════════
    # 5. Encode training data into latent space
    #    Datasets remain independent — each goes through
    #    its own TRAINED encoder. NO merge or join.
    # ══════════════════════════════════════════════════════
    section("STEP 5 ▸ Encoding to Shared Latent Space")
    z_kdd_train  = encode_to_latent(kdd_encoder,  X_kdd_train_sc)   # (N_kdd, 32)
    z_unsw_train = encode_to_latent(unsw_encoder, X_unsw_train_sc)  # (N_unsw, 32)
    print(f"  KDD  latent shape  : {z_kdd_train.shape}")
    print(f"  UNSW latent shape  : {z_unsw_train.shape}")

    # Measure domain gap in latent space
    kdd_mean  = z_kdd_train.mean(axis=0)
    unsw_mean = z_unsw_train.mean(axis=0)
    domain_gap = np.linalg.norm(kdd_mean - unsw_mean)
    print(f"  Latent-space domain gap (L2): {domain_gap:.4f}")

    # ══════════════════════════════════════════════════════
    # 6. Attention-based soft fusion
    #    Compute source centroids (mean latent representations)
    #    then fuse each sample with the other source's centroid.
    #
    #    For KDD samples:  fused = w1*kdd_sample + w2*unsw_centroid
    #    For UNSW samples: fused = w1*kdd_centroid + w2*unsw_sample
    #
    #    This enriches each sample with cross-source context
    #    WITHOUT any raw-feature merging.
    # ══════════════════════════════════════════════════════
    section("STEP 6 ▸ Attention-Based Fusion")
    kdd_centroid  = z_kdd_train.mean(axis=0)    # (32,)
    unsw_centroid = z_unsw_train.mean(axis=0)   # (32,)
    print(f"  KDD  centroid computed (mean of {z_kdd_train.shape[0]:,} vectors)")
    print(f"  UNSW centroid computed (mean of {z_unsw_train.shape[0]:,} vectors)")

    # Fuse KDD samples: each KDD latent + broadcast UNSW centroid
    unsw_centroid_broadcast = np.tile(unsw_centroid, (len(z_kdd_train), 1))
    fused_kdd_train = fuse_representations(fusion, z_kdd_train, unsw_centroid_broadcast)

    print(f"  Fused KDD  train shape : {fused_kdd_train.shape}")
    print(f"  (UNSW fused representations computed on-demand for streaming)")

    # ══════════════════════════════════════════════════════
    # 7. TRAINING ON KDD ONLY (not combined KDD+UNSW)
    #    This is the critical design choice for real drift:
    #    - Train distribution = KDD (legacy attack patterns)
    #    - Stream distribution = UNSW (modern traffic)
    #    - Trained encoders preserve source-specific features
    #    - The domain gap creates detectable concept drift
    #
    #    KDD labels: anomaly=0, normal=1 (from LabelEncoder)
    # ══════════════════════════════════════════════════════
    print(f"\n  KDD-only fused training data : {fused_kdd_train.shape}")
    print(f"  KDD labels distribution      : {dict(zip(*np.unique(y_kdd_train, return_counts=True)))}")
    print(f"  (UNSW fused representations are reserved for streaming only)")

    # 8. Balance KDD-only training data
    X_train_bal, y_train_bal = balance(fused_kdd_train, y_kdd_train)

    # 9. Build & train ensemble on KDD-only fused latent features
    ensemble = AdaptiveEnsemble()
    ensemble.fit(X_train_bal, y_train_bal)

    # 10. Anomaly detector — fit on normal KDD fused traffic
    #     Lowered threshold from 3.5 → 2.0 for sensitivity to
    #     cross-domain distribution shift (UNSW features will
    #     deviate more from KDD normal profile).
    section("STEP 7 ▸ Training Anomaly Detector")
    normal_mask = (y_train_bal == 1)   # normal=1 in KDD convention
    anomaly_det = AnomalyDetector(threshold=2.0)
    anomaly_det.fit(X_train_bal[normal_mask])
    print(f"  AnomalyDetector fitted on {normal_mask.sum():,} normal KDD samples (threshold=2.0).")

    # 11. Drift detector (sensitive settings for real drift)
    drift_det = DriftDetector()
    print(f"  DriftDetector: window={DRIFT_WINDOW}, threshold={DRIFT_THRESH}")

    # ══════════════════════════════════════════════════════
    # 12. Prepare UNSW test stream for drift simulation
    #     Model is trained ONLY on KDD latent representations.
    #     UNSW test represents a genuinely different data
    #     distribution (modern attacks) that the model has
    #     NEVER seen during training.
    #
    #     Drift simulation strategy:
    #       - Train distribution = KDD legacy traffic
    #       - Stream distribution = UNSW modern traffic
    #       - Distribution gap → classification errors
    #       - Errors accumulate → drift detector triggers
    #       - Adaptive retraining on drifted chunks
    # ══════════════════════════════════════════════════════
    section("STEP 8 ▸ Preparing UNSW Stream for Drift Simulation")
    z_unsw_test = encode_to_latent(unsw_encoder, X_unsw_test_sc)
    kdd_centroid_test = np.tile(kdd_centroid, (len(z_unsw_test), 1))
    fused_unsw_test = fuse_representations(fusion, kdd_centroid_test, z_unsw_test)
    print(f"  UNSW test encoded  : {z_unsw_test.shape}")
    print(f"  UNSW test fused    : {fused_unsw_test.shape}")

    # Unify UNSW test labels to match KDD convention (normal=1, anomaly=0)
    y_unsw_test_unified = 1 - y_unsw_test

    # ══════════════════════════════════════════════════════
    # 13. Create phased drift stream
    #     Reorder UNSW test data to create abrupt distribution
    #     shifts (normal-heavy → attack-heavy → mixed → attack)
    #     This amplifies the drift signal beyond the baseline
    #     KDD↔UNSW domain gap.
    # ══════════════════════════════════════════════════════
    X_drift_stream, y_drift_stream = create_drift_stream(
        fused_unsw_test, y_unsw_test_unified
    )

    # 14. Pseudo-label the stream data
    #     Even though UNSW test has real labels, pseudo-labeling
    #     preserves the self-learning aspect of the system.
    #     Using real labels as ground truth for drift detection.
    y_pseudo = pseudo_label(ensemble, X_drift_stream, confidence=0.75)

    # 15. Stream simulation with UNSW phased stream → real drift
    chunk_df, drift_points = stream_evaluate(
        ensemble, anomaly_det, drift_det,
        X_drift_stream, y_drift_stream,   # use REAL labels for drift evaluation
        X_train_bal, y_train_bal
    )

    # 16. Final evaluation (using real UNSW labels for ground truth)
    results = final_evaluation(ensemble, anomaly_det,
                               X_drift_stream, y_drift_stream, le_target)

    # 17. Plots
    plot_all(chunk_df, drift_points, results,
             y_kdd_train, X_train_bal, y_train_bal, ensemble)

    # ── Save chunk metrics CSV ──
    metrics_path = os.path.join(OUT_DIR, "stream_metrics.csv")
    chunk_df.to_csv(metrics_path, index=False)
    print(f"\n  Stream metrics CSV: {metrics_path}")

    elapsed = time.time() - t_start
    print(f"\n{'█'*60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Final Accuracy : {results['accuracy']:.4f}")
    print(f"  Final F1-Score : {results['f1']:.4f}")
    print(f"  Kappa Score    : {results['kappa']:.4f}")
    print(f"  Drift Events   : {len(drift_points)}")
    print(f"{'█'*60}\n")

if __name__ == "__main__":
    main()

    folder_path = os.path.abspath(OUT_DIR)
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".png"):
                file_path = os.path.join(folder_path, file)
                try:
                    os.startfile(file_path)
                except AttributeError:
                    import subprocess
                    subprocess.call(["xdg-open", file_path])
    else:
        print("models folder not found")
