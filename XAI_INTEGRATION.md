# SHAP XAI Integration Guide for Adaptive Drift-Aware IDS

## 1) Installation

From project root:

```powershell
python -m pip install -r requirements.txt
```

## 2) New modular structure

```text
project/
¦
+-- xai/
¦   +-- __init__.py
¦   +-- shap_explainer.py
¦   +-- explanation_service.py
¦   +-- visualization.py
¦
+-- models/
¦   +-- __init__.py
¦   +-- ensemble_wrapper.py
¦
+-- preprocessing/
¦   +-- __init__.py
¦   +-- feature_names.py
¦
+-- drift_detection/
¦   +-- __init__.py
¦   +-- detector.py
¦
+-- results/
¦   +-- explainability/
¦
+-- ids_main.py
+-- main.py
+-- requirements.txt
```

## 3) Where to connect XAI in pipeline

Connect XAI **immediately after ensemble training** and before stream simulation:

1. `ensemble.fit(X_train_bal, y_train_bal)`
2. Initialize `ExplanationService` with fused latent training data
3. Call `configure_explanation_service(...)`
4. Generate global SHAP plots using `generate_global_explanations(...)`
5. During streaming prediction, call `generate_explanation(sample)` for each prediction

This ensures:
- Base model is trained before SHAP explainer initialization
- Same feature space (fused latent 32-d) is used for predictions and explanations
- Per-prediction explanations are available in real-time operations

## 4) Generated outputs

Artifacts are auto-saved in `results/explainability/`:

- `global_importance_attack.png`
- `global_importance_benign.png`
- `shap_summary_attack.png`
- `shap_summary_benign.png`
- `waterfall_<sample_id>.png`
- `force_<sample_id>.html`
- `prediction_explanations.jsonl`

## 5) Runtime API

Use the module-level function from `xai/explanation_service.py`:

```python
from xai.explanation_service import generate_explanation

explanation = generate_explanation(sample)
print(explanation)
```

Returns:

```python
{
    "prediction": "attack|benign",
    "confidence": 0.0,
    "top_features": [...],
    "feature_contributions": {...},
    "human_readable_explanation": "..."
}
```

## 6) Human-readable explanation behavior

Narrative text is constructed from top SHAP contributors and confidence, for example:

"Traffic was classified as Malicious / Attack with 94.2% confidence because Flow Duration was unusually low, Packet Rate was elevated, and Destination Host Count strongly pushed toward attack behavior."

## 7) Running the pipeline

```powershell
python main.py
```

or

```powershell
python ids_main.py
```

Both execute the integrated IDS + SHAP explainability flow.
