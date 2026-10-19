import json
from pathlib import Path
from collections import defaultdict
class FeatureProjection:

    def __init__(self, mapping_file: str):
        mapping_path = Path(mapping_file)

        if not mapping_path.exists():
            raise FileNotFoundError(mapping_file)

        with open(mapping_path, "r", encoding="utf-8") as f:
            self.mapping = json.load(f)

    # -----------------------------------------------------

    def project(self, latent_shap_values):

        feature_scores = defaultdict(float)

        for latent_idx, shap_value in enumerate(latent_shap_values):

            if latent_idx >= len(self.mapping):
                break

            latent = self.mapping[latent_idx]

            for feature in latent["top_original_features"]:

                feature_name = feature["feature"]

                weight = feature["importance"]

                feature_scores[feature_name] += abs(shap_value) * weight

        return dict(feature_scores)

    # -----------------------------------------------------

    def normalize(self, feature_scores):

        if len(feature_scores) == 0:
            return {}

        maximum = max(feature_scores.values())

        return {

            feature: round(score / maximum * 100, 2)

            for feature, score in feature_scores.items()

        }

    # -----------------------------------------------------

    def rank(self, normalized_scores, top_k=5):

        return sorted(

            normalized_scores.items(),

            key=lambda x: x[1],

            reverse=True

        )[:top_k]