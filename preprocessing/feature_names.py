"""Feature name helpers for human-readable SHAP explanations."""

from __future__ import annotations

from typing import List, Optional, Sequence

# Canonical KDD/NSL-KDD feature names (41 features, excluding target).
KDD_RAW_FEATURE_NAMES: List[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# Analyst-friendly aliases used in narrative explanations.
FEATURE_DISPLAY_ALIASES = {
    "duration": "Flow Duration",
    "src_bytes": "Source Bytes",
    "dst_bytes": "Destination Bytes",
    "count": "Connection Count",
    "srv_count": "Service Count",
    "same_srv_rate": "Same Service Rate",
    "diff_srv_rate": "Different Service Rate",
    "serror_rate": "SYN Error Rate",
    "rerror_rate": "REJ Error Rate",
    "dst_host_count": "Destination Host Count",
    "dst_host_srv_count": "Destination Host Service Count",
    "fused_latent_dim_00": "Fused Latent Dimension 1",
}


def kdd_raw_feature_names() -> List[str]:
    """Return a copy of canonical KDD feature names."""
    return list(KDD_RAW_FEATURE_NAMES)


def default_latent_feature_names(n_features: int) -> List[str]:
    """Names for fused latent-space inputs (ensemble input)."""
    return [f"fused_latent_dim_{i:02d}" for i in range(n_features)]


def resolve_feature_names(
    n_features: int,
    custom_names: Optional[Sequence[str]] = None,
) -> List[str]:
    """Resolve feature names; fall back to latent defaults if custom list is invalid."""
    if custom_names is not None and len(custom_names) == n_features:
        return list(custom_names)
    return default_latent_feature_names(n_features)


def display_name(feature: str) -> str:
    """Map internal feature name to analyst-friendly label."""
    if feature in FEATURE_DISPLAY_ALIASES:
        return FEATURE_DISPLAY_ALIASES[feature]
    return feature.replace("_", " ").title()
