from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

KDD_REQUIRED_COLUMNS = [
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
    "class",
]

UNSW_REQUIRED_COLUMNS = [
    "id",
    "dur",
    "proto",
    "service",
    "state",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
    "attack_cat",
    "label",
]


def detect_dataset_format(df: pd.DataFrame) -> Optional[str]:
    columns = set(df.columns.str.lower())
    if set(KDD_REQUIRED_COLUMNS).issubset(columns):
        return "NSL-KDD"
    if set(UNSW_REQUIRED_COLUMNS).issubset(columns):
        return "UNSW-NB15"
    if "class" in columns and "protocol_type" in columns and "flag" in columns:
        return "NSL-KDD"
    if "label" in columns and "proto" in columns and "state" in columns:
        return "UNSW-NB15"
    return None


def validate_uploaded_csv(df: pd.DataFrame) -> Dict[str, object]:
    dataset_type = detect_dataset_format(df)
    errors: List[str] = []
    if dataset_type == "NSL-KDD":
        missing = [col for col in KDD_REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            errors.append(f"Missing NSL-KDD columns: {', '.join(missing)}")
    elif dataset_type == "UNSW-NB15":
        missing = [col for col in UNSW_REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            errors.append(f"Missing UNSW-NB15 columns: {', '.join(missing)}")
    else:
        errors.append("Unable to automatically detect the dataset format. Check NSL-KDD or UNSW-NB15 headers.")
    return {
        "dataset_type": dataset_type,
        "errors": errors,
    }


def get_upload_summary(df: pd.DataFrame, dataset_type: Optional[str]) -> Dict[str, object]:
    return {
        "Dataset type": dataset_type or "Unknown",
        "Total records": len(df),
        "Total columns": len(df.columns),
        "Sample columns": list(df.columns[:10]),
    }
