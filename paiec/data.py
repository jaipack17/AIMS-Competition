"""Loading measurement-db tables and building competition-shaped [subject, item] inputs."""
import glob
import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "mdb")
SUBJECT_KEYS = ["normalized_name", "provider", "release_date", "access_date", "harness",
                "reasoning_effort", "harness_version", "subject_features_extra"]


def benchmarks() -> pd.DataFrame:
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{ROOT}/*/formatted_tables/benchmarks.parquet"))],
                     ignore_index=True)


def binary_benchmark_ids() -> list[str]:
    b = benchmarks()
    return sorted(b.loc[b.response_type == "binary", "benchmark_id"])


def load_responses(bench: str) -> pd.DataFrame:
    """Binary responses only (0/1). Non-binary benchmarks are excluded until the binarization rule is known."""
    r = pd.read_parquet(f"{ROOT}/{bench}/formatted_tables/responses.parquet",
                        columns=["subject_id", "item_id", "trial", "response"])
    r = r.dropna(subset=["response"])
    return r[r.response.isin([0.0, 1.0])].reset_index(drop=True)


def load_subjects(bench: str) -> dict[str, dict]:
    s = pd.read_parquet(f"{ROOT}/{bench}/formatted_tables/subjects.parquet")
    out = {}
    for _, row in s.iterrows():
        out[row.subject_id] = {k: ("" if pd.isna(row[k]) else str(row[k])) for k in SUBJECT_KEYS}
    return out


def load_items(bench: str) -> dict[str, dict]:
    """Item dicts keyed by item_id. Falls back to empty text if items.parquet is not downloaded."""
    path = f"{ROOT}/{bench}/formatted_tables/items.parquet"
    if not os.path.exists(path):
        return {}
    i = pd.read_parquet(path, columns=["item_id", "content", "item_features"])
    return {r.item_id: {"item_content": "" if pd.isna(r.content) else str(r.content),
                        "item_features": "" if pd.isna(r.item_features) else str(r.item_features),
                        "interactors": "", "benchmark_id": bench} for r in i.itertuples()}
