from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _safe_auc(y_true, prob) -> float:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    prob = np.asarray(prob).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, prob))


def compute_permutation_importance_binary(
    model,
    x: pd.DataFrame,
    y,
    feature_cols: list[str],
    n_repeats: int = 3,
    random_state: int = 42,
    positive_class: int = 1,
) -> pd.DataFrame:
    """Permutation importance for tabular binary classifiers using AUC drop."""
    rng = np.random.default_rng(random_state)
    x_base = x[feature_cols].copy()
    y_true = np.asarray(y).reshape(-1).astype(int)
    classes = list(getattr(model, "classes_", [0, 1]))
    if positive_class not in classes:
        base_prob = np.zeros(len(x_base), dtype=float)
    else:
        base_prob = model.predict_proba(x_base)[:, classes.index(positive_class)]
    base_auc = _safe_auc(y_true, base_prob)

    rows = []
    for feature in feature_cols:
        drops = []
        for _ in range(n_repeats):
            x_perm = x_base.copy()
            x_perm[feature] = rng.permutation(x_perm[feature].to_numpy())
            if positive_class not in classes:
                perm_prob = np.zeros(len(x_perm), dtype=float)
            else:
                perm_prob = model.predict_proba(x_perm)[:, classes.index(positive_class)]
            perm_auc = _safe_auc(y_true, perm_prob)
            drops.append(base_auc - perm_auc if np.isfinite(base_auc) and np.isfinite(perm_auc) else np.nan)
        rows.append(
            {
                "feature": feature,
                "base_auc": base_auc,
                "importance_auc_drop_mean": float(np.nanmean(drops)),
                "importance_auc_drop_std": float(np.nanstd(drops)),
                "n_repeats": int(n_repeats),
            }
        )
    return pd.DataFrame(rows).sort_values("importance_auc_drop_mean", ascending=False).reset_index(drop=True)


def compute_torch_gradient_importance(
    model,
    x_seq,
    feature_cols: list[str],
    device: str = "cpu",
    max_samples: int = 1024,
) -> pd.DataFrame:
    """Input-gradient importance for binary sequence models.

    The result aggregates absolute gradient * input by feature and by time step.
    This is a lightweight sanity check, not a substitute for full SHAP.
    """
    import torch

    model = model.to(device)
    model.eval()
    x_np = np.asarray(x_seq)
    if len(x_np) > max_samples:
        x_np = x_np[-max_samples:]
    x = torch.tensor(x_np, dtype=torch.float32, device=device, requires_grad=True)
    logits = model(x).reshape(-1)
    score = logits.mean()
    model.zero_grad(set_to_none=True)
    score.backward()
    grad_x_input = (x.grad.detach().abs() * x.detach().abs()).cpu().numpy()

    feature_importance = grad_x_input.mean(axis=(0, 1))
    rows = [
        {"feature": feature, "gradient_importance": float(value)}
        for feature, value in zip(feature_cols, feature_importance)
    ]
    feature_df = pd.DataFrame(rows).sort_values("gradient_importance", ascending=False).reset_index(drop=True)

    time_importance = grad_x_input.mean(axis=(0, 2))
    time_rows = [
        {"relative_step": int(i - len(time_importance)), "gradient_importance": float(value)}
        for i, value in enumerate(time_importance)
    ]
    feature_df.attrs["time_importance"] = pd.DataFrame(time_rows)
    return feature_df


def try_compute_shap_summary(model, x: pd.DataFrame, feature_cols: list[str], max_samples: int = 512):
    """Optional SHAP summary values. Returns None when shap is unavailable."""
    try:
        import shap
    except ImportError:
        return None

    x_sample = x[feature_cols].tail(max_samples)
    try:
        explainer = shap.Explainer(model, x_sample)
        values = explainer(x_sample)
        raw_values = getattr(values, "values", values)
        if isinstance(raw_values, list):
            raw_values = raw_values[-1]
        if np.asarray(raw_values).ndim == 3:
            raw_values = np.asarray(raw_values)[:, :, -1]
        importance = np.abs(raw_values).mean(axis=0)
    except Exception:
        return None

    return (
        pd.DataFrame({"feature": feature_cols, "shap_abs_mean": importance})
        .sort_values("shap_abs_mean", ascending=False)
        .reset_index(drop=True)
    )


def export_model_explainability(
    output_dir: str | Path,
    summary: dict,
    permutation_results: dict[str, pd.DataFrame] | None = None,
    gradient_results: dict[str, pd.DataFrame] | None = None,
    shap_results: dict[str, pd.DataFrame] | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"summary": summary, "files": {}}

    for name, df in (permutation_results or {}).items():
        path = output_dir / f"{name}_permutation_importance.csv"
        df.to_csv(path, index=False)
        manifest["files"][f"{name}_permutation_importance"] = str(path)

    for name, df in (gradient_results or {}).items():
        path = output_dir / f"{name}_gradient_feature_importance.csv"
        df.to_csv(path, index=False)
        manifest["files"][f"{name}_gradient_feature_importance"] = str(path)
        time_df = df.attrs.get("time_importance")
        if isinstance(time_df, pd.DataFrame):
            time_path = output_dir / f"{name}_gradient_time_importance.csv"
            time_df.to_csv(time_path, index=False)
            manifest["files"][f"{name}_gradient_time_importance"] = str(time_path)

    for name, df in (shap_results or {}).items():
        if df is None:
            continue
        path = output_dir / f"{name}_shap_summary.csv"
        df.to_csv(path, index=False)
        manifest["files"][f"{name}_shap_summary"] = str(path)

    with open(output_dir / "explainability_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest
