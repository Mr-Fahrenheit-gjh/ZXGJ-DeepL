from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - sklearn baselines should still be importable without torch.
    torch = None
    nn = None
    DataLoader = None
    Dataset = object


def get_logit_params(config: dict) -> dict:
    return {
        "max_iter": int(config.get("logit_max_iter", 2000)),
        "C": float(config.get("logit_c", 1.0)),
        "class_weight": config.get("logit_class_weight", "balanced"),
        "random_state": int(config.get("random_state", 42)),
    }


def get_rf_params(config: dict) -> dict:
    return {
        "n_estimators": int(config.get("rf_n_estimators", 300)),
        "max_depth": config.get("rf_max_depth", 8),
        "min_samples_leaf": int(config.get("rf_min_samples_leaf", 50)),
        "max_features": config.get("rf_max_features", "sqrt"),
        "class_weight": config.get("rf_class_weight", "balanced_subsample"),
        "n_jobs": int(config.get("rf_n_jobs", -1)),
        "random_state": int(config.get("random_state", 42)),
    }


def _positive_class_probability(model, x, positive_class: int = 1) -> np.ndarray:
    classes = list(model.classes_)
    if positive_class not in classes:
        return np.zeros(len(x), dtype=float)
    return model.predict_proba(x)[:, classes.index(positive_class)]


def binary_signal_metrics(y_true, prob, prefix: str) -> dict:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    prob = np.asarray(prob).reshape(-1)
    pred_05 = (prob >= 0.5).astype(int)

    return {
        f"{prefix}_auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else np.nan,
        f"{prefix}_positive_rate": float(np.mean(y_true)),
        f"{prefix}_prob_mean": float(np.mean(prob)),
        f"{prefix}_prob_std": float(np.std(prob)),
        f"{prefix}_prob_min": float(np.min(prob)),
        f"{prefix}_prob_max": float(np.max(prob)),
        f"{prefix}_acc_05": float(accuracy_score(y_true, pred_05)),
        f"{prefix}_precision_05": float(precision_score(y_true, pred_05, zero_division=0)),
        f"{prefix}_recall_05": float(recall_score(y_true, pred_05, zero_division=0)),
        f"{prefix}_f1_05": float(f1_score(y_true, pred_05, zero_division=0)),
        f"{prefix}_q90_threshold": float(np.quantile(prob, 0.90)),
        f"{prefix}_q95_threshold": float(np.quantile(prob, 0.95)),
    }


def build_signal_group_analysis(signal_df: pd.DataFrame) -> pd.DataFrame:
    group_rows = []
    for side, prob_col, label_col in [
        ("buy", "buy_prob", "buy_label_true"),
        ("sell", "sell_prob", "sell_label_true"),
    ]:
        tmp = signal_df.copy()
        tmp["prob_group"] = pd.qcut(tmp[prob_col], q=10, labels=False, duplicates="drop")
        grouped = tmp.groupby("prob_group", observed=True).agg(
            sample_count=(prob_col, "size"),
            pred_prob_mean=(prob_col, "mean"),
            label_mean=(label_col, "mean"),
            trade_return_mean=("trade_return", "mean"),
            future_return_mean=("future_return", "mean"),
        ).reset_index()
        grouped.insert(0, "side", side)
        group_rows.append(grouped)
    return pd.concat(group_rows, ignore_index=True)


def build_signal_threshold_diagnostics(
    signal_df: pd.DataFrame,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    quantiles = quantiles or [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    rows = []
    for side, prob_col, label_col in [
        ("buy", "buy_prob", "buy_label_true"),
        ("sell", "sell_prob", "sell_label_true"),
    ]:
        for q in quantiles:
            threshold = signal_df[prob_col].quantile(q)
            active = signal_df[signal_df[prob_col] >= threshold]
            rows.append({
                "side": side,
                "quantile": float(q),
                "threshold": float(threshold),
                "signal_count": int(len(active)),
                "signal_ratio": float(len(active) / len(signal_df)) if len(signal_df) else np.nan,
                "active_label_mean": float(active[label_col].mean()) if len(active) else np.nan,
                "active_trade_return_mean": float(active["trade_return"].mean()) if len(active) else np.nan,
                "active_future_return_mean": float(active["future_return"].mean()) if len(active) else np.nan,
            })
    return pd.DataFrame(rows)


def train_dual_sklearn_signals(
    train_scaled: pd.DataFrame,
    valid_scaled: pd.DataFrame,
    test_scaled: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    buy_target_col: str,
    sell_target_col: str,
    valid_index_seq,
    test_index_seq,
    config: dict,
    model_name: str,
    buy_model,
    sell_model,
    model_params: dict,
    output_dir: str | Path | None = None,
) -> dict:
    safe_model_name = model_name.lower().replace(" ", "_")
    output_dir = Path(output_dir or Path(config.get("diagnostics_dir", "outputs/diagnostics")) / "model_signals" / safe_model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_index = pd.to_datetime(valid_index_seq)
    test_index = pd.to_datetime(test_index_seq)

    x_train = train_scaled[feature_cols]
    x_valid = valid_scaled.loc[valid_index, feature_cols]
    x_test = test_scaled.loc[test_index, feature_cols]

    y_train_buy = train_scaled[buy_target_col].astype(int)
    y_valid_buy = valid_scaled.loc[valid_index, buy_target_col].astype(int)
    y_test_buy = test_scaled.loc[test_index, buy_target_col].astype(int)

    y_train_sell = train_scaled[sell_target_col].astype(int)
    y_valid_sell = valid_scaled.loc[valid_index, sell_target_col].astype(int)
    y_test_sell = test_scaled.loc[test_index, sell_target_col].astype(int)

    buy_model.fit(x_train, y_train_buy)
    sell_model.fit(x_train, y_train_sell)

    valid_buy_prob = _positive_class_probability(buy_model, x_valid, positive_class=1)
    test_buy_prob = _positive_class_probability(buy_model, x_test, positive_class=1)
    valid_sell_prob = _positive_class_probability(sell_model, x_valid, positive_class=1)
    test_sell_prob = _positive_class_probability(sell_model, x_test, positive_class=1)

    summary = {
        "model": model_name,
        "target_col_buy": buy_target_col,
        "target_col_sell": sell_target_col,
        "feature_count": len(feature_cols),
        "lookback_aligned_to_sequence_models": int(config.get("lookback", 0)),
        "params": model_params,
    }
    summary.update(binary_signal_metrics(y_valid_buy.values, valid_buy_prob, "valid_buy"))
    summary.update(binary_signal_metrics(y_test_buy.values, test_buy_prob, "test_buy"))
    summary.update(binary_signal_metrics(y_valid_sell.values, valid_sell_prob, "valid_sell"))
    summary.update(binary_signal_metrics(y_test_sell.values, test_sell_prob, "test_sell"))

    valid_signals = valid_df.loc[valid_index].copy()
    valid_signals["buy_prob"] = valid_buy_prob
    valid_signals["sell_prob"] = valid_sell_prob
    valid_signals["buy_label_true"] = y_valid_buy.values
    valid_signals["sell_label_true"] = y_valid_sell.values

    test_signals = test_df.loc[test_index].copy()
    test_signals["buy_prob"] = test_buy_prob
    test_signals["sell_prob"] = test_sell_prob
    test_signals["buy_label_true"] = y_test_buy.values
    test_signals["sell_label_true"] = y_test_sell.values

    signal_cols = ["buy_prob", "sell_prob", "buy_label_true", "sell_label_true", "trade_return", "future_return"]
    valid_signals[signal_cols].to_csv(output_dir / "valid_signals.csv")
    test_signals[signal_cols].to_csv(output_dir / "test_signals.csv")

    group_analysis = build_signal_group_analysis(test_signals)
    group_analysis.to_csv(output_dir / "signal_group_analysis.csv", index=False)

    threshold_diagnostics = build_signal_threshold_diagnostics(test_signals)
    threshold_diagnostics.to_csv(output_dir / "signal_threshold_diagnostics.csv", index=False)

    with open(output_dir / "signal_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "buy_model": buy_model,
        "sell_model": sell_model,
        "valid_signals": valid_signals,
        "test_signals": test_signals,
        "group_analysis": group_analysis,
        "threshold_diagnostics": threshold_diagnostics,
        "summary": summary,
        "output_dir": output_dir,
    }


def train_dual_logistic_signals(
    train_scaled: pd.DataFrame,
    valid_scaled: pd.DataFrame,
    test_scaled: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    buy_target_col: str,
    sell_target_col: str,
    valid_index_seq,
    test_index_seq,
    config: dict,
    output_dir: str | Path | None = None,
) -> dict:
    params = get_logit_params(config)
    result = train_dual_sklearn_signals(
        train_scaled=train_scaled,
        valid_scaled=valid_scaled,
        test_scaled=test_scaled,
        valid_df=valid_df,
        test_df=test_df,
        feature_cols=feature_cols,
        buy_target_col=buy_target_col,
        sell_target_col=sell_target_col,
        valid_index_seq=valid_index_seq,
        test_index_seq=test_index_seq,
        config=config,
        model_name="LogisticRegression",
        buy_model=LogisticRegression(**params),
        sell_model=LogisticRegression(**params),
        model_params=params,
        output_dir=output_dir,
    )
    with open(result["output_dir"] / "logit_signal_summary.json", "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, ensure_ascii=False, indent=2)
    return result


def train_dual_random_forest_signals(
    train_scaled: pd.DataFrame,
    valid_scaled: pd.DataFrame,
    test_scaled: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    buy_target_col: str,
    sell_target_col: str,
    valid_index_seq,
    test_index_seq,
    config: dict,
    output_dir: str | Path | None = None,
) -> dict:
    params = get_rf_params(config)
    result = train_dual_sklearn_signals(
        train_scaled=train_scaled,
        valid_scaled=valid_scaled,
        test_scaled=test_scaled,
        valid_df=valid_df,
        test_df=test_df,
        feature_cols=feature_cols,
        buy_target_col=buy_target_col,
        sell_target_col=sell_target_col,
        valid_index_seq=valid_index_seq,
        test_index_seq=test_index_seq,
        config=config,
        model_name="RandomForestClassifier",
        buy_model=RandomForestClassifier(**params),
        sell_model=RandomForestClassifier(**params),
        model_params=params,
        output_dir=output_dir,
    )

    buy_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": result["buy_model"].feature_importances_,
        "side": "buy",
    })
    sell_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": result["sell_model"].feature_importances_,
        "side": "sell",
    })
    feature_importance = pd.concat([buy_importance, sell_importance], ignore_index=True)
    feature_importance = feature_importance.sort_values(["side", "importance"], ascending=[True, False])
    feature_importance.to_csv(result["output_dir"] / "feature_importance.csv", index=False)
    result["feature_importance"] = feature_importance
    result["summary"]["feature_importance_file"] = str(result["output_dir"] / "feature_importance.csv")
    with open(result["output_dir"] / "signal_summary.json", "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, ensure_ascii=False, indent=2)
    return result


def _require_torch() -> None:
    if torch is None or nn is None or DataLoader is None:
        raise ImportError("PyTorch is required for sequence signal models.")


class SeqArrayDataset(Dataset):
    def __init__(self, x, y):
        _require_torch()
        self.x = torch.tensor(np.asarray(x), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(y).reshape(-1), dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class LSTMSequenceClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.3, output_dim=1):
        _require_torch()
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.dropout(out[:, -1, :])
        logits = self.fc(h)
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits


class TransformerLSTMSequenceClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=128,
        nhead=4,
        transformer_layers=2,
        transformer_dropout=0.2,
        lstm_hidden_dim=128,
        lstm_num_layers=1,
        output_dim=1,
    ):
        _require_torch()
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=transformer_dropout if lstm_num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(transformer_dropout)
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0)
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        h = self.input_norm(self.input_proj(x))
        h = self.transformer(h)
        out, _ = self.lstm(h)
        last_hidden = self.dropout(out[:, -1, :])
        logits = self.fc(last_hidden)
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits


class CNNSequenceClassifier(nn.Module):
    def __init__(self, input_dim, channels=64, kernel_size=3, dropout=0.3, output_dim=1):
        _require_torch()
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(channels, output_dim)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        h = self.net(x.transpose(1, 2)).squeeze(-1)
        logits = self.fc(h)
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits


class MLPSequenceClassifier(nn.Module):
    def __init__(self, input_dim, lookback, hidden_dim=256, dropout=0.3, output_dim=1):
        _require_torch()
        super().__init__()
        flat_dim = input_dim * lookback
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 8)),
            nn.BatchNorm1d(max(hidden_dim // 2, 8)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, 8), output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        logits = self.net(x)
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits


def _resolve_device(device: str | None = None) -> str:
    _require_torch()
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _init_binary_output_bias(model, y_train, device: str) -> tuple[float, float]:
    y_train = np.asarray(y_train).reshape(-1).astype(float)
    pos_rate = float(np.clip(np.mean(y_train), 1e-6, 1 - 1e-6))
    init_bias = float(np.log(pos_rate / (1 - pos_rate)))
    pos_weight = float((1 - pos_rate) / pos_rate)

    with torch.no_grad():
        linear = None
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            linear = model.fc
        elif hasattr(model, "net"):
            for module in reversed(list(model.net.modules())):
                if isinstance(module, nn.Linear):
                    linear = module
                    break
        if linear is not None and linear.bias is not None and linear.bias.numel() == 1:
            linear.bias.fill_(init_bias)

    return pos_rate, pos_weight


def predict_torch_binary(model, x, batch_size: int = 256, device: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    _require_torch()
    device = _resolve_device(device)
    loader = DataLoader(SeqArrayDataset(x, np.zeros(len(x))), batch_size=batch_size, shuffle=False)
    model.eval()
    probs = []
    logits_all = []
    with torch.no_grad():
        for x_batch, _ in loader:
            logits = model(x_batch.to(device))
            logits = logits.reshape(-1)
            prob = torch.sigmoid(logits)
            logits_all.append(logits.detach().cpu().numpy())
            probs.append(prob.detach().cpu().numpy())
    return np.concatenate(probs), np.concatenate(logits_all)


def train_binary_sequence_model(
    model,
    x_train,
    y_train,
    x_valid,
    y_valid,
    config: dict,
    device: str | None = None,
    model_label: str = "sequence_model",
) -> tuple[object, pd.DataFrame, dict]:
    _require_torch()
    device = _resolve_device(device)
    model = model.to(device)

    batch_size_train = int(config.get("batch_size_train", 128))
    batch_size_eval = int(config.get("batch_size_eval", 256))
    train_loader = DataLoader(SeqArrayDataset(x_train, y_train), batch_size=batch_size_train, shuffle=False)
    valid_loader = DataLoader(SeqArrayDataset(x_valid, y_valid), batch_size=batch_size_eval, shuffle=False)

    pos_rate, pos_weight = _init_binary_output_bias(model, y_train, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("learning_rate", 5e-4)),
        weight_decay=float(config.get("weight_decay", 1e-6)),
    )

    best_score = -np.inf
    best_state = None
    wait = 0
    history_rows = []

    max_epochs = int(config.get("max_epochs", 100))
    patience = int(config.get("early_stop_patience", 10))
    grad_clip_norm = float(config.get("grad_clip_norm", 1.0))

    for epoch in range(max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch).reshape(-1)
            loss = criterion(logits, y_batch.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(x_batch)
            train_count += len(x_batch)

        valid_loss_sum = 0.0
        valid_count = 0
        model.eval()
        valid_probs = []
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch).reshape(-1)
                loss = criterion(logits, y_batch.float())
                valid_loss_sum += float(loss.item()) * len(x_batch)
                valid_count += len(x_batch)
                valid_probs.append(torch.sigmoid(logits).detach().cpu().numpy())

        valid_prob = np.concatenate(valid_probs)
        y_valid_array = np.asarray(y_valid).reshape(-1).astype(int)
        valid_auc = float(roc_auc_score(y_valid_array, valid_prob)) if len(np.unique(y_valid_array)) > 1 else np.nan
        valid_loss = valid_loss_sum / max(valid_count, 1)
        score = valid_auc if np.isfinite(valid_auc) else -valid_loss

        row = {
            "model_label": model_label,
            "epoch": epoch + 1,
            "train_loss": train_loss_sum / max(train_count, 1),
            "valid_loss": valid_loss,
            "valid_auc": valid_auc,
            "valid_prob_mean": float(valid_prob.mean()),
            "valid_prob_max": float(valid_prob.max()),
        }
        history_rows.append(row)

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    fit_info = {
        "best_score": float(best_score),
        "best_valid_auc": float(max([r["valid_auc"] for r in history_rows if np.isfinite(r["valid_auc"])], default=np.nan)),
        "epochs_ran": int(len(history_rows)),
        "pos_rate": float(pos_rate),
        "pos_weight": float(pos_weight),
        "device": device,
    }
    return model, pd.DataFrame(history_rows), fit_info


def train_dual_torch_sequence_signals(
    x_train_seq,
    y_train_buy_seq,
    y_train_sell_seq,
    x_valid_seq,
    y_valid_buy_seq,
    y_valid_sell_seq,
    x_test_seq,
    y_test_buy_seq,
    y_test_sell_seq,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    valid_index_seq,
    test_index_seq,
    config: dict,
    model_name: str,
    buy_model,
    sell_model,
    model_params: dict,
    output_dir: str | Path | None = None,
    device: str | None = None,
) -> dict:
    _require_torch()
    safe_model_name = model_name.lower().replace(" ", "_")
    output_dir = Path(output_dir or Path(config.get("diagnostics_dir", "outputs/diagnostics")) / "model_signals" / safe_model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(device)

    buy_model, buy_history, buy_fit = train_binary_sequence_model(
        buy_model,
        x_train_seq,
        y_train_buy_seq,
        x_valid_seq,
        y_valid_buy_seq,
        config,
        device=device,
        model_label=f"{model_name}_buy",
    )
    sell_model, sell_history, sell_fit = train_binary_sequence_model(
        sell_model,
        x_train_seq,
        y_train_sell_seq,
        x_valid_seq,
        y_valid_sell_seq,
        config,
        device=device,
        model_label=f"{model_name}_sell",
    )

    batch_size_eval = int(config.get("batch_size_eval", 256))
    valid_buy_prob, valid_buy_logits = predict_torch_binary(buy_model, x_valid_seq, batch_size_eval, device)
    test_buy_prob, test_buy_logits = predict_torch_binary(buy_model, x_test_seq, batch_size_eval, device)
    valid_sell_prob, valid_sell_logits = predict_torch_binary(sell_model, x_valid_seq, batch_size_eval, device)
    test_sell_prob, test_sell_logits = predict_torch_binary(sell_model, x_test_seq, batch_size_eval, device)

    y_valid_buy = np.asarray(y_valid_buy_seq).reshape(-1).astype(int)
    y_test_buy = np.asarray(y_test_buy_seq).reshape(-1).astype(int)
    y_valid_sell = np.asarray(y_valid_sell_seq).reshape(-1).astype(int)
    y_test_sell = np.asarray(y_test_sell_seq).reshape(-1).astype(int)

    summary = {
        "model": model_name,
        "target_col_buy": config.get("buy_label_col", "buy_label"),
        "target_col_sell": config.get("sell_label_col", "sell_label"),
        "feature_count": int(np.asarray(x_train_seq).shape[-1]),
        "lookback": int(config.get("lookback", np.asarray(x_train_seq).shape[1])),
        "params": model_params,
        "buy_fit": buy_fit,
        "sell_fit": sell_fit,
    }
    summary.update(binary_signal_metrics(y_valid_buy, valid_buy_prob, "valid_buy"))
    summary.update(binary_signal_metrics(y_test_buy, test_buy_prob, "test_buy"))
    summary.update(binary_signal_metrics(y_valid_sell, valid_sell_prob, "valid_sell"))
    summary.update(binary_signal_metrics(y_test_sell, test_sell_prob, "test_sell"))

    valid_index = pd.to_datetime(valid_index_seq)
    test_index = pd.to_datetime(test_index_seq)
    valid_signals = valid_df.loc[valid_index].copy()
    valid_signals["buy_prob"] = valid_buy_prob
    valid_signals["sell_prob"] = valid_sell_prob
    valid_signals["buy_logit"] = valid_buy_logits
    valid_signals["sell_logit"] = valid_sell_logits
    valid_signals["buy_label_true"] = y_valid_buy
    valid_signals["sell_label_true"] = y_valid_sell

    test_signals = test_df.loc[test_index].copy()
    test_signals["buy_prob"] = test_buy_prob
    test_signals["sell_prob"] = test_sell_prob
    test_signals["buy_logit"] = test_buy_logits
    test_signals["sell_logit"] = test_sell_logits
    test_signals["buy_label_true"] = y_test_buy
    test_signals["sell_label_true"] = y_test_sell

    signal_cols = [
        "buy_prob",
        "sell_prob",
        "buy_logit",
        "sell_logit",
        "buy_label_true",
        "sell_label_true",
        "trade_return",
        "future_return",
    ]
    valid_signals[[c for c in signal_cols if c in valid_signals.columns]].to_csv(output_dir / "valid_signals.csv")
    test_signals[[c for c in signal_cols if c in test_signals.columns]].to_csv(output_dir / "test_signals.csv")

    group_analysis = build_signal_group_analysis(test_signals)
    threshold_diagnostics = build_signal_threshold_diagnostics(test_signals)
    group_analysis.to_csv(output_dir / "signal_group_analysis.csv", index=False)
    threshold_diagnostics.to_csv(output_dir / "signal_threshold_diagnostics.csv", index=False)
    buy_history.to_csv(output_dir / "buy_training_history.csv", index=False)
    sell_history.to_csv(output_dir / "sell_training_history.csv", index=False)
    torch.save(buy_model.state_dict(), output_dir / "buy_model.pt")
    torch.save(sell_model.state_dict(), output_dir / "sell_model.pt")
    with open(output_dir / "signal_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "buy_model": buy_model,
        "sell_model": sell_model,
        "valid_signals": valid_signals,
        "test_signals": test_signals,
        "group_analysis": group_analysis,
        "threshold_diagnostics": threshold_diagnostics,
        "summary": summary,
        "buy_history": buy_history,
        "sell_history": sell_history,
        "output_dir": output_dir,
    }


def train_dual_lstm_signals(
    x_train_seq,
    y_train_buy_seq,
    y_train_sell_seq,
    x_valid_seq,
    y_valid_buy_seq,
    y_valid_sell_seq,
    x_test_seq,
    y_test_buy_seq,
    y_test_sell_seq,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    valid_index_seq,
    test_index_seq,
    config: dict,
    output_dir: str | Path | None = None,
    device: str | None = None,
) -> dict:
    params = {
        "hidden_dim": int(config.get("lstm_hidden_dim", 256)),
        "num_layers": int(config.get("lstm_num_layers", 2)),
        "dropout": float(config.get("lstm_dropout", 0.3)),
        "output_dim": 1,
    }
    input_dim = int(np.asarray(x_train_seq).shape[-1])
    return train_dual_torch_sequence_signals(
        x_train_seq=x_train_seq,
        y_train_buy_seq=y_train_buy_seq,
        y_train_sell_seq=y_train_sell_seq,
        x_valid_seq=x_valid_seq,
        y_valid_buy_seq=y_valid_buy_seq,
        y_valid_sell_seq=y_valid_sell_seq,
        x_test_seq=x_test_seq,
        y_test_buy_seq=y_test_buy_seq,
        y_test_sell_seq=y_test_sell_seq,
        valid_df=valid_df,
        test_df=test_df,
        valid_index_seq=valid_index_seq,
        test_index_seq=test_index_seq,
        config=config,
        model_name="LSTM",
        buy_model=LSTMSequenceClassifier(input_dim=input_dim, **params),
        sell_model=LSTMSequenceClassifier(input_dim=input_dim, **params),
        model_params=params,
        output_dir=output_dir,
        device=device,
    )


def train_dual_transformer_lstm_signals(
    x_train_seq,
    y_train_buy_seq,
    y_train_sell_seq,
    x_valid_seq,
    y_valid_buy_seq,
    y_valid_sell_seq,
    x_test_seq,
    y_test_buy_seq,
    y_test_sell_seq,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    valid_index_seq,
    test_index_seq,
    config: dict,
    output_dir: str | Path | None = None,
    device: str | None = None,
) -> dict:
    params = {
        "d_model": int(config.get("transformer_d_model", 128)),
        "nhead": int(config.get("transformer_nhead", 4)),
        "transformer_layers": int(config.get("transformer_num_layers", 2)),
        "transformer_dropout": float(config.get("transformer_dropout", 0.2)),
        "lstm_hidden_dim": int(config.get("transformer_lstm_hidden_dim", 128)),
        "lstm_num_layers": int(config.get("transformer_lstm_num_layers", 1)),
        "output_dim": 1,
    }
    input_dim = int(np.asarray(x_train_seq).shape[-1])
    return train_dual_torch_sequence_signals(
        x_train_seq=x_train_seq,
        y_train_buy_seq=y_train_buy_seq,
        y_train_sell_seq=y_train_sell_seq,
        x_valid_seq=x_valid_seq,
        y_valid_buy_seq=y_valid_buy_seq,
        y_valid_sell_seq=y_valid_sell_seq,
        x_test_seq=x_test_seq,
        y_test_buy_seq=y_test_buy_seq,
        y_test_sell_seq=y_test_sell_seq,
        valid_df=valid_df,
        test_df=test_df,
        valid_index_seq=valid_index_seq,
        test_index_seq=test_index_seq,
        config=config,
        model_name="Transformer_LSTM",
        buy_model=TransformerLSTMSequenceClassifier(input_dim=input_dim, **params),
        sell_model=TransformerLSTMSequenceClassifier(input_dim=input_dim, **params),
        model_params=params,
        output_dir=output_dir,
        device=device,
    )


def train_dual_cnn_signals(
    x_train_seq,
    y_train_buy_seq,
    y_train_sell_seq,
    x_valid_seq,
    y_valid_buy_seq,
    y_valid_sell_seq,
    x_test_seq,
    y_test_buy_seq,
    y_test_sell_seq,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    valid_index_seq,
    test_index_seq,
    config: dict,
    output_dir: str | Path | None = None,
    device: str | None = None,
) -> dict:
    params = {
        "channels": int(config.get("cnn_channels", 64)),
        "kernel_size": int(config.get("cnn_kernel_size", 3)),
        "dropout": float(config.get("cnn_dropout", 0.3)),
        "output_dim": 1,
    }
    input_dim = int(np.asarray(x_train_seq).shape[-1])
    return train_dual_torch_sequence_signals(
        x_train_seq=x_train_seq,
        y_train_buy_seq=y_train_buy_seq,
        y_train_sell_seq=y_train_sell_seq,
        x_valid_seq=x_valid_seq,
        y_valid_buy_seq=y_valid_buy_seq,
        y_valid_sell_seq=y_valid_sell_seq,
        x_test_seq=x_test_seq,
        y_test_buy_seq=y_test_buy_seq,
        y_test_sell_seq=y_test_sell_seq,
        valid_df=valid_df,
        test_df=test_df,
        valid_index_seq=valid_index_seq,
        test_index_seq=test_index_seq,
        config=config,
        model_name="CNN",
        buy_model=CNNSequenceClassifier(input_dim=input_dim, **params),
        sell_model=CNNSequenceClassifier(input_dim=input_dim, **params),
        model_params=params,
        output_dir=output_dir,
        device=device,
    )


def train_dual_mlp_signals(
    x_train_seq,
    y_train_buy_seq,
    y_train_sell_seq,
    x_valid_seq,
    y_valid_buy_seq,
    y_valid_sell_seq,
    x_test_seq,
    y_test_buy_seq,
    y_test_sell_seq,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    valid_index_seq,
    test_index_seq,
    config: dict,
    output_dir: str | Path | None = None,
    device: str | None = None,
) -> dict:
    params = {
        "lookback": int(config.get("lookback", np.asarray(x_train_seq).shape[1])),
        "hidden_dim": int(config.get("mlp_hidden_dim", 256)),
        "dropout": float(config.get("mlp_dropout", 0.3)),
        "output_dim": 1,
    }
    input_dim = int(np.asarray(x_train_seq).shape[-1])
    return train_dual_torch_sequence_signals(
        x_train_seq=x_train_seq,
        y_train_buy_seq=y_train_buy_seq,
        y_train_sell_seq=y_train_sell_seq,
        x_valid_seq=x_valid_seq,
        y_valid_buy_seq=y_valid_buy_seq,
        y_valid_sell_seq=y_valid_sell_seq,
        x_test_seq=x_test_seq,
        y_test_buy_seq=y_test_buy_seq,
        y_test_sell_seq=y_test_sell_seq,
        valid_df=valid_df,
        test_df=test_df,
        valid_index_seq=valid_index_seq,
        test_index_seq=test_index_seq,
        config=config,
        model_name="MLP",
        buy_model=MLPSequenceClassifier(input_dim=input_dim, **params),
        sell_model=MLPSequenceClassifier(input_dim=input_dim, **params),
        model_params=params,
        output_dir=output_dir,
        device=device,
    )
