from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


DEFAULT_PERIODS_PER_YEAR = 48 * 242
DEFAULT_THRESHOLD_QUANTILES = [0.95]


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.1, output_dim=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
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
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        if logits.shape[-1] == 1:
            return logits.squeeze(-1)
        return logits


def is_multiclass_config(config):
    return int(config.get("num_classes", 2)) > 2


def get_buy_class(config):
    return int(config.get("buy_class", 2 if is_multiclass_config(config) else 1))


def get_sell_class(config):
    return int(config.get("sell_class", 0))


def get_hold_class(config):
    return int(config.get("hold_class", 1))


def make_sequence_data(data, feature_cols, target_col, lookback):
    X = data[feature_cols].values
    y = data[target_col].values

    X_seq = []
    y_seq = []
    index_seq = []
    for i in range(lookback, len(data)):
        X_seq.append(X[i - lookback : i])
        y_seq.append(y[i])
        index_seq.append(data.index[i])

    return np.array(X_seq), np.array(y_seq), np.array(index_seq)


def safe_auc_score(y_true, y_prob):
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def safe_action_metrics(y_true, prob_matrix, config):
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    prob_matrix = np.asarray(prob_matrix)
    pred_class = prob_matrix.argmax(axis=1)
    buy_class = get_buy_class(config)
    sell_class = get_sell_class(config)
    metrics = {
        "accuracy": float((pred_class == y_true).mean()),
        "buy_auc": safe_auc_score((y_true == buy_class).astype(int), prob_matrix[:, buy_class]),
        "sell_auc": safe_auc_score((y_true == sell_class).astype(int), prob_matrix[:, sell_class]),
        "buy_positive_rate": float((y_true == buy_class).mean()),
        "sell_positive_rate": float((y_true == sell_class).mean()),
        "hold_rate": float((y_true == get_hold_class(config)).mean()),
    }
    return metrics


def calc_max_drawdown(equity):
    if len(equity) == 0:
        return np.nan
    peak = equity.cummax()
    return (equity / peak - 1).min()


def calc_sharpe_from_trade_returns(ret, periods_per_year=None):
    std = ret.std()
    if std == 0 or pd.isna(std):
        return np.nan
    sharpe = ret.mean() / std
    if periods_per_year is not None:
        sharpe *= np.sqrt(periods_per_year)
    return sharpe


def calc_trade_frequency_stats(trades_df, periods_per_year=DEFAULT_PERIODS_PER_YEAR):
    if len(trades_df) < 2:
        return {
            "per_trade_sharpe": np.nan,
            "trades_per_year": np.nan,
            "bar_annualized_sharpe": np.nan,
            "sharpe": np.nan,
        }

    ret = trades_df["net_return"]
    per_trade_sharpe = calc_sharpe_from_trade_returns(ret)
    bar_annualized_sharpe = calc_sharpe_from_trade_returns(ret, periods_per_year)

    exit_time = pd.to_datetime(trades_df["exit_time"])
    duration_years = (exit_time.max() - exit_time.min()).total_seconds() / (365.25 * 24 * 60 * 60)
    if duration_years <= 0 or pd.isna(duration_years):
        trades_per_year = np.nan
        trade_frequency_sharpe = np.nan
    else:
        trades_per_year = len(trades_df) / duration_years
        trade_frequency_sharpe = per_trade_sharpe * np.sqrt(trades_per_year)

    return {
        "per_trade_sharpe": float(per_trade_sharpe),
        "trades_per_year": float(trades_per_year) if pd.notna(trades_per_year) else np.nan,
        "bar_annualized_sharpe": float(bar_annualized_sharpe),
        "sharpe": float(trade_frequency_sharpe) if pd.notna(trade_frequency_sharpe) else np.nan,
    }


def run_fixed_horizon_event_backtest(
    data,
    threshold,
    horizon,
    round_trip_cost,
    prob_col="pred_prob",
    entry_price_col="open",
    exit_price_col="close",
    no_overnight=True,
    periods_per_year=DEFAULT_PERIODS_PER_YEAR,
):
    data = data.sort_index().copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data index must be DatetimeIndex")

    required_cols = [prob_col, entry_price_col, exit_price_col]
    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        raise ValueError(f"missing columns: {missing_cols}")

    trades = []
    n = len(data)
    i = 0

    while i < n - horizon:
        prob = data[prob_col].iloc[i]
        if pd.isna(prob) or prob < threshold:
            i += 1
            continue

        signal_time = data.index[i]
        entry_i = i + 1
        exit_i = i + horizon
        if exit_i >= n:
            break

        entry_time = data.index[entry_i]
        exit_time = data.index[exit_i]
        if no_overnight and (
            signal_time.normalize() != entry_time.normalize()
            or entry_time.normalize() != exit_time.normalize()
        ):
            i += 1
            continue

        entry_price = data[entry_price_col].iloc[entry_i]
        exit_price = data[exit_price_col].iloc[exit_i]
        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
            i += 1
            continue

        gross_return = exit_price / entry_price - 1
        net_return = gross_return - round_trip_cost
        trades.append(
            {
                "signal_time": signal_time,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "holding_bars": horizon,
                "pred_prob": float(prob),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "gross_return": float(gross_return),
                "round_trip_cost": float(round_trip_cost),
                "net_return": float(net_return),
                "y_true": float(data["y_true"].iloc[i]) if "y_true" in data.columns else np.nan,
            }
        )

        # 不允许持仓重叠，下一次从出场后的下一根K线继续找信号。
        i = exit_i + 1

    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0:
        equity = pd.Series(dtype=float, name="equity")
        stats = {
            "threshold": threshold,
            "horizon": horizon,
            "round_trip_cost": round_trip_cost,
            "trade_count": 0,
            "win_rate": np.nan,
            "gross_return_mean": np.nan,
            "net_return_mean": np.nan,
            "total_return": 0.0,
            "max_drawdown": np.nan,
            "per_trade_sharpe": np.nan,
            "trades_per_year": np.nan,
            "bar_annualized_sharpe": np.nan,
            "sharpe": np.nan,
        }
        return trades_df, equity, stats

    trades_df["equity"] = (1 + trades_df["net_return"]).cumprod()
    equity = trades_df.set_index("exit_time")["equity"]
    trade_frequency_stats = calc_trade_frequency_stats(trades_df, periods_per_year)

    stats = {
        "threshold": float(threshold),
        "horizon": int(horizon),
        "round_trip_cost": float(round_trip_cost),
        "trade_count": int(len(trades_df)),
        "win_rate": float((trades_df["net_return"] > 0).mean()),
        "gross_return_mean": float(trades_df["gross_return"].mean()),
        "net_return_mean": float(trades_df["net_return"].mean()),
        "gross_return_median": float(trades_df["gross_return"].median()),
        "net_return_median": float(trades_df["net_return"].median()),
        "total_return": float(trades_df["equity"].iloc[-1] - 1),
        "max_drawdown": float(calc_max_drawdown(trades_df["equity"])),
        "per_trade_sharpe": trade_frequency_stats["per_trade_sharpe"],
        "trades_per_year": trade_frequency_stats["trades_per_year"],
        "bar_annualized_sharpe": trade_frequency_stats["bar_annualized_sharpe"],
        "sharpe": trade_frequency_stats["sharpe"],
        "avg_pred_prob": float(trades_df["pred_prob"].mean()),
        "positive_label_rate": float((trades_df["y_true"] > 0).mean()) if "y_true" in trades_df else np.nan,
        "buy_label_rate": float((trades_df["y_true"] == 2).mean()) if "y_true" in trades_df else np.nan,
    }
    return trades_df, equity, stats


def _resolve_device(device=None):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_horizon_dataset(base_df, feature_cols, target_col, config, horizon, lookback):
    label_threshold = config.get("label_threshold", config.get("one_side_cost", config["round_trip_cost"]))
    data = base_df.copy().sort_index()
    data["future_close"] = data["close"].shift(-horizon)
    data["future_entry_open"] = data["open"].shift(-1)
    data["future_return"] = data["future_close"] / data["close"] - 1
    data["trade_return"] = data["future_close"] / data["future_entry_open"] - 1
    if is_multiclass_config(config):
        data[target_col] = get_hold_class(config)
        data.loc[data["trade_return"] > label_threshold, target_col] = get_buy_class(config)
        data.loc[data["trade_return"] < -label_threshold, target_col] = get_sell_class(config)
        data[target_col] = data[target_col].astype(int)
    else:
        data[target_col] = (data["trade_return"] > label_threshold).astype(int)

    needed_cols = feature_cols + [target_col, "future_return", "trade_return", "future_close", "future_entry_open"]
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=needed_cols).copy()

    n_local = len(data)
    train_end_local = int(n_local * config["train_ratio"])
    valid_end_local = int(n_local * (config["train_ratio"] + config["valid_ratio"]))
    train_local = data.iloc[:train_end_local].copy()
    valid_local = data.iloc[train_end_local:valid_end_local].copy()
    test_local = data.iloc[valid_end_local:].copy()

    local_clip_bounds = {}
    for col in feature_cols:
        local_clip_bounds[col] = (train_local[col].quantile(0.001), train_local[col].quantile(0.999))
    for col in feature_cols:
        lower, upper = local_clip_bounds[col]
        train_local[col] = train_local[col].clip(lower, upper)
        valid_local[col] = valid_local[col].clip(lower, upper)
        test_local[col] = test_local[col].clip(lower, upper)

    local_scaler = StandardScaler()
    train_scaled_local = train_local.copy()
    valid_scaled_local = valid_local.copy()
    test_scaled_local = test_local.copy()
    train_scaled_local[feature_cols] = local_scaler.fit_transform(train_local[feature_cols])
    valid_scaled_local[feature_cols] = local_scaler.transform(valid_local[feature_cols])
    test_scaled_local[feature_cols] = local_scaler.transform(test_local[feature_cols])

    X_train, y_train, train_idx = make_sequence_data(train_scaled_local, feature_cols, target_col, lookback)
    X_valid, y_valid, valid_idx = make_sequence_data(valid_scaled_local, feature_cols, target_col, lookback)
    X_test, y_test, test_idx = make_sequence_data(test_scaled_local, feature_cols, target_col, lookback)

    loaders = {
        "train": DataLoader(SeqDataset(X_train, y_train), batch_size=config["batch_size_train"], shuffle=False),
        "valid": DataLoader(SeqDataset(X_valid, y_valid), batch_size=config["batch_size_eval"], shuffle=False),
        "test": DataLoader(SeqDataset(X_test, y_test), batch_size=config["batch_size_eval"], shuffle=False),
    }

    return {
        "horizon": horizon,
        "lookback": lookback,
        "data": data,
        "train_df": train_local,
        "valid_df": valid_local,
        "test_df": test_local,
        "train_scaled": train_scaled_local,
        "valid_scaled": valid_scaled_local,
        "test_scaled": test_scaled_local,
        "X_train_seq": X_train,
        "y_train_seq": y_train,
        "train_index_seq": train_idx,
        "X_valid_seq": X_valid,
        "y_valid_seq": y_valid,
        "valid_index_seq": valid_idx,
        "X_test_seq": X_test,
        "y_test_seq": y_test,
        "test_index_seq": test_idx,
        "loaders": loaders,
    }


def predict_loader(model_obj, loader, device):
    model_obj.eval()
    preds, labels, logits_all = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model_obj(X_batch)
            if logits.ndim == 2 and logits.shape[1] > 1:
                prob = torch.softmax(logits, dim=1).cpu().numpy()
            else:
                prob = torch.sigmoid(logits).cpu().numpy()
            preds.append(prob)
            labels.extend(y_batch.numpy())
            logits_all.append(logits.cpu().numpy())

    if len(preds) and np.asarray(preds[0]).ndim == 2:
        pred_array = np.vstack(preds)
    else:
        pred_array = np.concatenate([np.asarray(p).reshape(-1) for p in preds]) if preds else np.array([])

    if len(logits_all) and np.asarray(logits_all[0]).ndim == 2:
        logits_array = np.vstack(logits_all)
    else:
        logits_array = np.concatenate([np.asarray(x).reshape(-1) for x in logits_all]) if logits_all else np.array([])

    return pred_array, np.asarray(labels), logits_array


def train_lstm_for_dataset(dataset_pack, feature_count, config, device=None, verbose=True):
    device = _resolve_device(device)
    horizon = dataset_pack["horizon"]
    lookback = dataset_pack["lookback"]

    model = LSTMClassifier(
        input_dim=feature_count,
        hidden_dim=config["lstm_hidden_dim"],
        num_layers=config["lstm_num_layers"],
        dropout=config["lstm_dropout"],
        output_dim=int(config.get("num_classes", 1 if not is_multiclass_config(config) else 3)),
    ).to(device)

    y_train = dataset_pack["y_train_seq"]
    if is_multiclass_config(config):
        y_train_int = np.asarray(y_train).astype(int)
        counts = np.bincount(y_train_int, minlength=int(config["num_classes"])).astype(float)
        priors = np.clip(counts / counts.sum(), 1e-6, 1)
        class_weights = counts.sum() / np.maximum(counts, 1)
        class_weights = class_weights / class_weights.mean()
        with torch.no_grad():
            model.fc.bias.copy_(torch.tensor(np.log(priors), dtype=torch.float32, device=device))
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
        )
        pos_rate = float(priors[get_buy_class(config)])
    else:
        pos_rate = np.clip(np.mean(y_train), 1e-6, 1 - 1e-6)
        init_bias = np.log(pos_rate / (1 - pos_rate))
        with torch.no_grad():
            model.fc.bias.fill_(init_bias)
        pos_weight = (1 - pos_rate) / pos_rate
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    best_auc = -np.inf
    best_state = None
    wait = 0
    history_rows = []

    if verbose:
        print(f"\n===== LOOKBACK={lookback}, HORIZON={horizon} =====")
        print("positive_rate_train:", pos_rate)

    for epoch in range(config["max_epochs"]):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in dataset_pack["loaders"]["train"]:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            if is_multiclass_config(config):
                loss = criterion(logits, y_batch.long())
            else:
                loss = criterion(logits, y_batch.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_clip_norm"])
            optimizer.step()
            total_loss += loss.item()

        train_prob, train_label, _ = predict_loader(model, dataset_pack["loaders"]["train"], device)
        valid_prob, valid_label, _ = predict_loader(model, dataset_pack["loaders"]["valid"], device)
        if is_multiclass_config(config):
            train_auc = safe_auc_score((train_label == get_buy_class(config)).astype(int), train_prob[:, get_buy_class(config)])
            valid_auc = safe_auc_score((valid_label == get_buy_class(config)).astype(int), valid_prob[:, get_buy_class(config)])
            valid_prob_for_log = valid_prob[:, get_buy_class(config)]
        else:
            train_auc = safe_auc_score(train_label, train_prob)
            valid_auc = safe_auc_score(valid_label, valid_prob)
            valid_prob_for_log = valid_prob
        history_rows.append(
            {
                "epoch": epoch + 1,
                "loss": float(total_loss),
                "train_auc": float(train_auc) if pd.notna(train_auc) else np.nan,
                "valid_auc": float(valid_auc) if pd.notna(valid_auc) else np.nan,
                "valid_prob_mean": float(np.mean(valid_prob_for_log)),
                "valid_prob_max": float(np.max(valid_prob_for_log)),
            }
        )

        if verbose:
            print(
                f"LB={lookback} H={horizon} Epoch {epoch + 1}, "
                f"Loss={total_loss:.4f}, Train AUC={train_auc:.4f}, Valid AUC={valid_auc:.4f}"
            )

        auc_for_selection = valid_auc if pd.notna(valid_auc) else -np.inf
        if auc_for_selection > best_auc:
            best_auc = auc_for_selection
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= config["early_stop_patience"]:
            if verbose:
                print(f"LB={lookback} H={horizon} Early stopping")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, float(best_auc), pd.DataFrame(history_rows)


def build_prediction_frame(prob, label, index_seq, base_df):
    prob = np.asarray(prob)
    if prob.ndim == 2:
        prob_cols = {f"pred_prob_class_{i}": prob[:, i] for i in range(prob.shape[1])}
        pred_prob = prob[:, 2] if prob.shape[1] > 2 else prob[:, -1]
    else:
        prob_cols = {}
        pred_prob = np.asarray(prob).reshape(-1)
    pred_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(index_seq),
            "y_true": np.asarray(label).reshape(-1),
            "pred_prob": pred_prob,
            **prob_cols,
        }
    ).set_index("datetime")
    result = base_df.loc[pred_df.index].copy()
    result["pred_prob"] = pred_df["pred_prob"]
    result["y_true"] = pred_df["y_true"]
    for col in prob_cols:
        result[col] = pred_df[col]
    return result


def evaluate_lstm_horizon_model(
    model,
    best_valid_auc,
    dataset_pack,
    config,
    output_dir,
    device=None,
    threshold_quantiles=None,
):
    device = _resolve_device(device)
    threshold_quantiles = threshold_quantiles or DEFAULT_THRESHOLD_QUANTILES
    horizon = dataset_pack["horizon"]
    lookback = dataset_pack["lookback"]
    round_trip_cost = config["round_trip_cost"]
    exp_dir = Path(output_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    valid_prob, valid_label, _ = predict_loader(model, dataset_pack["loaders"]["valid"], device)
    test_prob, test_label, _ = predict_loader(model, dataset_pack["loaders"]["test"], device)
    valid_prob = np.asarray(valid_prob)
    valid_label = np.asarray(valid_label).reshape(-1)
    test_prob = np.asarray(test_prob)
    test_label = np.asarray(test_label).reshape(-1)

    if is_multiclass_config(config):
        buy_class = get_buy_class(config)
        valid_action_metrics = safe_action_metrics(valid_label, valid_prob, config)
        test_action_metrics = safe_action_metrics(test_label, test_prob, config)
        valid_auc = valid_action_metrics["buy_auc"]
        test_auc = test_action_metrics["buy_auc"]
    else:
        valid_action_metrics = {}
        test_action_metrics = {}
        valid_auc = safe_auc_score(valid_label, valid_prob)
        test_auc = safe_auc_score(test_label, test_prob)

    valid_result = build_prediction_frame(valid_prob, valid_label, dataset_pack["valid_index_seq"], dataset_pack["valid_df"])
    test_result = build_prediction_frame(test_prob, test_label, dataset_pack["test_index_seq"], dataset_pack["test_df"])

    threshold_rows_valid = []
    for q in threshold_quantiles:
        threshold = valid_result["pred_prob"].quantile(q)
        _, _, stats_q = run_fixed_horizon_event_backtest(
            valid_result,
            threshold=threshold,
            horizon=horizon,
            round_trip_cost=round_trip_cost,
        )
        stats_q["quantile"] = q
        stats_q["threshold"] = float(threshold)
        threshold_rows_valid.append(stats_q)

    valid_threshold_df = pd.DataFrame(threshold_rows_valid)
    valid_threshold_df.to_csv(exp_dir / "validation_threshold_diagnostics.csv", index=False)
    fixed_quantile = float(config.get("fixed_threshold_quantile", 0.95))
    selected_row = {
        "quantile": fixed_quantile,
        "threshold": float(valid_result["pred_prob"].quantile(fixed_quantile)),
    }

    selected_quantile = float(selected_row["quantile"])
    selected_threshold = float(selected_row["threshold"])
    selected_trades, selected_equity, selected_stats = run_fixed_horizon_event_backtest(
        test_result,
        threshold=selected_threshold,
        horizon=horizon,
        round_trip_cost=round_trip_cost,
    )
    selected_stats["selected_by"] = "validation_net_return_mean"
    selected_stats["selected_quantile"] = selected_quantile

    test_threshold_rows = []
    for q in threshold_quantiles:
        threshold = test_result["pred_prob"].quantile(q)
        _, _, stats_q = run_fixed_horizon_event_backtest(
            test_result,
            threshold=threshold,
            horizon=horizon,
            round_trip_cost=round_trip_cost,
        )
        stats_q["quantile"] = q
        stats_q["threshold"] = float(threshold)
        test_threshold_rows.append(stats_q)
    test_threshold_df = pd.DataFrame(test_threshold_rows)

    selected_trades.to_csv(exp_dir / "realistic_trades_selected_threshold.csv", index=False)
    pd.Series(selected_stats).to_csv(exp_dir / "realistic_backtest_stats_selected_threshold.csv")
    test_threshold_df.to_csv(exp_dir / "test_threshold_diagnostics_diagnostic_only.csv", index=False)
    test_result[["pred_prob", "y_true", "trade_return", "future_return"]].to_csv(exp_dir / "test_predictions.csv")

    signal_group = (
        test_result.assign(prob_group=pd.qcut(test_result["pred_prob"], q=10, labels=False, duplicates="drop"))
        .groupby("prob_group", observed=True)
        .agg(
            sample_count=("pred_prob", "size"),
            pred_prob_mean=("pred_prob", "mean"),
            label_mean=("y_true", "mean"),
            trade_return_mean=("trade_return", "mean"),
            trade_return_median=("trade_return", "median"),
        )
    )
    signal_group.to_csv(exp_dir / "signal_group_analysis.csv")

    summary = {
        "lookback": int(lookback),
        "lookback_minutes": int(lookback * config["bar_minutes"]),
        "horizon": int(horizon),
        "horizon_minutes": int(horizon * config["bar_minutes"]),
        "best_valid_auc": float(best_valid_auc),
        "valid_auc": float(valid_auc) if pd.notna(valid_auc) else np.nan,
        "test_auc": float(test_auc) if pd.notna(test_auc) else np.nan,
        "train_positive_rate": float((dataset_pack["y_train_seq"] == get_buy_class(config)).mean()) if is_multiclass_config(config) else float(np.mean(dataset_pack["y_train_seq"])),
        "valid_positive_rate": float((dataset_pack["y_valid_seq"] == get_buy_class(config)).mean()) if is_multiclass_config(config) else float(np.mean(dataset_pack["y_valid_seq"])),
        "test_positive_rate": float((dataset_pack["y_test_seq"] == get_buy_class(config)).mean()) if is_multiclass_config(config) else float(np.mean(dataset_pack["y_test_seq"])),
        "valid_action_metrics": valid_action_metrics,
        "test_action_metrics": test_action_metrics,
        "test_trade_return_mean_all": float(test_result["trade_return"].mean()),
        "test_trade_return_mean_label_1": float(test_result.loc[test_result["y_true"] == 1, "trade_return"].mean()),
        "selected_threshold": float(selected_threshold),
        "selected_quantile": float(selected_quantile),
        "selected_threshold_test_stats": selected_stats,
        "output_dir": str(exp_dir),
    }
    with open(exp_dir / "horizon_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    row = {
        "lookback": int(lookback),
        "lookback_minutes": int(lookback * config["bar_minutes"]),
        "horizon": int(horizon),
        "horizon_minutes": int(horizon * config["bar_minutes"]),
        "valid_auc": float(valid_auc) if pd.notna(valid_auc) else np.nan,
        "test_auc": float(test_auc) if pd.notna(test_auc) else np.nan,
        "train_positive_rate": float((dataset_pack["y_train_seq"] == get_buy_class(config)).mean()) if is_multiclass_config(config) else float(np.mean(dataset_pack["y_train_seq"])),
        "valid_positive_rate": float((dataset_pack["y_valid_seq"] == get_buy_class(config)).mean()) if is_multiclass_config(config) else float(np.mean(dataset_pack["y_valid_seq"])),
        "test_positive_rate": float((dataset_pack["y_test_seq"] == get_buy_class(config)).mean()) if is_multiclass_config(config) else float(np.mean(dataset_pack["y_test_seq"])),
        "selected_quantile": float(selected_quantile),
        "selected_threshold": float(selected_threshold),
        "trade_count": selected_stats.get("trade_count", np.nan),
        "win_rate": selected_stats.get("win_rate", np.nan),
        "gross_return_mean": selected_stats.get("gross_return_mean", np.nan),
        "net_return_mean": selected_stats.get("net_return_mean", np.nan),
        "total_return": selected_stats.get("total_return", np.nan),
        "max_drawdown": selected_stats.get("max_drawdown", np.nan),
        "per_trade_sharpe": selected_stats.get("per_trade_sharpe", np.nan),
        "trades_per_year": selected_stats.get("trades_per_year", np.nan),
        "bar_annualized_sharpe": selected_stats.get("bar_annualized_sharpe", np.nan),
        "sharpe": selected_stats.get("sharpe", np.nan),
        "test_trade_return_mean_all": float(test_result["trade_return"].mean()),
    }
    return row, summary


def run_single_lstm_horizon_experiment(
    base_df,
    feature_cols,
    target_col,
    config,
    horizon,
    lookback,
    output_dir,
    device=None,
    seed=None,
    verbose=True,
):
    if seed is not None:
        set_global_seed(seed)
    device = _resolve_device(device)
    dataset_pack = build_horizon_dataset(base_df, feature_cols, target_col, config, horizon, lookback)
    model, best_auc, history = train_lstm_for_dataset(
        dataset_pack,
        feature_count=len(feature_cols),
        config=config,
        device=device,
        verbose=verbose,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_dir / "training_history.csv", index=False)
    row, summary = evaluate_lstm_horizon_model(model, best_auc, dataset_pack, config, output_dir, device=device)
    if seed is not None:
        row["seed"] = int(seed)
        summary["seed"] = int(seed)
        with open(output_dir / "horizon_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return row, summary


def run_horizon_lookback_comparison(
    base_df,
    feature_cols,
    target_col,
    config,
    horizon_values,
    lookback_values,
    output_dir="outputs/diagnostics/horizon_compare",
    device=None,
    seed=None,
    verbose=True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for lookback in lookback_values:
        for horizon in horizon_values:
            exp_dir = output_dir / f"lb{lookback}_h{horizon}"
            row, _ = run_single_lstm_horizon_experiment(
                base_df=base_df,
                feature_cols=feature_cols,
                target_col=target_col,
                config=config,
                horizon=horizon,
                lookback=lookback,
                output_dir=exp_dir,
                device=device,
                seed=seed,
                verbose=verbose,
            )
            rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values(
        ["net_return_mean", "gross_return_mean", "test_auc"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    comparison.to_csv(output_dir / "horizon_lookback_comparison.csv", index=False)
    comparison.to_csv(output_dir / "horizon_comparison.csv", index=False)
    with open(output_dir / "horizon_lookback_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    with open(output_dir / "horizon_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    return comparison


def run_seed_robustness(
    base_df,
    feature_cols,
    target_col,
    config,
    horizon=12,
    lookback=6,
    seeds=None,
    output_dir="outputs/diagnostics/robustness/lb6_h12",
    device=None,
    verbose=True,
):
    seeds = seeds or [7, 11, 19, 23, 42]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        exp_dir = output_dir / f"seed_{seed}"
        row, _ = run_single_lstm_horizon_experiment(
            base_df=base_df,
            feature_cols=feature_cols,
            target_col=target_col,
            config=config,
            horizon=horizon,
            lookback=lookback,
            output_dir=exp_dir,
            device=device,
            seed=seed,
            verbose=verbose,
        )
        rows.append(row)

    runs = pd.DataFrame(rows)
    runs = runs.sort_values("seed").reset_index(drop=True)
    runs.to_csv(output_dir / "seed_robustness_runs.csv", index=False)

    metric_cols = [
        "valid_auc",
        "test_auc",
        "trade_count",
        "win_rate",
        "gross_return_mean",
        "net_return_mean",
        "total_return",
        "max_drawdown",
        "per_trade_sharpe",
        "trades_per_year",
        "bar_annualized_sharpe",
        "sharpe",
    ]
    available_metric_cols = [col for col in metric_cols if col in runs.columns]
    summary = runs[available_metric_cols].agg(["mean", "std", "min", "median", "max"]).T
    summary["positive_total_return_rate"] = np.nan
    summary["positive_net_return_mean_rate"] = np.nan
    if len(runs):
        summary.loc["total_return", "positive_total_return_rate"] = float((runs["total_return"] > 0).mean())
        summary.loc["net_return_mean", "positive_net_return_mean_rate"] = float((runs["net_return_mean"] > 0).mean())

    summary.to_csv(output_dir / "seed_robustness_summary.csv")
    with open(output_dir / "seed_robustness_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "horizon": int(horizon),
                "lookback": int(lookback),
                "seeds": [int(s) for s in seeds],
                "runs": runs.to_dict(orient="records"),
                "summary": summary.reset_index(names="metric").to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return runs, summary
