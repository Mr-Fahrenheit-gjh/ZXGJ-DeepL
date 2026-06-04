# Walk-forward Research Report

## Executive Summary

- Executed folds: 3
- Total trades: 5
- Median test buy AUC: 0.5301757066462949
- Median test sell AUC: 0.5158823529411765
- Median total return per fold: -0.02604630005834141
- Median alpha total return per fold: 0.00042197265599996925
- Stress min alpha total return: -0.007853865819999917
- Stress worst alpha drawdown: -0.007853865819999917
- Quality gate passed: True
- Live readiness: FAIL

## Quality Gate Checks

- enough_folds: True
- enough_trades: True
- buy_auc_above_gate: True
- alpha_return_above_gate: True
- drawdown_within_gate: True

## Observed Metrics

- fold_count: 3
- total_trades: 5
- median_test_buy_auc: 0.5301757066462949
- median_alpha_return: 0.00042197265599996925
- median_max_drawdown: -0.0006559698475407938
- drawdown_metric: alpha_max_drawdown
- return_metric: alpha_total_return

## Live Readiness Gates

- reproducibility_clean_source_tree: True
- feature_leakage_passed: True
- execution_feasibility_passed: True
- walk_forward_quality_passed: True
- min_executed_folds: False
- min_total_trades: False
- positive_median_alpha_return: True
- median_buy_auc_above_live_floor: True
- median_sell_auc_above_live_floor: False
- drawdown_within_live_limit: True
- stress_alpha_positive: False
- stress_drawdown_within_live_limit: True
- shadow_monitoring_passed: False

## Live Readiness Observed

- fold_count: 3
- total_trades: 5
- median_alpha_total_return: 0.00042197265599996925
- median_test_buy_auc: 0.5301757066462949
- median_test_sell_auc: 0.5158823529411765
- median_max_drawdown: -0.0006559698475407938
- stress_min_alpha_total_return: -0.007853865819999917
- stress_worst_alpha_drawdown: -0.007853865819999917
- shadow_monitoring_status: None
- execution_feasibility_status: PASS

## Execution Feasibility

- status: PASS
- row_count: 67968
- trade_day_count: 1416
- missing_required_cols: []
- missing_optional_cols: []
- duplicate_timestamps: 0
- outside_session_bars: 0
- outside_session_ratio: 0.0
- null_ohlcv_bars: 0
- nonpositive_price_bars: 0
- zero_volume_bars: 0
- zero_volume_ratio: 0.0
- high_low_inconsistent: 0
- open_outside_range: 0
- close_outside_range: 0
- short_day_count: 0
- short_day_ratio: 0.0
- max_bar_shortfall: 0
- extreme_daily_move_count: 0
- price_limit_checked_day_count: 1377
- price_limit_skipped_gap_day_count: 38
- max_abs_daily_return: 0.2940034512510785
- max_abs_price_limit_checked_return: 0.1999467518636846

## Methodology

- split_policy: chronological walk-forward by bar position; no random shuffle
- fold_policy: each train window is split into subtrain and calibration; fold validation is treated as out-of-sample test
- preprocessing_policy: winsorization and standardization are fit on subtrain only, then applied to calibration and test
- threshold_policy: buy/sell thresholds are calibration-set fixed quantiles, never selected from the fold test window
- ensemble_policy: model weights are proportional to validation AUC edge over 0.5; equal fallback if no model beats random
- execution_policy: signal at bar t, execute at next bar open with commission, slippage, lot size, no-overlap state machine, tp/sl and no-overnight handling
- a_share_t0_policy: default inventory mode starts with tradable base shares; same-day sells use existing inventory, buybacks restore inventory, and buy-first exits sell old inventory rather than same-day purchases
- liquidity_policy: single trade size is capped by max_participation_rate of next-bar volume when volume is available
- explainability_policy: optional per-fold permutation/SHAP for sklearn models and input-gradient importance for torch sequence models

## Fold Summary

|   fold | selected_model        |   test_buy_auc |   test_sell_auc |   total_return |   alpha_total_return |   max_drawdown |   alpha_max_drawdown |    sharpe |   alpha_sharpe |   sortino |   alpha_sortino |   calmar |   alpha_calmar |   trade_count |
|-------:|:----------------------|---------------:|----------------:|---------------:|---------------------:|---------------:|---------------------:|----------:|---------------:|----------:|----------------:|---------:|---------------:|--------------:|
|      0 | auc_weighted_ensemble |       0.534896 |        0.515882 |    -0.0260463  |          0.000832219 |     -0.0339183 |         -0.00065597  | -13.2871  |        4.88379 |  -16.3383 |         3.18581 | -25.1541 |        95.0354 |             1 |
|      1 | auc_weighted_ensemble |       0.530176 |        0.578069 |    -0.00572944 |          0.000421973 |     -0.0154335 |         -0.000566618 |  -3.01436 |        2.59678 |   -5.2999 |         1.48062 | -22.1237 |        54.9659 |             1 |
|      2 | auc_weighted_ensemble |       0.462295 |        0.412464 |    -0.0486167  |         -0.00626432  |     -0.0628632 |         -0.00626432  | -12.5187  |      -18.4477  |  -14.1961 |        -6.85664 | -15.8689 |       -84.8988 |             3 |

## Interpretation Guardrail

This report is a research artifact. A passing gate is not live-trading approval. Live readiness requires broader walk-forward coverage, stress tests, realistic venue constraints, and cross-checking with VeighNa event-driven backtests.
