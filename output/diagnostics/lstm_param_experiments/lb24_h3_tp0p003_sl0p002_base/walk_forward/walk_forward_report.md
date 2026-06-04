# Walk-forward Research Report

## Executive Summary

- Executed folds: 3
- Total trades: 30
- Median test buy AUC: 0.5230526883057662
- Median test sell AUC: 0.5149812068806264
- Median total return per fold: -0.04529606674186626
- Median alpha total return per fold: -7.811561100001807e-05
- Stress min alpha total return: -0.011409687737499619
- Stress worst alpha drawdown: -0.012558078218078084
- Quality gate passed: False
- Live readiness: FAIL

## Quality Gate Checks

- enough_folds: True
- enough_trades: True
- buy_auc_above_gate: True
- sell_auc_above_gate: False
- direction_auc_above_gate: False
- alpha_return_above_gate: False
- drawdown_within_gate: True

## Observed Metrics

- fold_count: 3
- total_trades: 30
- median_test_buy_auc: 0.5230526883057662
- median_test_sell_auc: 0.5149812068806264
- median_alpha_return: -7.811561100001807e-05
- median_max_drawdown: -0.003098672484468601
- drawdown_metric: alpha_max_drawdown
- return_metric: alpha_total_return

## Live Readiness Gates

- reproducibility_clean_source_tree: False
- feature_leakage_passed: True
- execution_feasibility_passed: True
- walk_forward_quality_passed: False
- min_executed_folds: False
- min_total_trades: False
- positive_median_alpha_return: False
- median_buy_auc_above_live_floor: False
- median_sell_auc_above_live_floor: False
- drawdown_within_live_limit: True
- stress_alpha_positive: False
- stress_drawdown_within_live_limit: True
- shadow_monitoring_passed: False

## Live Readiness Observed

- fold_count: 3
- total_trades: 30
- median_alpha_total_return: -7.811561100001807e-05
- median_test_buy_auc: 0.5230526883057662
- median_test_sell_auc: 0.5149812068806264
- median_max_drawdown: -0.003098672484468601
- stress_min_alpha_total_return: -0.011409687737499619
- stress_worst_alpha_drawdown: -0.012558078218078084
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

|   fold | selected_model   |   test_buy_auc |   test_sell_auc |   total_return |   alpha_total_return |   max_drawdown |   alpha_max_drawdown |   sharpe |   alpha_sharpe |   sortino |   alpha_sortino |   calmar |   alpha_calmar |   trade_count |
|-------:|:-----------------|---------------:|----------------:|---------------:|---------------------:|---------------:|---------------------:|---------:|---------------:|----------:|----------------:|---------:|---------------:|--------------:|
|      0 | lstm             |       0.55221  |        0.533211 |     -0.0452961 |         -7.81156e-05 |     -0.0785363 |          -0.00289545 | -2.29988 |     -0.0980816 |  -2.18813 |      -0.0568959 | -6.82996 |      -0.447142 |             9 |
|      1 | lstm             |       0.523053 |        0.508117 |      0.0786601 |          8.07943e-05 |     -0.0439233 |          -0.00309867 |  4.24205 |      0.0941709 |   7.21389 |       0.0235623 | 45.9847  |       0.380778 |             4 |
|      2 | lstm             |       0.515041 |        0.514981 |     -0.0945602 |         -0.00391332  |     -0.114455  |          -0.00692389 | -6.14814 |     -4.00127   |  -9.81169 |      -1.71623   | -7.05467 |      -9.09266  |            17 |

## Interpretation Guardrail

This report is a research artifact. A passing gate is not live-trading approval. Live readiness requires broader walk-forward coverage, stress tests, realistic venue constraints, and cross-checking with VeighNa event-driven backtests.
