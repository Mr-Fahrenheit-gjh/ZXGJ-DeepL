# Walk-forward Research Report

## Executive Summary

- Executed folds: 3
- Total trades: 26
- Median test buy AUC: 0.511844790617169
- Median test sell AUC: 0.5090610083831375
- Median total return per fold: -0.04434635275792398
- Median alpha total return per fold: -0.0008465480669997971
- Stress min alpha total return: -0.006377364826500198
- Stress worst alpha drawdown: -0.007115026660758916
- Quality gate passed: False
- Live readiness: FAIL

## Quality Gate Checks

- enough_folds: True
- enough_trades: False
- buy_auc_above_gate: False
- sell_auc_above_gate: False
- direction_auc_above_gate: False
- alpha_return_above_gate: False
- drawdown_within_gate: True

## Observed Metrics

- fold_count: 3
- total_trades: 26
- median_test_buy_auc: 0.511844790617169
- median_test_sell_auc: 0.5090610083831375
- median_alpha_return: -0.0008465480669997971
- median_max_drawdown: -0.002521157173164812
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
- total_trades: 26
- median_alpha_total_return: -0.0008465480669997971
- median_test_buy_auc: 0.511844790617169
- median_test_sell_auc: 0.5090610083831375
- median_max_drawdown: -0.002521157173164812
- stress_min_alpha_total_return: -0.006377364826500198
- stress_worst_alpha_drawdown: -0.007115026660758916
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
|      0 | lstm             |       0.511845 |        0.509061 |     -0.0443464 |          0.00039089  |     -0.0789314 |          -0.00231563 | -2.07656 |        0.49217 |  -1.97049 |        0.227211 | -6.86475 |        2.91401 |             5 |
|      1 | lstm             |       0.498078 |        0.510319 |      0.0837288 |         -0.000846548 |     -0.0419746 |          -0.00252116 |  4.42245 |       -1.10994 |   7.51509 |       -0.492905 | 53.3151  |       -4.87822 |             9 |
|      2 | lstm             |       0.51868  |        0.503069 |     -0.0953836 |         -0.00401619  |     -0.112975  |          -0.00566924 | -6.3948  |       -5.17574 | -10.076   |       -1.75789  | -7.17615 |      -11.4017  |            12 |

## Interpretation Guardrail

This report is a research artifact. A passing gate is not live-trading approval. Live readiness requires broader walk-forward coverage, stress tests, realistic venue constraints, and cross-checking with VeighNa event-driven backtests.
