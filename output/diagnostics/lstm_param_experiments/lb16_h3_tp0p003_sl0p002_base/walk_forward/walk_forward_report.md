# Walk-forward Research Report

## Executive Summary

- Executed folds: 3
- Total trades: 39
- Median test buy AUC: 0.5254303599374022
- Median test sell AUC: 0.4993732394366197
- Median total return per fold: -0.051544003660935034
- Median alpha total return per fold: -0.008633847113999726
- Stress min alpha total return: -0.018108536479000015
- Stress worst alpha drawdown: -0.019134434635037945
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
- total_trades: 39
- median_test_buy_auc: 0.5254303599374022
- median_test_sell_auc: 0.4993732394366197
- median_alpha_return: -0.008633847113999726
- median_max_drawdown: -0.008633847113999948
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
- total_trades: 39
- median_alpha_total_return: -0.008633847113999726
- median_test_buy_auc: 0.5254303599374022
- median_test_sell_auc: 0.4993732394366197
- median_max_drawdown: -0.008633847113999948
- stress_min_alpha_total_return: -0.018108536479000015
- stress_worst_alpha_drawdown: -0.019134434635037945
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
|      0 | lstm             |       0.501469 |        0.499052 |     -0.051544  |          -0.0053658  |     -0.0792366 |          -0.00661856 | -2.59216 |       -5.88762 |  -2.46249 |        -2.55617 | -7.36742 |       -12.8817 |            12 |
|      1 | lstm             |       0.52543  |        0.499373 |      0.0722454 |          -0.00863385 |     -0.0511741 |          -0.00863385 |  3.8169  |       -7.67084 |   6.48904 |        -1.38064 | 34.3514  |       -13.723  |             6 |
|      2 | lstm             |       0.529932 |        0.518348 |     -0.11089   |          -0.0145931  |     -0.122924  |          -0.0158887  | -7.46722 |      -10.8813  | -11.5977  |        -3.94144 | -6.97393 |       -13.6016 |            21 |

## Interpretation Guardrail

This report is a research artifact. A passing gate is not live-trading approval. Live readiness requires broader walk-forward coverage, stress tests, realistic venue constraints, and cross-checking with VeighNa event-driven backtests.
