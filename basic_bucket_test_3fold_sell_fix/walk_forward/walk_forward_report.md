# Walk-forward Research Report

## Executive Summary

- Executed folds: 3
- Total trades: 57
- Median test buy AUC: 0.5784761091592726
- Median test sell AUC: 0.5200016027121815
- Median total return per fold: -0.07281313942377443
- Median alpha total return per fold: -0.00217643421899949
- Stress min alpha total return: -0.039319696486499245
- Stress worst alpha drawdown: -0.03983832228034301
- Quality gate passed: False
- Live readiness: FAIL

## Quality Gate Checks

- enough_folds: True
- enough_trades: True
- buy_auc_above_gate: True
- sell_auc_above_gate: True
- direction_auc_above_gate: True
- alpha_return_above_gate: False
- drawdown_within_gate: True

## Observed Metrics

- fold_count: 3
- total_trades: 57
- median_test_buy_auc: 0.5784761091592726
- median_test_sell_auc: 0.5200016027121815
- median_alpha_return: -0.00217643421899949
- median_max_drawdown: -0.005115470776233755
- drawdown_metric: alpha_max_drawdown
- return_metric: alpha_total_return

## Live Readiness Gates

- reproducibility_clean_source_tree: True
- feature_leakage_passed: True
- execution_feasibility_passed: True
- walk_forward_quality_passed: False
- min_executed_folds: False
- min_total_trades: False
- positive_median_alpha_return: False
- median_buy_auc_above_live_floor: True
- median_sell_auc_above_live_floor: False
- drawdown_within_live_limit: True
- stress_alpha_positive: False
- stress_drawdown_within_live_limit: True
- shadow_monitoring_passed: False

## Live Readiness Observed

- fold_count: 3
- total_trades: 57
- median_alpha_total_return: -0.00217643421899949
- median_test_buy_auc: 0.5784761091592726
- median_test_sell_auc: 0.5200016027121815
- median_max_drawdown: -0.005115470776233755
- stress_min_alpha_total_return: -0.039319696486499245
- stress_worst_alpha_drawdown: -0.03983832228034301
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

|   fold | selected_model        |   test_buy_auc |   test_sell_auc |   total_return |   alpha_total_return |   max_drawdown |   alpha_max_drawdown |   sharpe |   alpha_sharpe |   sortino |   alpha_sortino |   calmar |   alpha_calmar |   trade_count |
|-------:|:----------------------|---------------:|----------------:|---------------:|---------------------:|---------------:|---------------------:|---------:|---------------:|----------:|----------------:|---------:|---------------:|--------------:|
|      0 | auc_weighted_ensemble |       0.565747 |        0.520002 |     -0.10923   |          -0.00217643 |     -0.151239  |          -0.00511547 | -2.43814 |       -1.13982 |  -3.41633 |        -0.29435 | -2.76383 |       -1.98302 |            10 |
|      1 | auc_weighted_ensemble |       0.578476 |        0.569687 |     -0.0356129 |          -0.022629   |     -0.0923399 |          -0.0242464  | -1.2015  |      -10.0819  |  -1.37688 |        -5.14692 | -1.89015 |       -4.70292 |            47 |
|      2 | auc_weighted_ensemble |       0.628051 |        0.517014 |     -0.0728131 |           0          |     -0.0998367 |           0          | -2.7842  |      nan       |  -3.22858 |       nan       | -3.01676 |      nan       |             0 |

## Interpretation Guardrail

This report is a research artifact. A passing gate is not live-trading approval. Live readiness requires broader walk-forward coverage, stress tests, realistic venue constraints, and cross-checking with VeighNa event-driven backtests.
