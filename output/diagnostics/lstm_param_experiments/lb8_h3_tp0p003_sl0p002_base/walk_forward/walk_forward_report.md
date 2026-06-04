# Walk-forward Research Report

## Executive Summary

- Executed folds: 3
- Total trades: 45
- Median test buy AUC: 0.5119135988775685
- Median test sell AUC: 0.5186295917521214
- Median total return per fold: -0.057476872296454706
- Median alpha total return per fold: -0.006859570603499621
- Stress min alpha total return: -0.014997425854999524
- Stress worst alpha drawdown: -0.014997425854999524
- Quality gate passed: False
- Live readiness: FAIL

## Quality Gate Checks

- enough_folds: True
- enough_trades: True
- buy_auc_above_gate: False
- sell_auc_above_gate: False
- direction_auc_above_gate: False
- alpha_return_above_gate: False
- drawdown_within_gate: True

## Observed Metrics

- fold_count: 3
- total_trades: 45
- median_test_buy_auc: 0.5119135988775685
- median_test_sell_auc: 0.5186295917521214
- median_alpha_return: -0.006859570603499621
- median_max_drawdown: -0.007926851772901644
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
- total_trades: 45
- median_alpha_total_return: -0.006859570603499621
- median_test_buy_auc: 0.5119135988775685
- median_test_sell_auc: 0.5186295917521214
- median_max_drawdown: -0.007926851772901644
- stress_min_alpha_total_return: -0.014997425854999524
- stress_worst_alpha_drawdown: -0.014997425854999524
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
|      0 | lstm             |       0.522631 |        0.491668 |     -0.0574769 |          -0.00855797 |     -0.0791477 |          -0.00860148 | -2.88546 |      -10.4068  |  -2.7493  |        -3.26048 | -7.88897 |       -15.4081 |            15 |
|      1 | lstm             |       0.509316 |        0.51863  |      0.0728988 |          -0.00685957 |     -0.0475277 |          -0.00685957 |  3.88284 |       -8.1933  |   6.56448 |        -3.52344 | 37.4371  |       -13.8722 |            17 |
|      2 | lstm             |       0.511914 |        0.522072 |     -0.0937244 |          -0.00649835 |     -0.115924  |          -0.00792685 | -6.29844 |       -7.13223 |  -9.8914  |        -2.13478 | -6.83713 |       -12.4846 |            13 |

## Interpretation Guardrail

This report is a research artifact. A passing gate is not live-trading approval. Live readiness requires broader walk-forward coverage, stress tests, realistic venue constraints, and cross-checking with VeighNa event-driven backtests.
