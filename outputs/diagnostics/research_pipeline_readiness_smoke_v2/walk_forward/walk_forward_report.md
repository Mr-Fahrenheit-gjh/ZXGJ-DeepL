# Walk-forward Research Report

## Executive Summary

- Executed folds: 1
- Total trades: 9
- Median test buy AUC: 0.6145833333333334
- Median test sell AUC: 0.6427450980392156
- Median total return per fold: -0.026491108788348594
- Median alpha total return per fold: 0.0003877263900000649
- Stress min alpha total return: -0.004300797458499983
- Stress worst alpha drawdown: -0.0047554364885000755
- Quality gate passed: True
- Live readiness: FAIL

## Quality Gate Checks

- enough_folds: True
- enough_trades: True
- buy_auc_above_gate: True
- alpha_return_above_gate: True
- drawdown_within_gate: True

## Observed Metrics

- fold_count: 1
- total_trades: 9
- median_test_buy_auc: 0.6145833333333334
- median_alpha_return: 0.0003877263900000649
- median_max_drawdown: -0.0021974289680001258
- drawdown_metric: alpha_max_drawdown
- return_metric: alpha_total_return

## Live Readiness Gates

- reproducibility_clean_source_tree: False
- feature_leakage_passed: True
- walk_forward_quality_passed: True
- min_executed_folds: False
- min_total_trades: False
- positive_median_alpha_return: True
- median_buy_auc_above_live_floor: True
- median_sell_auc_above_live_floor: True
- drawdown_within_live_limit: True
- stress_alpha_positive: False
- stress_drawdown_within_live_limit: True

## Live Readiness Observed

- fold_count: 1
- total_trades: 9
- median_alpha_total_return: 0.0003877263900000649
- median_test_buy_auc: 0.6145833333333334
- median_test_sell_auc: 0.6427450980392156
- median_max_drawdown: -0.0021974289680001258
- stress_min_alpha_total_return: -0.004300797458499983
- stress_worst_alpha_drawdown: -0.0047554364885000755

## Methodology

- split_policy: chronological walk-forward by bar position; no random shuffle
- fold_policy: each train window is split into subtrain and calibration; fold validation is treated as out-of-sample test
- preprocessing_policy: winsorization and standardization are fit on subtrain only, then applied to calibration and test
- threshold_policy: buy/sell thresholds are calibration-set fixed quantiles, never selected from the fold test window
- ensemble_policy: model weights are proportional to validation AUC edge over 0.5; equal fallback if no model beats random
- execution_policy: signal at bar t, execute at next bar open with commission, slippage, lot size, no-overlap state machine, tp/sl and no-overnight handling
- a_share_t0_policy: default inventory mode starts with tradable base shares; same-day sells use existing inventory, buybacks restore inventory, and buy-first exits sell old inventory rather than same-day purchases
- liquidity_policy: single trade size is capped by max_participation_rate of next-bar volume when volume is available

## Fold Summary

|   fold | selected_model        |   test_buy_auc |   test_sell_auc |   total_return |   alpha_total_return |   max_drawdown |   alpha_max_drawdown |   sharpe |   alpha_sharpe |   sortino |   alpha_sortino |   calmar |   alpha_calmar |   trade_count |
|-------:|:----------------------|---------------:|----------------:|---------------:|---------------------:|---------------:|---------------------:|---------:|---------------:|----------:|----------------:|---------:|---------------:|--------------:|
|      0 | auc_weighted_ensemble |       0.614583 |        0.642745 |     -0.0264911 |          0.000387726 |     -0.0330756 |          -0.00219743 | -14.1291 |        1.44175 |  -17.7174 |         1.45085 |   -25.94 |        13.0069 |             9 |

## Interpretation Guardrail

This report is a research artifact. A passing gate is not live-trading approval. Live readiness requires broader walk-forward coverage, stress tests, realistic venue constraints, and cross-checking with VeighNa event-driven backtests.
