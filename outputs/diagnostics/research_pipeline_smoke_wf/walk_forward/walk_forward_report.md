# Walk-forward Research Report

## Executive Summary

- Executed folds: 2
- Total trades: 14
- Median test buy AUC: 0.5254955675452
- Median test sell AUC: 0.6091888564235441
- Median total return per fold: -0.012450987388499424
- Quality gate passed: True

## Quality Gate Checks

- enough_folds: True
- enough_trades: True
- buy_auc_above_gate: True
- drawdown_within_gate: True

## Observed Metrics

- fold_count: 2
- total_trades: 14
- median_test_buy_auc: 0.5254955675452
- median_max_drawdown: -0.014283736442999517

## Methodology

- split_policy: chronological walk-forward by bar position; no random shuffle
- fold_policy: each train window is split into subtrain and calibration; fold validation is treated as out-of-sample test
- preprocessing_policy: winsorization and standardization are fit on subtrain only, then applied to calibration and test
- threshold_policy: buy/sell thresholds are calibration-set fixed quantiles, never selected from the fold test window
- ensemble_policy: model weights are proportional to validation AUC edge over 0.5; equal fallback if no model beats random
- execution_policy: signal at bar t, execute at next bar open with commission, slippage, lot size, no-overlap state machine, tp/sl and no-overnight handling

## Fold Summary

|   fold | selected_model        |   test_buy_auc |   test_sell_auc |   total_return |   max_drawdown |   sharpe |   sortino |   calmar |   trade_count |
|-------:|:----------------------|---------------:|----------------:|---------------:|---------------:|---------:|----------:|---------:|--------------:|
|      0 | auc_weighted_ensemble |       0.611979 |        0.642745 |       0        |      0         | nan      |  nan      | nan      |             0 |
|      1 | auc_weighted_ensemble |       0.439012 |        0.575633 |      -0.024902 |     -0.0285675 | -18.2822 |  -21.8719 | -29.4076 |            14 |

## Interpretation Guardrail

This report is a research artifact. A passing gate is not live-trading approval. Live readiness requires broader walk-forward coverage, stress tests, realistic venue constraints, and cross-checking with VeighNa event-driven backtests.
