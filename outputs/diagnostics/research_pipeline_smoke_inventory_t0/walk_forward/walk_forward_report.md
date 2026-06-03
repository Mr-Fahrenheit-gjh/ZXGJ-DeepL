# Walk-forward Research Report

## Executive Summary

- Executed folds: 2
- Total trades: 21
- Median test buy AUC: 0.5254955675452
- Median test sell AUC: 0.6091888564235441
- Median total return per fold: -0.020819547685523743
- Quality gate passed: True

## Quality Gate Checks

- enough_folds: True
- enough_trades: True
- buy_auc_above_gate: True
- drawdown_within_gate: True

## Observed Metrics

- fold_count: 2
- total_trades: 21
- median_test_buy_auc: 0.5254955675452
- median_max_drawdown: -0.02745179367989259

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

|   fold | selected_model        |   test_buy_auc |   test_sell_auc |   total_return |   alpha_total_return |   max_drawdown |   alpha_max_drawdown |    sharpe |   alpha_sharpe |   sortino |   alpha_sortino |   calmar |   alpha_calmar |   trade_count |
|-------:|:----------------------|---------------:|----------------:|---------------:|---------------------:|---------------:|---------------------:|----------:|---------------:|----------:|----------------:|---------:|---------------:|--------------:|
|      0 | auc_weighted_ensemble |       0.611979 |        0.642745 |     -0.0264911 |          0.000387726 |     -0.0330756 |          -0.00219743 | -14.1291  |        1.44175 |  -17.7174 |         1.45085 | -25.94   |        13.0069 |             9 |
|      1 | auc_weighted_ensemble |       0.439012 |        0.575633 |     -0.015148  |         -0.00899426  |     -0.021828  |          -0.00899426 |  -7.54146 |      -36.5006  |  -12.8951 |       -27.4682  | -30.7092 |       -53.5339 |            12 |

## Interpretation Guardrail

This report is a research artifact. A passing gate is not live-trading approval. Live readiness requires broader walk-forward coverage, stress tests, realistic venue constraints, and cross-checking with VeighNa event-driven backtests.
