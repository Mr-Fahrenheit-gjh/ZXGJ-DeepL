# Walk-forward Research Report

## Executive Summary

- Executed folds: 15
- Total trades: 71
- Median test buy AUC: 0.4823020117137764
- Median test sell AUC: 0.5331988261188555
- Median total return per fold: -0.022604848032000024
- Median alpha total return per fold: -0.0008205711840000296
- Stress min alpha total return: -0.01126560429549961
- Stress worst alpha drawdown: -0.011354952735999846
- Quality gate passed: False
- Live readiness: FAIL

## Quality Gate Checks

- enough_folds: True
- enough_trades: True
- buy_auc_above_gate: False
- sell_auc_above_gate: True
- direction_auc_above_gate: True
- alpha_return_above_gate: False
- drawdown_within_gate: True

## Observed Metrics

- fold_count: 15
- total_trades: 71
- median_test_buy_auc: 0.4823020117137764
- median_test_sell_auc: 0.5331988261188555
- median_alpha_return: -0.0008205711840000296
- median_max_drawdown: -0.0014712882593462817
- drawdown_metric: alpha_max_drawdown
- return_metric: alpha_total_return

## Live Readiness Gates

- reproducibility_clean_source_tree: True
- feature_leakage_passed: True
- execution_feasibility_passed: True
- walk_forward_quality_passed: False
- min_executed_folds: True
- min_total_trades: False
- positive_median_alpha_return: False
- median_buy_auc_above_live_floor: False
- median_sell_auc_above_live_floor: True
- drawdown_within_live_limit: True
- stress_alpha_positive: False
- stress_drawdown_within_live_limit: True
- shadow_monitoring_passed: False

## Live Readiness Observed

- fold_count: 15
- total_trades: 71
- median_alpha_total_return: -0.0008205711840000296
- median_test_buy_auc: 0.4823020117137764
- median_test_sell_auc: 0.5331988261188555
- median_max_drawdown: -0.0014712882593462817
- stress_min_alpha_total_return: -0.01126560429549961
- stress_worst_alpha_drawdown: -0.011354952735999846
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

|   fold | selected_model        |   test_buy_auc |   test_sell_auc |   total_return |   alpha_total_return |   max_drawdown |   alpha_max_drawdown |    sharpe |   alpha_sharpe |   sortino |   alpha_sortino |      calmar |   alpha_calmar |   trade_count |
|-------:|:----------------------|---------------:|----------------:|---------------:|---------------------:|---------------:|---------------------:|----------:|---------------:|----------:|----------------:|------------:|---------------:|--------------:|
|      0 | auc_weighted_ensemble |       0.478385 |        0.547647 |    -0.0277422  |         -0.000862491 |     -0.0349877 |         -0.00187507  | -14.2386  |      -5.30413  | -17.5522  |       -3.37049  |  -24.8846   |      -32.4256  |             5 |
|      1 | auc_weighted_ensemble |       0.482302 |        0.597938 |    -0.00606182 |          8.96675e-05 |     -0.0153442 |         -0.00061015  |  -3.15415 |       0.837657 |  -5.43063 |        0.451137 |  -23.2831   |       10.7179  |             1 |
|      2 | auc_weighted_ensemble |       0.414754 |        0.42152  |    -0.0473142  |         -0.0049623   |     -0.0636814 |         -0.00497047  | -12.1472  |     -24.7261   | -13.7668  |      -14.8696   |  -15.6581   |      -90.8617  |             6 |
|      3 | auc_weighted_ensemble |       0.596354 |        0.554031 |    -0.0227826  |         -0.000820571 |     -0.0321186 |         -0.001307    | -12.1201  |      -6.52449  | -20.2603  |       -1.99245  |  -25.305    |      -44.324   |             1 |
|      4 | auc_weighted_ensemble |       0.451062 |        0.49603  |    -0.0493939  |         -0.00157576  |     -0.0877143 |         -0.00435964  |  -7.30035 |      -4.37356  |  -6.31773 |       -2.14066  |  -11.1138   |      -24.8451  |             6 |
|      5 | auc_weighted_ensemble |       0.603525 |        0.53563  |    -0.0269813  |          0.000778482 |     -0.0565405 |         -0.000529917 |  -9.33334 |       6.64506  | -12.7682  |        6.00869  |  -17.0363   |      185.953   |             2 |
|      6 | auc_weighted_ensemble |       0.484796 |        0.461852 |     0.0151575  |         -0.00162346  |     -0.024248  |         -0.00251119  |   4.01958 |      -5.79746  |   8.18868 |       -4.11709  |   81.8661   |      -44.3642  |             7 |
|      7 | auc_weighted_ensemble |       0.546417 |        0.46725  |    -0.00014736 |          0.000902114 |     -0.0251097 |         -0.000493634 |   1.5637  |       4.83406  |   2.69446 |        3.17125  |   -0.424384 |      137.242   |             1 |
|      8 | auc_weighted_ensemble |       0.507257 |        0.583028 |    -0.0290092  |          0.000702781 |     -0.0300591 |         -0.00038914  | -14.6483  |       5.4671   | -26.3577  |        3.51557  |  -32.3173   |      227.552   |             1 |
|      9 | auc_weighted_ensemble |       0.398177 |        0.48356  |    -0.0477624  |         -5.92902e-05 |     -0.0507539 |         -0.00147129  | -10.4413  |      -0.242654 |  -8.68147 |       -0.184515 |  -19.1414   |       -2.92332 |             7 |
|     10 | auc_weighted_ensemble |       0.430946 |        0.448994 |     0.0602416  |         -0.00695011  |     -0.019328  |         -0.00725452  |  15.5762  |     -20.8652   |  25.5294  |      -17.2416   | 3584.57     |      -54.8221  |            13 |
|     11 | auc_weighted_ensemble |       0.427747 |        0.533199 |     0.0403521  |          0.00188596  |     -0.0236456 |         -0.000862707 |   8.85435 |       8.20667  |  20.8712  |       10.5134   |  707.951    |      170.16    |             5 |
|     12 | auc_weighted_ensemble |       0.516905 |        0.606079 |    -0.0226048  |          0.000195152 |     -0.0370776 |         -0.000493632 |  -7.8075  |       1.85228  | -10.1067  |        1.3351   |  -25.2657   |       48.308   |             3 |
|     13 | auc_weighted_ensemble |       0.592397 |        0.583529 |     0.0018791  |         -0.00275396  |     -0.0157056 |         -0.00275396  |   1.30429 |     -19.6632   |   2.153   |       -6.73451  |    9.31053  |      -65.9639  |             3 |
|     14 | auc_weighted_ensemble |       0.471753 |        0.484217 |     0.0356648  |         -0.00587146  |     -0.0465655 |         -0.00600205  |   8.30409 |     -17.0206   |  12.6834  |       -9.75274  |  252.884    |      -58.0217  |            10 |

## Interpretation Guardrail

This report is a research artifact. A passing gate is not live-trading approval. Live readiness requires broader walk-forward coverage, stress tests, realistic venue constraints, and cross-checking with VeighNa event-driven backtests.
