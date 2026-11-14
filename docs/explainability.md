# ENZERA - Explainability write-up

All numbers below come from the backend (`GET /customers`, `GET /customers/{id}`) running model
`xgb-20261114-1557` (17 features), from `reports/metrics.json`, and from `data/demo/demo_labels.csv` (the true labels of the
demo customers, which the app never shows). Demo customers come from the 15% demo split that the model never saw in training.

## How to read a risk score and its reasons

- **Risk score** = XGBoost's raw output (0-1), used for ranking. **Risk band**: High if score ≥ the threshold **0.656** (tuned for
  best F1 on the validation split), Medium if ≥ half the threshold (0.328), otherwise Low.
- **Estimated probability** = the risk score after **Platt calibration** (a logistic curve fitted on the validation split).
  Class weights inflate raw scores, so a raw 0.97 corresponds to roughly an 80% real chance of theft. On the test set, calibration
  lowers the Brier score (mean squared error of probabilities) from 0.122 to **0.065**. We chose Platt over isotonic regression
  because isotonic's top step claimed 100% based on only 4 validation customers.
- **SHAP reasons.** For every customer, XGBoost's TreeSHAP (`booster.predict(..., pred_contribs=True)`) splits the model output (in
  log-odds) into one exact contribution per feature plus a base value:

  `log-odds = base value + SHAP(feature 1) + ... + SHAP(feature 17)`,   `risk = 1 / (1 + e^(-log-odds))`

  The base value is **0.173 log-odds**. Positive contributions push towards theft, negative ones away from it. The app shows the
  top 3 positive contributions as "reasons", the **full waterfall** of all 17, and a **what-if** simulator that re-scores the customer
  after changing feature values. We verified that base + contributions reproduce the displayed score exactly for the customers
  below (an automated test checks this as well).

## Customer 1 - `448FE0108C91D70C0E906B7D5AE496E7` (highest risk)

**Risk 0.971 (High), estimated probability 80%, anomaly 0.31. True label: theft.**
Top reasons: day-to-day variation 13.86 kWh (**+0.63**), missing readings 29% (**+0.49**), long-term trend 0.05 (**+0.42**). Next:
weekly repetition 0.90 (+0.39), highest daily consumption 60.69 kWh (+0.38), recent vs earlier usage 2.35 (+0.28), largest
month-to-month drop 65% (+0.12). 15 of 17 contributions are positive. Base 0.173 + contributions = 3.517 log-odds → 0.971.

**What the consumption shows.** This is the classic theft pattern. In January-June 2014 the customer used 16-24 kWh/day. In July
2014 usage fell by 65% (24.1 → 8.4 kWh/day) and stayed at only 3-8 kWh/day for about nine months (July 2014 - March 2015).
Readings then stop for April-December 2015. In 2016 usage is back at 20-57 kWh/day, higher than ever. A sharp, sustained drop
followed by recovery is what a meter bypass that was later removed would look like. The model captures it mainly through
variability, trend and recent-vs-earlier usage. Its anomaly score is only moderate (0.31), so the supervised model catches
something the unsupervised detector does not flag.

## Customer 2 - `6FDB63D74BAC34833F2330EC3C897327` (second highest risk)

**Risk 0.970 (High), estimated probability 79%, anomaly 0.14. True label: theft.**
Top reasons: long-term trend 0.03 (**+0.49**), missing readings 30% (**+0.49**), day-to-day variation 7.05 kWh (**+0.43**). Next:
weekly repetition 0.90 (+0.42), highest daily consumption 33.15 kWh (+0.30), active-reporting share 100% (+0.29). Base 0.173 +
contributions = 3.461 log-odds → 0.970.

**What the consumption shows.** The same story as customer 1, slightly earlier. Usage was 18-23 kWh/day in January-March 2014,
then fell to 6.5 in April and 3.1 in May 2014 (largest monthly drop 63%). It stayed at only 2.8-7.3 kWh/day until February 2015,
then readings stop until December 2015. In 2016 it is back at 15-25 kWh/day. This customer is very ordinary in feature space
(anomaly 0.14), which shows again that "abnormal" and "theft" are different questions.

## Customer 3 - `F21D74D306C006E021DCF3AFD0D6CE9F` (lowest risk with complete data)

**Risk 0.0025 (Low), estimated probability 0.06%, anomaly 0.90. True label: normal.** This is the lowest-risk demo customer with
less than 10% missing readings.
The score is driven by strong *negative* contributions: highest daily consumption 0.01 kWh (**−1.44**), day-to-day variation 0.00
(**−1.05**), weekend ratio (−0.74), average consumption 0.00 (−0.71), missing readings 2% (−0.71). Base 0.173 + contributions =
−5.973 log-odds → 0.0025. The listed "reasons" (weekly repetition +0.18, longest zero run 514 days +0.08) are small and
outweighed by the negatives.

**What the consumption shows.** This meter reads **essentially zero for three years** (monthly means 0.0 kWh/day, maximum 0.05 kWh
on any day). It is labelled normal in SGCC, so it is probably a vacant property or an unused connection. The supervised model is
confident it is not a thief, because SGCC's caught thieves are mostly larger, variable consumers (see the EDA). Zero usage
pushes the score *down*. The **Isolation Forest, however, scores it 0.90**, among the most anomalous customers. This is the
complementary role of the two detectors in one example. For a utility, a meter recording nothing for three years is worth a
check (a dead meter or a full bypass), even if it does not look like past thieves. *Interpretation caveat:* the reasons list always
shows the top positive contributors, even when they are small. For low-risk customers, look at the waterfall.

## Global importance (`reports/shap_importance.png`)

Mean absolute SHAP value of each feature over the 6,356 test customers (from `reports/metrics.json`):

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | Day-to-day variation (kWh) | 0.363 |
| 2 | Share of days with missing readings | 0.330 |
| 3 | Long-term trend of monthly usage | 0.306 |
| 4 | Highest daily consumption (kWh) | 0.286 |
| 5 | Variability between months | 0.200 |
| 6 | Share of the period the meter was reporting *(new)* | 0.199 |
| 7 | Share of days with unusual spikes | 0.191 |
| 8 | Relative variability of consumption | 0.185 |
| 9 | Weekly repetition of the usage pattern *(new)* | 0.178 |
| ... | ... | ... |
| 16 | Strength of the day-of-week cycle *(new)* | 0.100 |
| 17 | Share of days far below the usual level | 0.080 |

**Interpretation.** Variability, data quality, trend and peak usage dominate, consistent with the EDA. Two of the three features we
added in notebook 03 rank 6th and 9th. Cautions: (1) importance is not causality. Missing readings may reflect tampering *or* how
SGCC collected the data. (2) Zero-usage features rank low (zero share 0.106), and as customer 3 shows, zero usage tends to *lower*
the risk. This matches notebook 04, where synthetic "zero" and "reduced" attacks were rarely detected.

## Isolation Forest (abnormal consumption)

**What it does.** 300 random trees split the 17-feature space at random cut points. Unusual customers are isolated in few splits,
so they get a high score. It is fitted **only on normal training customers** ("anomalous" = unlike a typical normal customer), and
the raw score is rescaled to 0-1 using the 1st-99th percentile range of the training split. It needs no labels.

**Results (test split, `reports/metrics.json`):** ROC-AUC **0.586**, PR-AUC **0.109**, theft caught in the top 10% **13.8%**
(precision there 11.8%). Random ranking gives ≈ 0.085 and 10%. XGBoost gives 0.376 and 43.5%.

**On the 1,000 demo customers:** anomaly and risk are essentially uncorrelated (Spearman −0.039). Median anomaly is 0.175 for normal
and 0.196 for theft customers. Of the 204 customers with anomaly ≥ 0.5, only 4.4% are thieves, *below* the 8.5% base rate. The 100
most anomalous include 7% thieves; the 100 highest-risk include 37%.

**Meaning and limits.** Isolation Forest is a weak theft *ranker* (barely better than random), because "abnormal" is not "theft".
Its value is as a second, label-free signal for strange behaviour the labels do not cover, like customer 3's three years of zero
readings (anomaly 0.90, theft risk 0.0025). It also works in a new region with no inspection history. It uses the same handcrafted
features and a relative 0-1 scale, so it only sees the kinds of abnormality those features describe.

## Overall: how good are the High-risk flags?

Using the hidden demo labels: of the 80 High-band demo customers, **34 (42.5%) are thieves**, vs 9.8% in Medium and 3.9% in Low.
The demo set has 85 thieves: 34 in High, 25 in Medium, 26 in Low. So the High band concentrates theft about 5× above the base rate,
but about 3 in 10 thieves still look normal to the model. SHAP makes every flag checkable; it does not make the model more
accurate.
