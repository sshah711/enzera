# ENZERA - Explainability write-up

All numbers below come from the running backend (`GET /customers`, `GET /customers/{id}`) for model
`xgb-20261114-1143`, from `reports/metrics.json`, and from `data/demo/demo_labels.csv` (the true labels of the demo
customers, which the app never shows). The demo customers come from the 15% demo split, which the model never saw in
training.

## How to read a risk score and its reasons

- **Risk score** = XGBoost's predicted theft probability (0-1). **Risk band**: High if score ≥ the threshold
  0.673 (tuned for best F1 on the validation split), Medium if ≥ half the threshold (0.337), otherwise Low.
- **SHAP reasons.** For each customer, XGBoost's built-in TreeSHAP (`booster.predict(..., pred_contribs=True)`)
  splits the model's output (in log-odds) into one contribution per feature plus a base value:

  `log-odds = base value + SHAP(feature 1) + ... + SHAP(feature 14)`,   `risk = 1 / (1 + e^(-log-odds))`

  The base value for this model is **0.159 log-odds** (it is not the 8.5% theft rate, because class weights
  re-balance the training). A positive SHAP value pushes the customer towards theft, a negative one away from it. The app
  shows the top 3 *positive* contributors as the "reasons". We checked that the contributions add up exactly to the
  displayed score for all three customers below.

## Customer 1 - `57531380F73D67943259F07E12498E1A` (highest risk)

**Risk 0.983 (High), anomaly score 1.00. True label: theft.**
SHAP reasons: day-to-day variation 228.11 kWh (**+1.75**), missing readings 57% (**+0.61**), average daily consumption
220.44 kWh (**+0.59**). Other positive contributors: long-term trend (+0.38) and highest daily consumption (+0.26).
Base 0.159 + all contributions = 4.066 log-odds → 0.983.

**What the consumption shows.** This is an extreme consumer: 220 kWh/day on average and a peak of 905 kWh in one day,
against a demo median of 5.2 kWh/day. It is probably a commercial or industrial connection. Its readings are patchy (57% of days
missing, with gaps of several months in 2014 and 2015), and its usage swings widely: monthly means range from about 37
to 645 kWh/day. In 2016 usage first jumps to 494-645 kWh/day and then falls to 109-279 kWh/day, so recent usage is only
56% of earlier usage. The model has learnt that very large, very variable consumers with patchy meter data are the
profile most often labelled as theft in SGCC. This customer is also the most anomalous in the demo set (anomaly score capped at
1.0). It is a sensible first inspection, but the "reason" is a risk *profile*, not proof of tampering.

## Customer 2 - `448FE0108C91D70C0E906B7D5AE496E7` (second highest risk)

**Risk 0.963 (High), anomaly score 0.33. True label: theft.**
SHAP reasons: day-to-day variation 13.86 kWh (**+0.80**), long-term trend of monthly usage 0.05 (**+0.66**), missing
readings 29% (**+0.33**). Almost every feature pushes upwards here (13 of 14 contributions are positive), including the
largest month-to-month drop of 65% (+0.10). Base 0.159 + contributions = 3.247 log-odds → 0.963.

**What the consumption shows.** This customer has the classic theft pattern. In January-June 2014 it used 16-24 kWh/day.
In July 2014 usage suddenly fell by 65% (24.1 → 8.4 kWh/day) and stayed at only 3-8 kWh/day for about nine months (July 2014 - March 2015).
Then readings disappear for April-December 2015. In 2016 usage is back at 20-57 kWh/day, higher than ever. A sharp, sustained
drop followed by a return to (or above) the old level is what we would expect from a meter bypass that was later removed.
The model captures this mainly through variability and trend rather than the drop feature itself: the 2016 recovery makes
the trend strongly positive and recent usage 2.35× the earlier level. Its anomaly score (0.33) is only moderate,
which shows that the supervised model catches theft patterns the unsupervised detector does not flag.

## Customer 3 - `50B1489EA4125827439E4C0E97B28DFC` (low risk)

**Risk 0.003 (Low), anomaly score 0.27. True label: normal.** We chose the lowest-risk demo customer with under 10% missing
data. Several low-risk customers with mostly missing data would be less informative examples.
The listed "reasons" are tiny: trend 0.01 (+0.06) and low-day share 0% (+0.01), so in practice nothing pushes this customer
towards theft. The score is driven by strong *negative* contributions: missing readings only 2.4% (**−1.15**), highest daily
consumption 0.71 kWh (**−1.13**), day-to-day variation 0.10 kWh (**−0.88**), spike share 0% (−0.60). Base 0.159 +
contributions = −5.803 log-odds → 0.003.

**What the consumption shows.** A very small, very stable household-type load: monthly means sit at 0.5 kWh/day from January 2014
to September 2015 and creep up to 0.6-0.7 kWh/day from October 2015 onwards. There are no gaps, spikes, zero runs or drops. Everything about this curve is
regular, so the model is confident. *Caveat for interpretation:* the reasons list always shows the top positive contributors,
even when they are negligible. For a low-risk customer, the size of the SHAP values (+0.06 here vs +1.75 for customer 1)
matters more than the labels.

## Global importance (`reports/shap_importance.png`)

The chart shows the mean absolute SHAP value of each feature over the 6,356 test customers, i.e. how much each feature
moves the prediction on average, in either direction (values from `reports/metrics.json`):

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | Share of days with missing readings | 0.485 |
| 2 | Day-to-day variation (kWh) | 0.412 |
| 3 | Highest daily consumption (kWh) | 0.331 |
| 4 | Long-term trend of monthly usage | 0.293 |
| 5 | Variability between months | 0.216 |
| 6 | Share of days with unusual spikes | 0.210 |
| ... | ... | ... |
| 14 | Share of days far below the usual level | 0.103 |

**Interpretation.** Data quality (missing readings), consumption magnitude and variability (std, peak), and change over time
(trend) dominate. Zero-usage and "far below usual" features matter least. This matches the EDA (`notebooks/01_eda.ipynb`):
in SGCC, theft-labelled customers are on average larger, more variable consumers with more missing data and rising recent
usage, and zero-usage runs are rare in both classes. Two cautions: (1) importance is not causality. Missing readings may
reflect faulty or tampered meters, *or* just how SGCC collected the data. (2) `missing_ratio` ranks first in SHAP but only
8th in single-feature AUC (0.568), because the model uses it mostly *in combination* with other features (for example,
it pushes strongly downwards for clean, stable customers like customer 3).

## Isolation Forest (abnormal consumption)

**What it does.** Isolation Forest builds 300 random trees that repeatedly split the 14-feature space at random cut points.
Unusual customers get isolated in few splits (short paths), so they receive a high anomaly score. We fit it **only on normal
customers** from the training split, so "anomalous" means "unlike a typical normal customer". The raw score is rescaled to 0-1
using the 1st-99th percentile range on the training split (so a few customers sit exactly at 0 or 1). It needs no labels.

**Results (test split, from `reports/metrics.json`):** ROC-AUC **0.595**, PR-AUC **0.115**, theft caught in the top 10% **16.0%**
(precision in the top 10%: 13.7%). Random ranking would give PR-AUC ≈ 0.085 and 10%. XGBoost gives 0.332 and 40.5%.

**On the 1,000 demo customers:** risk score and anomaly score are essentially uncorrelated (Spearman 0.001). Median anomaly score is
0.150 for normal and 0.161 for theft customers. Of the 198 customers with anomaly ≥ 0.5, only 5.1% are thieves, which is
*below* the 8.5% base rate. The 100 most anomalous customers include 6% thieves, while the 100 highest-risk customers include
36%.

**What this means and its limits.**
- Isolation Forest is a weak theft detector on its own: only slightly better than random. "Abnormal" is not the same as "theft".
  Many normal customers are unusual (holiday homes, businesses, new connections), and many thieves look ordinary.
- Its value in ENZERA is as a **second, label-free signal**. It can point at strange behaviour that no historical label covers
  (new theft methods, meter faults, data-collection errors), and it would still work in a new region with no inspection
  history. It should not be used to rank inspections; the supervised risk score should.
- It uses the same 14 handcrafted features, so it can only see the kinds of abnormality those features describe. The 0-1 scaling
  is relative to our training data, not an absolute probability.

## Overall: how good are the High-risk flags?

Using the hidden demo labels: of the 73 High-band demo customers, **31 (42.5%) are thieves**, vs 10.3% in Medium and 4.1% in
Low. The demo set has 85 thieves in total: 31 are in High, 27 in Medium and 27 in Low. So the model concentrates theft about 5×
above the base rate in its High band, but about a third of thieves still look normal to it. SHAP explanations make each flag
checkable by an inspector. They do not make the model more accurate.
