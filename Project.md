# Project Overview

## Working Title
Loan default risk modeling and decision support

## Description
This project uses large-scale historical loan data to analyze borrower behavior, identify patterns associated with repayment risk, and build a machine learning model that predicts the probability of loan default.

The workflow starts with data exploration and cleaning on a large Lending Club style dataset using scalable tools such as PySpark. From there, the project moves through feature engineering, model development, temporal evaluation, and threshold analysis so that predicted probabilities can support real credit decisions.

The intended scoring layer supports three business decisions:

1. Estimating the probability that an applicant will default.
2. Recommending whether the loan should be approved, reviewed, or declined.
3. Suggesting a risk-adjusted interest rate for approved applicants.

## Core Objectives
1. Understand the structure and quality of the raw loan dataset.
2. Build a reliable preprocessing and feature engineering pipeline.
3. Train and compare models for default prediction.
4. Evaluate model performance using metrics appropriate for imbalanced credit risk data.
5. Translate predicted risk into practical lending decisions.
6. Prepare the solution for deployment as a scoring service or application.

## Expected Deliverables
1. A scalable exploratory data analysis workflow in Jupyter and Spark.
2. A cleaned and modeled dataset ready for experimentation.
3. A baseline and improved default prediction model.
4. A documented decision framework for approvals and pricing.
5. A deployable scoring component for new loan applications.

## Why This Matters
Accurate default prediction helps reduce credit losses, improve pricing decisions, and make lending more consistent. A strong model can support better portfolio quality while still allowing the lender to serve qualified borrowers efficiently.

# Credit Risk Modeling Report

## Executive Summary
This project now has a complete temporal credit-risk evaluation workflow. The workflow prepares more than two million historical loan records, defines a clean resolved-loan default target, selects features using training-period data only, compares multiple model families on a later validation period, tunes the leading gradient-boosted tree, and evaluates the selected model once on the newest untouched test period.

The authoritative result is:

```text
tuned_gbt_full_temporal
```

| Evaluation | Rows | Default rate | ROC AUC | PR AUC | Brier score | Top-decile default rate | Top-decile lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Final untouched temporal test | `148,972` | `20.23%` | `0.6843` | `0.3453` | `0.1511` | `41.62%` | `2.0569` |

The model's highest-risk 10% had an actual default rate of `41.62%`, compared with `20.23%` across the complete test period. Therefore, the highest-risk decile was slightly more than twice as risky as the average test loan.

The earlier random-sample tuned GBT achieved ROC AUC `0.7032` and top-decile lift `2.3230`. Those results remain useful as model-development evidence, but the lower temporal test result is the more credible estimate of future-period performance because the model is evaluated on loans originated after the training and validation periods.

## Target Definition And Eligible Population
The original CSV contains `2,260,668` loans. Cleaning and conversion to Parquet preserve all source rows. Rows are removed from the modeling population only when defining a target that requires a known final repayment outcome.

The binary target is:

| `default_flag` | Loan statuses |
|---:|---|
| `0` | `Fully Paid`, `Does not meet the credit policy. Status:Fully Paid` |
| `1` | `Charged Off`, `Default`, `Does not meet the credit policy. Status:Charged Off` |

Unresolved statuses such as `Current`, `In Grace Period`, and late loans are excluded. A current loan cannot safely be labeled as non-default because it may still default later.

Resolved target counts:

| Loan status | `default_flag` | Count |
|---|---:|---:|
| `Fully Paid` | `0` | `1,041,952` |
| `Does not meet the credit policy. Status:Fully Paid` | `0` | `1,988` |
| `Charged Off` | `1` | `261,654` |
| `Default` | `1` | `31` |
| `Does not meet the credit policy. Status:Charged Off` | `1` | `761` |
| **Total resolved modeling population** | | **`1,306,386`** |

The complete resolved population has an overall default rate of approximately `20.09%`.

## Data Preparation And Leakage Control
The PySpark workflow:

1. Trims and standardizes raw string values.
2. Normalizes fields such as `term`, `emp_length`, and ZIP prefixes.
3. Parses valid month-year fields into dates and flags malformed raw values.
4. Groups normalized loan-title values into `title_grouped`.
5. Creates a faster Parquet working snapshot.
6. Creates meaningful missingness indicators where absence may carry risk information.
7. Excludes fields that would not be available at application time.

The main independent model excludes:

- repayment outcomes and servicing fields;
- payments, recoveries, outstanding principal, and late fees;
- hardship and settlement fields;
- future payment and credit-pull dates;
- identifiers and unstructured raw text;
- Lending Club pricing and underwriting outputs such as `grade`, `sub_grade`, `int_rate`, and `installment`.

This prevents the model from appearing strong by learning information that becomes available only after the lending decision or by copying Lending Club's existing underwriting decision.

## Rigorous Temporal Evaluation Design
The final experiment uses complete issue months and preserves chronological order:

| Split | Rows | Default rate | First issue month | Last issue month | Purpose |
|---|---:|---:|---|---|---|
| Training | `1,007,621` | `19.47%` | June 2007 | July 2016 | Feature selection and model fitting |
| Validation | `149,793` | `24.13%` | August 2016 | April 2017 | Architecture comparison, GBT tuning, threshold selection |
| Test | `148,972` | `20.23%` | May 2017 | December 2018 | One final untouched evaluation |

The training set was requested to contain at least one million records. Because complete origination months remain together, the training split contains `1,007,621` loans rather than splitting July 2016 between datasets.

The changing default rates across time are important. Validation default rate increased to `24.13%`, while the later test period returned to `20.23%`. This is evidence of population or outcome drift and explains why chronological evaluation is stricter than a random split.

## Training-Only Feature Shortlisting
Feature selection is performed using a reproducible sample of `200,000` loans from the training period only. Validation and test outcomes do not influence the shortlist.

### Numeric Statistical Utility
Numeric features are measured using:

| Statistic | Definition | Purpose |
|---|---|---|
| Information value | `sum((bad_distribution - good_distribution) * WOE)` | Measures separation across feature ranges |
| KS statistic | `max abs(F_default(x) - F_non_default(x))` | Measures maximum distribution separation |
| Standardized mean difference | `(mean_default - mean_non_default) / pooled_std` | Measures group difference on a comparable scale |
| Absolute Pearson correlation | `abs(corr(feature, default_flag))` | Measures linear association with default |

The measurements use different scales, so each is converted into a percentile rank relative to the other candidate features:

```text
numeric statistical utility =
    0.35 * IV percentile rank
  + 0.25 * KS percentile rank
  + 0.20 * standardized-effect percentile rank
  + 0.20 * absolute-correlation percentile rank
```

### Categorical Statistical Utility
Categorical features are measured using:

| Statistic | Definition | Purpose |
|---|---|---|
| Information value | IV calculated across grouped categories | Measures good/bad separation |
| Cramer's V | Chi-square association strength | Measures overall association with default |
| Default-rate range | Highest category default rate minus lowest | Measures practical risk differences between categories |

```text
categorical statistical utility =
    0.45 * IV percentile rank
  + 0.35 * Cramer's V percentile rank
  + 0.20 * default-rate-range percentile rank
```

The statistical utility score is a relative ranking score, not a probability or accuracy measure. A utility score of `0.98` means a feature ranked very strongly across the selected statistics compared with other candidates.

### Diversification Across Risk Families
Taking the raw top 15 could over-select many closely related features. The workflow assigns features to business families such as debt burden, credit utilization, credit activity, loan size, income, housing, and inquiries. It then prioritizes the strongest feature from each family before selecting additional features from the same family.

This creates a model that uses multiple risk themes instead of relying heavily on many versions of one borrower behavior.

## Final Training-Only Feature Set
The temporal model uses these 15 application-time features:

| Rank | Feature | Type | Risk family | Statistical utility | Information value |
|---:|---|---|---|---:|---:|
| 1 | `acc_open_past_24mths` | Numeric | Credit activity | `0.9841` | `0.0888` |
| 2 | `dti` | Numeric | Debt burden | `0.9598` | `0.0795` |
| 3 | `all_util` | Numeric | Credit utilization | `0.9159` | `0.0579` |
| 4 | `verification_status` | Categorical | Income verification | `0.8857` | `0.0518` |
| 5 | `title_grouped` | Categorical | Loan purpose | `0.8571` | `0.0319` |
| 6 | `tot_cur_bal` | Numeric | Total balance | `0.8549` | `0.0418` |
| 7 | `mort_acc` | Numeric | Mortgage history | `0.8293` | `0.0330` |
| 8 | `mo_sin_rcnt_tl` | Numeric | Credit recency | `0.8183` | `0.0413` |
| 9 | `loan_amnt` | Numeric | Loan size | `0.7329` | `0.0323` |
| 10 | `num_rev_tl_bal_gt_0` | Numeric | Active accounts | `0.7256` | `0.0296` |
| 11 | `home_ownership` | Categorical | Housing | `0.6857` | `0.0276` |
| 12 | `annual_inc` | Numeric | Income | `0.6841` | `0.0330` |
| 13 | `term` | Numeric | Loan structure | `0.6799` | `0.0000` |
| 14 | `total_rev_hi_lim` | Numeric | Revolving capacity | `0.6390` | `0.0283` |
| 15 | `inq_fi` | Numeric | Credit inquiries | `0.6037` | `0.0170` |

`term` has zero information value in the univariate IV calculation but remains useful through other statistics, diversification, and interactions with other model inputs. This illustrates why feature selection should not rely on one statistic alone.

## Validation Architecture Comparison
Three independent model families were fitted on all `1,007,621` training loans and compared on the later validation period:

| Model | Validation ROC AUC | Validation PR AUC | Brier score | Top-decile default rate | Top-decile lift |
|---|---:|---:|---:|---:|---:|
| `gradient_boosted_trees_full_temporal` | `0.6747` | `0.3928` | `0.1706` | `47.61%` | `1.9733` |
| `logistic_regression_full_temporal` | `0.6687` | `0.3840` | `0.1729` | `46.67%` | `1.9343` |
| `random_forest_full_temporal` | `0.6378` | `0.3612` | `0.1784` | `44.52%` | `1.8452` |

Gradient-boosted trees won the validation comparison across ROC AUC, PR AUC, Brier score, top-decile default rate, and top-decile lift. Logistic regression remains the interpretability benchmark because coefficients and odds ratios are easier to explain. Random forest was materially weaker and was not carried forward.

The notebook also compares the models visually using:

- ROC curves;
- precision-recall curves;
- cumulative lift curves;
- cumulative gains curves;
- calibration curves.

These plots show model behavior across many possible thresholds rather than relying on a single classification cutoff.

## Gradient-Boosted Tree Tuning
Four GBT configurations were evaluated on validation:

| Profile | Depth | Iterations | Step size | Minimum instances per node | Subsampling | ROC AUC | PR AUC | Brier | Top-decile lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `deeper_subsampled` | `5` | `100` | `0.040` | `300` | `0.7` | `0.6785` | `0.3969` | `0.1700` | `1.9940` |
| `deeper_slow` | `5` | `140` | `0.025` | `400` | `0.8` | `0.6769` | `0.3951` | `0.1702` | `1.9824` |
| `balanced_regularized` | `4` | `100` | `0.040` | `300` | `0.8` | `0.6733` | `0.3903` | `0.1710` | `1.9429` |
| `shallow_regularized` | `3` | `100` | `0.040` | `400` | `0.8` | `0.6690` | `0.3866` | `0.1717` | `1.9376` |

The selected profile was:

```text
deeper_subsampled
maxDepth = 5
maxIter = 100
stepSize = 0.04
minInstancesPerNode = 300
subsamplingRate = 0.7
```

It produced the strongest validation ROC AUC, PR AUC, Brier score, and top-decile lift. The final model was then retrained on the combined training and validation periods before being evaluated on the untouched test period.

## Final Untouched Test Result
The selected GBT was refitted on `1,157,414` combined training and validation records. It was then evaluated once on the `148,972` newest-period test loans.

| Model | Rows | Default rate | ROC AUC | PR AUC | Brier score | Top-decile default rate | Top-decile lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tuned_gbt_full_temporal` | `148,972` | `20.23%` | `0.6843` | `0.3453` | `0.1511` | `41.62%` | `2.0569` |

### Interpretation
- **ROC AUC `0.6843`:** the model ranks a randomly selected default above a randomly selected non-default approximately `68.43%` of the time.
- **PR AUC `0.3453`:** the model improves meaningfully over the test default-rate baseline of `20.23%` when identifying defaults.
- **Brier score `0.1511`:** predicted probabilities contain useful information, but calibration can still be improved.
- **Top-decile default rate `41.62%`:** approximately 42 of every 100 loans in the model's highest-risk decile actually defaulted.
- **Top-decile lift `2.0569`:** the highest-risk decile was approximately 2.06 times as risky as the average test loan.

## Threshold Policy Analysis
A working probability threshold was selected using validation only. The validation threshold targeted approximately the highest-risk 10%:

```text
selected probability threshold = 0.387615
```

When this unchanged threshold was applied to the final test period:

| Metric | Test result |
|---|---:|
| Review or rejection share | `10.72%` |
| Approval share | `89.28%` |
| Flagged-group default rate / precision | `41.12%` |
| Share of all defaults captured / recall | `21.79%` |
| Approved-group default rate | `17.72%` |
| Correctly flagged defaults | `6,568` |
| Flagged loans that did not default | `9,406` |
| Approved loans that defaulted | `23,573` |
| Approved loans that did not default | `109,425` |

The threshold flags `15,974` of the `148,972` test loans. Of those flagged loans, `6,568` defaulted. This concentrates risk effectively, but it still leaves `23,573` defaults in the approved group.

The threshold is therefore a useful working policy example, not a production approval rule. A final lending threshold must include:

- expected loan revenue;
- exposure at default;
- loss given default;
- review and rejection costs;
- approval-volume targets;
- risk appetite and regulatory constraints.

## Metric Definitions

| Metric | Definition | Interpretation |
|---|---|---|
| ROC AUC | `P(score_default > score_non_default)` | Overall ranking quality across thresholds |
| PR AUC | Area under the precision-recall curve | Default-class identification under imbalance |
| Brier score | `mean((predicted_probability - actual_outcome)^2)` | Probability accuracy and calibration; lower is better |
| Top-decile default rate | Default rate among the highest-risk 10% | Risk concentration in the model's highest-ranked group |
| Top-decile lift | `top_decile_default_rate / overall_default_rate` | How many times riskier the top decile is than average |
| Precision | `TP / (TP + FP)` | Share of flagged loans that default |
| Recall | `TP / (TP + FN)` | Share of all defaults captured by the threshold |

## Exploratory Sample Versus Temporal Evaluation

| Evaluation design | Tuned GBT ROC AUC | PR AUC | Brier | Top-decile lift |
|---|---:|---:|---:|---:|
| Earlier random-sample test | `0.7032` | `0.3742` | `0.1456` | `2.3230` |
| Final chronological test | `0.6843` | `0.3453` | `0.1511` | `2.0569` |

The temporal result is weaker but more trustworthy. Random splits mix loans from different origination periods and make training and testing populations more similar. Temporal testing asks the harder and more realistic question: can a model trained on older loans rank risk for loans originated later?

## Final Modeling Decision

| Role | Model | Decision |
|---|---|---|
| Performance champion | `tuned_gbt_full_temporal` | Retain as the current final candidate |
| Interpretability benchmark | `logistic_regression_full_temporal` | Retain for coefficient and odds-ratio explanation |
| Challenger not retained | `random_forest_full_temporal` | Weaker validation ranking and calibration |
| Historical exploratory result | `tuned_gradient_boosted_trees_top_15` | Keep as evidence from the earlier random-sample experiment |

## Limitations
1. Only resolved loans are included. More recent origination periods may exclude unresolved loans that have not yet had enough time to default or repay, creating outcome-maturity bias.
2. The final probabilities have not yet been formally calibrated.
3. The working threshold does not yet include expected loss, revenue, or operational costs.
4. Segment-level fairness and error analysis have not yet been completed.
5. The fitted model pipeline has not yet been serialized into a production scoring artifact.
6. No production monitoring framework currently exists for data drift, score drift, calibration drift, or realized default performance.

## Recommended Next Steps
1. Define a fixed performance window so every loan has equal time to reach the target outcome.
2. Calibrate the final GBT probabilities using validation-period data.
3. Add expected-loss and profitability calculations to the threshold policy table.
4. Evaluate performance and calibration by `home_ownership`, `verification_status`, `term`, loan purpose, and other important segments.
5. Compare the tuned GBT and logistic benchmark for stability across origination periods.
6. Save the final preprocessing and model pipeline as a repeatable scoring artifact.
7. Define the application-time input contract and build a scoring service.
8. Add production monitoring for drift, calibration, threshold outcomes, and realized losses.

## Bottom Line
The project has progressed from exploratory analysis to a rigorous full-data temporal evaluation. The final tuned GBT demonstrates useful risk-ranking ability on future-period loans: its highest-risk decile defaults at `41.62%`, approximately `2.06` times the test-period average.

The model is a credible decision-support candidate, but it is not yet a complete automated lending policy. The next stage is to improve probability calibration, incorporate lending economics into threshold selection, verify performance across borrower segments and time periods, and package the model for repeatable scoring and monitoring.
