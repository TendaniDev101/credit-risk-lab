# Project Overview

## Working Title
Loan default risk modeling and decision support

## Description
This project uses large-scale historical loan data to analyze borrower behavior, identify patterns associated with repayment risk, and build a machine learning model that predicts the probability of loan default.

The workflow starts with data exploration and cleaning on a large Lending Club style dataset using scalable tools such as PySpark. From there, the project will move into feature engineering, model development, evaluation, and calibration so that predicted probabilities are useful in a real credit decision setting.

If model performance is strong enough, the final stage of the project will be deployment into an application environment where new loan applications can be scored in real time. That scoring layer is intended to support three business decisions:

1. Estimating the probability that an applicant will default.
2. Recommending whether the loan should be approved or declined.
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
This project built a credit-risk modeling workflow for predicting whether a resolved loan will default. The work started with scalable data preparation in PySpark, moved through statistical feature analysis, and then compared multiple model architectures on the same default-prediction task.

The target variable was `default_flag`, where resolved loans were labeled as:

| Target value | Loan statuses |
|---:|---|
| `0` | `Fully Paid`, `Does not meet the credit policy. Status:Fully Paid` |
| `1` | `Charged Off`, `Default`, `Does not meet the credit policy. Status:Charged Off` |

Unresolved statuses such as `Current`, `In Grace Period`, and late-stage active loans were excluded because the final repayment outcome was not yet known.

The final modeling dataset contained `1,306,386` resolved loans with a default rate of `20.09%`. Model experiments used a reproducible sample of `350,000` rows, split into `279,890` training rows and `70,110` held-out test rows. The modeling sample default rate was `19.99%`.

The strongest model so far is:

| Champion model | Feature set | ROC AUC | PR AUC | Brier score | Top-decile default rate | Top-decile lift |
|---|---|---:|---:|---:|---:|---:|
| Tuned gradient-boosted trees | Top 15 independent features | `0.7032` | `0.3742` | `0.1456` | `46.34%` | `2.3230` |

The tuned gradient-boosted tree is the best performance model. Logistic regression remains the best interpretability benchmark.

## Data Preparation And Target Definition
The dataset was first standardized and converted to a faster Parquet working snapshot. The workflow cleaned or transformed fields such as `term`, `emp_length`, date columns, and missingness indicators.

Outcome, servicing, and post-origination fields were excluded from the modeling feature set to reduce leakage. Examples include fields related to payments, recoveries, loan balance after origination, final payment dates, and settlement outcomes. Pricing/risk-grade fields such as `grade`, `sub_grade`, `int_rate`, and `installment` were kept out of the main independent model and used only as a benchmark comparison.

Resolved target counts:

| Loan status | `default_flag` | Count |
|---|---:|---:|
| `Fully Paid` | `0` | `1,041,952` |
| `Does not meet the credit policy. Status:Fully Paid` | `0` | `1,988` |
| `Charged Off` | `1` | `261,654` |
| `Default` | `1` | `31` |
| `Does not meet the credit policy. Status:Charged Off` | `1` | `761` |

Candidate modeling features after leakage review:

| Feature type | Count |
|---|---:|
| Numeric baseline candidates | `41` |
| Categorical baseline candidates | `7` |
| Excluded leakage/pricing/raw columns present | `50` |

## How Feature Usefulness Was Determined
Feature usefulness was not based only on raw correlation with the target. Correlation is useful, but it mainly captures linear relationships and can miss nonlinear risk patterns, categorical effects, and feature interactions. The feature-selection process used several complementary statistics.

### Numeric Feature Statistics
For numeric features, the following statistics were calculated:

| Statistic | Formula / definition | Why it matters |
|---|---|---|
| Pearson correlation | `corr(x, y) = cov(x, y) / (std(x) * std(y))` | Measures linear association with default. |
| Standardized mean difference | `(mean_default - mean_non_default) / pooled_std` | Measures how far apart default and non-default borrowers are on the feature scale. |
| KS statistic | `max_x |F_default(x) - F_non_default(x)|` | Measures the maximum separation between the default and non-default distributions. |
| Information value | `sum((bad_dist_b - good_dist_b) * WOE_b)` | Measures how much a binned variable separates good and bad loans. |
| Weight of evidence | `WOE_b = ln(bad_dist_b / good_dist_b)` | Measures the direction and strength of risk in each bin. |

The numeric utility score combined percentile ranks:

```text
numeric_score =
    0.35 * IV_rank
  + 0.25 * KS_rank
  + 0.20 * standardized_effect_rank
  + 0.20 * abs_correlation_rank
```

### Categorical Feature Statistics
For categorical features, categories were profiled by default rate and grouped where needed. The following statistics were used:

| Statistic | Formula / definition | Why it matters |
|---|---|---|
| Information value | Same IV framework after grouping categories | Measures categorical separation of good and bad loans. |
| Cramer's V | `sqrt((chi_square / n) / min(r - 1, c - 1))` | Measures association strength between category and default. |
| Default-rate range | `max_category_default_rate - min_category_default_rate` | Shows how much risk differs between categories. |
| Category lift | `category_default_rate / overall_default_rate` | Shows categories with unusually high or low default risk. |

The categorical utility score combined percentile ranks:

```text
categorical_score =
    0.45 * IV_rank
  + 0.35 * Cramers_V_rank
  + 0.20 * default_rate_range_rank
```

### Diversified Feature Shortlist
After raw ranking, features were grouped into business families such as debt burden, credit utilization, income verification, housing, loan size, credit recency, and credit inquiries. The shortlist was then diversified so the first model did not over-concentrate on many versions of the same borrower behavior.

This produced a top-15 feature set that balanced statistical strength with business coverage.

## Most Useful Features Found
The top-15 independent feature set used for the main models was:

| Rank | Feature | Type | Business interpretation |
|---:|---|---|---|
| 1 | `dti` | Numeric | Debt burden relative to income. |
| 2 | `acc_open_past_24mths` | Numeric | Recent credit account-opening activity. |
| 3 | `all_util` | Numeric | Overall credit utilization. |
| 4 | `mort_acc` | Numeric | Mortgage history and depth of credit profile. |
| 5 | `tot_cur_bal` | Numeric | Total current balance across accounts. |
| 6 | `num_rev_tl_bal_gt_0` | Numeric | Active revolving accounts with balances. |
| 7 | `verification_status` | Categorical | Whether reported income was verified. |
| 8 | `loan_amnt` | Numeric | Requested loan size. |
| 9 | `home_ownership` | Categorical | Housing status and stability signal. |
| 10 | `title_grouped` | Categorical | Grouped loan title or stated borrower intent. |
| 11 | `mo_sin_rcnt_tl` | Numeric | Months since most recent account opened. |
| 12 | `annual_inc` | Numeric | Borrower income. |
| 13 | `inq_last_6mths` | Numeric | Recent credit inquiries. |
| 14 | `term` | Numeric | Loan duration, usually 36 or 60 months. |
| 15 | `total_rev_hi_lim` | Numeric | Total revolving credit limit. |

The strongest raw statistical features were:

| Raw rank | Feature | Utility score | IV | KS | Correlation / Cramer's V |
|---:|---|---:|---:|---:|---:|
| 1 | `dti` | `0.9793` | `0.0758` | `0.1136` | `0.0995` correlation |
| 2 | `acc_open_past_24mths` | `0.9646` | `0.0620` | `0.1022` | `0.1005` correlation |
| 3 | `all_util` | `0.9354` | `0.0564` | `0.1018` | `0.0943` correlation |
| 4 | `mort_acc` | `0.8976` | `0.0390` | `0.0917` | `-0.0769` correlation |
| 5 | `tot_cur_bal` | `0.8854` | `0.0419` | `0.0907` | `-0.0728` correlation |
| 6 | `open_rv_24m` | `0.8659` | `0.0377` | `0.0807` | `0.0812` correlation |
| 7 | `num_rev_tl_bal_gt_0` | `0.8305` | `0.0340` | `0.0763` | `0.0742` correlation |
| 8 | `verification_status` | `0.8286` | `0.0601` | N/A | `0.0954` Cramer's V |
| 9 | `loan_amnt` | `0.8061` | `0.0339` | `0.0819` | `0.0657` correlation |
| 10 | `home_ownership` | `0.8000` | `0.0338` | N/A | `0.0737` Cramer's V |

Important categorical patterns:

| Categorical feature | IV | Cramer's V | Notable risk pattern |
|---|---:|---:|---|
| `verification_status` | `0.0601` | `0.0954` | `Verified` loans had a higher sampled default rate (`24.06%`) than `Not Verified` loans (`14.53%`). |
| `home_ownership` | `0.0338` | `0.0737` | `RENT` had a higher sampled default rate (`23.45%`) than `MORTGAGE` (`17.18%`). |
| `title_grouped` | `0.0272` | `0.0661` | Business-related and missing titles showed elevated risk. |
| `purpose` | `0.0213` | `0.0583` | Small-business and renewable-energy purposes showed higher risk than car and wedding purposes. |

Useful interaction patterns were also found. For example, high recent account-opening activity combined with missing or risky title groups showed materially higher default rates. High loan amount combined with business-related purpose/title was also elevated. These interaction patterns help explain why tree-based and boosted-tree models can improve over a purely linear model.

## Feature Importance From The Final Model
The tuned gradient-boosted tree learned a different but related importance ordering. This ranking is model-based, not purely statistical.

| Rank | Feature | Tuned GBT importance |
|---:|---|---:|
| 1 | `total_rev_hi_lim` | `0.1472` |
| 2 | `loan_amnt` | `0.1349` |
| 3 | `num_rev_tl_bal_gt_0` | `0.0887` |
| 4 | `dti` | `0.0805` |
| 5 | `annual_inc` | `0.0695` |
| 6 | `acc_open_past_24mths` | `0.0639` |
| 7 | `inq_last_6mths` | `0.0637` |
| 8 | `all_util` | `0.0593` |
| 9 | `term` | `0.0536` |
| 10 | `verification_status` | `0.0509` |
| 11 | `title_grouped` | `0.0478` |
| 12 | `mort_acc` | `0.0362` |
| 13 | `home_ownership` | `0.0354` |
| 14 | `mo_sin_rcnt_tl` | `0.0347` |
| 15 | `tot_cur_bal` | `0.0338` |

This profile is healthy because the model is not relying on a single variable. It uses a broad combination of revolving capacity, requested loan size, active revolving debt, debt burden, income, recent credit activity, utilization, loan term, verification status, and borrower intent.

## Model Architectures Tested
Three modeling stages were tested:

1. Logistic regression as the baseline and interpretability benchmark.
2. Tree-based models: decision tree and random forest.
3. Gradient-boosted trees, then tuned gradient-boosted trees.

All main architecture comparisons used the same top-15 independent feature set.

## Architecture 1: Logistic Regression
Logistic regression models the log-odds of default as a linear function of the features:

```text
z_i = beta_0 + beta_1*x_i1 + ... + beta_p*x_ip
p_i = P(y_i = 1 | x_i) = 1 / (1 + exp(-z_i))
logit(p_i) = log(p_i / (1 - p_i)) = z_i
```

The model is trained by minimizing binary log loss, with regularization:

```text
Loss = -sum(y_i*log(p_i) + (1 - y_i)*log(1 - p_i)) + lambda * ||beta||_2^2
```

Interpretation:

```text
odds_ratio_j = exp(beta_j)
```

If `odds_ratio_j > 1`, the feature increases estimated default odds, holding other features constant. If `odds_ratio_j < 1`, it reduces estimated default odds.

Logistic regression results:

| Logistic feature set | Features | ROC AUC | PR AUC | Brier score | Top-decile default rate | Top-decile lift |
|---|---:|---:|---:|---:|---:|---:|
| `top_5_plus_pricing_benchmark` | `9` | `0.6965` | `0.3588` | `0.1469` | `43.70%` | `2.1908` |
| `top_15` | `15` | `0.6870` | `0.3585` | `0.1477` | `44.57%` | `2.2344` |
| `top_10` | `10` | `0.6475` | `0.3091` | `0.1528` | `37.84%` | `1.8969` |
| `top_5` | `5` | `0.6202` | `0.2861` | `0.1551` | `34.83%` | `1.7460` |

Interpretation:
The `top_5_plus_pricing_benchmark` had the best logistic ROC AUC, but it includes Lending Club pricing/risk-grade variables. The `top_15` logistic model is the more useful independent benchmark because it avoids relying on those pricing fields while still giving strong top-decile risk concentration. Logistic regression is easy to explain and provides stable directional insight, but it is limited because it assumes mostly linear additive effects unless interactions are explicitly engineered.

## Architecture 2: Decision Tree And Random Forest
### Decision Tree Math
A decision tree recursively splits the data into regions. At each split, it chooses the feature and threshold that best reduce impurity. For binary classification, the Gini impurity is:

```text
Gini(node) = 1 - p_default^2 - p_non_default^2
```

The split objective is impurity reduction:

```text
gain = impurity(parent)
       - weighted_impurity(left_child)
       - weighted_impurity(right_child)
```

Interpretation:
A single tree is easy to explain because it follows a sequence of rules, but it can be unstable and may over-focus on a small number of dominant variables.

### Random Forest Math
A random forest averages many decision trees trained on randomized samples and feature subsets:

```text
p_forest(x) = (1 / T) * sum_t p_tree_t(x)
```

This reduces variance compared with a single tree. Random forests are usually more stable than a single decision tree, but they can still be less effective than boosted trees when the signal requires sequential correction of errors.

Tree-based model results:

| Model | ROC AUC | PR AUC | Brier score | Top-decile default rate | Top-decile lift | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| `logistic_regression_top_15` | `0.6870` | `0.3585` | `0.1477` | `44.57%` | `2.2344` | Strong independent benchmark. |
| `random_forest_top_15` | `0.6766` | `0.3523` | `0.1525` | `43.26%` | `2.1686` | Reasonable ranking, but weaker than logistic. |
| `decision_tree_top_15` | `0.3983` | `0.1670` | `0.1530` | `38.88%` | `1.9491` | Not competitive. |

Interpretation:
The decision tree performed poorly and was dominated by a small number of splits, especially `term`. The random forest improved stability but still underperformed the top-15 logistic model. This showed that tree-based nonlinearities were potentially useful, but the plain tree and random forest were not the best way to capture them.

## Architecture 3: Gradient-Boosted Trees
Gradient-boosted trees build an additive sequence of trees. Each new tree is trained to improve the errors made by the current ensemble.

The model has the form:

```text
F_M(x) = F_0(x) + eta * h_1(x) + eta * h_2(x) + ... + eta * h_M(x)
p(x) = 1 / (1 + exp(-F_M(x)))
```

Where:

| Term | Meaning |
|---|---|
| `F_M(x)` | Final ensemble score after `M` trees. |
| `h_m(x)` | Tree added at boosting round `m`. |
| `eta` | Learning rate, controlled by `stepSize`. |
| `M` | Number of boosting rounds, controlled by `maxIter`. |

For binary classification, the model is trained to reduce logistic loss:

```text
Loss = -sum(y_i*log(p_i) + (1 - y_i)*log(1 - p_i))
```

The key tuning parameters were:

| Parameter | Role |
|---|---|
| `maxDepth` | Controls interaction depth and tree complexity. |
| `maxIter` | Number of boosting rounds. |
| `stepSize` | Learning rate for each added tree. |
| `minInstancesPerNode` | Regularization through minimum leaf size. |
| `subsamplingRate` | Row subsampling for each boosting iteration. |

Untuned GBT result:

| Model | ROC AUC | PR AUC | Brier score | Top-decile default rate | Top-decile lift |
|---|---:|---:|---:|---:|---:|
| `gradient_boosted_trees_top_15` | `0.7009` | `0.3714` | `0.1460` | `45.47%` | `2.2794` |
| `logistic_regression_top_15` | `0.6870` | `0.3585` | `0.1477` | `44.57%` | `2.2344` |
| `random_forest_top_15` | `0.6766` | `0.3523` | `0.1525` | `43.26%` | `2.1686` |
| `decision_tree_top_15` | `0.3983` | `0.1670` | `0.1530` | `38.88%` | `1.9491` |

Interpretation:
The first GBT model became the best performer at that point. The gain over logistic regression was real but not enormous, which suggests the model was capturing useful nonlinear effects and interactions, but the tabular signal was still relatively hard.

## Gradient-Boosted Tree Tuning
The GBT was tuned using an inner train/validation split from the training data:

| Split | Rows |
|---|---:|
| GBT tuning train | `224,012` |
| GBT validation | `55,878` |

Validation tuning results:

| Profile | maxDepth | maxIter | stepSize | minInstancesPerNode | subsamplingRate | ROC AUC | PR AUC | Brier | Top-decile lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `deeper_subsampled` | `5` | `100` | `0.04` | `200` | `0.7` | `0.7020` | `0.3670` | `0.1450` | `2.2431` |
| `deeper_regularized` | `5` | `80` | `0.05` | `250` | `0.8` | `0.7016` | `0.3654` | `0.1451` | `2.2449` |
| `balanced_regularized` | `4` | `80` | `0.05` | `150` | `0.8` | `0.6995` | `0.3640` | `0.1454` | `2.2449` |
| `wider_sampling` | `4` | `80` | `0.05` | `100` | `0.9` | `0.6992` | `0.3635` | `0.1455` | `2.2530` |
| `balanced_more_trees` | `4` | `120` | `0.03` | `150` | `0.8` | `0.6988` | `0.3632` | `0.1455` | `2.2467` |
| `shallow_slow_regularized` | `3` | `100` | `0.03` | `150` | `0.8` | `0.6951` | `0.3574` | `0.1464` | `2.1969` |

The chosen tuning profile was `deeper_subsampled`. Although `wider_sampling` had the highest validation top-decile lift, `deeper_subsampled` was selected because it had the best broader combination of ROC AUC, PR AUC, and Brier score.

Final held-out test comparison:

| Model | Family | ROC AUC | PR AUC | Brier score | Top-decile default rate | Top-decile lift | Precision at 0.5 | Recall at 0.5 | F1 at 0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `tuned_gradient_boosted_trees_top_15` | Tuned GBT | `0.7032` | `0.3742` | `0.1456` | `46.34%` | `2.3230` | `0.5569` | `0.0840` | `0.1460` |
| `gradient_boosted_trees_top_15` | GBT | `0.7009` | `0.3714` | `0.1460` | `45.47%` | `2.2794` | `0.5544` | `0.0739` | `0.1305` |
| `logistic_regression_top_15` | Logistic regression | `0.6870` | `0.3585` | `0.1477` | `44.57%` | `2.2344` | `0.5571` | `0.0405` | `0.0755` |
| `random_forest_top_15` | Random forest | `0.6766` | `0.3523` | `0.1525` | `43.26%` | `2.1686` | N/A | `0.0000` | N/A |
| `decision_tree_top_15` | Decision tree | `0.3983` | `0.1670` | `0.1530` | `38.88%` | `1.9491` | `0.5239` | `0.0533` | `0.0968` |

## Metric Definitions
The model comparison used metrics suitable for imbalanced credit-risk classification.

| Metric | Formula / definition | Interpretation |
|---|---|---|
| ROC AUC | `P(score_default > score_non_default)` | Ranking quality across all classification thresholds. |
| PR AUC | Area under precision-recall curve | More sensitive to default-class performance under class imbalance. |
| Brier score | `mean((p_i - y_i)^2)` | Probability calibration and accuracy. Lower is better. |
| Accuracy at 0.5 | `(TP + TN) / (TP + FP + TN + FN)` | Share of correct labels at default threshold. Can be misleading in imbalanced data. |
| Precision at 0.5 | `TP / (TP + FP)` | Among predicted defaults, share that actually defaulted. |
| Recall at 0.5 | `TP / (TP + FN)` | Share of actual defaults caught at threshold 0.5. |
| F1 at 0.5 | `2 * precision * recall / (precision + recall)` | Harmonic mean of precision and recall. |
| Top-decile default rate | Default rate in highest-scored 10% | Business ranking quality for risk review or rejection. |
| Top-decile lift | `top_decile_default_rate / overall_default_rate` | How much risk is concentrated in the highest-risk decile. |

For this project, ROC AUC, PR AUC, Brier score, and top-decile lift are more important than default `0.5` threshold accuracy. The default rate is about 20%, so a threshold of `0.5` is too conservative for many credit-risk decisions.

## Interpretation Across Architectures
### Logistic Regression
Logistic regression gave the strongest simple benchmark. It is stable, transparent, and easy to communicate. However, it is limited because feature effects are mostly additive and linear unless interactions are manually engineered.

### Decision Tree
The decision tree was not competitive. It over-relied on a small number of variables, especially `term`, and had poor ROC AUC and PR AUC. It is useful for interpretability but not strong enough as the final model.

### Random Forest
Random forest improved stability compared with the single decision tree, but it still underperformed logistic regression and gradient-boosted trees. It captured some nonlinear structure, but not enough to become the leading model.

### Gradient-Boosted Trees
Gradient-boosted trees provided the best performance because they can model nonlinearities and feature interactions while correcting errors sequentially. Tuning improved the model further, but the gain over the untuned GBT was modest. This means the project has moved from architecture selection into refinement.

## Final Modeling Decision
The current champion model is:

```text
tuned_gradient_boosted_trees_top_15
```

The current interpretability benchmark is:

```text
logistic_regression_top_15
```

This is the right pairing for the project:

| Role | Model | Reason |
|---|---|---|
| Performance champion | Tuned GBT | Best ROC AUC, PR AUC, Brier score, and top-decile lift. |
| Interpretability benchmark | Logistic regression | Easier coefficient and odds-ratio interpretation. |
| Not retained | Decision tree | Too weak and unstable. |
| Not retained | Random forest | Reasonable but weaker than logistic and GBT. |

## Recommended Next Steps
The next work should not be another architecture yet. Neural networks or more complex models are probably overkill at this stage because the tuned GBT is already the best practical model for this structured tabular problem.

Recommended next steps:

1. Calibrate `tuned_gradient_boosted_trees_top_15` probabilities.
2. Compare calibrated versus uncalibrated Brier score and calibration curves.
3. Build threshold tables across possible cutoffs.
4. Add business decision columns such as approval share, rejection share, captured defaults, expected loss, and review capacity.
5. Perform segment-level error analysis by `purpose`, `home_ownership`, `verification_status`, `term`, and other important borrower groups.
6. Decide on a deployment threshold using business cost, not the default `0.5` cutoff.

## Bottom Line
The project has progressed from exploratory statistics to a working model comparison framework. The feature analysis identified debt burden, credit activity, utilization, loan size, income, verification, housing, and borrower intent as the most useful predictive themes. Logistic regression established a strong transparent benchmark, tree-based models tested nonlinear alternatives, and tuned gradient-boosted trees became the best model so far.

The next major question is no longer "Which architecture should we try next?" It is "How should the tuned model's probabilities be calibrated and converted into a lending decision?"
