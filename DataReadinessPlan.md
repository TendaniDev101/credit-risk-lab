# Data Readiness Plan

This document captures the first-pass interpretation of the loan dataset columns we flagged during manual review, plus the cleanup steps needed before modeling.

Sources used:
- `data/LCDataDictionary.xlsx`
- profile checks against `data/loan.csv`

## Quick Audit Findings

- Total rows in `loan.csv`: `2,260,668`
- `id`: `100%` null
- `member_id`: `100%` null
- `url`: `100%` null
- `desc`: `125,815` non-null, `2,134,853` null, so only about `5.56%` populated
- `pymnt_plan`: `n = 2,259,986`, `y = 682`
- `title`: `2,237,346` non-null, `23,322` null
- `loan_status` distribution is dominated by:
  - `Fully Paid`: `1,041,952`
  - `Current`: `919,695`
  - `Charged Off`: `261,655`
- `term` normalized cleanly into two valid values only:
  - `36`: `1,609,754`
  - `60`: `650,914`
- `emp_length` normalized cleanly to integer years `0` through `10`, with one malformed raw value:
  - unexpected raw value: `reactors"` with count `1`
  - handling: convert to `NULL` rather than force an invalid numeric value

## High Missingness Snapshot

The following columns were flagged from the missingness profile as having extremely high null rates:

| Column | missing_rows | missing_pct | Initial interpretation |
|---|---:|---:|---|
| `id` | `2,260,668` | `1.000000` | Structurally unusable in this extract. Drop. |
| `member_id` | `2,260,668` | `1.000000` | Structurally unusable in this extract. Drop. |
| `url` | `2,260,667` | `1.000000` | Effectively empty in this extract. Drop. |
| `orig_projected_additional_accrued_interest` | `2,252,240` | `0.996272` | Very likely conditional field tied to hardship/relief cases. |
| `hardship_last_payment_amount` | `2,250,054` | `0.995305` | Likely only populated when hardship is active. |
| `hardship_payoff_balance_amount` | `2,250,054` | `0.995305` | Likely only populated when hardship is active. |
| `hardship_loan_status` | `2,250,048` | `0.995302` | Likely only populated for hardship subset. |
| `hardship_dpd` | `2,250,048` | `0.995302` | Likely only populated for hardship subset. |
| `hardship_length` | `2,250,044` | `0.995301` | Likely only populated for hardship subset. |
| `hardship_start_date` | `2,250,044` | `0.995301` | Likely only populated for hardship subset. |
| `hardship_end_date` | `2,250,042` | `0.995300` | Likely only populated for hardship subset. |
| `payment_plan_start_date` | `2,250,042` | `0.995300` | Likely only populated for hardship/payment-plan subset. |
| `hardship_amount` | `2,250,039` | `0.995298` | Likely only populated for hardship subset. |
| `deferral_term` | `2,250,032` | `0.995295` | Likely only populated for hardship subset. |
| `hardship_status` | `2,250,013` | `0.995287` | Likely only populated for hardship subset. |
| `hardship_reason` | `2,250,008` | `0.995285` | Likely only populated for hardship subset. |
| `hardship_type` | `2,249,996` | `0.995279` | Likely only populated for hardship subset. |
| `settlement_term` | `2,227,592` | `0.985369` | Likely only populated for settled loans subset. |
| `settlement_percentage` | `2,227,588` | `0.985367` | Likely only populated for settled loans subset. |
| `settlement_amount` | `2,227,574` | `0.985361` | Likely only populated for settled loans subset. |

Important note:
- We cannot treat nulls the same way across the whole dataset.
- Some nulls mean the concept does not apply to that loan.
- Some nulls mean no event on file.
- Some nulls mean the data is genuinely absent.

So for each field, null handling must be decided from business meaning, not just from the missing percentage.

## Column Notes

| Column | Meaning | Action |
|---|---|---|
| `term` | Number of payments on the loan. Dictionary says values are `36` or `60` months. | Strip `"months"`, trim whitespace, cast to integer. |
| `grade` | LC assigned loan grade. In practice this is the coarse risk band, typically `A` to `G`, where `A` is stronger credit quality and `G` is weaker. | Keep as ordered categorical. Also check relationship with `sub_grade`. |
| `id` | Unique LC loan listing ID. | Drop. It is null for all `2,260,668` rows. |
| `member_id` | Unique LC borrower member ID. | Drop. It is null for all `2,260,668` rows. |
| `emp_length` | Employment length in years. Dictionary says `0` means less than one year and `10` means ten or more years. | Normalize to integer years: `< 1 year -> 0`, `10+ years -> 10`, `"4 years" -> 4`, `"1 year" -> 1`. Preserve nulls. |
| `issue_d` | Month the loan was funded. | Parse `MMM-YYYY` to date, usually first day of month. Example: `Dec-2018 -> 2018-12-01`. |
| `loan_status` | Current status of the loan. This is the main repayment outcome field. | Keep, document carefully, and define modeling target from it. |
| `pymnt_plan` | Indicates whether a payment plan has been put in place for the loan. | Keep initially, but it is nearly constant. Consider binary encoding or dropping if low signal. |
| `url` | URL for the Lending Club listing page. | Drop. It is null for all rows in this extract. |
| `desc` | Borrower-provided loan description. | Treat as optional free text. Because it is very sparse, either drop for baseline modeling or reserve for a later NLP pass. |
| `title` | Borrower-provided loan title. | Likely overlaps heavily with `purpose`. Keep for comparison first, then probably drop if redundant/high-cardinality. |
| `zip_code` | First 3 numbers of the borrower ZIP code, masked as `109xx`, `713xx`, etc. | Strip `xx`. Store as 3-digit ZIP prefix string, not integer, to preserve leading zeroes. |
| `dti` | Debt-to-income ratio. Dictionary defines it as monthly debt obligations divided by self-reported monthly income, excluding mortgage and the requested LC loan. | Keep as numeric. Check outliers, negatives, and missing values. |
| `delinq_2yrs` | Number of 30+ day delinquencies in the borrower credit file during the last 2 years. | Keep as count variable. |
| `earliest_cr_line` | Month the borrower’s earliest reported credit line was opened. | Parse `MMM-YYYY` to date. Later derive credit history age relative to `issue_d`. |
| `mths_since_last_delinq` | Months since the borrower’s last delinquency. | Keep as numeric. Null likely means no delinquency on file or not reported. Add a missingness flag. |
| `mths_since_last_record` | Months since the last public record. | Keep as numeric. Null likely means no public record on file. Add a missingness flag. |
| `pub_rec` | Number of derogatory public records. | Keep as count variable. Public records can include serious legal/credit events. |
| `revol_util` | Revolving utilization rate, meaning revolving credit used relative to total revolving limit. | Keep as numeric percentage. Remove `%` if present, cast to numeric, inspect outliers over `100`. |
| `initial_list_status` | Initial listing status of the loan. Dictionary says values are `W` and `F`. By Lending Club convention this is typically `whole` vs `fractional`. | Encode as categorical. Keep unless we later find it has no predictive value. |
| `total_rec_late_fee` | Late fees received to date. | Keep with caution. This is post-origination behavior and may leak future information if used to score new applications. Exclude from origination-time model features. |
| `last_pymnt_d` | Last month payment was received. | Parse `MMM-YYYY` to date. This is post-origination and should not be used in an application-time model. |
| `next_pymnt_d` | Next scheduled payment date. | Parse `MMM-YYYY` to date. This is post-origination and should not be used in an application-time model. |
| `last_credit_pull_d` | Most recent month LC pulled credit for this loan. | Parse `MMM-YYYY` to date. Likely post-origination metadata. Exclude from application-time model. |
| `collections_12_mths_ex_med` | Number of collections in last 12 months excluding medical collections. | Keep as count variable. |
| `mths_since_last_major_derog` | Months since most recent 90-day delinquency or worse. | Keep as numeric. Null likely means no major derogatory event on file. Add a missingness flag. |
| `tot_coll_amt` | Total collection amounts ever owed. This is not collateral. | Keep as numeric count/amount field. Treat as collections history, not pledged asset value. |
| `open_acc_6m` | Number of open trades in last 6 months. | Keep as count variable. |
| `open_act_il` | Number of currently active installment trades. | Keep as count variable. |
| `open_il_12m` | Number of installment accounts opened in the last 12 months. | Keep as count variable. |
| `open_il_24m` | Number of installment accounts opened in the last 24 months. | Keep as count variable. |
| `mths_since_rcnt_il` | Months since most recent installment account opened. | Keep as numeric recency feature. Null likely means no installment account history visible. |
| `il_util` | Ratio of current balance to credit limit on installment accounts. | Keep as numeric percentage. |
| `open_rv_12m` | Number of revolving trades opened in the last 12 months. | Keep as count variable. |
| `open_rv_24m` | Number of revolving trades opened in the last 24 months. | Keep as count variable. |
| `all_util` | Balance-to-credit-limit ratio across all trades. | Keep as numeric percentage. |
| `inq_fi` | Number of personal finance inquiries. | Keep as count variable. |
| `total_cu_tl` | Number of finance trades. | Keep as count variable. Treat carefully because the name is not self-explanatory without the dictionary. |
| `bc_util` | Ratio of current balance to high credit or limit for all bankcard accounts. | Keep as numeric percentage. |
| `chargeoff_within_12_mths` | Number of charge-offs within 12 months. | Keep as count variable. |
| `mo_sin_old_il_acct` | Months since oldest bank installment account opened. | Keep as age feature. |
| `mo_sin_old_rev_tl_op` | Months since oldest revolving account opened. | Keep as age feature. |
| `mo_sin_rcnt_rev_tl_op` | Months since most recent revolving account opened. | Keep as recency feature. |
| `mo_sin_rcnt_tl` | Months since most recent account opened. | Keep as recency feature. |

## Special Interpretation Notes

### `loan_status`

Observed values in the full dataset:
- `Fully Paid`
- `Current`
- `Charged Off`
- `Late (31-120 days)`
- `In Grace Period`
- `Late (16-30 days)`
- `Does not meet the credit policy. Status:Fully Paid`
- `Does not meet the credit policy. Status:Charged Off`
- `Default`

What this tells us:
- `Fully Paid` means the borrower repaid the loan.
- `Charged Off` means the loan was written off as a loss.
- `Default` is an adverse outcome and should be grouped with default-like statuses.
- `Late` and `In Grace Period` indicate repayment distress but are not necessarily final outcomes yet.
- `Current` means the loan is still active, so final outcome is unresolved.

Modeling implication:
- For a clean default model, we should not blindly use `Current` loans as non-defaults.
- A practical first target is:
  - bad = `Charged Off`, `Default`, `Does not meet the credit policy. Status:Charged Off`
  - good = `Fully Paid`, `Does not meet the credit policy. Status:Fully Paid`
  - exclude for baseline target = `Current`, `In Grace Period`, `Late (16-30 days)`, `Late (31-120 days)`

### `pymnt_plan`

Observed values in the full dataset:
- `n = 2,259,986`
- `y = 682`

Interpretation:
- `y` means a payment plan was put in place.
- This is very rare and may represent distressed loans.

Modeling implication:
- It is likely post-origination behavior. For a model that scores a new applicant at application time, this should probably be excluded.

### `desc`

Interpretation:
- Free-text narrative from the borrower explaining the loan.

Modeling implication:
- Very sparse.
- Can be useful later for text modeling, but it should not block the first structured-data baseline.

### `title`

Interpretation:
- Short borrower-entered text label such as `Debt consolidation`.

Modeling implication:
- Likely overlaps with `purpose`.
- We should compare `title` against `purpose` before deciding whether to drop it.

### `emp_length`

Validation notes:
- The expected categories mapped correctly to integer values from `0` to `10`.
- One malformed raw value was found: `reactors"`.
- Because it does not match the documented category set, it should be treated as a data-quality anomaly and mapped to `NULL`.

## Concrete Data Preparation Steps

### Step 1: Standardize raw values

1. Trim whitespace on all string columns.
2. Convert known placeholder strings such as empty strings to nulls.
3. Normalize `term` to integer months.
4. Normalize `emp_length` to integer years.
5. Normalize `zip_code` to a 3-character ZIP prefix.

### Step 2: Parse dates

Convert these `MMM-YYYY` columns to proper dates:
- `issue_d`
- `earliest_cr_line`
- `last_pymnt_d`
- `next_pymnt_d`
- `last_credit_pull_d`

Recommended convention:
- Parse to the first day of the month.

### Step 3: Remove unusable or low-value columns

Drop immediately:
- `id`
- `member_id`
- `url`

Candidate drop after review:
- `desc`
- `title`

### Step 4: Preserve missingness information

Null interpretation must be feature-specific:
- structural null: the field is effectively unusable in this extract, such as `id`, `member_id`, and `url`
- conditional null: the field only applies to a subset of loans, such as hardship and settlement columns
- semantic null: null may mean no event exists, such as `mths_since_last_delinq`
- true missing data: the value should exist in principle, but is absent

For columns where null is meaningful, create companion flags:
- `mths_since_last_delinq_is_missing`
- `mths_since_last_record_is_missing`
- `mths_since_last_major_derog_is_missing`
- `mths_since_rcnt_il_is_missing`

Reason:
- In credit data, null often means no event on file rather than bad data.
- In hardship and settlement fields, null often means the loan never entered that process.

For the high-missingness hardship and settlement fields:
1. Do not impute them blindly.
2. First verify whether null means "not in hardship" or "not settled".
3. If they are post-origination servicing fields, exclude them from the application-time default model even if they are populated.

### Step 5: Separate application-time features from future-information fields

Do not use clear post-origination fields in a model that scores new applicants:
- `last_pymnt_d`
- `next_pymnt_d`
- `last_credit_pull_d`
- `total_rec_late_fee`
- `pymnt_plan`

These can still be kept for portfolio analysis, collections analysis, or outcome interpretation.

### Step 6: Define the target carefully

For the first default model:
1. Build a binary target from final outcomes only.
2. Exclude unresolved statuses such as `Current`.
3. Document exactly which statuses map to good, bad, and excluded.

### Step 7: Feature engineering

Create derived features such as:
- `credit_history_months = months_between(issue_d, earliest_cr_line)`
- grouped employment buckets from `emp_length`
- grouped utilization bands for `revol_util`, `bc_util`, `all_util`, `il_util`
- delinquency and charge-off indicators from the count fields

### Step 8: Validation before modeling

Before training:
1. Check null percentages.
2. Check cardinality of categorical features.
3. Check numeric outliers and impossible values.
4. Verify that no leakage features are included.
5. Freeze a reproducible modeling dataset, ideally in Parquet.

## Immediate Next Implementation Tasks

1. Create a Spark cleaning notebook section that standardizes the fields above.
2. Produce a cleaned Parquet dataset.
3. Build a target-label table from `loan_status`.
4. Create a feature whitelist for application-time modeling.
