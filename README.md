# Credit Risk & Fraud Loss Prevention Engine

Fraud transaction scoring built around a real business tradeoff: catching fraud at the lowest cost to good customers, not just maximizing detection. Framed for a subprime-issuer Risk/Fraud Strategy context, where a false decline risks losing the customer relationship entirely.

## The problem

Accuracy is meaningless when fraud is 0.17% of transactions — a model that predicts "not fraud" every time is 99.83% accurate and useless. The real question a fraud strategy team asks isn't "did we catch it," it's: **given a fixed review capacity, where's the operating point that maximizes net savings?**

## Approach

- **Dataset**: [ULB/Worldline Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 European cardholder transactions, 492 fraud (0.17%)
- **Model**: XGBoost with class weighting (`scale_pos_weight`), benchmarked against a logistic regression baseline
- **Rules layer**: an amount-threshold rule simulating a legacy fraud system, run alongside the model — real fraud stacks are always ML + rules, never ML alone
- **Evaluation**: Precision@k (maps directly to analyst review capacity) and AUPRC, not accuracy or plain ROC-AUC, since those are misleading at this class imbalance
- **Cost framework**: explicit dollar assumptions for false-decline cost and manual review cost, swept across review-capacity budgets to find the net-savings-optimal operating point
- **Explainability**: SHAP feature attribution, since fraud flags need to be defensible to compliance/audit

## Results

| Metric | Value |
|---|---|
| XGBoost AUPRC | 0.83 (vs. 0.70 logistic regression baseline) |
| Precision@100 | 98.0% |
| Optimal review budget | Top-150 flagged transactions |
| Net savings at optimal budget | $11,492.73 per 85,443-transaction test window |
| False-positive reduction vs. rules-only baseline (matched 80% recall) | 99.96% (31 vs. 80,223 false positives) |
| Top SHAP drivers | V4, V14, V12 |

**Stated cost assumptions**: $15 per false decline, $2 per manual review — adjustable in the dashboard.

## Try it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Adjust the review-capacity slider and cost assumptions to see how the net-savings-optimal operating point shifts in real time.

## Repo structure

```
├── fraud_detection_analysis.ipynb   # Full analysis: EDA, modeling, cost framework, SHAP
├── app.py                           # Streamlit dashboard
├── scored_test_set.csv              # Precomputed model scores (dashboard doesn't retrain live)
├── requirements.txt
└── README.md
```

Note: `creditcard.csv` (~144MB) isn't included in this repo due to size — download it from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the root folder if you want to rerun the notebook from scratch.

## Honest limitations

- Train/test split is random-stratified, not time-based — a production system should split by time to avoid leakage
- Dollar figures are per test window (85K transactions over the dataset's 2-day span), not annualized
- The 99.96% false-positive reduction reflects how weak a single-variable amount rule is at high recall (it has to flag nearly the entire test set) — this is a genuine finding about legacy rule systems, not an artifact to hide
