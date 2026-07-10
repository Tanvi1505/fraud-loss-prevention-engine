# Credit Risk & Fraud Loss Prevention Engine

## Why I picked this

I kept seeing fraud detection projects on GitHub that reported 99% accuracy and called it a day. But the dataset they were all using, the ULB credit card fraud dataset — has fraud in only 0.17% of transactions. A model that literally predicts "not fraud" every single time gets 99.83% accuracy. That's not a model, that's just saying no to everything.

I wanted to build something that actually answers the question a fraud team asks in real life: given that we can only review a certain number of flagged transactions per day, which ones should we look at, and at what point does flagging more transactions start costing us more than it saves?

## What the problem actually is

When fraud is this rare, the standard metrics break. Accuracy is useless. Even AUC-ROC can look great while your model still lets through most fraud. The metric that actually matters here is **precision at k** — out of the top k transactions your model flags, how many are actually fraud? Because k is your review team's capacity. You can't review 80,000 transactions a day.

The other thing real fraud systems deal with: a false positive (flagging a good customer) isn't free. For a subprime card issuer especially, a false decline can end the customer relationship. So I built in a cost framework — you put in what a false decline costs you and what a manual review costs, and the dashboard shows you where the net savings peak before the false-positive costs eat into them.

## What I built

- Trained an **XGBoost model** with `scale_pos_weight` to handle the class imbalance, benchmarked against a logistic regression baseline
- Added a **rules layer** alongside the model (a simple amount threshold), because real fraud stacks always combine ML with rules, its never just one
- Evaluated on **AUPRC and Precision@k**, not accuracy
- Used **SHAP** to explain which features drive each score — fraud decisions need to be explainable to compliance
- Built a **Streamlit dashboard** where you can move a slider for review capacity and adjust the cost assumptions, and it recalculates net savings in real time

## Results

| Metric | Result |
|---|---|
| XGBoost AUPRC | 0.83 (logistic regression baseline: 0.70) |
| Precision at top 100 | 98% |
| Optimal review budget | Top 150 flagged transactions |
| Net savings at that point | $11,492 per 85,443-transaction test window |
| False positives vs. rules-only baseline (at matched recall) | 31 vs. 80,223 |

The false-positive comparison is the one I find most interesting like a simple amount-threshold rule that's tuned to catch 80% of fraud has to flag nearly the entire test set to do it. The model gets there with 31 false positives.

## How to run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard loads precomputed scores so it doesn't retrain anything. Move the sliders to see how the optimal review point shifts under different cost assumptions.

## Files

```
├── fraud_detection_analysis.ipynb   # full walkthrough: EDA, model, cost framework, SHAP
├── app.py                           # Streamlit dashboard
├── scored_test_set.csv              # precomputed scores (so the dashboard runs without the raw data)
├── requirements.txt
└── README.md
```

The raw dataset (`creditcard.csv`, ~144MB) isn't in the repo — you can download it from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and drop it in the root folder if you want to rerun the notebook.

## Honest caveats

- I used a random stratified train/test split, not time-based. In production you'd always split by time so you're not accidentally training on future data.
- The dollar figures are per test window (2 days of transactions in the dataset), not annualized.
- The 99.96% false-positive reduction against the rules baseline sounds dramatic, but it's a genuine finding about how bad single-variable rules are when you push them to high recall, not something I'm inflating.
