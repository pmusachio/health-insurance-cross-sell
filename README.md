# Health Insurance Cross-Sell — Propensity Ranking

> Imbalanced classification · Propensity ranking · Lift and cumulative gains

## Business Problem

A health-insurance company wants to cross-sell vehicle insurance to its existing customers.
The sales team can only make a fixed number of calls per campaign, and calling at random is
inefficient because the base conversion rate is low (~12%).

The decision the model informs is **who to call first**: it scores every customer by their
propensity to buy and the team works the list top-down until capacity runs out. This is a
**ranking** problem, not a yes/no classification — the absolute probability matters less than
the ordering. The cost of error is asymmetric: a false positive wastes one call, while a false
negative leaves a willing buyer uncontacted, forfeiting a sale. Because outreach capacity is the
binding constraint, the model is optimized and judged on **lift at capacity**, not accuracy.

A plain heuristic ("call everyone with a damaged vehicle who is not already insured") was
rejected: it captures the strongest signals but ignores their interaction with premium, age,
channel and tenure, which the model exploits to order customers within those coarse buckets.

## Dataset

[Health Insurance Cross-Sell Prediction](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)

| Property | Value |
|----------|-------|
| Rows | 381,109 customers |
| Target | `Response` (1 = interested in vehicle insurance) |
| Positive rate | 12.3% (imbalanced) |
| Key features | `Previously_Insured`, `Vehicle_Damage`, `Vehicle_Age`, `Age`, `Annual_Premium`, `Policy_Sales_Channel`, `Vintage` |

## Solution Strategy

1. **Acquisition** — pull the dataset from Kaggle on demand; a versioned stratified sample backs an offline run.
2. **Leakage control** — every feature is known before the customer is contacted, so there is no target leakage; `id` is dropped from the feature set and `Response` is the target.
3. **Encoding** — `Gender` and `Vehicle_Damage` map to binary, `Vehicle_Age` stays categorical; all inside the model `Pipeline` (a custom first step) so serving reuses the exact transform.
4. **Imbalance** — handled with `class_weight="balanced"` applied only inside the fitted folds, never by resampling the holdout.
5. **Model selection** — `StratifiedKFold` cross-validation compares a logistic baseline, random forest and histogram gradient boosting on ROC AUC; the winner is tuned with `RandomizedSearchCV`.
6. **Evaluation** — ROC AUC and average precision on a stratified holdout, plus precision/lift at fixed contact capacities and ROC AUC by segment.

## Top Insights & Hypotheses

- **Already-insured customers almost never convert.** `Previously_Insured` is by far the strongest feature (permutation importance 0.18); it alone explains most of the ranking power.
- **Past vehicle damage signals intent.** `Vehicle_Damage` is the second strongest driver — customers who have experienced damage are far more receptive.
- **The model is sharpest where the answer is "no".** ROC AUC is 0.93 on the no-damage segment but only 0.69 on the damaged segment, where buyers and non-buyers look more alike — a known limitation, flagged in Next Steps.
- **Newer vehicles and younger customers skew negative**, consistent with lower perceived risk.

## Model

A histogram gradient boosting classifier (selected by cross-validation, tuned with randomized
search) inside a `Pipeline` that owns the cleaning and encoding. The logistic baseline sets the
bar the final model must clear.

| Model | CV ROC AUC | Holdout ROC AUC | Holdout AP |
|-------|-----------:|----------------:|-----------:|
| Logistic baseline | 0.836 | 0.839 | 0.323 |
| Random forest | 0.835 | — | — |
| **Hist gradient boosting (final)** | **0.854** | **0.858** | **0.367** |

Tuned parameters: `learning_rate=0.17`, `max_leaf_nodes=127`, `max_depth=4`, `l2_regularization=10`, `max_iter=600`.

## Business Results

Ranking the holdout by score and contacting top-N customers:

| Contacts | Precision@k | Lift vs random | Interested buyers captured |
|----------|------------:|---------------:|---------------------------:|
| 5,000 | 40.4% | 3.30x | 21.6% |
| 10,000 | 38.0% | 3.10x | 40.6% |
| 20,000 | 33.6% | 2.74x | 71.9% |

Contacting the top 20,000 ranked customers reaches **71.9% of all interested buyers** at
**2.74x** the efficiency of random outreach — the team captures roughly three quarters of the
addressable demand while calling a fraction of the base.

## How to Run

1. **Clone**
   ```
   git clone https://github.com/pmusachio/health-insurance-cross-sell.git
   cd health-insurance-cross-sell
   ```
2. **Environment**
   ```
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Kaggle access** — place a Kaggle API token at `~/.kaggle/`; the pipeline falls back to the versioned sample if none is present.
4. **Run the pipeline**
   ```
   python -m src.pipeline
   ```
5. **Tests**
   ```
   pytest tests/
   ```
6. **App (local)**
   ```
   streamlit run app/streamlit_app.py
   ```
7. **Live app** — [huggingface.co/spaces/pmusachio/health-insurance-cross-sell](https://huggingface.co/spaces/pmusachio/health-insurance-cross-sell) — score a customer and explore the campaign view.

## Next Steps

- Improve discrimination on the damaged-vehicle segment (ROC AUC 0.69), where the current features do not separate buyers well; richer behavioural or pricing features are the likely lever.
- Calibrate probabilities (isotonic or Platt) if the scores are ever used as expected-value inputs rather than purely for ranking; deferred because the campaign decision only needs the ordering.
- Cost-sensitive thresholding tied to call cost and policy margin would convert lift into an explicit profit-optimal capacity; deferred until per-call economics are available.
