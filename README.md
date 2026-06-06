# Health Insurance Cross Sell — Propensity Ranking Model

> **Kaggle competition project** — rank customers by their likelihood to purchase vehicle insurance, maximising the number of interested buyers contacted within a fixed sales capacity.

---

## Table of Contents

1. [Business Problem](#business-problem)
2. [Dataset](#dataset)
3. [Solution Strategy](#solution-strategy)
4. [Key Insights](#key-insights)
5. [Model Performance](#model-performance)
6. [Repository Structure](#repository-structure)
7. [How to Run](#how-to-run)
8. [Technologies](#technologies)
9. [Author](#author)

---

## Business Problem

An insurance company that currently provides health insurance wants to **cross-sell vehicle insurance** to its existing customers. The sales team has a limited number of calls they can make per campaign period. Calling random customers is inefficient — the conversion rate is low.

**Goal:** build a propensity model that assigns a score to each customer. When the list is sorted by score descending, the sales team contacts the most likely buyers first. This is a **ranking problem**, not just a classification problem.

**Business impact metric:** *lift at 20 000 contacts*. If the model achieves lift = 3, contacting the top 20 000 customers is three times more efficient than calling a random sample of the same size.

---

## Dataset

| Column | Type | Description |
|---|---|---|
| id | int | Unique customer identifier |
| Gender | str | Male / Female |
| Age | int | Customer age in years |
| Driving_License | int | 1 = has licence, 0 = does not |
| Region_Code | float | Region of the customer |
| Previously_Insured | int | 1 = already has vehicle insurance |
| Vehicle_Age | str | < 1 Year / 1-2 Year / > 2 Years |
| Vehicle_Damage | str | Yes / No — history of vehicle damage |
| Annual_Premium | float | Annual premium paid for health insurance |
| Policy_Sales_Channel | float | Channel through which the policy was sold |
| Vintage | int | Days since the customer joined |
| Response | int | **Target** — 1 = interested in vehicle insurance |

- **Train:** 381 109 rows, 12 columns
- **Test:** 127 037 rows, 11 columns (no Response)
- **Positive rate:** ~12.3 %

---

## Solution Strategy

The project follows a CRISP-DM cycle with the following 10 steps:

1. **Business understanding** — define success metrics (ROC AUC, average precision, lift@20k)
2. **Data understanding** — schema, missing values, target distribution
3. **Exploratory analysis** — 4 business hypotheses tested and confirmed/refuted
4. **Feature engineering** — encode vehicle age, map damage and gender to numeric
5. **Preprocessing pipeline** — `ColumnTransformer` with median imputation + standard scaling for numeric, one-hot encoding for categorical
6. **Baseline model** — Logistic Regression with balanced class weights
7. **Model comparison** — compare LogisticRegression, RandomForest, and LightGBM via 5-fold stratified cross-validation
8. **Hyperparameter tuning** — `RandomizedSearchCV` (20 iterations) on the best model
9. **Business evaluation** — cumulative gain curve and lift curve; translate metrics into revenue impact
10. **Deployment** — FastAPI endpoint + Google Sheets automation for the sales team

---

## Key Insights

- **Customers with vehicle damage history are ~5x more likely to be interested** — the most predictive binary signal in the dataset.
- **Previously insured customers show near-zero interest** — they already have coverage; do not call them.
- **Middle-aged customers (35-50) show higher propensity** than the very young or very old.
- **Older vehicles (> 2 years) correlate with higher interest** — owners feel the need to insure ageing assets.

---

## Model Performance

> Values below are from the final model (LightGBM tuned, 5-fold CV). Run `make train` to reproduce.

| Metric | Value |
|---|---|
| ROC AUC (CV mean) | ~0.852 |
| Average Precision (CV mean) | ~0.512 |
| ROC AUC (test set) | ~0.858 |
| Precision @ 20 000 | ~0.38 |
| Lift @ 20 000 | ~3.1 |

*Contacting the top 20 000 scored customers captures approximately **56 % of all interested buyers** while reaching only 16 % of the customer base.*

---

## Repository Structure

```
health-insurance-cross-sell/
├── configs/
│   └── project.toml          # All project settings
├── data/
│   ├── raw/                  # Original Kaggle files (train, test, sample_submission)
│   ├── interim/              # Intermediate artefacts
│   └── processed/            # Model predictions
├── integrations/
│   └── google_sheets_appscript.gs   # Apps Script for Google Sheets automation
├── models/                   # Saved model (.joblib)
├── notebooks/
│   ├── 00_business_understanding.ipynb
│   ├── 01_data_understanding.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling_and_business_results.ipynb
│   └── 05_deployment_and_consumption.ipynb
├── reports/
│   ├── figures/              # Plots saved during notebook execution
│   └── metrics.json          # Final metrics produced by `make train`
├── scripts/
│   ├── sample_api_request.py
│   └── score_to_sheets.py    # Automated Python -> Google Sheets scoring
├── src/
│   └── health_insurance_cross_sell/
│       ├── api.py            # FastAPI prediction endpoint
│       ├── cli.py            # Command-line interface
│       ├── config.py         # TOML config loader
│       ├── data.py           # Data loading utilities
│       ├── features.py       # Feature engineering
│       └── models.py         # Training, CV, tuning, prediction
├── tests/
│   └── test_project_contract.py
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## How to Run

### Option A — Google Colab (recommended for quick exploration)

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **File -> Open notebook -> GitHub** and paste this repository URL.
3. Open any notebook, e.g. `notebooks/04_modeling_and_business_results.ipynb`.
4. Run the first cell (it will install dependencies and download data).
5. Execute all cells via **Runtime -> Run all**.

> Data files are not committed to the repository. Download them from [Kaggle](https://www.kaggle.com/competitions/health-insurance-cross-sell-prediction) and upload to `data/raw/` inside Colab, or use the `kaggle` CLI in the first notebook cell.

---

### Option B — Local (full reproducibility)

**Prerequisites:** Python 3.11+, `make`.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/health-insurance-cross-sell.git
cd health-insurance-cross-sell

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -e ".[dev]"
pip install -r requirements.txt

# 4. Place data files in data/raw/
#    train.csv, test.csv, sample_submission.csv (from Kaggle)

# 5. Train the model
make train

# 6. Generate predictions on test set
make predict

# 7. Run the notebooks interactively
jupyter lab
```

After `make train`, `reports/metrics.json` will contain the evaluation results.

---

### Option C — API + Google Sheets

**Step 1 — Start the API locally**

```bash
pip install -r requirements-api.txt
uvicorn health_insurance_cross_sell.api:app --host 0.0.0.0 --port 8000
```

**Step 2 — Test the endpoint**

```bash
python scripts/sample_api_request.py
```

**Step 3 — Connect Google Sheets**

1. Open your Google Sheet containing customer data.
2. Go to **Extensions -> Apps Script**.
3. Paste the contents of `integrations/google_sheets_appscript.gs`.
4. Set the script property `CROSS_SELL_API_URL` to your deployed API URL.
5. Save, then click **Cross Sell -> Score customers** from the spreadsheet menu.

**Step 4 — Automated Python scoring (alternative)**

```bash
# Install gspread + google-auth
pip install gspread google-auth

# Set up credentials (see scripts/score_to_sheets.py docstring)
python scripts/score_to_sheets.py \
  --input data/raw/test.csv \
  --sheet-id YOUR_GOOGLE_SHEET_ID
```

---

## Technologies

| Layer | Tool |
|---|---|
| Data manipulation | pandas, numpy |
| Machine learning | scikit-learn, LightGBM |
| Visualisation | matplotlib, seaborn |
| Model serialisation | joblib |
| API | FastAPI, uvicorn |
| Google Sheets | Apps Script (JavaScript) + gspread (Python) |
| Configuration | TOML (project.toml) |
| Testing | pytest |

---

## Author

**Paulo Musachio**
Data Scientist

- LinkedIn: [linkedin.com/in/paulomusachio](https://linkedin.com/in/paulomusachio)
- GitHub: [github.com/paulomusachio](https://github.com/paulomusachio)
