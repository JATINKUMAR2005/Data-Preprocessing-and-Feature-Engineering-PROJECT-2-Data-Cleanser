# 🏥 Data Cleanser — Data Preprocessing & Feature Engineering

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)

**Red & White Skill Education | Data Science & AI/ML Program**

*A complete data preprocessing pipeline covering missing value imputation and outlier treatment on real-world patient health records*

</div>

Video Link : ([https://drive.google.com/drive/u/0/folders/13Wag0vMzH4mo2MnU_4GAXM49hH2RTPFE](https://drive.google.com/file/d/1p9FTRSRd9SaGpoPMFvSdAcUjlVgQkG-b/view?usp=sharing))

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Part A — Missing Value Imputation](#-part-a--missing-value-imputation)
- [Part B — Outlier Detection & Treatment](#-part-b--outlier-detection--treatment)
- [Part C — Final Clean Dataset](#-part-c--final-clean-dataset)
- [Key Findings](#-key-findings)
- [Technologies Used](#-technologies-used)
- [How to Run](#-how-to-run)
- [Output Files](#-output-files)
- [Results Summary](#-results-summary)

---

## 🎯 Project Overview

As a **Data Analyst** for a healthcare company, this project processes a dataset of **500 patient health records** that contain missing values and outlier measurements caused by inconsistent reporting and measurement errors.

The objective is to:
- **Identify** and handle all missing values using 6 imputation strategies
- **Detect** and treat outliers using 4 different methods
- **Produce** a clean, ML-ready dataset for predicting disease risk (binary classification)
- **Generate** an interactive HTML report with all charts and findings

**Problem Statement:**
> *Given patient health records with missing values and extreme measurements — apply imputation and outlier treatment techniques to produce a clean dataset ready for predicting `disease_risk` (0 = Low Risk, 1 = High Risk)*

---

## 📊 Dataset

**File:** `patient_health_records_500.csv`

| Column | Type | Description | Missing |
|--------|------|-------------|---------|
| `patient_id` | Int | Unique patient identifier | 0 |
| `age` | Float | Patient age in years (18–79) | 24 (4.8%) |
| `gender` | Categorical | Male / Female | 25 (5.0%) |
| `region` | Categorical | North / South / East / West | 24 (4.8%) |
| `bmi` | Float | Body Mass Index | 25 (5.0%) |
| `blood_pressure` | Float | Systolic blood pressure (mmHg) | 0 |
| `cholesterol` | Float | Cholesterol level (mg/dL) | 25 (5.0%) |
| `glucose` | Float | Fasting glucose (mg/dL) | 24 (4.8%) |
| `disease_risk` | Binary | **TARGET** — 0 = Low Risk, 1 = High Risk | 0 |

**Dataset Stats:**
- **Total Records:** 500 patients
- **Total Missing Values:** 147 cells (~3.3% overall)
- **Class Balance:** 53.8% Low Risk (269) | 46.2% High Risk (231)
- **Gender Split:** Female 265 | Male 235
- **Region Split:** South 155 | North 124 | East 117 | West 104

---

## 📁 Project Structure

```
Data-Cleanser/
│
├── 📓 Data_Cleanser_Complete.ipynb    ← Main Jupyter notebook
├── 📊 patient_health_records_500.csv  ← Raw dataset (input)
├── 📊 patient_health_cleaned.csv      ← Cleaned dataset (output)
├── 🌐 Data_Cleanser_Report.html       ← Interactive HTML report
├── 📄 README.md                       ← This file
│
└── 📸 Charts (auto-generated on run)
    ├── missing_values_overview.png
    ├── task2a_bmi_imputation.png
    ├── task2b_region_imputation.png
    ├── task2c_gender_imputation.png
    ├── task2d_random_sample_imputation.png
    ├── task2e_knn_imputation.png
    ├── task2f_mice_comparison.png
    ├── task3a_zscore_outliers.png
    ├── task3b_iqr_outliers.png
    ├── task3c_percentile_method.png
    ├── task4_winsorization.png
    ├── task5_before_after_comparison.png
    ├── task6_final_dataset_overview.png
    └── final_summary_dashboard.png
```

---

## 🔴 Part A — Missing Value Imputation

### Task 1 — Missing Value Summary Report

Generates a complete report showing missing count and percentage per column.

```python
missing_count = df.isnull().sum()
missing_pct   = (df.isnull().sum() / len(df) * 100).round(2)

missing_report = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing (%)':   missing_pct,
    'Data Type':     df.dtypes
}).sort_values('Missing (%)', ascending=False)
```

---

### Task 2a — Simple Imputer (Numerical) — BMI

Replaces missing BMI values with **Mean** and **Median** and compares both.

```python
bmi_mean_imp   = SimpleImputer(strategy='mean')
bmi_median_imp = SimpleImputer(strategy='median')

df_mean['bmi']   = bmi_mean_imp.fit_transform(df_mean[['bmi']])
df_median['bmi'] = bmi_median_imp.fit_transform(df_median[['bmi']])
```

| Method | Value Used | Verdict |
|--------|-----------|---------|
| Mean   | 26.57     | ⚠️ Affected by outliers |
| Median | 26.30     | ✅ Preferred — robust to outliers |

---

### Task 2b — Simple Imputer (Categorical) — Region

Replaces missing Region values with the **most frequent category**.

```python
region_imp = SimpleImputer(strategy='most_frequent')
df['region'] = region_imp.fit_transform(df[['region']]).ravel()
# Most frequent → "South" | Missing: 24 → 0
```

---

### Task 2c — Most Frequent Imputation — Gender

Replaces missing Gender values with the **mode**.

```python
gender_imp = SimpleImputer(strategy='most_frequent')
df['gender'] = gender_imp.fit_transform(df[['gender']]).ravel()
# Most frequent → "Female" | Missing: 25 → 0
```

---

### Task 2d — Missing Indicator + Random Sample Imputation

**Two-step approach** — marks missingness first, then fills from observed values.

```python
# Step 1: Create binary indicator columns
for col in numerical_cols:
    df_indicator[f'{col}_missing'] = df[col].isnull().astype(int)

# Step 2: Fill NaN by randomly sampling real observed values
for col in numerical_cols:
    null_mask   = df[col].isnull()
    observed    = df[col].dropna()
    random_fill = observed.sample(null_mask.sum(), replace=True, random_state=42)
    df.loc[null_mask, col] = random_fill.values
```

> **Why better than mean?** Random sampling preserves the original distribution shape instead of creating an artificial spike at the mean value.

---

### Task 2e — KNN Imputer (k=5 Neighbors)

**Multivariate imputation** — uses 5 most similar patients to estimate missing values.

```python
knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
df_knn[num_cols] = knn_imputer.fit_transform(df_knn[num_cols])
```

> For a patient with missing BMI, KNN finds the 5 patients with the most similar age, blood pressure, cholesterol, and glucose — and averages their BMI to fill the gap.

---

### Task 2f — MICE Algorithm 🏆 Best Method

**Multiple Imputation by Chained Equations** — iteratively models each column using all other columns.

```python
mice_imputer = IterativeImputer(max_iter=10, random_state=42)
df_mice[num_cols] = mice_imputer.fit_transform(df_mice[num_cols])

# Clip to valid medical ranges after imputation
df_mice['age']            = df_mice['age'].clip(18, 90).round(0)
df_mice['bmi']            = df_mice['bmi'].clip(10, 60).round(1)
df_mice['blood_pressure'] = df_mice['blood_pressure'].clip(60, 200).round(1)
df_mice['cholesterol']    = df_mice['cholesterol'].clip(50, 400).round(1)
df_mice['glucose']        = df_mice['glucose'].clip(50, 400).round(1)
```

**Why MICE wins (Score: 10/10):**
- Runs 10 iterations — each column is modelled as a function of all other columns
- Captures correlations between BMI, cholesterol, glucose, and blood pressure
- Most statistically sound imputation for correlated clinical data
- Zero missing values after imputation

### Imputation Methods Comparison

| Method | Score | Columns | Data Loss | Verdict |
|--------|-------|---------|-----------|---------|
| Mean Imputer | 3/10 | BMI | None | ⚠️ Shifts distribution |
| Median Imputer | 4/10 | BMI | None | ✅ Better than mean |
| Most Frequent | 4/10 | Gender, Region | None | ✅ Correct for categorical |
| Random Sample | 6/10 | Age, BMI, Chol, Gluc | None | ✅ Preserves distribution |
| KNN (k=5) | 8/10 | All numerical | None | ✅✅ Multivariate |
| **MICE (10 iter)** | **10/10** | **All numerical** | **None** | **🏆 Best** |

---

## 🔍 Part B — Outlier Detection & Treatment

All outlier methods are applied on `df_mice` — the MICE-imputed dataset with zero missing values.

---

### Task 3a — Z-Score Method

Detects patients with extreme **Cholesterol** or **Glucose** values.

```python
from scipy import stats

z_scores     = np.abs(stats.zscore(df_zscore[col]))
outlier_mask = z_scores > 3   # Flag anything beyond 3 standard deviations

df_zscore_clean = df_zscore[~outlier_mask].copy()
```

| Column | Outliers Detected | Action |
|--------|------------------|--------|
| Cholesterol | 5 patients | Removed |
| Glucose | 1 patient | Removed |
| **Total** | **6 rows** | **494 rows remain** |

---

### Task 3b — IQR Method

Detects unusual **BMI** and **Blood Pressure** values using interquartile range.

```python
Q1  = df_iqr[col].quantile(0.25)
Q3  = df_iqr[col].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

mask = (df_iqr[col] < lower) | (df_iqr[col] > upper)
df_iqr_clean = df_iqr[~mask].copy()
```

| Column | Q1 | Q3 | IQR | Outliers |
|--------|----|----|-----|---------|
| BMI | 24.00 | 29.12 | 5.12 | 14 rows |
| Blood Pressure | 114.20 | 136.22 | 22.02 | 3 rows |
| **Total** | | | | **17 rows removed** |

---

### Task 3c — Percentile Method

Caps values below **1st percentile** and above **99th percentile** — no rows removed.

```python
p1  = df_percentile[col].quantile(0.01)
p99 = df_percentile[col].quantile(0.99)

df_percentile[col] = df_percentile[col].clip(lower=p1, upper=p99)
# All 500 rows preserved — only extreme edges are clipped
```

---

### Task 4 — Winsorization 🏆 Best Method

Replaces the most extreme **5% from each tail** with boundary values — zero data loss.

```python
from scipy.stats.mstats import winsorize

df_wins[col] = winsorize(df_wins[col], limits=(0.05, 0.05))
```

| Column | Max Before | Max After | Min Before | Min After |
|--------|-----------|-----------|-----------|-----------|
| BMI | 60.00 | 32.80 | 11.70 | 19.20 |
| Blood Pressure | 200.00 | 150.80 | 74.20 | 98.40 |
| Cholesterol | 400.00 | 267.80 | 101.80 | 146.90 |
| Glucose | 165.20 | 135.50 | 50.00 | 65.00 |

### Outlier Methods Comparison

| Method | Rows Removed | Data Loss | Verdict |
|--------|-------------|-----------|---------|
| Z-Score (|z|>3) | 6 | 1.2% | ✅ Good for normal data |
| IQR Method | 17 | 3.4% | ✅ No normality needed |
| Percentile Cap | 0 | 0% | ✅✅ Non-destructive |
| **Winsorization** | **0** | **0%** | **🏆 Best — smooth capping** |

---

## 🟢 Part C — Final Clean Dataset

### Task 6 — Final Dataset Construction

Combined the two best methods — **MICE** for imputation and **Winsorization** for outliers.

```python
df_final = df_wins.copy()

# Label encode categorical columns for ML
le_gender = LabelEncoder()
le_region = LabelEncoder()

df_final['gender_encoded'] = le_gender.fit_transform(df_final['gender'])
df_final['region_encoded'] = le_region.fit_transform(df_final['region'])

# Save cleaned dataset
df_final.to_csv('patient_health_cleaned.csv', index=False)
```

**Final Dataset Specs:**

| Metric | Value |
|--------|-------|
| Shape | 500 rows × 11 columns |
| Missing Values | **0** |
| Duplicate Rows | 0 |
| Low Risk (0) | 269 patients (53.8%) |
| High Risk (1) | 231 patients (46.2%) |
| ML Ready | ✅ Yes |

---

### Task 7 — Interactive HTML Report

A self-contained 5-tab HTML report is generated and linked inside the notebook.

```python
from IPython.display import FileLink, display

display(FileLink('Data_Cleanser_Report.html',
                 result_html_prefix='📥 Click to open report: '))
```

**Report Tabs:**
- 📊 **Overview** — Dataset schema, stats, region & risk distribution
- 💉 **Part A: Imputation** — All 6 methods with charts and comparison table
- 🔍 **Part B: Outliers** — All 4 methods with box plots and heatmap
- 📝 **Part C: Findings** — Feature correlations, scatter plots, key insights
- 🎛 **Dashboard** — Full pipeline summary with all charts in one view

---

## 📝 Key Findings

| # | Finding | Value |
|---|---------|-------|
| 1 | **Top predictor of disease risk** | BMI (r = 0.287) |
| 2 | **Second predictor** | Glucose (r = 0.268) |
| 3 | **Third predictor** | Blood Pressure (r = 0.247) |
| 4 | **Fourth predictor** | Cholesterol (r = 0.234) |
| 5 | **Class balance** | 46.2% High Risk — no resampling needed |
| 6 | **Best imputation** | MICE — 0 missing, preserves correlations |
| 7 | **Best outlier method** | Winsorization — 0 data loss |
| 8 | **High-risk pattern** | Higher BMI + higher glucose = higher risk |

---

## 🛠 Technologies Used

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | 2.x | Data loading, manipulation, missing value analysis |
| `numpy` | 1.x | Numerical operations, array handling |
| `matplotlib` | 3.x | All chart generation and visualization |
| `seaborn` | 0.x | Statistical plots and heatmaps |
| `scikit-learn` | 1.x | SimpleImputer, KNNImputer, IterativeImputer (MICE), LabelEncoder |
| `scipy` | 1.x | Z-score outlier detection, Winsorization |

---

## ▶️ How to Run

**1. Clone or download the project**
```bash
git clone https://github.com/JATINKUMAR2005/data-cleanser.git
cd data-cleanser
```

**2. Install required libraries**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

**3. Place the dataset in the project folder**
```
patient_health_records_500.csv  ← must be in the same folder as the notebook
```

**4. Open and run the notebook**
```bash
jupyter notebook Data_Cleanser_Complete.ipynb
```

**5. Run all cells** — `Kernel → Restart & Run All`

**6. Open the HTML report** — Click the link generated by the last cell

> ⚠️ **Note:** Keep `Data_Cleanser_Report.html` in the **same folder** as the notebook for the FileLink to work.

---

## 📤 Output Files

| File | Description |
|------|-------------|
| `patient_health_cleaned.csv` | Final ML-ready cleaned dataset (500×11) |
| `Data_Cleanser_Report.html` | Interactive 5-tab HTML report with all charts |
| `missing_values_overview.png` | Missing value bar chart + heatmap |
| `task2a_bmi_imputation.png` | BMI: Original vs Mean vs Median |
| `task2b_region_imputation.png` | Region: Before vs After imputation |
| `task2c_gender_imputation.png` | Gender: Before vs After imputation |
| `task2d_random_sample_imputation.png` | Random sample distribution comparison |
| `task2e_knn_imputation.png` | KNN vs Mean distribution comparison |
| `task2f_mice_comparison.png` | All strategies compared |
| `task3a_zscore_outliers.png` | Z-Score histogram and box plots |
| `task3b_iqr_outliers.png` | IQR scatter with fence lines |
| `task3c_percentile_method.png` | Percentile capping before vs after |
| `task4_winsorization.png` | Winsorization before vs after |
| `task5_before_after_comparison.png` | Box plot comparison all methods |
| `task6_final_dataset_overview.png` | Final dataset charts |
| `final_summary_dashboard.png` | Complete project dashboard |

---

## ✅ Results Summary

| Part | Task | Description | Status |
|------|------|-------------|--------|
| A – Missing Values | Task 1 | Missing value summary report | ✅ Done |
| A – Missing Values | Task 2a | Simple Imputer – BMI (Mean & Median) | ✅ Done |
| A – Missing Values | Task 2b | Simple Imputer – Region (Most Frequent) | ✅ Done |
| A – Missing Values | Task 2c | Most Frequent – Gender | ✅ Done |
| A – Missing Values | Task 2d | Missing Indicator + Random Sample | ✅ Done |
| A – Missing Values | Task 2e | KNN Imputer (k=5) | ✅ Done |
| A – Missing Values | Task 2f | MICE Algorithm (10 iterations) | ✅ Done |
| B – Outliers | Task 3a | Z-Score Method – Cholesterol & Glucose | ✅ Done |
| B – Outliers | Task 3b | IQR Method – BMI & Blood Pressure | ✅ Done |
| B – Outliers | Task 3c | Percentile Method (1st–99th cap) | ✅ Done |
| B – Outliers | Task 4 | Winsorization (5% each tail) | ✅ Done |
| B – Outliers | Task 5 | Before vs After comparison | ✅ Done |
| C – Final Dataset | Task 6 | Final clean ML-ready dataset | ✅ Done |
| C – Final Dataset | Task 7 | Brief report + Interactive HTML | ✅ Done |

---

<div align="center">

**🏆 Best Imputation → MICE Algorithm &nbsp;|&nbsp; 🏆 Best Outlier Treatment → Winsorization**

*Red & White Skill Education — Data Cleanser Project*

**Student:** Jatin Kumar &nbsp;|&nbsp; **GitHub:** [@JATINKUMAR2005](https://github.com/JATINKUMAR2005) &nbsp;|&nbsp; **Portfolio:** [jk-portfolio.lovable.app](https://jk-portfolio.lovable.app/)

</div>
