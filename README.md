# hospital_sim

AI simulation of post-surgery survival rates in a New Zealand context, focusing
on **Days Alive and Out of Hospital in 90 days post-surgery (DAOH90)**.

---

## Overview

DAOH90 is an increasingly used composite outcome in surgical research. It
captures the 90-day window after surgery and counts days the patient was both
alive **and** not in hospital. The distribution is characteristically
bimodal, leptokurtic, and has ties (many patients at the ceiling), making it
difficult to model with standard parametric approaches.

This project:

1. Generates a realistic synthetic cohort of **5,000 NZ surgical patients**
   (`generate_data.py`).
2. Analyses the data to identify determinants of DAOH90 and 90-day mortality
   (`analyse_data.py`).
3. Fits a Cox Proportional-Hazards model (survival) and an OLS regression model
   (DAOH90), producing interpretable outputs including forest plots.

---

## Running the code

```bash
# Install dependencies
pip install pandas numpy scipy matplotlib seaborn lifelines statsmodels scikit-learn

# Step 1 – generate 5 000 synthetic patient records
python generate_data.py            # writes patient_data.csv

# Optional: change sample size or output file
python generate_data.py -n 10000 -o my_data.csv

# Step 2 – run statistical analysis and generate plots
python analyse_data.py             # reads patient_data.csv, writes plots/
```

All output plots are saved to the `plots/` directory.

---

## Synthetic data generation (`generate_data.py`)

### Patient covariates

| Variable | Description |
|---|---|
| `nhi` | NZ National Health Index (3 upper-case letters + 4 digits) |
| `age` | Age at surgery (years, 18–95) |
| `sex` | Male / Female |
| `ethnicity` | NZ prioritised ethnicity (NZ European, Māori, Pacific, Asian, MELAA, Other) |
| `nz_dep` | NZ Deprivation Index decile (1 = least deprived, 10 = most) |
| `dhb` | District Health Board (all 20 NZ DHBs) |
| `rural` | Rural resident (bool) |
| `surgery_type` | Type of surgery (10 types, see below) |
| `urgency` | Elective / Acute |
| `asa_grade` | ASA Physical Status classification (1–5) |
| `comorbidities` | Count of comorbidities (0–5) |
| `diabetes` | Diabetes mellitus (bool) |
| `hypertension` | Hypertension (bool) |
| `copd` | Chronic obstructive pulmonary disease (bool) |
| `cardiac_disease` | Cardiac disease (bool) |
| `renal_disease` | Renal disease (bool) |
| `bmi` | Body-mass index (kg/m²) |
| `smoker` | Current smoker (bool) |

**Outcomes:**

| Variable | Description |
|---|---|
| `daoh90` | Days Alive and Out of Hospital in 90 days post-surgery |
| `died_90d` | Died within 90 days (bool) |
| `readmitted` | ≥1 readmission within 90 days (bool) |
| `hospital_days` | Total days spent in hospital in the 90-day window |

### Simulation methodology

Each patient's post-operative trajectory is simulated via a **day-by-day
Markov-like state machine** with three states:

- `IN_HOSPITAL` – patient is an inpatient
- `HOME` – patient is alive and not in hospital
- `DEAD` – absorbing state

All patients begin in `IN_HOSPITAL`. Daily transition probabilities are derived
from a **latent health score** that incorporates age, ASA grade, comorbidity
burden, BMI, smoking, deprivation, urgency, and surgery risk. This produces the
autoregressive property described in the problem statement (being in hospital
today makes it more likely you are in hospital tomorrow).

Ethnodemographic distributions follow approximate NZ 2023 census proportions.
Comorbidity prevalences, BMI distributions, and deprivation indices are
modulated by ethnicity and socioeconomic factors to reflect known health
disparities.

---

## Statistical analysis (`analyse_data.py`)

### Cohort summary (n = 5,000)

| Variable | Value |
|---|---|
| Mean age | 62.8 years |
| Male / Female | 49 / 51% |
| Elective / Acute | 65 / 35% |
| 90-day mortality | 13.3% |
| Readmission rate | 86.8% |
| Diabetes prevalence | 11.9% |
| Hypertension prevalence | 18.8% |
| Rural residence | 24.9% |

---

## Findings

### 1. DAOH90 distribution

![DAOH90 distribution](plots/01_daoh90_distribution.png)

The DAOH90 distribution is **left-skewed** (skewness = −1.29) and
**leptokurtic** (excess kurtosis = 0.99), consistent with a bimodal-like
pattern. Most patients achieve a high number of days out of hospital, but a
substantial minority have poor outcomes (near 0 days) due to prolonged
hospitalisation or death.

- **Mean DAOH90:** 65.3 days &nbsp;|&nbsp; **Median:** 73 days
- **2.1%** of patients had 0 DAOH90 (continuous hospitalisation or immediate death)
- There is no ceiling at 90 days in this simulation (the maximum simulated is 89
  days), as all patients begin in hospital on day 1.

---

### 2. DAOH90 by patient group

![DAOH90 by groups](plots/02_daoh90_by_groups.png)

Surgery type dominates variation in DAOH90:

| Surgery | Mean DAOH90 | Median |
|---|---|---|
| Laparoscopic Cholecystectomy | 81.3 d | 84 d |
| Appendicectomy | 78.6 d | 83 d |
| Total Hip Replacement | 78.1 d | 81 d |
| Total Knee Replacement | 77.8 d | 81 d |
| Spinal Fusion | 70.8 d | 75 d |
| Nephrectomy | 61.9 d | 67 d |
| Gastrectomy | 57.8 d | 64 d |
| Bowel Resection | 56.9 d | 61 d |
| Coronary Artery Bypass Graft | 48.9 d | 52 d |
| **Abdominal Aortic Aneurysm Repair** | **44.1 d** | **46 d** |

ASA grade shows a monotonic decreasing trend with DAOH90 (Grade 1: 68.8 d →
Grade 5: 57.9 d), and acute admissions have ~2.8 fewer DAOH90 than elective.

---

### 3. Kaplan-Meier 90-day survival curves

![Kaplan-Meier survival curves](plots/03_km_survival_curves.png)

Survival curves stratified by sex, ethnicity, ASA grade, and urgency. Key
observations:

- **ASA Grade:** Higher ASA grade is associated with lower 90-day survival,
  though the effect is gradual across the 90-day window.
- **Urgency:** Acute admissions have modestly lower survival than elective.
- **Sex and Ethnicity:** No statistically significant survival differences were
  detected in the Cox model (see below), consistent with the synthetic data
  construction where ethnicity primarily affects comorbidity burden rather than
  directly affecting mortality probability.

---

### 4. Correlation matrix

![Correlation heatmap](plots/04_correlation_heatmap.png)

DAOH90 is negatively correlated with hospital days (r ≈ −0.97; by construction)
and positively correlated with health-promoting factors. Comorbidities,
ASA grade, and age are moderately inter-correlated.

---

### 5. Cox Proportional-Hazards model — 90-day mortality

![Cox forest plot](plots/05_cox_forest_plot.png)

Statistically significant predictors of 90-day mortality (p < 0.05):

| Predictor | HR | 95% CI | p-value | Direction |
|---|---|---|---|---|
| Coronary Artery Bypass Graft | 1.70 | 1.42–2.03 | <0.001 | ↑ risk |
| Abdominal Aortic Aneurysm Repair | 1.58 | 1.32–1.88 | <0.001 | ↑ risk |
| Gastrectomy | 1.36 | 1.12–1.65 | 0.002 | ↑ risk |
| Bowel Resection | 1.26 | 1.04–1.53 | 0.019 | ↑ risk |
| Nephrectomy | 1.25 | 1.03–1.51 | 0.026 | ↑ risk |
| Total Knee Replacement | 0.71 | 0.57–0.87 | 0.001 | ↓ risk |
| Total Hip Replacement | 0.68 | 0.55–0.85 | <0.001 | ↓ risk |
| Laparoscopic Cholecystectomy | 0.64 | 0.51–0.80 | <0.001 | ↓ risk |

**Surgery type** is the dominant predictor of 90-day mortality in this cohort.
Patient-level factors (age, ASA grade, comorbidities, sex, ethnicity,
deprivation) showed trends in the expected directions but did not reach
statistical significance individually, likely because surgery-type risk
overwhelms these effects in the Cox model at n=5,000.

---

### 6. OLS regression — DAOH90

![OLS coefficient plot](plots/06_ols_coefficient_plot.png)

The OLS model (R² = 0.358) confirms the following **statistically significant**
predictors of DAOH90 (all p < 0.05):

| Predictor | β (days) | Direction |
|---|---|---|
| Abdominal Aortic Aneurysm Repair | −34.6 | Fewer days |
| Coronary Artery Bypass Graft | −29.6 | Fewer days |
| Bowel Resection | −21.6 | Fewer days |
| Gastrectomy | −20.5 | Fewer days |
| Nephrectomy | −16.5 | Fewer days |
| Spinal Fusion | −8.3 | Fewer days |
| ASA Grade (per unit ↑) | −1.6 | Fewer days |
| No. comorbidities (per unit ↑) | −1.6 | Fewer days |
| Current smoker | −2.0 | Fewer days |
| Rural residence | −1.4 | Fewer days |
| Age (per year ↑) | −0.09 | Fewer days |
| Laparoscopic Cholecystectomy | +2.6 | More days |

**Ethnicity** was not a significant predictor in either model after adjusting for
other covariates. However, Māori and Pacific patients face higher pre-operative
burdens (greater deprivation, higher comorbidity rates) that indirectly reduce
DAOH90 — the pathways are mediated rather than direct.

---

### 7. Deprivation and ethnicity

![Deprivation and ethnicity](plots/07_deprivation_ethnicity.png)

There is a **negative relationship between NZ Deprivation Index and DAOH90**
(r = −0.03, p < 0.05), indicating that patients from more deprived areas tend to
have worse post-surgical outcomes. This is consistent with reduced access to
primary care, higher comorbidity burden, and greater distances to hospital in
rural/deprived areas.

---

## Key conclusions

1. **Surgery type is the strongest determinant** of both 90-day mortality and
   DAOH90 in this synthetic cohort. High-risk procedures (CABG, AAA repair)
   are associated with substantially worse outcomes.

2. **Patient physiology matters:** ASA grade, comorbidity count, age, and
   smoking status each independently reduce DAOH90, even after accounting for
   surgery type.

3. **Social determinants matter:** Rural residence (−1.4 days) and high
   deprivation are associated with fewer DAOH90, likely reflecting healthcare
   access barriers.

4. **Ethnicity effects are mediated:** No direct independent effect of ethnicity
   on mortality or DAOH90 was found after adjustment; however, Māori and Pacific
   patients have higher baseline deprivation and comorbidity burden, driving
   indirect disparities.

5. **DAOH90 is leptokurtic and left-skewed,** confirming the distributional
   challenges described in the problem statement. Standard linear regression is
   an approximation; beta regression, two-part models, or distributional
   regression may better capture this outcome in real-world applications.

---

## Limitations

- This is **synthetic data** — all patterns are artefacts of the simulation
  parameters. Results should not be interpreted as real clinical evidence.
- The day-by-day Markov model is a simplification; real post-surgical
  trajectories involve more complex temporal dependencies.
- The OLS model violates normality assumptions (the residuals are non-normal);
  a Tobit or fractional regression model would be more appropriate for bounded
  outcomes.
- With real data, one would also model **time-varying covariates** (e.g.,
  in-hospital interventions) and account for **competing risks** (death vs.
  readmission).
