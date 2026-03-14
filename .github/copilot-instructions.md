# GitHub Copilot Instructions

## Project Overview

This repository is a Python simulation of post-surgery patient outcomes in a New Zealand context, focused on **DAOH90** (Days Alive and Out of Hospital in the 90 days following surgery). It generates a synthetic cohort of NZ surgical patients and analyses determinants of DAOH90 and 90-day mortality.

## Repository Structure

- **`generate_data.py`** – Generates synthetic patient data and writes `patient_data.csv`.
  - Uses a day-by-day Markov-like state machine (states: `IN_HOSPITAL`, `HOME`, `DEAD`) to simulate each patient's 90-day trajectory.
  - Patient covariates reflect NZ-specific distributions (ethnicity, NZ Deprivation Index, DHBs).
  - Run with: `python generate_data.py` (optional flags: `-n <num_patients>`, `-o <output_file>`)
- **`analyse_data.py`** – Reads `patient_data.csv` and produces statistical analyses and plots in `plots/`.
  - Fits a Cox Proportional-Hazards model (survival) and an OLS regression model (DAOH90).
  - Saves 7 plots to `plots/`.
- **`patient_data.csv`** – Synthetic patient dataset (output of `generate_data.py`).
- **`plots/`** – Directory for output plots (output of `analyse_data.py`).

## Domain Context

- **DAOH90**: The primary outcome. Counts the number of days a patient was alive AND not in hospital within the 90-day post-surgery window. Ranges from 0 to 89 in this simulation (patients start in hospital on day 1).
- **NHI**: NZ National Health Index — a unique patient identifier (3 uppercase letters + 4 digits).
- **NZ Deprivation Index (`nz_dep`)**: Decile 1 = least deprived, 10 = most deprived.
- **ASA Grade**: ASA Physical Status classification (1–5; higher = sicker).
- **DHB**: District Health Board — one of 20 NZ regional health authorities.
- The simulation reflects real-world NZ health disparities (Māori/Pacific patients have higher deprivation and comorbidity burden).

## Tech Stack & Dependencies

- **Python 3** with the following packages:
  - `pandas`, `numpy`, `scipy` – data manipulation and statistics
  - `matplotlib`, `seaborn` – plotting
  - `lifelines` – survival analysis (Kaplan-Meier, Cox PH)
  - `statsmodels` – OLS regression

Install dependencies with:
```bash
pip install pandas numpy scipy matplotlib seaborn lifelines statsmodels
```

## Coding Conventions

- Use `numpy.random.default_rng(SEED)` (not `np.random.seed`) for reproducibility.
- Global RNG instance is `rng`; set seed via `SEED = 42`.
- Patient-level logic belongs in `generate_patient()` in `generate_data.py`.
- Analysis logic belongs in `analyse_data.py`; plots are saved to the `plots/` directory.
- Use non-interactive matplotlib backend: `matplotlib.use("Agg")` at the top of analysis scripts.
- Column names use `snake_case`. Boolean columns (e.g., `rural`, `smoker`, `died_90d`) are Python `bool` in the dataframe.
- Surgery risk multipliers are defined in `SURGERY_RISK` dict; add new surgery types there first.
- Preserve the existing output format (CSV column order, plot filenames) to maintain compatibility.

## Running the Project

```bash
# Step 1 – generate synthetic patient data
python generate_data.py            # default: 5000 patients → patient_data.csv

# Step 2 – analyse data and generate plots
python analyse_data.py             # reads patient_data.csv, writes to plots/
```
