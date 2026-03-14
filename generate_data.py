"""
generate_data.py
================
Generates synthetic post-surgery patient data in a New Zealand context.

The primary outcome is DAOH90 (Days Alive and Out of Hospital in the 90 days
following surgery). This is modelled via a day-by-day Markov-like state
simulation (states: IN_HOSPITAL, DEAD, HOME) with autoregressive properties,
driven by a latent patient-health score.

Patient covariates
------------------
nhi             : NZ National Health Index (3 upper-case letters + 4 digits)
age             : age at surgery (years)
sex             : Male / Female
ethnicity       : NZ prioritised ethnicity (NZ European, Māori, Pacific,
                  Asian, MELAA, Other)
nzDep           : NZ Deprivation Index decile (1 = least deprived, 10 = most)
dhb             : District Health Board (20 NZ DHBs)
rural           : Rural resident (bool)
surgery_type    : type of surgery performed
asa_grade       : ASA Physical Status (1–5; higher = sicker)
urgency         : Elective / Acute
comorbidities   : number of comorbidities present (0–5)
diabetes        : bool
hypertension    : bool
copd            : bool
cardiac_disease : bool
renal_disease   : bool
bmi             : body-mass index (kg/m²)
smoker          : bool
daoh90          : days alive and out of hospital in 90 days post-surgery
died_90d        : died within 90 days (bool)
readmitted      : at least one readmission within 90 days (bool)
hospital_days   : total days spent in hospital in the 90-day window
"""

import argparse
import random
import string
import numpy as np
import pandas as pd

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)


# ── helpers ──────────────────────────────────────────────────────────────────

def _nhi() -> str:
    """Generate a realistic NZ NHI number (3 letters + 4 digits)."""
    letters = "".join(random.choices(string.ascii_uppercase, k=3))
    digits = f"{random.randint(0, 9999):04d}"
    return letters + digits


NZ_DHBS = [
    "Northland", "Waitemata", "Auckland", "Counties Manukau",
    "Waikato", "Lakes", "Bay of Plenty", "Tairawhiti",
    "Taranaki", "Hawke's Bay", "Whanganui", "MidCentral",
    "Hutt Valley", "Capital & Coast", "Wairarapa", "Nelson Marlborough",
    "West Coast", "Canterbury", "South Canterbury", "Southern",
]

SURGERY_TYPES = [
    "Total Hip Replacement",
    "Total Knee Replacement",
    "Coronary Artery Bypass Graft",
    "Bowel Resection",
    "Laparoscopic Cholecystectomy",
    "Appendicectomy",
    "Spinal Fusion",
    "Abdominal Aortic Aneurysm Repair",
    "Gastrectomy",
    "Nephrectomy",
]

# relative surgical risk multiplier (affects transition probabilities)
SURGERY_RISK = {
    "Total Hip Replacement":           1.0,
    "Total Knee Replacement":          1.0,
    "Coronary Artery Bypass Graft":    2.5,
    "Bowel Resection":                 2.2,
    "Laparoscopic Cholecystectomy":    0.8,
    "Appendicectomy":                  0.9,
    "Spinal Fusion":                   1.4,
    "Abdominal Aortic Aneurysm Repair": 3.0,
    "Gastrectomy":                     2.0,
    "Nephrectomy":                     1.8,
}

# NZ ethnicity distribution (prioritised), approximate 2023 census proportions
ETHNICITY_PROBS = {
    "NZ European": 0.62,
    "Māori":       0.17,
    "Pacific":     0.08,
    "Asian":       0.15,
    "MELAA":       0.03,
    "Other":       0.02,
}
# normalise
_eth_keys = list(ETHNICITY_PROBS.keys())
_eth_vals = np.array(list(ETHNICITY_PROBS.values()))
_eth_vals /= _eth_vals.sum()


def generate_patient(i: int) -> dict:
    """Generate a single patient's covariates and 90-day outcome."""

    nhi = _nhi()

    # ── demographics ────────────────────────────────────────────────────────
    ethnicity = rng.choice(_eth_keys, p=_eth_vals)

    # Age: skewed toward older patients for elective surgery;
    # different distributions by ethnicity (Māori/Pacific present younger)
    if ethnicity in ("Māori", "Pacific"):
        age = int(np.clip(rng.normal(58, 14), 18, 95))
    else:
        age = int(np.clip(rng.normal(65, 14), 18, 95))

    sex = rng.choice(["Male", "Female"], p=[0.50, 0.50])

    # NZ Deprivation Index (1–10); Māori and Pacific over-represented in 8-10
    if ethnicity in ("Māori", "Pacific"):
        nz_dep = int(np.clip(rng.normal(7.5, 2), 1, 10))
    else:
        nz_dep = int(np.clip(rng.normal(5.0, 2.5), 1, 10))

    dhb = rng.choice(NZ_DHBS)
    rural = bool(rng.random() < 0.25)

    # ── surgery ─────────────────────────────────────────────────────────────
    surgery_type = rng.choice(SURGERY_TYPES)
    urgency = rng.choice(["Elective", "Acute"], p=[0.65, 0.35])

    # ASA grade (1–5): higher age / comorbidities → higher ASA
    asa_base = 1 + age / 40 + (1 if urgency == "Acute" else 0)
    asa_grade = int(np.clip(rng.normal(asa_base, 0.8), 1, 5))

    # ── comorbidities ───────────────────────────────────────────────────────
    # Prevalence modulated by age, ethnicity, deprivation
    dep_factor = nz_dep / 10          # 0.1 (low dep) → 1.0 (high dep)
    eth_risk = 1.3 if ethnicity in ("Māori", "Pacific") else 1.0
    age_risk = np.clip(age / 70, 0.3, 1.5)

    p_dm   = np.clip(0.12 * age_risk * eth_risk * dep_factor * 2, 0, 0.70)
    p_htn  = np.clip(0.25 * age_risk * eth_risk * dep_factor * 1.5, 0, 0.80)
    p_copd = np.clip(0.08 * age_risk * (1.4 if sex == "Male" else 1.0), 0, 0.45)
    p_card = np.clip(0.10 * age_risk * (1.3 if sex == "Male" else 1.0), 0, 0.50)
    p_renal = np.clip(0.06 * age_risk * eth_risk, 0, 0.40)

    diabetes        = bool(rng.random() < p_dm)
    hypertension    = bool(rng.random() < p_htn)
    copd            = bool(rng.random() < p_copd)
    cardiac_disease = bool(rng.random() < p_card)
    renal_disease   = bool(rng.random() < p_renal)

    comorbidities = sum([diabetes, hypertension, copd, cardiac_disease, renal_disease])

    # BMI: higher in Pacific / Māori, increases with deprivation
    bmi_mean = 27.5 + dep_factor * 3
    if ethnicity == "Pacific":
        bmi_mean += 4
    elif ethnicity == "Māori":
        bmi_mean += 2
    bmi = round(float(np.clip(rng.normal(bmi_mean, 5), 16, 60)), 1)

    smoker = bool(rng.random() < np.clip(0.12 * eth_risk * dep_factor * 2.5, 0, 0.65))

    # ── latent health score (higher = healthier) ─────────────────────────────
    # Penalise: age, asa_grade, comorbidities, high BMI, smoking, deprivation
    # Reward: younger, elective, lower comorbidity burden
    health_score = (
        100
        - (age - 50) * 0.5          # age penalty above 50
        - asa_grade * 8             # ASA
        - comorbidities * 5         # comorbidity count
        - (bmi - 25) * 0.3          # BMI deviation
        - (10 if smoker else 0)
        - nz_dep * 1.0              # deprivation
        - (5 if urgency == "Acute" else 0)
        - (5 if rural else 0)       # rural access penalty
        + rng.normal(0, 5)          # individual random variation
    )

    # ── day-by-day state simulation ──────────────────────────────────────────
    # States: 0 = HOME, 1 = IN_HOSPITAL, 2 = DEAD
    # All patients start in hospital (just had surgery)
    surg_risk = SURGERY_RISK[surgery_type]

    # Base daily death probability (very small, scaled by risk)
    p_die_daily = np.clip(
        0.0008 * surg_risk * (1 + (100 - health_score) / 80), 0, 0.05
    )
    # Base probability of being discharged on any given hospital day
    p_discharge = np.clip(
        0.20 / surg_risk * (1 + health_score / 150), 0.05, 0.55
    )
    # Base probability of readmission on a HOME day
    p_readmit = np.clip(
        0.015 * surg_risk * (1 + (100 - health_score) / 100), 0.002, 0.12
    )

    state = 1  # start in hospital
    days_in_hospital = 0
    days_dead = 0
    readmitted = False
    was_home_once = False

    for _ in range(90):
        if state == 2:          # dead → absorbing
            days_dead += 1
            continue

        if state == 1:          # IN_HOSPITAL
            days_in_hospital += 1
            # Autoregressiveness: staying in hospital is the default;
            # discharge possible each day
            if rng.random() < p_die_daily * 1.5:
                state = 2
            elif rng.random() < p_discharge:
                state = 0

        else:                   # HOME (state == 0)
            was_home_once = True
            if rng.random() < p_die_daily * 0.5:
                state = 2
            elif rng.random() < p_readmit:
                state = 1
                readmitted = True

    died_90d = (state == 2) or (days_dead > 0)
    hospital_days = days_in_hospital
    daoh90 = 90 - hospital_days - days_dead

    return {
        "nhi":            nhi,
        "age":            age,
        "sex":            sex,
        "ethnicity":      ethnicity,
        "nz_dep":         nz_dep,
        "dhb":            dhb,
        "rural":          rural,
        "surgery_type":   surgery_type,
        "urgency":        urgency,
        "asa_grade":      asa_grade,
        "comorbidities":  comorbidities,
        "diabetes":       diabetes,
        "hypertension":   hypertension,
        "copd":           copd,
        "cardiac_disease": cardiac_disease,
        "renal_disease":  renal_disease,
        "bmi":            bmi,
        "smoker":         smoker,
        "daoh90":         daoh90,
        "died_90d":       died_90d,
        "readmitted":     readmitted,
        "hospital_days":  hospital_days,
    }


def generate_dataset(n: int = 5000, output: str = "patient_data.csv") -> pd.DataFrame:
    print(f"Generating {n} synthetic patient records …")
    patients = [generate_patient(i) for i in range(n)]
    df = pd.DataFrame(patients)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} records to '{output}'")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic NZ patient data.")
    parser.add_argument("-n", "--num-patients", type=int, default=5000,
                        help="Number of patients to generate (default: 5000)")
    parser.add_argument("-o", "--output", type=str, default="patient_data.csv",
                        help="Output CSV file path (default: patient_data.csv)")
    args = parser.parse_args()
    generate_dataset(n=args.num_patients, output=args.output)
