"""
analyse_data.py
===============
Analyses the synthetic NZ post-surgery patient dataset produced by
generate_data.py.

Sections
--------
1. Descriptive statistics
2. Distribution of DAOH90 (bimodal / leptokurtic characterisation)
3. Kaplan-Meier 90-day survival curves stratified by key covariates
4. Cox Proportional-Hazards model for 90-day mortality
5. Tobit-style (OLS on clipped outcome) regression for DAOH90
6. Summary of statistically significant predictors
7. Forest plot of Cox HR and regression coefficients
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")          # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from lifelines import KaplanMeierFitter, CoxPHFitter
from scipy import stats

warnings.filterwarnings("ignore")

# ── configuration ─────────────────────────────────────────────────────────────
DATA_FILE   = "patient_data.csv"
PLOTS_DIR   = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

ALPHA = 0.05          # significance threshold
SEED  = 42

sns.set_theme(style="whitegrid", palette="muted")

# ── 0. load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_FILE)

# Binary indicator columns
for col in ["rural", "diabetes", "hypertension", "copd",
            "cardiac_disease", "renal_disease", "smoker",
            "died_90d", "readmitted"]:
    df[col] = df[col].astype(bool)

print(f"Loaded {len(df):,} patient records.\n")
print("── Column dtypes ──")
print(df.dtypes, "\n")

# ── 1. descriptive statistics ─────────────────────────────────────────────────
print("═" * 60)
print("SECTION 1 – DESCRIPTIVE STATISTICS")
print("═" * 60)

num_summary = df[["age", "bmi", "nz_dep", "asa_grade",
                   "comorbidities", "hospital_days", "daoh90"]].describe().round(2)
print(num_summary, "\n")

for col in ["sex", "ethnicity", "surgery_type", "urgency"]:
    pct = df[col].value_counts(normalize=True).mul(100).round(1)
    print(f"  {col}:\n{pct.to_string()}\n")

binary_cols = ["died_90d", "readmitted", "diabetes", "hypertension",
               "copd", "cardiac_disease", "renal_disease", "smoker", "rural"]
print("  Binary outcomes / comorbidities (%):")
print(df[binary_cols].mean().mul(100).round(1).to_string(), "\n")

# ── 2. DAOH90 distribution ────────────────────────────────────────────────────
print("═" * 60)
print("SECTION 2 – DAOH90 DISTRIBUTION")
print("═" * 60)

daoh = df["daoh90"]
print(f"  Mean:      {daoh.mean():.1f}")
print(f"  Median:    {daoh.median():.1f}")
print(f"  Std dev:   {daoh.std():.1f}")
print(f"  Skewness:  {daoh.skew():.3f}")
print(f"  Kurtosis:  {daoh.kurtosis():.3f}  (excess, leptokurtic if >0)")
print(f"  % at 90d:  {(daoh == 90).mean()*100:.1f}%  (ceiling / right-pile)")
print(f"  % at 0d:   {(daoh == 0).mean()*100:.1f}%  (died day 0 or never left)")
print()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(daoh, bins=45, color="#4C72B0", edgecolor="white", linewidth=0.4)
axes[0].set_title("Distribution of DAOH90")
axes[0].set_xlabel("Days Alive & Out of Hospital (0–90)")
axes[0].set_ylabel("Count")
axes[0].axvline(daoh.mean(), color="red",    linestyle="--", label=f"Mean {daoh.mean():.1f}")
axes[0].axvline(daoh.median(), color="green", linestyle=":",  label=f"Median {daoh.median():.1f}")
axes[0].legend(fontsize=8)

# by alive vs died
ax2 = axes[1]
for label, grp in df.groupby("died_90d")["daoh90"]:
    ax2.hist(grp, bins=40, alpha=0.6,
             label=("Died within 90d" if label else "Survived 90d"))
ax2.set_title("DAOH90 by 90-day Survival")
ax2.set_xlabel("Days Alive & Out of Hospital")
ax2.set_ylabel("Count")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_daoh90_distribution.png", dpi=150)
plt.close()
print(f"  → Saved {PLOTS_DIR}/01_daoh90_distribution.png\n")

# ── 3. DAOH90 by key groups ───────────────────────────────────────────────────
print("═" * 60)
print("SECTION 3 – DAOH90 BY KEY GROUPS")
print("═" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3a ethnicity
eth_order = df.groupby("ethnicity")["daoh90"].median().sort_values().index
sns.boxplot(data=df, x="daoh90", y="ethnicity",
            order=eth_order, ax=axes[0, 0], palette="Set2")
axes[0, 0].set_title("DAOH90 by Ethnicity")
axes[0, 0].set_xlabel("Days")

# 3b surgery type
surg_order = df.groupby("surgery_type")["daoh90"].median().sort_values().index
sns.boxplot(data=df, x="daoh90", y="surgery_type",
            order=surg_order, ax=axes[0, 1], palette="Set3")
axes[0, 1].set_title("DAOH90 by Surgery Type")
axes[0, 1].set_xlabel("Days")

# 3c ASA grade
sns.boxplot(data=df, x="asa_grade", y="daoh90",
            ax=axes[1, 0], palette="RdYlGn_r")
axes[1, 0].set_title("DAOH90 by ASA Grade")
axes[1, 0].set_ylabel("Days")

# 3d age groups
df["age_group"] = pd.cut(df["age"], bins=[17, 40, 55, 65, 75, 96],
                         labels=["18–40", "41–55", "56–65", "66–75", "76+"])
sns.boxplot(data=df, x="age_group", y="daoh90",
            ax=axes[1, 1], palette="Blues")
axes[1, 1].set_title("DAOH90 by Age Group")
axes[1, 1].set_ylabel("Days")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/02_daoh90_by_groups.png", dpi=150)
plt.close()
print(f"  → Saved {PLOTS_DIR}/02_daoh90_by_groups.png\n")

# Group means
for col in ["ethnicity", "surgery_type", "asa_grade", "urgency", "sex"]:
    grp = df.groupby(col)["daoh90"].agg(["mean", "median", "std"]).round(1)
    print(f"  DAOH90 by {col}:\n{grp.to_string()}\n")

# ── 4. Kaplan-Meier survival curves ──────────────────────────────────────────
print("═" * 60)
print("SECTION 4 – KAPLAN-MEIER SURVIVAL CURVES")
print("═" * 60)

# For KM we need a time-to-event column.
# Use hospital_days as a proxy for "time in hospital" or create
# a 90-day binary (died_90d).  We treat death as event; DAOH90 as duration.
# duration = days until death OR 90 (censored)
df["km_duration"] = np.where(df["died_90d"], 90 - df["daoh90"], 90)
df["km_event"]    = df["died_90d"].astype(int)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

def _km_plot(ax, groups, title, palette=None):
    kmf = KaplanMeierFitter()
    colors = plt.cm.tab10.colors if palette is None else palette
    for i, (label, grp_df) in enumerate(groups):
        kmf.fit(grp_df["km_duration"], grp_df["km_event"], label=str(label))
        kmf.plot_survival_function(ax=ax, ci_show=True, color=colors[i % len(colors)])
    ax.set_title(title)
    ax.set_xlabel("Days post-surgery")
    ax.set_ylabel("Survival probability")
    ax.legend(fontsize=8)

_km_plot(axes[0, 0],
         df.groupby("sex"),
         "90-day Survival by Sex")

_km_plot(axes[0, 1],
         df.groupby("ethnicity"),
         "90-day Survival by Ethnicity")

_km_plot(axes[1, 0],
         df.groupby("asa_grade"),
         "90-day Survival by ASA Grade")

df["urgency_label"] = df["urgency"]
_km_plot(axes[1, 1],
         df.groupby("urgency_label"),
         "90-day Survival by Urgency")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_km_survival_curves.png", dpi=150)
plt.close()
print(f"  → Saved {PLOTS_DIR}/03_km_survival_curves.png\n")

# ── 5. Cox Proportional-Hazards model (90-day mortality) ─────────────────────
print("═" * 60)
print("SECTION 5 – COX PROPORTIONAL-HAZARDS MODEL (90-day mortality)")
print("═" * 60)

# Prepare Cox data: one-hot encode categoricals
cox_df = df.copy()
cox_df["female"]    = (cox_df["sex"] == "Female").astype(int)
cox_df["acute"]     = (cox_df["urgency"] == "Acute").astype(int)
cox_df["rural_int"] = cox_df["rural"].astype(int)

# Ethnicity dummies (reference = NZ European)
eth_dummies = pd.get_dummies(cox_df["ethnicity"], prefix="eth", drop_first=False)
eth_dummies = eth_dummies.drop(columns=["eth_NZ European"], errors="ignore")
cox_df = pd.concat([cox_df, eth_dummies], axis=1)

# Surgery dummies (reference = Appendicectomy – lowest risk)
surg_dummies = pd.get_dummies(cox_df["surgery_type"], prefix="surg", drop_first=False)
surg_dummies = surg_dummies.drop(columns=["surg_Appendicectomy"], errors="ignore")
cox_df = pd.concat([cox_df, surg_dummies], axis=1)

eth_cols  = [c for c in cox_df.columns if c.startswith("eth_")]
surg_cols = [c for c in cox_df.columns if c.startswith("surg_")]

cox_covs = (
    ["age", "female", "nz_dep", "asa_grade", "comorbidities",
     "bmi", "acute", "rural_int", "smoker"]
    + eth_cols
    + surg_cols
)
# cast booleans to int for lifelines
for col in ["smoker"]:
    cox_df[col] = cox_df[col].astype(int)
for col in eth_cols + surg_cols:
    cox_df[col] = cox_df[col].astype(int)

cox_input = cox_df[cox_covs + ["km_duration", "km_event"]].dropna()

cph = CoxPHFitter(penalizer=0.1)
cph.fit(cox_input, duration_col="km_duration", event_col="km_event")

print(cph.summary.to_string(), "\n")

# significant predictors
cox_sig = cph.summary[cph.summary["p"] < ALPHA].copy()
print(f"\n  Significant Cox predictors (p < {ALPHA}):")
print(cox_sig[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].to_string(), "\n")

# ── 6. OLS regression for DAOH90 ─────────────────────────────────────────────
print("═" * 60)
print("SECTION 6 – OLS REGRESSION FOR DAOH90")
print("═" * 60)

import statsmodels.api as sm

ols_df = cox_df.copy()
# clip outcome away from hard boundaries slightly for OLS
ols_df["daoh90_clipped"] = np.clip(ols_df["daoh90"], 0, 90)

base_cols = ["age", "female", "nz_dep", "asa_grade", "comorbidities",
             "bmi", "acute", "rural_int", "smoker"]

# Cast all predictor columns to numeric
for col in base_cols + eth_cols + surg_cols:
    ols_df[col] = pd.to_numeric(ols_df[col], errors="coerce")

X_cols = base_cols + eth_cols + surg_cols
X = sm.add_constant(ols_df[X_cols].astype(float))
y = ols_df["daoh90_clipped"].astype(float)

ols_model = sm.OLS(y, X).fit()
print(ols_model.summary(), "\n")

ols_sig = ols_model.pvalues[ols_model.pvalues < ALPHA].sort_values()
print(f"\n  Significant OLS predictors (p < {ALPHA}):")
print(ols_sig.to_string(), "\n")

# ── 7. Comorbidity heatmap ────────────────────────────────────────────────────
print("═" * 60)
print("SECTION 7 – COMORBIDITY / COVARIATE HEATMAP")
print("═" * 60)

corr_cols = ["age", "bmi", "nz_dep", "asa_grade", "comorbidities",
             "daoh90", "hospital_days", "died_90d",
             "diabetes", "hypertension", "copd", "cardiac_disease", "renal_disease"]
corr_df = df[corr_cols].copy()
for c in ["died_90d", "diabetes", "hypertension", "copd", "cardiac_disease", "renal_disease"]:
    corr_df[c] = corr_df[c].astype(int)

corr_matrix = corr_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, linewidths=0.5, ax=ax,
            annot_kws={"size": 8})
ax.set_title("Correlation Matrix – Patient Covariates & Outcomes", fontsize=13)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/04_correlation_heatmap.png", dpi=150)
plt.close()
print(f"  → Saved {PLOTS_DIR}/04_correlation_heatmap.png\n")

# ── 8. Forest plot – Cox hazard ratios ───────────────────────────────────────
print("═" * 60)
print("SECTION 8 – FOREST PLOT (Cox HRs)")
print("═" * 60)

cox_sum = cph.summary.copy()
cox_sum = cox_sum.sort_values("exp(coef)", ascending=True)

# Shorten labels for display
label_map = {
    "age":         "Age (per year)",
    "female":      "Female sex",
    "nz_dep":      "NZ Deprivation (per decile)",
    "asa_grade":   "ASA Grade",
    "comorbidities": "No. comorbidities",
    "bmi":         "BMI (per kg/m²)",
    "acute":       "Acute urgency",
    "rural_int":   "Rural residence",
    "smoker":      "Current smoker",
}
for eth in ["NZ European", "Māori", "Pacific", "Asian", "MELAA", "Other"]:
    label_map[f"eth_{eth}"] = f"Ethnicity: {eth}"

SURGERY_TYPES_LOCAL = [
    "Total Hip Replacement", "Total Knee Replacement",
    "Coronary Artery Bypass Graft", "Bowel Resection",
    "Laparoscopic Cholecystectomy", "Appendicectomy",
    "Spinal Fusion", "Abdominal Aortic Aneurysm Repair",
    "Gastrectomy", "Nephrectomy",
]
for surg in SURGERY_TYPES_LOCAL:
    label_map[f"surg_{surg}"] = f"Surgery: {surg}"

cox_sum["label"] = cox_sum.index.map(lambda x: label_map.get(x, x))

fig, ax = plt.subplots(figsize=(10, max(6, len(cox_sum) * 0.35)))
y_pos = range(len(cox_sum))

colors = ["#d62728" if p < ALPHA else "#aec7e8"
          for p in cox_sum["p"]]

ax.barh(list(y_pos), cox_sum["exp(coef)"] - 1,
        left=1, color=colors, edgecolor="grey", linewidth=0.4, height=0.7)
ax.errorbar(cox_sum["exp(coef)"].values,
            list(y_pos),
            xerr=[
                (cox_sum["exp(coef)"] - cox_sum["exp(coef) lower 95%"]).values,
                (cox_sum["exp(coef) upper 95%"] - cox_sum["exp(coef)"]).values,
            ],
            fmt="none", color="black", capsize=3, linewidth=1.2)

ax.axvline(1.0, color="black", linewidth=1.0, linestyle="--")
ax.set_yticks(list(y_pos))
ax.set_yticklabels(cox_sum["label"], fontsize=8)
ax.set_xlabel("Hazard Ratio (95% CI)")
ax.set_title("Cox PH – Hazard Ratios for 90-day Mortality\n"
             "(red = p < 0.05, blue = ns)")
sig_patch  = mpatches.Patch(color="#d62728", label="p < 0.05")
ns_patch   = mpatches.Patch(color="#aec7e8", label="p ≥ 0.05 (ns)")
ax.legend(handles=[sig_patch, ns_patch], fontsize=8)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/05_cox_forest_plot.png", dpi=150)
plt.close()
print(f"  → Saved {PLOTS_DIR}/05_cox_forest_plot.png\n")

# ── 9. OLS coefficient plot ───────────────────────────────────────────────────
ols_coef = pd.DataFrame({
    "coef":  ols_model.params,
    "lower": ols_model.conf_int()[0],
    "upper": ols_model.conf_int()[1],
    "p":     ols_model.pvalues,
}).drop(index="const", errors="ignore")

ols_coef["label"] = ols_coef.index.map(lambda x: label_map.get(x, x))
ols_coef = ols_coef.sort_values("coef")

fig, ax = plt.subplots(figsize=(10, max(6, len(ols_coef) * 0.35)))
y_pos = range(len(ols_coef))
colors_ols = ["#d62728" if p < ALPHA else "#aec7e8" for p in ols_coef["p"]]

ax.barh(list(y_pos), ols_coef["coef"], color=colors_ols,
        edgecolor="grey", linewidth=0.4, height=0.7)
ax.errorbar(ols_coef["coef"].values, list(y_pos),
            xerr=[
                (ols_coef["coef"] - ols_coef["lower"]).values,
                (ols_coef["upper"] - ols_coef["coef"]).values,
            ],
            fmt="none", color="black", capsize=3, linewidth=1.2)

ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
ax.set_yticks(list(y_pos))
ax.set_yticklabels(ols_coef["label"], fontsize=8)
ax.set_xlabel("Coefficient (days change in DAOH90, 95% CI)")
ax.set_title("OLS Regression – Predictors of DAOH90\n"
             "(red = p < 0.05, blue = ns)")
ax.legend(handles=[sig_patch, ns_patch], fontsize=8)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/06_ols_coefficient_plot.png", dpi=150)
plt.close()
print(f"  → Saved {PLOTS_DIR}/06_ols_coefficient_plot.png\n")

# ── 10. DAOH90 by NZ Deprivation & Ethnicity ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# scatter with trend line: deprivation vs daoh90
axes[0].scatter(df["nz_dep"], df["daoh90"],
                alpha=0.15, s=8, color="#4C72B0")
m, b, r, p_val, se = stats.linregress(df["nz_dep"], df["daoh90"])
x_line = np.linspace(1, 10, 100)
axes[0].plot(x_line, m * x_line + b, color="red", linewidth=2,
             label=f"r={r:.3f}, p={p_val:.4f}")
axes[0].set_xlabel("NZ Deprivation Decile (1=least, 10=most)")
axes[0].set_ylabel("DAOH90")
axes[0].set_title("Deprivation vs DAOH90")
axes[0].legend(fontsize=9)

# violin by ethnicity
eth_order2 = df.groupby("ethnicity")["daoh90"].median().sort_values(ascending=False).index
sns.violinplot(data=df, x="ethnicity", y="daoh90",
               order=eth_order2, ax=axes[1], palette="Set2",
               inner="quartile")
axes[1].set_title("DAOH90 by Ethnicity")
axes[1].set_xlabel("")
axes[1].set_ylabel("Days")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/07_deprivation_ethnicity.png", dpi=150)
plt.close()
print(f"  → Saved {PLOTS_DIR}/07_deprivation_ethnicity.png\n")

# ── 11. 30/60/90-day mortality summary table ──────────────────────────────────
print("═" * 60)
print("SECTION 9 – MORTALITY SUMMARY BY KEY GROUPS")
print("═" * 60)

mort_rows = []
for grp_col in ["sex", "ethnicity", "asa_grade", "urgency"]:
    for grp_val, sub in df.groupby(grp_col):
        mort_rows.append({
            "Group":    grp_col,
            "Category": grp_val,
            "N":        len(sub),
            "Died_90d_%": sub["died_90d"].mean() * 100,
            "Mean_DAOH90": sub["daoh90"].mean(),
            "Median_DAOH90": sub["daoh90"].median(),
        })

mort_df = pd.DataFrame(mort_rows)
print(mort_df.round(2).to_string(index=False), "\n")

# ── 12. Collect key findings ──────────────────────────────────────────────────
print("═" * 60)
print("KEY FINDINGS SUMMARY")
print("═" * 60)

print("\n  DAOH90 distribution:")
print(f"    Mean {daoh.mean():.1f}d, Median {daoh.median():.0f}d, "
      f"Skewness {daoh.skew():.2f}, Kurtosis {daoh.kurtosis():.2f}")
print(f"    {(daoh == 90).mean()*100:.1f}% of patients achieved perfect DAOH90 (90 days)")
print(f"    {(daoh == 0).mean()*100:.1f}% achieved 0 days (died immediately or continuous hospitalisation)")

print("\n  Significant Cox predictors of 90-day mortality:")
for idx, row in cox_sig.iterrows():
    direction = "↑ risk" if row["exp(coef)"] > 1 else "↓ risk"
    label = label_map.get(idx, idx)
    print(f"    {label}: HR={row['exp(coef)']:.2f} "
          f"[{row['exp(coef) lower 95%']:.2f}–{row['exp(coef) upper 95%']:.2f}], "
          f"p={row['p']:.4f}  ({direction})")

print("\n  Significant OLS predictors of DAOH90 (days change per unit):")
for idx, pv in ols_sig.items():
    coef = ols_model.params[idx]
    label = label_map.get(idx, idx)
    print(f"    {label}: β={coef:.2f}d, p={pv:.4f}")

print("\n  ── Analysis complete. Plots saved to:", PLOTS_DIR)
