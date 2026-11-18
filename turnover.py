"""Employee turnover risk segmenting, 2026 version.

Usage: python turnover.py data/HR_comma_sep.csv output

Reproduces the 2017 two-way turnover rate matrices with pandas, then goes further:
a scatter of the three leaver clusters, and a gradient boosted model with SHAP to
show which traits actually drive turnover once they are considered together.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import plotstyle as ps

DATA = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
results = {}

hr = pd.read_csv(DATA)
results["rows"] = int(len(hr))
results["base_rate"] = round(float(hr["left"].mean()), 4)
print(hr.shape, "base turnover", results["base_rate"])


# ---------------------------------------------------------------- rate matrices
def rate_matrix(df, y, c1, c2=None, bins=20):
    """Turnover rate for every combination of two traits.

    Continuous columns are cut into equal-width ranges; integer and text columns
    keep one group per value. Returns a DataFrame of rates (rows = c1, cols = c2).
    """
    def group(col):
        s = df[col]
        if s.dtype.kind == "f":
            return pd.cut(s, bins=bins)
        return s
    if c2 is None:
        return df.groupby(group(c1), observed=True)[y].mean().to_frame(y)
    return df.pivot_table(index=group(c1), columns=group(c2), values=y,
                          aggfunc="mean", observed=True)


# 1. satisfaction alone: rate and count per bin
sat_bins = pd.cut(hr["satisfaction_level"], bins=20)
by_sat = hr.groupby(sat_bins, observed=True)["left"].agg(rate="mean", n="size")
results["satisfaction_bins"] = [
    {"range": f"{iv.left:.3f} to {iv.right:.3f}", "rate": round(float(r.rate), 3), "n": int(r.n)}
    for iv, r in by_sat.iterrows()
]
fig, ax = ps.figure(9.5, 5)
centers = [iv.mid for iv in by_sat.index]
widths = [iv.length * 0.92 for iv in by_sat.index]
bars = ax.bar(centers, by_sat["rate"], width=widths, color=ps.BLUE)
ax.axhline(results["base_rate"], color=ps.MUTED, ls="--", lw=1)
ax.text(1.0, results["base_rate"] + 0.015, f"company rate {results['base_rate']:.0%}",
        ha="right", color=ps.MUTED, fontsize=9)
for b, (iv, r) in zip(bars, by_sat.iterrows()):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{int(r.n):,}",
            ha="center", va="bottom", fontsize=7.5, color=ps.MUTED, rotation=90)
ax.set_xlabel("Satisfaction level (20 equal-width ranges, employee count above each bar)")
ax.set_ylabel("Share who left")
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_title("Turnover rate by satisfaction level")
ps.save(fig, OUT / "satisfaction_rate.png")


def heatmap(matrix, path, title, xlabel, ylabel, fmt=".0%", figsize=(10, 7.5), counts=None):
    fig, ax = plt.subplots(figsize=figsize)
    annot = matrix.map(lambda v: "" if pd.isna(v) else f"{v:.0%}") if fmt == ".0%" else True
    sns.heatmap(matrix, ax=ax, cmap=ps.SEQUENTIAL, vmin=0, vmax=1, annot=annot, fmt="",
                annot_kws={"size": 8}, linewidths=0.5, linecolor="white",
                cbar_kws={"format": plt.FuncFormatter(lambda v, _: f"{v:.0%}"), "shrink": 0.7},
                mask=matrix.isna())
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    ax.set_yticklabels([f"{iv.left:.2f} to {iv.right:.2f}" if hasattr(iv, "left") else str(iv)
                        for iv in matrix.index], rotation=0, fontsize=8)
    ax.set_xticklabels([f"{iv.left:.0f} to {iv.right:.0f}" if hasattr(iv, "left") else str(iv)
                        for iv in matrix.columns], rotation=45, ha="right", fontsize=8)
    ps.save(fig, path)


# 2. satisfaction x department
m_dept = rate_matrix(hr, "left", "satisfaction_level", "department")
heatmap(m_dept, OUT / "satisfaction_department.png",
        "Turnover rate by satisfaction level and department",
        "Department", "Satisfaction level")

# 3. satisfaction x monthly hours (hours cut into ranges of 20)
hr["hours_range"] = pd.cut(hr["average_monthly_hours"], bins=range(80, 330, 20))
m_hours = hr.pivot_table(index=sat_bins, columns="hours_range", values="left",
                         aggfunc="mean", observed=True)
n_hours = hr.pivot_table(index=sat_bins, columns="hours_range", values="left",
                         aggfunc="size", observed=True)
heatmap(m_hours, OUT / "satisfaction_hours.png",
        "Turnover rate by satisfaction level and average monthly hours",
        "Average monthly hours", "Satisfaction level")

# 4. satisfaction x number of projects
m_proj = rate_matrix(hr, "left", "satisfaction_level", "number_project")
heatmap(m_proj, OUT / "satisfaction_projects.png",
        "Turnover rate by satisfaction level and number of projects",
        "Number of projects", "Satisfaction level", figsize=(8, 7.5))

# 5. the same, employees with a workplace accident only
acc = hr[hr["work_accident"] == 1]
m_acc = rate_matrix(acc, "left", "satisfaction_level", "number_project")
heatmap(m_acc, OUT / "satisfaction_projects_accident.png",
        "Turnover rate by satisfaction and projects, employees with a workplace accident",
        "Number of projects", "Satisfaction level", figsize=(8, 7.5))

# numbers the write-up needs
band = hr[(hr.satisfaction_level > 0.36) & (hr.satisfaction_level <= 0.46)]
results["mid_band"] = {
    "n": int(len(band)),
    "rate": round(float(band.left.mean()), 3),
    "rate_by_projects": {int(k): round(float(v), 3) for k, v in band.groupby("number_project").left.mean().items()},
    "n_by_projects": {int(k): int(v) for k, v in band.groupby("number_project").size().items()},
    "hours_median_leavers": float(band[band.left == 1].average_monthly_hours.median()),
    "hours_median_stayers": float(band[band.left == 0].average_monthly_hours.median()),
    "accident_rate": round(float(acc[(acc.satisfaction_level > 0.36) & (acc.satisfaction_level <= 0.46)].left.mean()), 3),
}
results["accident_overall"] = {
    "rate_with_accident": round(float(acc.left.mean()), 3),
    "rate_without": round(float(hr[hr.work_accident == 0].left.mean()), 3),
}
results["dept_rates"] = {k: round(float(v), 3) for k, v in hr.groupby("department").left.mean().sort_values().items()}

# ---------------------------------------------------------------- the cluster view
fig, ax = ps.figure(9.5, 6)
stay = hr[hr.left == 0]
leave = hr[hr.left == 1]
ax.scatter(stay.average_monthly_hours, stay.satisfaction_level, s=6, alpha=0.25,
           color="#a0aec0", label=f"stayed ({len(stay):,})", linewidths=0)
ax.scatter(leave.average_monthly_hours, leave.satisfaction_level, s=6, alpha=0.5,
           color=ps.RED, label=f"left ({len(leave):,})", linewidths=0)
ax.set_xlabel("Average monthly hours")
ax.set_ylabel("Satisfaction level")
ax.set_title("Who left: satisfaction against hours worked")
ax.legend(loc="upper left", markerscale=3)
ax.grid(True, axis="both")
ax.annotate("overworked and unhappy", xy=(275, 0.12), xytext=(195, 0.2), fontsize=9,
            color=ps.TEXT, arrowprops=dict(arrowstyle="-", color=ps.MUTED, lw=0.8))
ax.annotate("under-used, mildly unhappy", xy=(145, 0.42), xytext=(100, 0.60), fontsize=9,
            color=ps.TEXT, arrowprops=dict(arrowstyle="-", color=ps.MUTED, lw=0.8))
ax.annotate("busy and happy, left anyway", xy=(250, 0.85), xytext=(150, 0.95), fontsize=9,
            color=ps.TEXT, arrowprops=dict(arrowstyle="-", color=ps.MUTED, lw=0.8))
ps.save(fig, OUT / "clusters.png")

# cluster sizes (rough boxes, for the prose)
c1 = leave[(leave.satisfaction_level < 0.12) & (leave.average_monthly_hours > 240)]
c2 = leave[(leave.satisfaction_level.between(0.35, 0.47)) & (leave.average_monthly_hours.between(125, 165))]
c3 = leave[(leave.satisfaction_level > 0.7) & (leave.average_monthly_hours > 210)]
results["clusters"] = {
    "overworked_unhappy": int(len(c1)),
    "underused_mildly_unhappy": int(len(c2)),
    "busy_happy": int(len(c3)),
    "leavers_total": int(len(leave)),
    "c2_projects_mode": int(c2.number_project.mode().iloc[0]),
    "c2_share_two_projects": round(float((c2.number_project == 2).mean()), 3),
    "c1_projects_median": float(c1.number_project.median()),
}

# ---------------------------------------------------------------- a model, and SHAP
features = ["satisfaction_level", "last_evaluation", "number_project", "average_monthly_hours",
            "time_spent_company", "work_accident", "promotion_last_5years", "department", "salary"]
X = hr[features].copy()
X["department"] = X["department"].astype("category")
X["salary"] = pd.Categorical(X["salary"], categories=["low", "medium", "high"], ordered=True)
y = hr["left"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=2017)

model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                           min_child_samples=40, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=2017, verbose=-1)
model.fit(X_tr, y_tr)
auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
results["model"] = {"type": "LightGBM", "test_auc": round(float(auc), 4),
                    "train_rows": int(len(X_tr)), "test_rows": int(len(X_te))}
print("test AUC", auc)

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_te)
if isinstance(sv, list):
    sv = sv[1]
mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=features).sort_values(ascending=False)
results["shap_mean_abs"] = {k: round(float(v), 4) for k, v in mean_abs.items()}

pretty = {"satisfaction_level": "Satisfaction level", "last_evaluation": "Last evaluation score",
          "number_project": "Number of projects", "average_monthly_hours": "Average monthly hours",
          "time_spent_company": "Years at the company", "work_accident": "Workplace accident",
          "promotion_last_5years": "Promoted in last 5 years", "department": "Department",
          "salary": "Salary band"}
X_plot = X_te.copy()
X_plot["department"] = X_plot["department"].cat.codes
X_plot["salary"] = X_plot["salary"].cat.codes
X_plot.columns = [pretty[c] for c in X_plot.columns]
plt.figure(figsize=(9, 5.5))
shap.summary_plot(sv, X_plot, show=False, max_display=9, color_bar_label="Feature value (low to high)",
                  cmap=plt.get_cmap("coolwarm"))
fig = plt.gcf()
ax = plt.gca()
ax.set_xlabel("SHAP value (effect on the log-odds of leaving)")
ax.set_title("What drives the model's turnover predictions", loc="left")
fig.set_size_inches(9, 5.5)
ps.save(fig, OUT / "shap_summary.png")

# dependence: satisfaction, coloured by hours
fig, ax = ps.figure(9, 5)
i = features.index("satisfaction_level")
sc = ax.scatter(X_te["satisfaction_level"], sv[:, i], c=X_te["average_monthly_hours"],
                cmap="coolwarm", s=7, alpha=0.6, linewidths=0)
cb = fig.colorbar(sc, ax=ax, shrink=0.8)
cb.set_label("Average monthly hours")
ax.axhline(0, color=ps.MUTED, lw=0.8)
ax.set_xlabel("Satisfaction level")
ax.set_ylabel("SHAP value for satisfaction")
ax.set_title("The satisfaction effect is not a straight line")
ax.grid(True, axis="both")
ps.save(fig, OUT / "shap_satisfaction.png")

json.dump(results, open(OUT / "results.json", "w"), indent=1)
print(json.dumps(results, indent=1)[:3000])
