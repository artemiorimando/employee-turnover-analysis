# Employee Turnover: Risk Segmenting

Two-way turnover rate matrices on the Kaggle Human Resources Analytics dataset, with a scatter view of the three leaver groups and a LightGBM model explained with SHAP.

This is the 2026 Python version of an analysis I first did in R in 2017. The write-up, with the figures and what changed between the two versions, is at [artemiorimando.com/writing/employee-turnover-risk-segmenting](https://artemiorimando.com/writing/employee-turnover-risk-segmenting/).

## What it does

1. Builds a `rate_matrix(df, y, c1, c2)` helper on top of `pandas.pivot_table`: continuous traits are cut into equal-width ranges, integer and text traits keep one group per value, and each cell is the turnover rate for that combination.
2. Runs the 2017 investigation in order: satisfaction alone, then crossed with department, monthly hours, number of projects, and the same for employees with a workplace accident.
3. Plots every employee as satisfaction against monthly hours to show the three distinct groups of leavers.
4. Fits a LightGBM classifier on all nine traits and uses SHAP to rank them and to show that the satisfaction effect is not monotonic.
5. Writes every number the write-up uses to `output/results.json`.

## Data

`HR_comma_sep.csv` from the Kaggle [Human Resources Analytics](https://www.kaggle.com/datasets/ludobenistant/hr-analytics) dataset: 14,999 employees, 10 columns, `left` is the outcome. Download it to `data/` (not committed). The script expects the cleaned column names (`average_monthly_hours`, `time_spent_company`, `work_accident`, `department`); if your copy has the original typos, rename the columns first.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python turnover.py data/HR_comma_sep.csv output
```

Takes under a minute. Figures land in `output/` as PNG at 200 dpi.

## Output

| File | What it shows |
|---|---|
| `satisfaction_rate.png` | Turnover rate by satisfaction level, 20 ranges, with counts |
| `satisfaction_department.png` | Satisfaction by department heatmap |
| `satisfaction_hours.png` | Satisfaction by average monthly hours heatmap |
| `satisfaction_projects.png` | Satisfaction by number of projects heatmap |
| `satisfaction_projects_accident.png` | Same, employees with a workplace accident only |
| `clusters.png` | Every employee, satisfaction against hours, leavers in red |
| `shap_summary.png` | SHAP beeswarm for the LightGBM model |
| `shap_satisfaction.png` | SHAP dependence plot for satisfaction, coloured by hours |
| `results.json` | Rates, counts, cluster sizes, model AUC and SHAP ranking |

## Stack

pandas, seaborn and matplotlib for the matrices and figures; scikit-learn for the split; LightGBM and SHAP for the model. `plotstyle.py` holds the figure style shared with my other write-ups.

## License

MIT
