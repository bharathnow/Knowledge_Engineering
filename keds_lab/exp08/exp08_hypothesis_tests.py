"""Experiment 8: Hypothesis testing, ANOVA and chi-square with evidence
Input : exam_scores.csv
Output: results table (printed + results_table.csv), box plot
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

os.chdir(os.path.dirname(os.path.abspath(__file__)))
ALPHA = 0.05
df = pd.read_csv("exam_scores.csv")
print("Data:", df.shape, "| methods:", df["method"].value_counts().to_dict())
print(df.groupby("method")[["score_before", "score_after"]].mean().round(2), "\n")
results = []


def record(test, h0, h1, stat, p, concl_sig, concl_ns):
    decision = "Reject H0" if p < ALPHA else "Fail to reject H0"
    results.append({"test": test, "H0": h0, "H1": h1, "statistic": round(float(stat), 3),
                    "p_value": float(p), "decision": decision,
                    "conclusion": concl_sig if p < ALPHA else concl_ns})
    print(f"{test}\n  H0: {h0}\n  H1: {h1}\n  alpha = {ALPHA} | statistic = {stat:.3f} | p = {p:.4g}"
          f"\n  Decision: {decision}\n  Conclusion: {results[-1]['conclusion']}\n")


# 2. One-sample t-test ----------------------------------------------------------
t, p = stats.ttest_1samp(df["score_after"], 60)
record("One-sample t (score_after vs 60)", "mean = 60", "mean != 60", t, p,
       f"Mean score_after ({df['score_after'].mean():.2f}) differs significantly from 60.",
       "No evidence that mean score_after differs from 60.")

# 3. Independent t-test A vs B -------------------------------------------------
a = df.loc[df["method"] == "A", "score_after"]
b = df.loc[df["method"] == "B", "score_after"]
t, p = stats.ttest_ind(a, b)
record("Independent t (method A vs B)", "mean_A = mean_B", "mean_A != mean_B", t, p,
       f"Methods A ({a.mean():.2f}) and B ({b.mean():.2f}) give significantly different scores.",
       "No significant difference between methods A and B.")

# 4. Paired t-test --------------------------------------------------------------
t, p = stats.ttest_rel(df["score_before"], df["score_after"])
record("Paired t (before vs after)", "mean difference = 0", "mean difference != 0", t, p,
       f"Scores changed significantly (before {df['score_before'].mean():.2f}, after {df['score_after'].mean():.2f}).",
       "No significant change between before and after.")

# 5. One-way ANOVA ---------------------------------------------------------------
c = df.loc[df["method"] == "C", "score_after"]
F, p = stats.f_oneway(a, b, c)
record("One-way ANOVA (methods A,B,C)", "mean_A = mean_B = mean_C", "at least one mean differs", F, p,
       "At least one teaching method has a different mean score.",
       "No evidence that the teaching methods differ.")
if p < ALPHA:
    means = df.groupby("method")["score_after"].mean().round(2)
    print("  Group means:", means.to_dict())
    for x, y, nm in [(a, b, "A vs B"), (a, c, "A vs C"), (b, c, "B vs C")]:
        tt, pp = stats.ttest_ind(x, y)
        print(f"  Post-hoc {nm}: t = {tt:.3f}, p = {pp:.4g} (Bonferroni threshold {ALPHA / 3:.4f}) -> "
              f"{'differ' if pp < ALPHA / 3 else 'no clear difference'}")
    print()

# 6. Chi-square ---------------------------------------------------------------------
ct = pd.crosstab(df["gender"], df["result"])
chi2, p, dof, exp = stats.chi2_contingency(ct)
print("Contingency table gender x result:\n", ct, "\n")
record("Chi-square (gender vs result)", "gender and result are independent",
       "gender and result are associated", chi2, p,
       "Result is associated with gender.", "No evidence of association between gender and result.")

# 7. Results table and box plot ---------------------------------------------------------
tab = pd.DataFrame(results)
tab["p_value"] = tab["p_value"].map(lambda v: f"{v:.4g}")
tab[["test", "statistic", "p_value", "decision", "conclusion"]].to_csv("results_table.csv", index=False)
print("=== Results table ===")
print(tab[["test", "statistic", "p_value", "decision"]].to_string(index=False))

df.boxplot(column="score_after", by="method")
plt.title("score_after by teaching method"); plt.suptitle(""); plt.ylabel("score_after")
plt.savefig("boxplot_by_method.png", dpi=100); plt.close()

# 8. 95% CI --------------------------------------------------------------------------------
x = df["score_after"]
ci = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
print(f"\n95% confidence interval for mean score_after: ({ci[0]:.2f}, {ci[1]:.2f}) | mean = {x.mean():.2f}")
print("Box plot saved: boxplot_by_method.png | table saved: results_table.csv")
