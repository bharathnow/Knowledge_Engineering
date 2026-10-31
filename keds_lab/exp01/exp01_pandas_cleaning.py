"""Experiment 1: Data Exploration and Cleaning with pandas
Input : student_marks.csv
Output: student_marks_clean.csv + 3 charts (png)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)

# 1. Load and inspect -------------------------------------------------------
df = pd.read_csv("student_marks.csv")
print("=== head() ===")
print(df.head())
print("\nshape:", df.shape)
print("\n=== info() ===")
df.info()
print("\n=== describe() ===")
print(df.describe().round(2))

# 2. Missing values ---------------------------------------------------------
print("\n=== Missing values per column ===")
print(df.isna().sum())
mean_before = df["final_exam"].mean()

# 3. Fill: median for numbers, mode for categories --------------------------
# Median is used because it is not pulled by outliers (e.g. 140, 250).
for col in ["attendance", "internal", "final_exam", "study_hours"]:
    df[col] = df[col].fillna(df[col].median())
for col in ["department", "gender"]:
    df[col] = df[col].fillna(df[col].mode()[0])

# 4. Duplicates and impossible values ---------------------------------------
dups = df.duplicated().sum()
df = df.drop_duplicates()
print(f"\nDuplicate rows removed: {dups}")
bad_att = (df["attendance"] > 100) | (df["attendance"] < 0)
bad_fin = (df["final_exam"] > 100) | (df["final_exam"] < 0)
print(f"Impossible attendance values: {bad_att.sum()} | impossible final_exam: {bad_fin.sum()}")
# correct impossible values by replacing with the column median of valid values
for col, bad in [("attendance", bad_att), ("final_exam", bad_fin)]:
    df.loc[bad, col] = df.loc[~bad, col].median()
df.to_csv("student_marks_clean.csv", index=False)
print("Saved student_marks_clean.csv, shape:", df.shape)

# 5. Summary statistics -----------------------------------------------------
fe = df["final_exam"]
summary = pd.DataFrame({"mean": [fe.mean()], "median": [fe.median()],
                        "std": [fe.std()], "min": [fe.min()], "max": [fe.max()]}).round(2)
print("\n=== Summary of final_exam ===")
print(summary.to_string(index=False))
print("\n=== Mean final_exam by department ===")
print(df.groupby("department")["final_exam"].mean().round(2))
print(f"\nMean final_exam before cleaning: {mean_before:.2f} | after: {fe.mean():.2f}")

# 6. Charts -----------------------------------------------------------------
plt.figure(); plt.hist(fe, bins=10, edgecolor="black")
plt.title("Histogram of final_exam"); plt.xlabel("final_exam"); plt.ylabel("students")
plt.savefig("hist_final_exam.png", dpi=100); plt.close()

df.boxplot(column="final_exam", by="department")
plt.title("final_exam by department"); plt.suptitle(""); plt.ylabel("final_exam")
plt.savefig("box_by_department.png", dpi=100); plt.close()

plt.figure(); plt.scatter(df["study_hours"], fe)
plt.title("study_hours vs final_exam"); plt.xlabel("study_hours"); plt.ylabel("final_exam")
plt.savefig("scatter_study_final.png", dpi=100); plt.close()

# 7. Correlation ------------------------------------------------------------
r = df["study_hours"].corr(df["final_exam"])
print(f"\nCorrelation(study_hours, final_exam) = {r:.3f}")
print("Charts saved: hist_final_exam.png, box_by_department.png, scatter_study_final.png")
