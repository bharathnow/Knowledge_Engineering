"""Generates every CSV / text input file used by the 10 experiments.
Run once:  python make_data.py
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(42)


def folder(name):
    p = os.path.join(BASE, name)
    os.makedirs(p, exist_ok=True)
    return p


# ---------------------------------------------------------------- Experiment 1
n = 60
dept = rng.choice(["CSE", "ECE", "MECH", "AIDS"], n)
gender = rng.choice(["M", "F"], n)
attendance = rng.integers(60, 100, n).astype(float)
internal = rng.integers(20, 50, n).astype(float)
study = np.round(rng.uniform(1, 8, n), 1)
final = np.clip(np.round(30 + 7 * study + 0.4 * internal + rng.normal(0, 6, n)), 0, 100)
df = pd.DataFrame({
    "reg_no": [f"21BT{1000 + i}" for i in range(n)],
    "department": dept, "gender": gender, "attendance": attendance,
    "internal": internal, "final_exam": final, "study_hours": study})
# dirt: blanks, one repeated row, one impossible value
df.loc[[3, 11, 25], "attendance"] = np.nan
df.loc[[7, 30], "final_exam"] = np.nan
df.loc[[14], "study_hours"] = np.nan
df.loc[[19], "department"] = np.nan
df.loc[[22], "gender"] = np.nan
df.loc[40, "attendance"] = 140            # impossible value
df.loc[50, "final_exam"] = 250            # outlier / impossible
df = pd.concat([df, df.iloc[[5]]], ignore_index=True)   # repeated row
df.to_csv(os.path.join(folder("exp01"), "student_marks.csv"), index=False)

# ---------------------------------------------------------------- Experiment 7
f7 = folder("exp07")
people = pd.DataFrame({
    "person_id": range(1, 11),
    "name": ["Asha", "Arun Kumar", "Priya", "A. Kumar", "Ravi", "Meena",
             "Karthik", "Divya", "Suresh", "Lakshmi"],
    "department": ["CSE", "CSE", "ECE", "CSE", "MECH", "ECE", "CSE", "AIDS", "MECH", "AIDS"],
    "city": ["Chennai", "Chennai", "Madurai", "Chennai", "Coimbatore", "Chennai",
             "Trichy", "Chennai", "Coimbatore", "Madurai"]})
people.to_csv(os.path.join(f7, "people.csv"), index=False)
works = pd.DataFrame({
    "person_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 8],
    "project": ["Project Alpha", "Project Beta", "Project Alpha", "Project Beta",
                "Project Gamma", "Project Delta", "Project Alpha", "Project Delta",
                "Project Gamma", "Project Beta", "Project Delta", "Project Alpha"],
    "role": ["Lead", "Developer", "Analyst", "Developer", "Engineer", "Tester",
             "Developer", "Lead", "Engineer", "Analyst", "Reviewer", "Analyst"]})
works.to_csv(os.path.join(f7, "works.csv"), index=False)
sentences = [
    "Asha works on Project Alpha at Anna University",
    "Karthik works on Project Epsilon at Anna University",
    "Ravi works on Project Gamma at SIMATS",
    "Priya works on Project Zeta at IIT Madras",
    "Meena works on Project Delta at SIMATS",
    "Divya works on Project Beta at Anna University",
    "Suresh works on Project Gamma at NIT Trichy",
    "Lakshmi works on Project Alpha at SIMATS",
    "Arun Kumar works on Project Beta at Anna University",
    "Asha works on Project Zeta at IIT Madras",
]
with open(os.path.join(f7, "sentences.txt"), "w") as fh:
    fh.write("\n".join(sentences) + "\n")

# ---------------------------------------------------------------- Experiment 8
m = 90
method = np.repeat(["A", "B", "C"], m // 3)
base = {"A": 55, "B": 62, "C": 70}
before = np.array([rng.normal(52, 8) for _ in range(m)])
after = np.array([base[x] + rng.normal(0, 8) for x in method]) + 0.2 * (before - 52)
after = np.clip(np.round(after, 1), 0, 100)
before = np.clip(np.round(before, 1), 0, 100)
gender8 = rng.choice(["M", "F"], m)
result = np.where(after >= 60, "pass", "fail")
pd.DataFrame({"student": [f"S{i + 1:03d}" for i in range(m)], "method": method,
              "score_before": before, "score_after": after, "gender": gender8,
              "result": result}).to_csv(os.path.join(folder("exp08"), "exam_scores.csv"), index=False)

# ---------------------------------------------------------------- Experiment 10
# Course recommender: student profile -> best elective track
f10 = folder("exp10")
tracks = ["AI", "Data", "Security", "Web"]
rows = []
for i in range(200):
    maths = rng.integers(40, 100)
    prog = rng.integers(40, 100)
    net = rng.integers(40, 100)
    design = rng.integers(40, 100)
    cgpa = round(float(np.clip(rng.normal(7.5, 1.0), 5, 10)), 2)
    interest = rng.choice(tracks)
    score = {"AI": 0.5 * maths + 0.4 * prog, "Data": 0.5 * maths + 0.2 * prog + 0.2 * design,
             "Security": 0.6 * net + 0.2 * prog, "Web": 0.5 * design + 0.4 * prog}
    score[interest] += 15
    label = max(score, key=score.get)
    if rng.random() < 0.08:            # label noise
        label = rng.choice(tracks)
    rows.append([f"ST{i + 1:03d}", maths, prog, net, design, cgpa, interest, label])
pd.DataFrame(rows, columns=["student", "maths", "programming", "networks", "design",
                            "cgpa", "interest", "best_track"]).to_csv(
    os.path.join(f10, "students.csv"), index=False)

print("Data files created under", BASE)
for root, _, files in os.walk(BASE):
    for f in sorted(files):
        if f != "make_data.py":
            print(" ", os.path.relpath(os.path.join(root, f), BASE))
