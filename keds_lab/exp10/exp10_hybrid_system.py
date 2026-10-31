"""Experiment 10: Hybrid decision-support system - elective-track recommender
Knowledge graph (rdflib) + expert rules with certainty factors + ML model (random forest).
Input : students.csv  (200 synthetic students)
Output: system_diagram.png, results_table.csv, printed explanations and failure analysis.
EDUCATIONAL USE ONLY - not for real academic counselling decisions.

COMBINATION LOGIC
  1. Graph  -> supplies facts. Each student's marks are turned into symbolic facts
              (strongIn / weakIn a subject, interestedIn a track) stored as triples, and the
              graph also holds domain knowledge (track requires subject, course belongs to track).
  2. Rules  -> read those facts and give each track a certainty factor (CF in [-1, 1]); several
              rules for the same track are merged with the MYCIN formula; rules with negative CF
              are evidence AGAINST a track.
  3. Model  -> random forest gives a probability for each track (second opinion).
  4. Hybrid -> score(track) = 0.5 * max(CF, 0) + 0.5 * model_probability.
              If rules and model DISAGREE the higher blended score wins, with one safeguard:
              a track with strong evidence against it (CF <= -0.5) is vetoed unless the model
              probability for it is >= 0.7. Disagreements are flagged in the explanation.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, RDFS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score

os.chdir(os.path.dirname(os.path.abspath(__file__)))
TRACKS = ["AI", "Data", "Security", "Web"]
SUBJECTS = ["maths", "programming", "networks", "design"]
EX = Namespace("http://example.org/rec/")

df = pd.read_csv("students.csv")
train, test = df.iloc[:160].copy(), df.iloc[160:].copy()          # 160 train, 40 test
print(f"Dataset: {len(df)} students | train {len(train)} | test {len(test)} cases")

# ------------------------------------------------------------------ 1. Knowledge graph
kg = Graph(); kg.bind("ex", EX)
schema_classes = ["Student", "Track", "Subject", "Course"]
for c in schema_classes:
    kg.add((EX[c], RDF.type, RDFS.Class))
requires = {"AI": ["maths", "programming"], "Data": ["maths", "design"],
            "Security": ["networks", "programming"], "Web": ["design", "programming"]}
courses = {"AI": ["MachineLearning", "NeuralNetworks"], "Data": ["DataMining", "BigData"],
           "Security": ["CryptoBasics", "EthicalHacking"], "Web": ["WebEngineering", "UXDesign"]}
for t in TRACKS:
    kg.add((EX[t], RDF.type, EX.Track))
    kg.add((EX[t], RDFS.label, Literal(t + " track")))
    for s in requires[t]:
        kg.add((EX[t], EX.requires, EX[s]))
    for c in courses[t]:
        kg.add((EX[c], RDF.type, EX.Course))
        kg.add((EX[c], RDFS.label, Literal(c)))
        kg.add((EX[c], EX.belongsTo, EX[t]))
for s in SUBJECTS:
    kg.add((EX[s], RDF.type, EX.Subject))
    kg.add((EX[s], RDFS.label, Literal(s)))
domain_triples = len(kg)
print(f"Knowledge graph: {domain_triples} domain triples (>= 40 required)")


def student_facts(row):
    """Write the student's symbolic facts into the graph and read them back."""
    s = EX[row["student"]]
    kg.add((s, RDF.type, EX.Student))
    for sub in SUBJECTS:
        if row[sub] >= 70:
            kg.add((s, EX.strongIn, EX[sub]))
        elif row[sub] < 55:
            kg.add((s, EX.weakIn, EX[sub]))
    kg.add((s, EX.interestedIn, EX[row["interest"]]))
    if row["cgpa"] >= 8.5:
        kg.add((s, EX.hasStanding, EX.high_cgpa))
    facts = set()
    for p, o in kg.predicate_objects(s):
        if p != RDF.type:
            facts.add((p.split("/")[-1], o.split("/")[-1]))
    return facts


# ------------------------------------------------------------------ 2. Rule base with CFs
# (id, relation, object, track, CF)   CF < 0 means evidence AGAINST the track
RULES = [
    ("R1", "strongIn", "maths", "AI", 0.5),
    ("R2", "strongIn", "programming", "AI", 0.4),
    ("R3", "strongIn", "maths", "Data", 0.5),
    ("R4", "strongIn", "design", "Data", 0.2),
    ("R5", "strongIn", "networks", "Security", 0.6),
    ("R6", "strongIn", "programming", "Security", 0.2),
    ("R7", "strongIn", "design", "Web", 0.5),
    ("R8", "strongIn", "programming", "Web", 0.4),
    ("R9", "interestedIn", "AI", "AI", 0.5),
    ("R10", "interestedIn", "Data", "Data", 0.5),
    ("R11", "interestedIn", "Security", "Security", 0.5),
    ("R12", "interestedIn", "Web", "Web", 0.5),
    ("R13", "weakIn", "networks", "Security", -0.5),       # evidence against
    ("R14", "weakIn", "maths", "AI", -0.5),                # evidence against
    ("R15", "weakIn", "design", "Web", -0.4),              # evidence against
    ("R16", "weakIn", "maths", "Data", -0.4),              # evidence against
]


def combine(cf1, cf2):
    """MYCIN certainty-factor combination."""
    if cf1 >= 0 and cf2 >= 0:
        return cf1 + cf2 * (1 - cf1)
    if cf1 < 0 and cf2 < 0:
        return cf1 + cf2 * (1 + cf1)
    return (cf1 + cf2) / (1 - min(abs(cf1), abs(cf2)))


def run_rules(facts):
    cf = {t: 0.0 for t in TRACKS}
    fired = []
    for rid, rel, obj, track, val in RULES:
        if (rel, obj) in facts:
            cf[track] = combine(cf[track], val)
            fired.append((rid, rel, obj, track, val))
    return cf, fired


# ------------------------------------------------------------------ 3. ML model
FEATS = SUBJECTS + ["cgpa"]


def to_X(frame):
    X = frame[FEATS].copy()
    for t in TRACKS:
        X[f"interest_{t}"] = (frame["interest"] == t).astype(int)
    return X


model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
Xtr, ytr = to_X(train), train["best_track"]
cv = cross_val_score(RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
                     Xtr, ytr, cv=5, scoring="accuracy")
print(f"Model: random forest | 5-fold CV accuracy on training data = {cv.mean():.3f} (+/- {cv.std():.3f})")
model.fit(Xtr, ytr)
classes = list(model.classes_)

# ------------------------------------------------------------------ 4. Four systems
majority = train["best_track"].mode()[0]


def hybrid(cf, proba):
    scores = {}
    for t in TRACKS:
        p = proba[classes.index(t)] if t in classes else 0.0
        s = 0.5 * max(cf[t], 0) + 0.5 * p
        if cf[t] <= -0.5 and p < 0.7:
            s = -1                                          # veto: strong evidence against
        scores[t] = s
    return max(scores, key=scores.get), scores


records = []
for _, row in test.iterrows():
    facts = student_facts(row)
    cf, fired = run_rules(facts)
    proba = model.predict_proba(to_X(pd.DataFrame([row])))[0]
    rule_pick = max(cf, key=cf.get) if max(cf.values()) > 0 else majority
    model_pick = classes[int(np.argmax(proba))]
    h_pick, h_scores = hybrid(cf, proba)
    records.append({"student": row["student"], "truth": row["best_track"], "baseline": majority,
                    "rules": rule_pick, "model": model_pick, "hybrid": h_pick,
                    "facts": sorted(facts), "fired": fired, "cf": cf, "proba": proba,
                    "model_conf": float(proba.max()), "agree": rule_pick == model_pick})
res = pd.DataFrame(records)

rows = []
for sysname in ["baseline", "rules", "model", "hybrid"]:
    rows.append({"system": sysname,
                 "accuracy": accuracy_score(res["truth"], res[sysname]),
                 "macro_F1": f1_score(res["truth"], res[sysname], average="macro", zero_division=0)})
table = pd.DataFrame(rows)
print(f"\n=== Evaluation on {len(res)} held-out test cases ===")
print(table.round(3).to_string(index=False))
table.round(3).to_csv("results_table.csv", index=False)
print(f"Rules and model agreed on {res['agree'].sum()} of {len(res)} cases.")
d = res[~res["agree"]]
if len(d):
    print(f"When they disagreed ({len(d)} cases): rules correct {(d['rules'] == d['truth']).sum()}, "
          f"model correct {(d['model'] == d['truth']).sum()}, hybrid correct {(d['hybrid'] == d['truth']).sum()}")

# ------------------------------------------------------------------ 5. Explanations
def explain(r):
    print(f"Student {r['student']}  ->  RECOMMENDED TRACK: {r['hybrid']}   (true best track: {r['truth']})")
    print("  Facts from graph :", ", ".join(f"{a} {b}" for a, b in r["facts"]))
    for rid, rel, obj, track, val in r["fired"]:
        print(f"  Rule {rid}: {rel} {obj} -> {track} (CF {val:+.2f})")
    print("  Combined CF      :", {k: round(v, 2) for k, v in r["cf"].items()})
    print(f"  Model confidence : {r['model']} at {r['model_conf']:.2f}  | probabilities "
          f"{ {c: round(float(p), 2) for c, p in zip(classes, r['proba'])} }")
    print("  Rules vs model   :", "AGREE" if r["agree"] else f"DISAGREE (rules {r['rules']}, model {r['model']})")
    sugg = [str(s).split("/")[-1] for s, _, o in kg.triples((None, EX.belongsTo, EX[r["hybrid"]]))]
    print("  Suggested courses:", ", ".join(sorted(sugg)))
    print()


print("\n=== 5 sample explanations ===")
samples = list(res[res["hybrid"] == res["truth"]].head(3).index) + list(res[~res["agree"]].head(2).index)
for i in dict.fromkeys(samples):
    explain(res.loc[i])

# ------------------------------------------------------------------ 6. Failure analysis
print("=== Failure case analysis ===")
fails = res[res["hybrid"] != res["truth"]]
print(f"Hybrid wrong on {len(fails)} of {len(res)} cases.")
if len(fails):
    f = fails.iloc[0]
    explain(f)
    print(f"  Why it failed: the true track '{f['truth']}' received rule CF {f['cf'][f['truth']]:.2f} and model "
          f"probability {f['proba'][classes.index(f['truth'])]:.2f}, while the chosen '{f['hybrid']}' scored "
          f"CF {f['cf'][f['hybrid']]:.2f} / probability {f['proba'][classes.index(f['hybrid'])]:.2f}.")
    print("  The labels were generated with 8% random noise and an interest bonus, so part of such errors is\n"
          "  label noise; the rules also use hard 70 / 55 mark thresholds, so students near a threshold are misjudged.")

# ------------------------------------------------------------------ system diagram
fig, ax = plt.subplots(figsize=(10, 4.5)); ax.axis("off")
boxes = {"Student marks\n(students.csv)": (0.02, 0.4), "Knowledge graph\n(rdflib triples)": (0.27, 0.4),
         "Rule base\n(16 rules + CF)": (0.52, 0.65), "ML model\n(random forest)": (0.52, 0.15),
         "Hybrid combiner\n0.5*CF + 0.5*prob\n+ veto": (0.76, 0.4)}
for t, (x, y) in boxes.items():
    ax.text(x, y, t, ha="left", va="center", fontsize=9, bbox=dict(boxstyle="round", fc="#dbe9ff", ec="k"))
for a, b in [((0.14, 0.43), (0.26, 0.43)), ((0.40, 0.46), (0.51, 0.65)), ((0.14, 0.40), (0.51, 0.18)),
             ((0.65, 0.65), (0.75, 0.48)), ((0.65, 0.18), (0.75, 0.40))]:
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->"))
ax.text(0.76, 0.12, "-> track + explanation", fontsize=9)
plt.savefig("system_diagram.png", dpi=110, bbox_inches="tight"); plt.close()
print("\nSystem diagram saved: system_diagram.png | results saved: results_table.csv")

# ------------------------------------------------------------------ 7. Limitations / ethics
print("\n=== Limitations and ethics ===")
print("- Data are SYNTHETIC; results show the method works, not that it suits real students.")
print("- Must NOT be used to deny a student a course, rank students, or make any admission decision.")
print("- Before real use: real, representative data; expert (faculty) review of every rule and CF;\n"
      "  fairness checks across groups; a way for students to see the explanation and override the advice.")
