"""Experiment 9: Acquiring rules from data with decision trees (Breast Cancer dataset)
LEARNING USE ONLY - NOT FOR CLINICAL USE.
Outputs: comparison table, accuracy_vs_depth.png, readable rule list, critique.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

os.chdir(os.path.dirname(os.path.abspath(__file__)))
SEED = 42
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target            # target: 0 = malignant, 1 = benign
names = list(X.columns)
print("Classes:", dict(zip(data.target_names, [int((y == 0).sum()), int((y == 1).sum())])))

# 1. 70:30 stratified split -------------------------------------------------------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
print(f"Train {len(Xtr)} | Test {len(Xte)}\n")

# Positive class for recall/precision = malignant (0): missing a malignant case is the costly error
def metrics(ytrue, ypred):
    cm = confusion_matrix(ytrue, ypred, labels=[0, 1])
    return {"acc": accuracy_score(ytrue, ypred),
            "prec_mal": precision_score(ytrue, ypred, pos_label=0, zero_division=0),
            "rec_mal": recall_score(ytrue, ypred, pos_label=0, zero_division=0),
            "f1_mal": f1_score(ytrue, ypred, pos_label=0, zero_division=0), "cm": cm}

# 2. Baseline ---------------------------------------------------------------------------
base = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
mb = metrics(yte, base.predict(Xte))
print(f"Baseline (always majority class = benign): test accuracy = {mb['acc']:.3f}")

# 3. Hand-written expert rules ------------------------------------------------------------
print("\nTraining-data look (mean by class) used to set thresholds:")
print(Xtr.groupby(ytr)[["worst radius", "mean concave points", "mean area", "worst concave points"]]
      .mean().round(3).rename(index={0: "malignant", 1: "benign"}))

EXPERT_RULES = [
    ("E1", "worst concave points > 0.15", lambda r: r["worst concave points"] > 0.15),
    ("E2", "worst radius > 16.8",           lambda r: r["worst radius"] > 16.8),
    ("E3", "mean concave points > 0.06",    lambda r: r["mean concave points"] > 0.06),
    ("E4", "mean area > 700",               lambda r: r["mean area"] > 700),
]


def expert_predict(frame):
    # any rule firing -> malignant (0); otherwise benign (1)
    fired = np.zeros(len(frame), dtype=bool)
    for _, _, f in EXPERT_RULES:
        fired |= frame.apply(f, axis=1).to_numpy()
    return np.where(fired, 0, 1)


me = metrics(yte, expert_predict(Xte))
print("\nExpert rules (IF any of these THEN malignant ELSE benign):")
for rid, txt, _ in EXPERT_RULES:
    print(f"  {rid}: IF {txt} THEN malignant")
print(f"Expert rules test accuracy = {me['acc']:.3f}")

# 4. Trees of different depth ---------------------------------------------------------------
rows = []
for d in [2, 3, 4, 6]:
    t = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(Xtr, ytr)
    cv = cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=SEED), Xtr, ytr, cv=5).mean()
    rows.append({"depth": d, "train_acc": t.score(Xtr, ytr), "test_acc": t.score(Xte, yte), "cv_acc": cv})
depth_tab = pd.DataFrame(rows)
print("\n=== Accuracy versus depth ===")
print(depth_tab.round(3).to_string(index=False))
# also an unrestricted tree to show overfitting
full = DecisionTreeClassifier(random_state=SEED).fit(Xtr, ytr)
print(f"Unrestricted tree (depth {full.get_depth()}): train = {full.score(Xtr, ytr):.3f}, "
      f"test = {full.score(Xte, yte):.3f}  <- 100% on training = memorising")

plt.figure()
plt.plot(depth_tab["depth"], depth_tab["train_acc"], "o-", label="train")
plt.plot(depth_tab["depth"], depth_tab["cv_acc"], "s-", label="5-fold CV")
plt.plot(depth_tab["depth"], depth_tab["test_acc"], "^-", label="test")
plt.xlabel("max_depth"); plt.ylabel("accuracy"); plt.title("Accuracy versus tree depth"); plt.legend()
plt.savefig("accuracy_vs_depth.png", dpi=100); plt.close()

# 5. Choose depth with evidence -----------------------------------------------------------------
best_cv = depth_tab["cv_acc"].max()
chosen = int(depth_tab[depth_tab["cv_acc"] >= best_cv - 0.01]["depth"].min())   # simplest within 1% of best CV
print(f"\nChosen depth = {chosen} (smallest depth whose CV accuracy is within 1% of the best {best_cv:.3f})")
tree = DecisionTreeClassifier(max_depth=chosen, random_state=SEED).fit(Xtr, ytr)
print("\n=== export_text ===")
print(export_text(tree, feature_names=names))


def tree_to_rules(clf, feature_names, class_names):
    t = clf.tree_
    out = []

    def walk(node, conds):
        if t.feature[node] == -2:                     # leaf
            cls = class_names[int(np.argmax(t.value[node]))]
            out.append((conds, cls, int(t.n_node_samples[node])))
            return
        f, thr = feature_names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [f"{f} <= {thr:.3f}"])
        walk(t.children_right[node], conds + [f"{f} > {thr:.3f}"])
    walk(0, [])
    return out


rules = tree_to_rules(tree, names, list(data.target_names))
print("=== IF-THEN rules from the tree ===")
for i, (conds, cls, n) in enumerate(rules, 1):
    print(f"T{i}: IF {' AND '.join(conds)} THEN {cls}   ({n} training cases)")
n_cond = sum(len(c) for c, _, _ in rules)
print(f"Rule count: {len(rules)} | total conditions: {n_cond}")

# 6. Metrics for tree and expert rules --------------------------------------------------------------
mt = metrics(yte, tree.predict(Xte))
print("\n=== Test-set comparison (positive class = malignant) ===")
tab = pd.DataFrame([
    ["Baseline (majority)", mb], ["Expert rules (4)", me], [f"Tree depth {chosen}", mt]],
    columns=["system", "m"])
for _, r in depth_tab.iterrows():
    t_ = DecisionTreeClassifier(max_depth=int(r["depth"]), random_state=SEED).fit(Xtr, ytr)
    tab.loc[len(tab)] = [f"Tree depth {int(r['depth'])}", metrics(yte, t_.predict(Xte))]
out = pd.DataFrame({"system": tab["system"],
                    "accuracy": [m["acc"] for m in tab["m"]],
                    "precision": [m["prec_mal"] for m in tab["m"]],
                    "recall": [m["rec_mal"] for m in tab["m"]],
                    "F1": [m["f1_mal"] for m in tab["m"]]}).drop_duplicates("system")
print(out.round(3).to_string(index=False))
print("\nConfusion matrices (rows = true [malignant, benign]; cols = predicted):")
print("Expert rules:\n", me["cm"])
print(f"Tree depth {chosen}:\n", mt["cm"])

# 7. Critique (150-200 words) ---------------------------------------------------------------------------
critique = (
    f"The majority baseline reaches {mb['acc']:.1%} accuracy but never detects a malignant case, so accuracy "
    f"alone is misleading here. The depth-{chosen} tree scores {mt['acc']:.1%} on the test set with malignant "
    f"recall {mt['rec_mal']:.1%}, using only {len(rules)} rules and {n_cond} conditions. The four hand-written "
    f"expert rules reach {me['acc']:.1%} accuracy and malignant recall {me['rec_mal']:.1%}. "
    f"The tree is only marginally better on accuracy and F1, while the expert rules use fewer conditions and, "
    f"because any single rule fires malignant, lean towards catching positives (higher recall, lower precision). "
    f"Missing a malignant case is the costly error, so recall matters more than accuracy. A deeper tree reaches "
    f"100% training accuracy but its test and cross-validation accuracy stay flat or dip slightly, which signals "
    f"overfitting; the depth was therefore chosen by cross-validation. "
    f"A domain expert would probably accept rules built on size and concavity measures, since these match "
    f"clinical intuition, but would question splits on odd features or tiny leaves with few cases. All learned "
    f"rules need validation on new data, and this system must never be used for real diagnosis."
)
print(f"\n=== Critique ({len(critique.split())} words) ===")
print(critique)
print("\nPlot saved: accuracy_vs_depth.png")
