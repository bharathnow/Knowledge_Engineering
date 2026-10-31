# Knowledge Engineering and Data Science - Programs, Inputs and Outputs

Every program below was executed (Python 3.13). The OUTPUT sections are the real console output.

To rerun everything: `pip install pandas numpy matplotlib scipy scikit-learn networkx rdflib owlready2`, then `python make_data.py` and `bash run_all.sh`.


---

## Experiment 1: Data Exploration and Cleaning with pandas

### Program: `exp01/exp01_pandas_cleaning.py`

```python
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

```

### Input file: `exp01/student_marks.csv` (62 lines)

```
reg_no,department,gender,attendance,internal,final_exam,study_hours
21BT1000,CSE,F,77.0,48.0,86.0,5.1
21BT1001,AIDS,F,86.0,24.0,77.0,5.5
21BT1002,MECH,M,86.0,34.0,57.0,1.6
21BT1003,ECE,F,,40.0,81.0,3.9
21BT1004,ECE,M,94.0,34.0,54.0,1.3
21BT1005,AIDS,M,82.0,33.0,72.0,4.5
21BT1006,CSE,F,63.0,24.0,69.0,3.3
21BT1007,MECH,M,90.0,31.0,,2.0
21BT1008,CSE,M,82.0,27.0,62.0,1.7
21BT1009,CSE,M,85.0,29.0,78.0,5.1
21BT1010,MECH,F,82.0,40.0,54.0,2.2
21BT1011,AIDS,M,,38.0,89.0,7.5
21BT1012,MECH,M,63.0,38.0,91.0,5.1
21BT1013,AIDS,M,82.0,30.0,76.0,3.4
... (47 more rows, full file in the zip)
```

### Output

```
=== head() ===
     reg_no department gender  attendance  internal  final_exam  study_hours
0  21BT1000        CSE      F        77.0      48.0        86.0          5.1
1  21BT1001       AIDS      F        86.0      24.0        77.0          5.5
2  21BT1002       MECH      M        86.0      34.0        57.0          1.6
3  21BT1003        ECE      F         NaN      40.0        81.0          3.9
4  21BT1004        ECE      M        94.0      34.0        54.0          1.3

shape: (61, 7)

=== info() ===
<class 'pandas.DataFrame'>
RangeIndex: 61 entries, 0 to 60
Data columns (total 7 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   reg_no       61 non-null     str    
 1   department   60 non-null     str    
 2   gender       60 non-null     str    
 3   attendance   58 non-null     float64
 4   internal     61 non-null     float64
 5   final_exam   59 non-null     float64
 6   study_hours  60 non-null     float64
dtypes: float64(4), str(3)
memory usage: 3.5 KB

=== describe() ===
       attendance  internal  final_exam  study_hours
count       58.00     61.00       59.00        60.00
mean        79.98     34.46       77.36         4.37
std         13.48      7.86       27.14         2.07
min         60.00     20.00       45.00         1.20
25%         71.00     29.00       62.00         2.75
50%         80.00     33.00       76.00         4.40
75%         87.75     41.00       87.00         6.12
max        140.00     49.00      250.00         7.90

=== Missing values per column ===
reg_no         0
department     1
gender         1
attendance     3
internal       0
final_exam     2
study_hours    1
dtype: int64

Duplicate rows removed: 1
Impossible attendance values: 1 | impossible final_exam: 1
Saved student_marks_clean.csv, shape: (60, 7)

=== Summary of final_exam ===
 mean  median  std  min   max
 74.5    76.0 14.5 45.0 100.0

=== Mean final_exam by department ===
department
AIDS    75.17
CSE     75.29
ECE     70.58
MECH    76.00
Name: final_exam, dtype: float64

Mean final_exam before cleaning: 77.36 | after: 74.50

Correlation(study_hours, final_exam) = 0.894
Charts saved: hist_final_exam.png, box_by_department.png, scatter_study_final.png

```

Files produced: `box_by_department.png`, `hist_final_exam.png`, `scatter_study_final.png`, `student_marks_clean.csv`


---

## Experiment 2: Semantic Network and Inheritance Queries

### Program: `exp02/exp02_semantic_network.py`

```python
"""Experiment 2: Semantic Network and Inheritance Queries (networkx)"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

os.chdir(os.path.dirname(os.path.abspath(__file__)))

G = nx.DiGraph()
# class hierarchy / instance links
G.add_edge("Bird", "Animal", relation="is-a")
G.add_edge("Fish", "Animal", relation="is-a")
G.add_edge("Canary", "Bird", relation="is-a")
G.add_edge("Penguin", "Bird", relation="is-a")
G.add_edge("Tweety", "Canary", relation="instance-of")
G.add_edge("Pingu", "Penguin", relation="instance-of")
# properties (value nodes)
G.add_edge("Animal", "skin", relation="has-part")
G.add_edge("Bird", "wings", relation="has-part")
G.add_edge("Bird", "can fly", relation="can")
G.add_edge("Fish", "swim", relation="can")
G.add_edge("Canary", "yellow", relation="colour")
G.add_edge("Penguin", "cannot fly", relation="can")      # exception (overrides Bird)
G.add_edge("Penguin", "swim", relation="can")

UP = {"is-a", "instance-of"}


def inherits(node, relation):
    """Walk up is-a / instance-of links; return (values, path) of the most specific node
    that has the relation."""
    path = [node]
    current = node
    while True:
        vals = [v for _, v, d in G.out_edges(current, data=True) if d["relation"] == relation]
        if vals:
            return vals, path
        parents = [v for _, v, d in G.out_edges(current, data=True) if d["relation"] in UP]
        if not parents:
            return None, path
        current = parents[0]
        path.append(current)


def ask(node, relation, text):
    vals, path = inherits(node, relation)
    print(f"Q: {text}")
    print(f"   links followed: {' -> '.join(path)}")
    print(f"   answer: {vals}  (found at '{path[-1]}')\n")


print("=== Inheritance queries ===")
ask("Tweety", "can", "Can Tweety fly?")
ask("Tweety", "has-part", "Does Tweety have wings (what parts)?")
ask("Penguin", "can", "Can a penguin fly? (exception overrides Bird)")
ask("Pingu", "can", "What can Pingu do?")
ask("Tweety", "colour", "What colour is Tweety?")
ask("Fish", "has-part", "What parts does a Fish have (inherited from Animal)?")


def concept_path(a, b):
    try:
        return nx.shortest_path(G, a, b)
    except nx.NetworkXNoPath:
        return None


print("=== Path search ===")
print("Tweety -> Animal :", concept_path("Tweety", "Animal"))
print("Pingu  -> Animal :", concept_path("Pingu", "Animal"))
print("Fish   -> Tweety :", concept_path("Fish", "Tweety"))

print("\nException handling note: the search starts at the most specific node and stops at\n"
      "the first node that has the relation, so 'Penguin: cannot fly' overrides 'Bird: can fly'.")

pos = nx.spring_layout(G, seed=3)
plt.figure(figsize=(10, 7))
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=2200, font_size=8, arrows=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "relation"), font_size=7)
plt.savefig("semantic_network.png", dpi=110)
print("Network drawing saved: semantic_network.png")

```

### Output

```
=== Inheritance queries ===
Q: Can Tweety fly?
   links followed: Tweety -> Canary -> Bird
   answer: ['can fly']  (found at 'Bird')

Q: Does Tweety have wings (what parts)?
   links followed: Tweety -> Canary -> Bird
   answer: ['wings']  (found at 'Bird')

Q: Can a penguin fly? (exception overrides Bird)
   links followed: Penguin
   answer: ['cannot fly', 'swim']  (found at 'Penguin')

Q: What can Pingu do?
   links followed: Pingu -> Penguin
   answer: ['cannot fly', 'swim']  (found at 'Penguin')

Q: What colour is Tweety?
   links followed: Tweety -> Canary
   answer: ['yellow']  (found at 'Canary')

Q: What parts does a Fish have (inherited from Animal)?
   links followed: Fish -> Animal
   answer: ['skin']  (found at 'Animal')

=== Path search ===
Tweety -> Animal : ['Tweety', 'Canary', 'Bird', 'Animal']
Pingu  -> Animal : ['Pingu', 'Penguin', 'Bird', 'Animal']
Fish   -> Tweety : None

Exception handling note: the search starts at the most specific node and stops at
the first node that has the relation, so 'Penguin: cannot fly' overrides 'Bird: can fly'.
Network drawing saved: semantic_network.png

```

Files produced: `semantic_network.png`


---

## Experiment 3: Frames with Defaults, Inheritance and If-Needed Procedures

### Program: `exp03/exp03_frames.py`

```python
"""Experiment 3: Frames with defaults, inheritance and if-needed procedures"""


class Frame:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.slots = {}          # slot -> {"value","default","if_needed","if_added"}

    def define(self, slot, value=None, default=None, if_needed=None, if_added=None):
        self.slots[slot] = {"value": value, "default": default,
                            "if_needed": if_needed, "if_added": if_added}

    def get(self, slot, origin=None):
        """Order: own value -> if-needed procedure -> default -> parent frame.
        Returns (value, source). `origin` is the frame the query started from, so an
        inherited if-needed procedure computes from the *asking* frame's slots."""
        origin = origin or self
        s = self.slots.get(slot)
        if s:
            if s["value"] is not None:
                return s["value"], f"frame value ({self.name})"
            if s["if_needed"] is not None:
                return s["if_needed"](origin), f"if-needed procedure ({self.name})"
            if s["default"] is not None:
                return s["default"], f"default ({self.name})"
        if self.parent:
            v, src = self.parent.get(slot, origin)
            return v, (src if src.startswith(("none", "inherited")) else f"inherited: {src}")
        return None, "none"

    def set(self, slot, value):
        s = self.slots.setdefault(slot, {"value": None, "default": None,
                                         "if_needed": None, "if_added": None})
        demon = s["if_added"] or self._inherited_demon(slot)
        if demon:
            demon(self, value)
        s["value"] = value

    def _inherited_demon(self, slot):
        f = self.parent
        while f:
            if slot in f.slots and f.slots[slot]["if_added"]:
                return f.slots[slot]["if_added"]
            f = f.parent
        return None


# --- procedures --------------------------------------------------------------
def compute_mileage(frame):
    odo, _ = frame.get("odometer")
    fuel, _ = frame.get("fuel_used")
    if odo is None or not fuel:
        return None
    return round(odo / fuel, 2)


def warn_negative(frame, value):
    if value < 0:
        print(f"   [DEMON] WARNING: {frame.name}: negative value {value} set!")


# --- hierarchy ---------------------------------------------------------------
vehicle = Frame("Vehicle")
vehicle.define("wheels", default=4)
vehicle.define("fuel_type", default="petrol")
vehicle.define("odometer")
vehicle.define("fuel_used")
vehicle.define("mileage", if_needed=compute_mileage, if_added=warn_negative)

car = Frame("Car", parent=vehicle)
car.define("doors", default=4)
car.define("category", value="passenger")

my_car = Frame("MyCar", parent=car)
my_car.define("colour", value="red")
my_car.define("odometer", value=12000)
my_car.define("fuel_used", value=800)
my_car.define("doors", value=2)

print("=== Slot values of MyCar and their source ===")
print(f"{'Slot':<12}{'Value':<10}Source")
print("-" * 55)
for slot in ["colour", "doors", "category", "wheels", "fuel_type", "odometer", "fuel_used", "mileage"]:
    v, src = my_car.get(slot)
    print(f"{slot:<12}{str(v):<10}{src}")

print("\n=== Demon demonstration ===")
print("Setting mileage = -5 on MyCar:")
my_car.set("mileage", -5)
print("Setting mileage = 15 on MyCar (no warning):")
my_car.set("mileage", 15)
print("mileage now ->", my_car.get("mileage"))

```

### Output

```
=== Slot values of MyCar and their source ===
Slot        Value     Source
-------------------------------------------------------
colour      red       frame value (MyCar)
doors       2         frame value (MyCar)
category    passenger inherited: frame value (Car)
wheels      4         inherited: default (Vehicle)
fuel_type   petrol    inherited: default (Vehicle)
odometer    12000     frame value (MyCar)
fuel_used   800       frame value (MyCar)
mileage     15.0      inherited: if-needed procedure (Vehicle)

=== Demon demonstration ===
Setting mileage = -5 on MyCar:
   [DEMON] WARNING: MyCar: negative value -5 set!
Setting mileage = 15 on MyCar (no warning):
mileage now -> (15, 'frame value (MyCar)')

```


---

## Experiment 4: Rule-Based System with Forward Chaining

### Program: `exp04/exp04_forward_chaining.py`

```python
"""Experiment 4: Rule-based system with forward chaining (loan eligibility, 10 rules)"""

RULES = [
    ("R1", {"salary_high"}, "income_ok"),
    ("R2", {"salary_medium", "has_coapplicant"}, "income_ok"),
    ("R3", {"no_defaults", "repaid_before"}, "credit_good"),
    ("R4", {"credit_score_high"}, "credit_good"),
    ("R5", {"permanent_job"}, "stable_job"),
    ("R6", {"income_ok", "credit_good", "stable_job"}, "approve"),
    ("R7", {"income_ok", "credit_good"}, "conditional_approve"),
    ("R8", {"has_defaults"}, "reject"),
    ("R9", {"income_ok", "credit_good", "stable_job", "has_collateral"}, "approve_low_interest"),
    ("R10", {"salary_low"}, "reject"),
]


def forward_chain(facts, strategy="first", verbose=True):
    facts = set(facts)
    order = []
    cycle = 0
    if verbose:
        print(f"{'Cycle':<6}{'Conflict set':<34}{'Fired':<8}Facts after")
        print("-" * 100)
    while True:
        conflict = [r for r in RULES if r[1] <= facts and r[2] not in facts]
        if not conflict:
            break
        cycle += 1
        if strategy == "specific":
            chosen = max(conflict, key=lambda r: len(r[1]))      # most conditions first
        else:
            chosen = conflict[0]                                 # first match
        facts.add(chosen[2])
        order.append(chosen[0])
        if verbose:
            print(f"{cycle:<6}{str([r[0] for r in conflict]):<34}{chosen[0]:<8}{sorted(facts)}")
    if verbose:
        print(f"Stopped: no rule can add a new fact.  Firing order: {order}")
    return facts, order


TESTS = {
    "Test 1 (strong applicant)": {"salary_high", "no_defaults", "repaid_before", "permanent_job"},
    "Test 2 (medium salary, co-applicant, collateral)": {"salary_medium", "has_coapplicant",
                                                         "credit_score_high", "permanent_job",
                                                         "has_collateral"},
    "Test 3 (no conclusion)": {"has_collateral"},
}

for name, start in TESTS.items():
    print("\n" + "=" * 100)
    print(name, "| starting facts:", sorted(start))
    print("--- Strategy: first match ---")
    f1, o1 = forward_chain(start, "first")
    print("--- Strategy: most specific rule first ---")
    f2, o2 = forward_chain(start, "specific")
    concl = sorted(x for x in f1 if x in {"approve", "approve_low_interest", "conditional_approve", "reject"})
    print("Final conclusions:", concl if concl else "none")
    print("Order same in both strategies?", o1 == o2)

```

### Output

```

====================================================================================================
Test 1 (strong applicant) | starting facts: ['no_defaults', 'permanent_job', 'repaid_before', 'salary_high']
--- Strategy: first match ---
Cycle Conflict set                      Fired   Facts after
----------------------------------------------------------------------------------------------------
1     ['R1', 'R3', 'R5']                R1      ['income_ok', 'no_defaults', 'permanent_job', 'repaid_before', 'salary_high']
2     ['R3', 'R5']                      R3      ['credit_good', 'income_ok', 'no_defaults', 'permanent_job', 'repaid_before', 'salary_high']
3     ['R5', 'R7']                      R5      ['credit_good', 'income_ok', 'no_defaults', 'permanent_job', 'repaid_before', 'salary_high', 'stable_job']
4     ['R6', 'R7']                      R6      ['approve', 'credit_good', 'income_ok', 'no_defaults', 'permanent_job', 'repaid_before', 'salary_high', 'stable_job']
5     ['R7']                            R7      ['approve', 'conditional_approve', 'credit_good', 'income_ok', 'no_defaults', 'permanent_job', 'repaid_before', 'salary_high', 'stable_job']
Stopped: no rule can add a new fact.  Firing order: ['R1', 'R3', 'R5', 'R6', 'R7']
--- Strategy: most specific rule first ---
Cycle Conflict set                      Fired   Facts after
----------------------------------------------------------------------------------------------------
1     ['R1', 'R3', 'R5']                R3      ['credit_good', 'no_defaults', 'permanent_job', 'repaid_before', 'salary_high']
2     ['R1', 'R5']                      R1      ['credit_good', 'income_ok', 'no_defaults', 'permanent_job', 'repaid_before', 'salary_high']
3     ['R5', 'R7']                      R7      ['conditional_approve', 'credit_good', 'income_ok', 'no_defaults', 'permanent_job', 'repaid_before', 'salary_high']
4     ['R5']                            R5      ['conditional_approve', 'credit_good', 'income_ok', 'no_defaults', 'permanent_job', 'repaid_before', 'salary_high', 'stable_job']
5     ['R6']                            R6      ['approve', 'conditional_approve', 'credit_good', 'income_ok', 'no_defaults', 'permanent_job', 'repaid_before', 'salary_high', 'stable_job']
Stopped: no rule can add a new fact.  Firing order: ['R3', 'R1', 'R7', 'R5', 'R6']
Final conclusions: ['approve', 'conditional_approve']
Order same in both strategies? False

====================================================================================================
Test 2 (medium salary, co-applicant, collateral) | starting facts: ['credit_score_high', 'has_coapplicant', 'has_collateral', 'permanent_job', 'salary_medium']
--- Strategy: first match ---
Cycle Conflict set                      Fired   Facts after
----------------------------------------------------------------------------------------------------
1     ['R2', 'R4', 'R5']                R2      ['credit_score_high', 'has_coapplicant', 'has_collateral', 'income_ok', 'permanent_job', 'salary_medium']
2     ['R4', 'R5']                      R4      ['credit_good', 'credit_score_high', 'has_coapplicant', 'has_collateral', 'income_ok', 'permanent_job', 'salary_medium']
3     ['R5', 'R7']                      R5      ['credit_good', 'credit_score_high', 'has_coapplicant', 'has_collateral', 'income_ok', 'permanent_job', 'salary_medium', 'stable_job']
4     ['R6', 'R7', 'R9']                R6      ['approve', 'credit_good', 'credit_score_high', 'has_coapplicant', 'has_collateral', 'income_ok', 'permanent_job', 'salary_medium', 'stable_job']
5     ['R7', 'R9']                      R7      ['approve', 'conditional_approve', 'credit_good', 'credit_score_high', 'has_coapplicant', 'has_collateral', 'income_ok', 'permanent_job', 'salary_medium', 'stable_job']
6     ['R9']                            R9      ['approve', 'approve_low_interest', 'conditional_approve', 'credit_good', 'credit_score_high', 'has_coapplicant', 'has_collateral', 'income_ok', 'permanent_job', 'salary_medium', 'stable_job']
Stopped: no rule can add a new fact.  Firing order: ['R2', 'R4', 'R5', 'R6', 'R7', 'R9']
--- Strategy: most specific rule first ---
Cycle Conflict set                      Fired   Facts after
----------------------------------------------------------------------------------------------------
1     ['R2', 'R4', 'R5']                R2      ['credit_score_high', 'has_coapplicant', 'has_collateral', 'income_ok', 'permanent_job', 'salary_medium']
2     ['R4', 'R5']                      R4      ['credit_good', 'credit_score_high', 'has_coapplicant', 'has_collateral', 'income_ok', 'permanent_job', 'salary_medium']
3     ['R5', 'R7']                      R7      ['conditional_approve', 'credit_good', 'credit_score_high', 'has_coapplicant', 'has_collateral', 'income_ok', 'permanent_job', 'salary_medium']
4     ['R5']                            R5      ['conditional_approve', 'credit_good', 'credit_score_high', 'has_coapplicant', 'has_collateral', 'income_ok', 'permanent_job', 'salary_medium', 'stable_job']
5     ['R6', 'R9']                      R9      ['approve_low_interest', 'conditional_approve', 'credit_good', 'credit_score_high', 'has_coapplicant', 'has_collateral', 'income_ok', 'permanent_job', 'salary_medium', 'stable_job']
6     ['R6']                            R6      ['approve', 'approve_low_interest', 'conditional_approve', 'credit_good', 'credit_score_high', 'has_coapplicant', 'has_collateral', 'income_ok', 'permanent_job', 'salary_medium', 'stable_job']
Stopped: no rule can add a new fact.  Firing order: ['R2', 'R4', 'R7', 'R5', 'R9', 'R6']
Final conclusions: ['approve', 'approve_low_interest', 'conditional_approve']
Order same in both strategies? False

====================================================================================================
Test 3 (no conclusion) | starting facts: ['has_collateral']
--- Strategy: first match ---
Cycle Conflict set                      Fired   Facts after
----------------------------------------------------------------------------------------------------
Stopped: no rule can add a new fact.  Firing order: []
--- Strategy: most specific rule first ---
Cycle Conflict set                      Fired   Facts after
----------------------------------------------------------------------------------------------------
Stopped: no rule can add a new fact.  Firing order: []
Final conclusions: none
Order same in both strategies? True

```


---

## Experiment 5: Backward Chaining with Why/How Explanation

### Program: `exp05/exp05_backward_chaining.py`

```python
"""Experiment 5: Backward chaining with Why / How explanation (plant disease diagnosis)
Educational use only.

Run interactively:      python exp05_backward_chaining.py
Run the scripted demo:  python exp05_backward_chaining.py --demo   (default for the record)
"""
import sys

# rule_id, [conditions], conclusion
RULES = [
    ("R1", ["leaf_spots", "yellow_halo"], "bacterial_blight"),
    ("R2", ["leaf_spots", "wet_weather"], "bacterial_blight"),
    ("R3", ["white_powder_on_leaves", "dry_warm_weather"], "powdery_mildew"),
    ("R4", ["white_powder_on_leaves", "leaf_curling"], "powdery_mildew"),
    ("R5", ["wilting", "brown_stem_inside"], "fusarium_wilt"),
    ("R6", ["wilting", "yellow_lower_leaves", "soil_borne_history"], "fusarium_wilt"),
    ("R7", ["leaf_spots", "concentric_rings"], "early_blight"),
    ("R8", ["early_blight_suspect", "lower_leaves_first"], "early_blight"),
    ("R9", ["leaf_spots", "lower_leaves_first"], "early_blight_suspect"),
    ("R10", ["yellow_leaves", "stunted_growth", "pale_veins"], "nitrogen_deficiency"),
    ("R11", ["stunted_growth", "purple_leaves"], "phosphorus_deficiency"),
    ("R12", ["yellow_leaves", "green_veins"], "iron_deficiency"),
    ("R13", ["bacterial_blight"], "apply_copper_spray"),
    ("R14", ["powdery_mildew"], "apply_sulphur_fungicide"),
]
CONCLUSIONS = ["bacterial_blight", "powdery_mildew", "fusarium_wilt", "early_blight",
               "nitrogen_deficiency", "phosphorus_deficiency", "iron_deficiency"]
DERIVED = {r[2] for r in RULES}
ASKABLE = {c for r in RULES for c in r[1]} - DERIVED


class Consultation:
    def __init__(self, answers=None):
        self.facts = {}          # fact -> True/False
        self.trace = []          # proof trace
        self.asked = []          # questions asked (in order)
        self.proof = {}          # derived goal -> (rule_id, conditions)
        self.answers = answers   # scripted answers (dict) or None for interactive
        self.stack = []          # (goal, rule) pairs for WHY
        self.rules_checked = 0

    def ask(self, fact):
        if fact in self.facts:                     # never ask twice
            return self.facts[fact]
        while True:
            if self.answers is not None:
                if fact in self.answers.get("__why__", ()):
                    print(f"  ? Is it true that '{fact}'? (yes/no/why) -> why [scripted]")
                    self.why()
                ans = "yes" if self.answers.get(fact, False) else "no"
                print(f"  ? Is it true that '{fact}'? (yes/no/why) -> {ans} [scripted]")
            else:
                ans = input(f"  ? Is it true that '{fact}'? (yes/no/why): ").strip().lower()
            if ans == "why":
                self.why()
                if self.answers is not None:
                    ans = "yes" if self.answers.get(fact, False) else "no"
                else:
                    continue
            if ans in ("yes", "y", "no", "n"):
                break
        self.facts[fact] = ans.startswith("y")
        self.asked.append(fact)
        self.trace.append(f"asked {fact} -> {self.facts[fact]}")
        return self.facts[fact]

    def why(self):
        if not self.stack:
            print("  WHY: no goal on stack.")
            return
        goal, rid, conds = self.stack[-1]
        print(f"  WHY: I am trying to prove '{goal}' using rule {rid}: "
              f"IF {' AND '.join(conds)} THEN {goal}. I need this fact to test that rule.")

    def prove(self, goal):
        if goal in self.facts:
            return self.facts[goal]
        if goal in ASKABLE:
            return self.ask(goal)
        for rid, conds, concl in RULES:
            if concl != goal:
                continue
            self.rules_checked += 1
            self.stack.append((goal, rid, conds))
            ok = True
            for c in conds:
                if not self.prove(c):
                    ok = False
                    break
            self.stack.pop()
            if ok:
                self.facts[goal] = True
                self.proof[goal] = (rid, conds)
                self.trace.append(f"{rid} proved {goal}")
                return True
        self.facts[goal] = False
        return False

    def how(self, goal, indent=2):
        pad = " " * indent
        if goal in self.proof:
            rid, conds = self.proof[goal]
            print(f"{pad}{goal} was proved by {rid}: IF {' AND '.join(conds)}")
            for c in conds:
                self.how(c, indent + 4)
        else:
            print(f"{pad}{goal} = {self.facts.get(goal)} (answered by the user)")


def consult(title, answers):
    print("\n" + "=" * 78)
    print(f"Consultation: {title}")
    print("=" * 78)
    c = Consultation(answers)
    found = []
    for goal in CONCLUSIONS:
        if c.prove(goal):
            found.append(goal)
            break                              # stop at first diagnosis
    if found:
        print(f"\nCONCLUSION: {found[0]}")
        print("HOW:")
        c.how(found[0])
        advice = "apply_copper_spray" if found[0] == "bacterial_blight" else (
            "apply_sulphur_fungicide" if found[0] == "powdery_mildew" else None)
        if advice and c.prove(advice):
            print(f"ADVICE: {advice} (rule {c.proof[advice][0]})")
    else:
        print("\nCONCLUSION: nothing could be proved from the answers given.")
    print(f"Questions asked: {len(c.asked)} | rules tried: {c.rules_checked}")
    return len(c.asked), c.rules_checked


def forward_stats(facts):
    """Forward chaining on the same rule base: count rules checked."""
    facts = set(facts)
    checked = 0
    changed = True
    while changed:
        changed = False
        for _, conds, concl in RULES:
            checked += 1
            if concl not in facts and set(conds) <= facts:
                facts.add(concl)
                changed = True
    return checked, facts


if __name__ == "__main__":
    if "--interactive" in sys.argv:
        consult("interactive", None)
        sys.exit()

    scenarios = [
        ("Bacterial blight (spots + wet weather)",
         {"leaf_spots": True, "yellow_halo": False, "wet_weather": True,
          "__why__": {"yellow_halo"}}),
        ("Powdery mildew",
         {"white_powder_on_leaves": True, "dry_warm_weather": True}),
        ("Fusarium wilt",
         {"wilting": True, "brown_stem_inside": True}),
        ("Early blight via intermediate rule R9 -> R8",
         {"leaf_spots": True, "yellow_halo": False, "wet_weather": False,
          "concentric_rings": False, "lower_leaves_first": True}),
        ("Nothing provable",
         {}),
    ]
    stats = []
    for title, ans in scenarios:
        stats.append((title, *consult(title, ans)))

    print("\n" + "=" * 78)
    print("Demonstration of the WHY facility (asked when system poses a question)")
    print("=" * 78)
    c = Consultation({"leaf_spots": True, "yellow_halo": True})
    c.stack.append(("bacterial_blight", "R1", ["leaf_spots", "yellow_halo"]))
    print("  ? Is it true that 'yellow_halo'? (yes/no/why) -> why")
    c.why()

    print("\n" + "=" * 78)
    print("Backward vs forward chaining on the same rule base")
    print("=" * 78)
    print(f"{'Scenario':<48}{'Questions (BC)':<16}{'Rules tried (BC)':<18}Rules checked (FC)")
    for (title, q, r), (_, ans) in zip(stats, scenarios):
        fc, _ = forward_stats({k for k, v in ans.items() if v is True})
        print(f"{title:<48}{q:<16}{r:<18}{fc}")

```

### Output

```

==============================================================================
Consultation: Bacterial blight (spots + wet weather)
==============================================================================
  ? Is it true that 'leaf_spots'? (yes/no/why) -> yes [scripted]
  ? Is it true that 'yellow_halo'? (yes/no/why) -> why [scripted]
  WHY: I am trying to prove 'bacterial_blight' using rule R1: IF leaf_spots AND yellow_halo THEN bacterial_blight. I need this fact to test that rule.
  ? Is it true that 'yellow_halo'? (yes/no/why) -> no [scripted]
  ? Is it true that 'wet_weather'? (yes/no/why) -> yes [scripted]

CONCLUSION: bacterial_blight
HOW:
  bacterial_blight was proved by R2: IF leaf_spots AND wet_weather
      leaf_spots = True (answered by the user)
      wet_weather = True (answered by the user)
ADVICE: apply_copper_spray (rule R13)
Questions asked: 3 | rules tried: 3

==============================================================================
Consultation: Powdery mildew
==============================================================================
  ? Is it true that 'leaf_spots'? (yes/no/why) -> no [scripted]
  ? Is it true that 'white_powder_on_leaves'? (yes/no/why) -> yes [scripted]
  ? Is it true that 'dry_warm_weather'? (yes/no/why) -> yes [scripted]

CONCLUSION: powdery_mildew
HOW:
  powdery_mildew was proved by R3: IF white_powder_on_leaves AND dry_warm_weather
      white_powder_on_leaves = True (answered by the user)
      dry_warm_weather = True (answered by the user)
ADVICE: apply_sulphur_fungicide (rule R14)
Questions asked: 3 | rules tried: 4

==============================================================================
Consultation: Fusarium wilt
==============================================================================
  ? Is it true that 'leaf_spots'? (yes/no/why) -> no [scripted]
  ? Is it true that 'white_powder_on_leaves'? (yes/no/why) -> no [scripted]
  ? Is it true that 'wilting'? (yes/no/why) -> yes [scripted]
  ? Is it true that 'brown_stem_inside'? (yes/no/why) -> yes [scripted]

CONCLUSION: fusarium_wilt
HOW:
  fusarium_wilt was proved by R5: IF wilting AND brown_stem_inside
      wilting = True (answered by the user)
      brown_stem_inside = True (answered by the user)
Questions asked: 4 | rules tried: 5

==============================================================================
Consultation: Early blight via intermediate rule R9 -> R8
==============================================================================
  ? Is it true that 'leaf_spots'? (yes/no/why) -> yes [scripted]
  ? Is it true that 'yellow_halo'? (yes/no/why) -> no [scripted]
  ? Is it true that 'wet_weather'? (yes/no/why) -> no [scripted]
  ? Is it true that 'white_powder_on_leaves'? (yes/no/why) -> no [scripted]
  ? Is it true that 'wilting'? (yes/no/why) -> no [scripted]
  ? Is it true that 'concentric_rings'? (yes/no/why) -> no [scripted]
  ? Is it true that 'lower_leaves_first'? (yes/no/why) -> yes [scripted]

CONCLUSION: early_blight
HOW:
  early_blight was proved by R8: IF early_blight_suspect AND lower_leaves_first
      early_blight_suspect was proved by R9: IF leaf_spots AND lower_leaves_first
          leaf_spots = True (answered by the user)
          lower_leaves_first = True (answered by the user)
      lower_leaves_first = True (answered by the user)
Questions asked: 7 | rules tried: 9

==============================================================================
Consultation: Nothing provable
==============================================================================
  ? Is it true that 'leaf_spots'? (yes/no/why) -> no [scripted]
  ? Is it true that 'white_powder_on_leaves'? (yes/no/why) -> no [scripted]
  ? Is it true that 'wilting'? (yes/no/why) -> no [scripted]
  ? Is it true that 'yellow_leaves'? (yes/no/why) -> no [scripted]
  ? Is it true that 'stunted_growth'? (yes/no/why) -> no [scripted]

CONCLUSION: nothing could be proved from the answers given.
Questions asked: 5 | rules tried: 12

==============================================================================
Demonstration of the WHY facility (asked when system poses a question)
==============================================================================
  ? Is it true that 'yellow_halo'? (yes/no/why) -> why
  WHY: I am trying to prove 'bacterial_blight' using rule R1: IF leaf_spots AND yellow_halo THEN bacterial_blight. I need this fact to test that rule.

==============================================================================
Backward vs forward chaining on the same rule base
==============================================================================
Scenario                                        Questions (BC)  Rules tried (BC)  Rules checked (FC)
Bacterial blight (spots + wet weather)          3               3                 28
Powdery mildew                                  3               4                 28
Fusarium wilt                                   4               5                 28
Early blight via intermediate rule R9 -> R8     7               9                 42
Nothing provable                                5               12                14

```


---

## Experiment 6: Building and Reasoning over an Ontology

### Program: `exp06/exp06_ontology.py`

```python
"""Experiment 6: Building and reasoning over an ontology (owlready2 + HermiT)
University domain. Produces university.owl, class hierarchy diagram, inferred facts,
inconsistency report and answers to the competency questions.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from owlready2 import (get_ontology, Thing, ObjectProperty, DataProperty, AllDisjoint,
                       sync_reasoner_hermit, default_world, OwlReadyInconsistentOntologyError,
                       destroy_entity)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
IRI = "http://example.org/university.owl"

print("Competency questions the ontology must answer:")
CQ = ["CQ1: Which courses does Asha attend?",
      "CQ2: Who teaches Data Science?",
      "CQ3: Which department does Dr. Rao belong to?",
      "CQ4: Which individuals are Students (including inferred ones)?",
      "CQ5: Which courses have more than 3 credits?"]
for q in CQ:
    print("  ", q)

onto = get_ontology(IRI)
with onto:
    # ---- classes (9) ----
    class Person(Thing): pass
    class Student(Person): pass
    class Faculty(Person): pass
    class UGStudent(Student): pass
    class PGStudent(Student): pass
    class Professor(Faculty): pass
    class Course(Thing): pass
    class Department(Thing): pass
    class Programme(Thing): pass
    AllDisjoint([Student, Faculty])

    # ---- properties with domain / range ----
    class enrolledIn(ObjectProperty):
        domain = [Person]
        range = [Course]
    class teaches(ObjectProperty):
        domain = [Faculty]
        range = [Course]
    class belongsTo(ObjectProperty):
        domain = [Person]
        range = [Department]
    class taughtBy(ObjectProperty):
        inverse_property = teaches
    class hasName(DataProperty):
        range = [str]
    class hasCredits(DataProperty):
        domain = [Course]
        range = [int]

    # restriction: a Course must have at least one teacher
    Course.is_a.append(taughtBy.min(1, Faculty))

    # defined class: anyone enrolled in a course is a Student (domain-like axiom)
    class EnrolledPerson(Person):
        equivalent_to = [Person & enrolledIn.some(Course)]
    # make enrolledIn's subject a Student through a defined-class subsumption
    EnrolledPerson.is_a.append(Student)

    # ---- individuals (12) ----
    cse = Department("CSE"); ece = Department("ECE")
    ds = Course("DataScience"); ds.hasCredits = [4]
    kr = Course("KnowledgeRepresentation"); kr.hasCredits = [3]
    ml = Course("MachineLearning"); ml.hasCredits = [4]
    dc = Course("DigitalCircuits"); dc.hasCredits = [3]

    rao = Professor("DrRao"); rao.hasName = ["Dr. Rao"]; rao.belongsTo = [cse]
    meena = Professor("DrMeena"); meena.hasName = ["Dr. Meena"]; meena.belongsTo = [ece]
    rao.teaches = [ds, kr, ml]; meena.teaches = [dc]

    asha = UGStudent("Asha"); asha.hasName = ["Asha"]; asha.enrolledIn = [ds, kr]; asha.belongsTo = [cse]
    ravi = UGStudent("Ravi"); ravi.hasName = ["Ravi"]; ravi.enrolledIn = [ml]; ravi.belongsTo = [cse]
    divya = PGStudent("Divya"); divya.hasName = ["Divya"]; divya.enrolledIn = [ds, ml]; divya.belongsTo = [cse]
    # Karthik is NOT declared a Student: the reasoner must infer it from enrolledIn
    karthik = Person("Karthik"); karthik.hasName = ["Karthik"]; karthik.enrolledIn = [dc]; karthik.belongsTo = [ece]

onto.save(file="university.owl", format="rdfxml")
print("\nSaved university.owl")

# ---- Run HermiT (needs Java) ---------------------------------------------------
print("\n=== Running HermiT reasoner ===")
before = {i.name: [c.name for c in i.is_a] for i in onto.individuals()}
with onto:
    sync_reasoner_hermit(infer_property_values=True)
after = {i.name: [getattr(c, "name", str(c)) for c in i.is_a] for i in onto.individuals()}

print("\n=== Inferred class membership (new types found by the reasoner) ===")
found = False
for name in sorted(after):
    new = [c for c in after[name] if c not in before.get(name, [])]
    if new:
        found = True
        print(f"  {name}: now also {new}")
if not found:
    print("  (no new types)")
print("\nKarthik is a Student?", Student in karthik.is_a or Student in karthik.INDIRECT_is_a)
print("DrRao is a Faculty?   ", Faculty in rao.INDIRECT_is_a)
print("Inferred inverse (taughtBy) for DataScience:", [x.name for x in ds.taughtBy])

# ---- Competency questions ---------------------------------------------------------
print("\n=== Answers to competency questions ===")
print("CQ1 courses Asha attends      :", [c.name for c in asha.enrolledIn])
print("CQ2 who teaches DataScience   :", [t.name for t in onto.Faculty.instances() if ds in t.teaches] or
      [t.name for t in ds.taughtBy])
print("CQ3 Dr. Rao's department      :", [d.name for d in rao.belongsTo])
print("CQ4 Students (incl. inferred) :", sorted(s.name for s in onto.Student.instances()))
print("CQ5 courses with credits > 3  :", sorted(c.name for c in onto.Course.instances()
                                                  if c.hasCredits and c.hasCredits[0] > 3))

# ---- Diagram of class hierarchy ----------------------------------------------------
G = nx.DiGraph()
for cls in onto.classes():
    for parent in cls.is_a:
        if hasattr(parent, "name") and parent in onto.classes():
            G.add_edge(cls.name, parent.name)
G.add_edge("Person", "Thing"); G.add_edge("Course", "Thing"); G.add_edge("Department", "Thing")
G.add_edge("Programme", "Thing")
pos = nx.spring_layout(G, seed=4)
plt.figure(figsize=(9, 6))
nx.draw(G, pos, with_labels=True, node_color="lightgreen", node_size=2200, font_size=8)
plt.title("University ontology: class hierarchy (arrow = subclass-of)")
plt.savefig("class_hierarchy.png", dpi=110); plt.close()
print("Class hierarchy diagram saved: class_hierarchy.png")

# ---- Deliberate inconsistency ------------------------------------------------------
print("\n=== Deliberate inconsistency: a person who is both Student and Faculty ===")
with onto:
    bad = Person("BadPerson")
    bad.is_a.append(Student)
    bad.is_a.append(Faculty)
try:
    with onto:
        sync_reasoner_hermit()
    print("No inconsistency reported (unexpected).")
except OwlReadyInconsistentOntologyError:
    print("HermiT reports: INCONSISTENT ONTOLOGY.")
    print("Reason: BadPerson is declared a Student and a Faculty, but Student and Faculty")
    print("        were declared disjoint (AllDisjoint), so no model can satisfy both.")
except Exception as e:                 # other reasoner messages
    print("Reasoner message:", type(e).__name__, str(e)[:200])
destroy_entity(bad)

```

### Output

```
Competency questions the ontology must answer:
   CQ1: Which courses does Asha attend?
   CQ2: Who teaches Data Science?
   CQ3: Which department does Dr. Rao belong to?
   CQ4: Which individuals are Students (including inferred ones)?
   CQ5: Which courses have more than 3 credits?

Saved university.owl

=== Running HermiT reasoner ===
* Owlready2 * Running HermiT...
    java -Xmx2000M -cp /usr/local/lib/python3.13/site-packages/owlready2/hermit:/usr/local/lib/python3.13/site-packages/owlready2/hermit/HermiT.jar org.semanticweb.HermiT.cli.CommandLine -c -O -D -I file:////tmp/tmp8_xw9zy4 -Y
* Owlready2 * HermiT took 0.604189395904541 seconds
* Owlready * Reparenting university.EnrolledPerson: {university.Person, university.Student} => {university.Student}
* Owlready * Reparenting university.Karthik: {university.Person} => {university.EnrolledPerson}
* Owlready * Reparenting university.Ravi: {university.UGStudent} => {university.EnrolledPerson, university.UGStudent}
* Owlready * Reparenting university.Divya: {university.PGStudent} => {university.EnrolledPerson, university.PGStudent}
* Owlready * Reparenting university.Asha: {university.UGStudent} => {university.EnrolledPerson, university.UGStudent}
* Owlready * (NB: only changes on entities loaded in Python are shown, other changes are done but not listed)

=== Inferred class membership (new types found by the reasoner) ===
  Asha: now also ['EnrolledPerson']
  Divya: now also ['EnrolledPerson']
  Karthik: now also ['EnrolledPerson']
  Ravi: now also ['EnrolledPerson']

Karthik is a Student? True
DrRao is a Faculty?    True
Inferred inverse (taughtBy) for DataScience: ['DrRao']

=== Answers to competency questions ===
CQ1 courses Asha attends      : ['DataScience', 'KnowledgeRepresentation']
CQ2 who teaches DataScience   : ['DrRao']
CQ3 Dr. Rao's department      : ['CSE']
CQ4 Students (incl. inferred) : ['Asha', 'Divya', 'Karthik', 'Ravi']
CQ5 courses with credits > 3  : ['DataScience', 'MachineLearning']
Class hierarchy diagram saved: class_hierarchy.png

=== Deliberate inconsistency: a person who is both Student and Faculty ===
* Owlready2 * Running HermiT...
    java -Xmx2000M -cp /usr/local/lib/python3.13/site-packages/owlready2/hermit:/usr/local/lib/python3.13/site-packages/owlready2/hermit/HermiT.jar org.semanticweb.HermiT.cli.CommandLine -c -O -D -I file:////tmp/tmpsq_nav19
HermiT reports: INCONSISTENT ONTOLOGY.
Reason: BadPerson is declared a Student and a Faculty, but Student and Faculty
        were declared disjoint (AllDisjoint), so no model can satisfy both.

```

Files produced: `class_hierarchy.png`, `university.owl`


---

## Experiment 7: Building and Querying a Knowledge Graph

### Program: `exp07/exp07_knowledge_graph.py`

```python
"""Experiment 7: Building and querying a knowledge graph (rdflib + networkx)
Inputs : people.csv, works.csv, sentences.txt
Outputs: knowledge_graph.ttl, kg_drawing.png
"""
import os
import re
from difflib import SequenceMatcher

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import XSD
from rdflib.extras.external_graph_libs import rdflib_to_networkx_digraph

os.chdir(os.path.dirname(os.path.abspath(__file__)))
EX = Namespace("http://example.org/kg/")
g = Graph()
g.bind("ex", EX)


def uri(text):
    return EX[re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_")]


# 1. Small schema ----------------------------------------------------------------
for c in ["Person", "Project", "Department", "City", "University"]:
    g.add((EX[c], RDF.type, RDFS.Class))
for p, d, r in [("worksOn", "Person", "Project"), ("memberOf", "Person", "Department"),
                ("livesIn", "Person", "City"), ("studiesAt", "Person", "University")]:
    g.add((EX[p], RDF.type, RDF.Property))
    g.add((EX[p], RDFS.domain, EX[d]))
    g.add((EX[p], RDFS.range, EX[r]))

# 2. CSV -> triples --------------------------------------------------------------------
people = pd.read_csv("people.csv")
works = pd.read_csv("works.csv")
pid2uri = {}
for _, r in people.iterrows():
    s = uri(r["name"])
    pid2uri[r["person_id"]] = s
    g.add((s, RDF.type, EX.Person))
    g.add((s, RDFS.label, Literal(r["name"])))
    g.add((s, EX.memberOf, uri(r["department"])))
    g.add((uri(r["department"]), RDF.type, EX.Department))
    g.add((s, EX.livesIn, uri(r["city"])))
    g.add((uri(r["city"]), RDF.type, EX.City))
for _, r in works.iterrows():
    s = pid2uri[r["person_id"]]
    pr = uri(r["project"])
    g.add((s, EX.worksOn, pr))
    g.add((pr, RDF.type, EX.Project))
    g.add((pr, RDFS.label, Literal(r["project"])))
    g.add((s, EX["role_" + re.sub(r"\W+", "", r["project"])], Literal(r["role"])))
print(f"Triples after loading CSV files: {len(g)}")

# 3. Text -> triples by pattern matching -----------------------------------------------------
pattern = re.compile(r"^(?P<who>.+?) works on (?P<proj>Project \w+) at (?P<org>.+)$")
added_from_text = 0
for line in open("sentences.txt").read().splitlines():
    m = pattern.match(line.strip())
    if not m:
        continue
    who = uri(m["who"])
    for t in [(who, EX.worksOn, uri(m["proj"])), (who, EX.studiesAt, uri(m["org"])),
              (uri(m["org"]), RDF.type, EX.University), (uri(m["proj"]), RDF.type, EX.Project)]:
        if t not in g:
            g.add(t); added_from_text += 1
print(f"Triples added from text: {added_from_text} | total now: {len(g)}")

# 4. Entity resolution -------------------------------------------------------------------------
def normalise(name):
    # 'A. Kumar' -> compare initial + surname with 'Arun Kumar'
    parts = name.replace(".", "").split()
    return parts


def similar(n1, n2):
    p1, p2 = normalise(n1), normalise(n2)
    if len(p1) >= 2 and len(p2) >= 2 and p1[-1].lower() == p2[-1].lower() and p1[0][0].lower() == p2[0][0].lower():
        return 0.9 + 0.1 * SequenceMatcher(None, n1.lower(), n2.lower()).ratio()
    return SequenceMatcher(None, n1.lower(), n2.lower()).ratio()


print("\n=== Entity resolution (same department + similar name) ===")
merged = []
rows = people.to_dict("records")
for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        a, b = rows[i], rows[j]
        if a["department"] == b["department"]:
            sc = similar(a["name"], b["name"])
            if sc >= 0.85:
                print(f"  candidate: '{a['name']}' vs '{b['name']}'  dept={a['department']}  score={sc:.2f}  -> MERGE")
                merged.append((uri(b["name"]), uri(a["name"])))   # (duplicate, canonical)
for dup, canon in merged:
    for p, o in list(g.predicate_objects(dup)):
        if p != RDFS.label:
            g.add((canon, p, o))
    for s, p in list(g.subject_predicates(dup)):
        g.add((s, p, canon))
    g.remove((dup, None, None)); g.remove((None, None, dup))
    original = next(r["name"] for r in rows if uri(r["name"]) == dup)
    g.add((canon, EX.alsoKnownAs, Literal(original)))
    print(f"  merged {dup.split('/')[-1]} into {canon.split('/')[-1]}")
print("Total triples after merge:", len(g))

g.serialize("knowledge_graph.ttl", format="turtle")
print("Graph saved: knowledge_graph.ttl")

# 5. SPARQL queries ------------------------------------------------------------------------------
PREFIX = "PREFIX ex: <http://example.org/kg/> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
queries = {
    "Q1 Who works on Project Alpha?":
        "SELECT DISTINCT ?n WHERE { ?p ex:worksOn ex:Project_Alpha . ?p rdfs:label ?n } ORDER BY ?n",
    "Q2 Who is in the same department as Asha?":
        "SELECT DISTINCT ?n WHERE { ex:Asha ex:memberOf ?d . ?p ex:memberOf ?d . ?p rdfs:label ?n "
        "FILTER(?p != ex:Asha) } ORDER BY ?n",
    "Q3 Which projects involve people living in Chennai?":
        "SELECT DISTINCT ?pn WHERE { ?p ex:livesIn ex:Chennai . ?p ex:worksOn ?pr . ?pr rdfs:label ?pn } ORDER BY ?pn",
    "Q4 How many projects does each person work on?":
        "SELECT ?n (COUNT(DISTINCT ?pr) AS ?k) WHERE { ?p ex:worksOn ?pr . ?p rdfs:label ?n } "
        "GROUP BY ?n ORDER BY DESC(?k) ?n",
    "Q5 Which universities are linked to Project Alpha workers?":
        "SELECT DISTINCT ?u WHERE { ?p ex:worksOn ex:Project_Alpha . ?p ex:studiesAt ?u }",
    "Q6 Who is also known under another name (resolved duplicates)?":
        "SELECT ?n ?alias WHERE { ?p ex:alsoKnownAs ?alias . ?p rdfs:label ?n }",
}
print("\n=== SPARQL results ===")
for q, text in queries.items():
    print(q)
    res = list(g.query(PREFIX + text))
    if not res:
        print("   (no rows)")
    for row in res:
        print("   ", " | ".join(str(x).split("/")[-1] for x in row))

# 6. networkx analysis -------------------------------------------------------------------------------
nxg = nx.Graph()
for s, p, o in g:
    if isinstance(o, URIRef) and p not in (RDF.type, RDFS.domain, RDFS.range) and s != o \
            and not str(s).endswith(("Person", "Project", "Department", "City", "University")):
        nxg.add_edge(s.split("/")[-1], o.split("/")[-1], rel=p.split("/")[-1])
deg = dict(nxg.degree())
print("\n=== Degree of each node (top 10) ===")
for n, d in sorted(deg.items(), key=lambda x: (-x[1], x[0]))[:10]:
    print(f"   {n:<18}{d}")
top = max(deg, key=deg.get)
print(f"Most connected entity: {top} (degree {deg[top]})")

# 7. Drawing -----------------------------------------------------------------------------------------
plt.figure(figsize=(13, 9))
pos = nx.spring_layout(nxg, seed=7, k=0.6)
nx.draw(nxg, pos, with_labels=True, node_size=[300 + 150 * deg[n] for n in nxg], font_size=7,
        node_color="skyblue", edge_color="gray")
plt.savefig("kg_drawing.png", dpi=110); plt.close()
print("Graph drawing saved: kg_drawing.png")

```

### Input file: `exp07/people.csv` (11 lines)

```
person_id,name,department,city
1,Asha,CSE,Chennai
2,Arun Kumar,CSE,Chennai
3,Priya,ECE,Madurai
4,A. Kumar,CSE,Chennai
5,Ravi,MECH,Coimbatore
6,Meena,ECE,Chennai
7,Karthik,CSE,Trichy
8,Divya,AIDS,Chennai
9,Suresh,MECH,Coimbatore
10,Lakshmi,AIDS,Madurai
```

### Input file: `exp07/sentences.txt` (10 lines)

```
Asha works on Project Alpha at Anna University
Karthik works on Project Epsilon at Anna University
Ravi works on Project Gamma at SIMATS
Priya works on Project Zeta at IIT Madras
Meena works on Project Delta at SIMATS
Divya works on Project Beta at Anna University
Suresh works on Project Gamma at NIT Trichy
Lakshmi works on Project Alpha at SIMATS
Arun Kumar works on Project Beta at Anna University
Asha works on Project Zeta at IIT Madras
```

### Input file: `exp07/works.csv` (13 lines)

```
person_id,project,role
1,Project Alpha,Lead
2,Project Beta,Developer
3,Project Alpha,Analyst
4,Project Beta,Developer
5,Project Gamma,Engineer
6,Project Delta,Tester
7,Project Alpha,Developer
8,Project Delta,Lead
9,Project Gamma,Engineer
10,Project Beta,Analyst
1,Project Delta,Reviewer
8,Project Alpha,Analyst
```

### Output

```
Triples after loading CSV files: 97
Triples added from text: 21 | total now: 118

=== Entity resolution (same department + similar name) ===
  candidate: 'Arun Kumar' vs 'A. Kumar'  dept=CSE  score=0.98  -> MERGE
  merged A_Kumar into Arun_Kumar
Total triples after merge: 113
Graph saved: knowledge_graph.ttl

=== SPARQL results ===
Q1 Who works on Project Alpha?
    Asha
    Divya
    Karthik
    Lakshmi
    Priya
Q2 Who is in the same department as Asha?
    Arun Kumar
    Karthik
Q3 Which projects involve people living in Chennai?
    Project Alpha
    Project Beta
    Project Delta
Q4 How many projects does each person work on?
    Asha | 3
    Divya | 3
    Karthik | 2
    Lakshmi | 2
    Priya | 2
    Arun Kumar | 1
    Meena | 1
    Ravi | 1
    Suresh | 1
Q5 Which universities are linked to Project Alpha workers?
    Anna_University
    IIT_Madras
    SIMATS
Q6 Who is also known under another name (resolved duplicates)?
    Arun Kumar | A. Kumar

=== Degree of each node (top 10) ===
   Asha              7
   Divya             6
   Karthik           5
   Lakshmi           5
   Priya             5
   Project_Alpha     5
   Anna_University   4
   Arun_Kumar        4
   Chennai           4
   Meena             4
Most connected entity: Asha (degree 7)
Graph drawing saved: kg_drawing.png

```

Files produced: `kg_drawing.png`, `knowledge_graph.ttl`


---

## Experiment 8: Hypothesis Testing, ANOVA and Chi-Square

### Program: `exp08/exp08_hypothesis_tests.py`

```python
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

```

### Input file: `exp08/exam_scores.csv` (91 lines)

```
student,method,score_before,score_after,gender,result
S001,A,50.7,54.2,F,fail
S002,A,43.5,43.9,F,fail
S003,A,47.8,50.0,M,fail
S004,A,45.0,65.7,M,pass
S005,A,51.2,59.9,M,fail
S006,A,37.9,46.6,M,fail
S007,A,40.3,44.5,M,fail
S008,A,69.0,58.7,M,fail
S009,A,41.7,43.2,F,fail
S010,A,43.2,47.9,F,fail
S011,A,66.7,60.4,F,pass
S012,A,75.2,68.9,F,pass
S013,A,42.6,58.0,F,fail
S014,A,49.1,36.1,F,fail
... (76 more rows, full file in the zip)
```

### Output

```
Data: (90, 6) | methods: {'A': 30, 'B': 30, 'C': 30}
        score_before  score_after
method                           
A              51.46        54.84
B              52.27        62.52
C              54.16        68.87 

One-sample t (score_after vs 60)
  H0: mean = 60
  H1: mean != 60
  alpha = 0.05 | statistic = 1.982 | p = 0.05061
  Decision: Fail to reject H0
  Conclusion: No evidence that mean score_after differs from 60.

Independent t (method A vs B)
  H0: mean_A = mean_B
  H1: mean_A != mean_B
  alpha = 0.05 | statistic = -3.476 | p = 0.0009719
  Decision: Reject H0
  Conclusion: Methods A (54.84) and B (62.52) give significantly different scores.

Paired t (before vs after)
  H0: mean difference = 0
  H1: mean difference != 0
  alpha = 0.05 | statistic = -7.720 | p = 1.624e-11
  Decision: Reject H0
  Conclusion: Scores changed significantly (before 52.63, after 62.08).

One-way ANOVA (methods A,B,C)
  H0: mean_A = mean_B = mean_C
  H1: at least one mean differs
  alpha = 0.05 | statistic = 22.095 | p = 1.739e-08
  Decision: Reject H0
  Conclusion: At least one teaching method has a different mean score.

  Group means: {'A': 54.84, 'B': 62.52, 'C': 68.87}
  Post-hoc A vs B: t = -3.476, p = 0.0009719 (Bonferroni threshold 0.0167) -> differ
  Post-hoc A vs C: t = -6.763, p = 7.315e-09 (Bonferroni threshold 0.0167) -> differ
  Post-hoc B vs C: t = -3.092, p = 0.003054 (Bonferroni threshold 0.0167) -> differ

Contingency table gender x result:
 result  fail  pass
gender            
F         18    25
M         21    26 

Chi-square (gender vs result)
  H0: gender and result are independent
  H1: gender and result are associated
  alpha = 0.05 | statistic = 0.003 | p = 0.9547
  Decision: Fail to reject H0
  Conclusion: No evidence of association between gender and result.

=== Results table ===
                            test  statistic   p_value          decision
One-sample t (score_after vs 60)      1.982   0.05061 Fail to reject H0
   Independent t (method A vs B)     -3.476 0.0009719         Reject H0
      Paired t (before vs after)     -7.720 1.624e-11         Reject H0
   One-way ANOVA (methods A,B,C)     22.095 1.739e-08         Reject H0
   Chi-square (gender vs result)      0.003    0.9547 Fail to reject H0

95% confidence interval for mean score_after: (59.99, 64.16) | mean = 62.08
Box plot saved: boxplot_by_method.png | table saved: results_table.csv

```

Files produced: `boxplot_by_method.png`, `results_table.csv`


---

## Experiment 9: Acquiring Rules from Data with Decision Trees

### Program: `exp09/exp09_decision_tree_rules.py`

```python
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

```

### Output

```
Classes: {np.str_('malignant'): 212, np.str_('benign'): 357}
Train 398 | Test 171

Baseline (always majority class = benign): test accuracy = 0.626

Training-data look (mean by class) used to set thresholds:
           worst radius  mean concave points  mean area  worst concave points
target                                                                       
malignant        21.180                0.087    973.026                 0.185
benign           13.358                0.025    460.017                 0.073

Expert rules (IF any of these THEN malignant ELSE benign):
  E1: IF worst concave points > 0.15 THEN malignant
  E2: IF worst radius > 16.8 THEN malignant
  E3: IF mean concave points > 0.06 THEN malignant
  E4: IF mean area > 700 THEN malignant
Expert rules test accuracy = 0.912

=== Accuracy versus depth ===
 depth  train_acc  test_acc  cv_acc
     2      0.965     0.918   0.940
     3      0.980     0.924   0.942
     4      0.992     0.924   0.935
     6      1.000     0.918   0.935
Unrestricted tree (depth 6): train = 1.000, test = 0.918  <- 100% on training = memorising

Chosen depth = 2 (smallest depth whose CV accuracy is within 1% of the best 0.942)

=== export_text ===
|--- worst radius <= 16.80
|   |--- worst concave points <= 0.14
|   |   |--- class: 1
|   |--- worst concave points >  0.14
|   |   |--- class: 0
|--- worst radius >  16.80
|   |--- texture error <= 0.45
|   |   |--- class: 1
|   |--- texture error >  0.45
|   |   |--- class: 0

=== IF-THEN rules from the tree ===
T1: IF worst radius <= 16.795 AND worst concave points <= 0.143 THEN benign   (241 training cases)
T2: IF worst radius <= 16.795 AND worst concave points > 0.143 THEN malignant   (26 training cases)
T3: IF worst radius > 16.795 AND texture error <= 0.455 THEN benign   (3 training cases)
T4: IF worst radius > 16.795 AND texture error > 0.455 THEN malignant   (128 training cases)
Rule count: 4 | total conditions: 8

=== Test-set comparison (positive class = malignant) ===
             system  accuracy  precision  recall    F1
Baseline (majority)     0.626      0.000   0.000 0.000
   Expert rules (4)     0.912      0.855   0.922 0.887
       Tree depth 2     0.918      0.891   0.891 0.891
       Tree depth 3     0.924      0.918   0.875 0.896
       Tree depth 4     0.924      0.905   0.891 0.898
       Tree depth 6     0.918      0.891   0.891 0.891

Confusion matrices (rows = true [malignant, benign]; cols = predicted):
Expert rules:
 [[59  5]
 [10 97]]
Tree depth 2:
 [[ 57   7]
 [  7 100]]

=== Critique (173 words) ===
The majority baseline reaches 62.6% accuracy but never detects a malignant case, so accuracy alone is misleading here. The depth-2 tree scores 91.8% on the test set with malignant recall 89.1%, using only 4 rules and 8 conditions. The four hand-written expert rules reach 91.2% accuracy and malignant recall 92.2%. The tree is only marginally better on accuracy and F1, while the expert rules use fewer conditions and, because any single rule fires malignant, lean towards catching positives (higher recall, lower precision). Missing a malignant case is the costly error, so recall matters more than accuracy. A deeper tree reaches 100% training accuracy but its test and cross-validation accuracy stay flat or dip slightly, which signals overfitting; the depth was therefore chosen by cross-validation. A domain expert would probably accept rules built on size and concavity measures, since these match clinical intuition, but would question splits on odd features or tiny leaves with few cases. All learned rules need validation on new data, and this system must never be used for real diagnosis.

Plot saved: accuracy_vs_depth.png

```

Files produced: `accuracy_vs_depth.png`


---

## Experiment 10: Hybrid Decision-Support System

### Program: `exp10/exp10_hybrid_system.py`

```python
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

```

### Input file: `exp10/students.csv` (201 lines)

```
student,maths,programming,networks,design,cgpa,interest,best_track
ST001,64,78,98,64,6.67,Web,Web
ST002,53,76,59,95,7.21,AI,Web
ST003,48,53,45,74,7.85,Security,Web
ST004,82,75,79,76,7.16,Data,Data
ST005,51,65,71,77,10.0,Web,Web
ST006,70,73,93,47,7.44,Web,Security
ST007,56,87,54,58,6.65,Data,Data
ST008,69,43,44,92,7.87,Web,Web
ST009,73,60,69,50,8.44,AI,Security
ST010,66,89,82,91,8.86,Security,Security
ST011,49,80,47,70,6.96,Data,Data
ST012,66,86,65,58,7.97,AI,AI
ST013,82,95,78,60,8.29,AI,AI
ST014,68,49,82,49,8.4,Web,Web
... (186 more rows, full file in the zip)
```

### Output

```
Dataset: 200 students | train 160 | test 40 cases
Knowledge graph: 52 domain triples (>= 40 required)
Model: random forest | 5-fold CV accuracy on training data = 0.762 (+/- 0.042)

=== Evaluation on 40 held-out test cases ===
  system  accuracy  macro_F1
baseline     0.325     0.123
   rules     0.725     0.725
   model     0.800     0.798
  hybrid     0.850     0.855
Rules and model agreed on 31 of 40 cases.
When they disagreed (9 cases): rules correct 2, model correct 5, hybrid correct 7

=== 5 sample explanations ===
Student ST161  ->  RECOMMENDED TRACK: Data   (true best track: Data)
  Facts from graph : interestedIn Data, strongIn design, strongIn maths
  Rule R1: strongIn maths -> AI (CF +0.50)
  Rule R3: strongIn maths -> Data (CF +0.50)
  Rule R4: strongIn design -> Data (CF +0.20)
  Rule R7: strongIn design -> Web (CF +0.50)
  Rule R10: interestedIn Data -> Data (CF +0.50)
  Combined CF      : {'AI': 0.5, 'Data': 0.8, 'Security': 0.0, 'Web': 0.5}
  Model confidence : Data at 0.92  | probabilities {'AI': 0.0, 'Data': 0.92, 'Security': 0.02, 'Web': 0.06}
  Rules vs model   : AGREE
  Suggested courses: BigData, DataMining

Student ST162  ->  RECOMMENDED TRACK: Web   (true best track: Web)
  Facts from graph : interestedIn Web, strongIn design, strongIn programming, weakIn maths
  Rule R2: strongIn programming -> AI (CF +0.40)
  Rule R4: strongIn design -> Data (CF +0.20)
  Rule R6: strongIn programming -> Security (CF +0.20)
  Rule R7: strongIn design -> Web (CF +0.50)
  Rule R8: strongIn programming -> Web (CF +0.40)
  Rule R12: interestedIn Web -> Web (CF +0.50)
  Rule R14: weakIn maths -> AI (CF -0.50)
  Rule R16: weakIn maths -> Data (CF -0.40)
  Combined CF      : {'AI': -0.17, 'Data': -0.25, 'Security': 0.2, 'Web': 0.85}
  Model confidence : Web at 0.88  | probabilities {'AI': 0.07, 'Data': 0.03, 'Security': 0.03, 'Web': 0.88}
  Rules vs model   : AGREE
  Suggested courses: UXDesign, WebEngineering

Student ST164  ->  RECOMMENDED TRACK: Web   (true best track: Web)
  Facts from graph : interestedIn Web, strongIn design, weakIn maths, weakIn networks, weakIn programming
  Rule R4: strongIn design -> Data (CF +0.20)
  Rule R7: strongIn design -> Web (CF +0.50)
  Rule R12: interestedIn Web -> Web (CF +0.50)
  Rule R13: weakIn networks -> Security (CF -0.50)
  Rule R14: weakIn maths -> AI (CF -0.50)
  Rule R16: weakIn maths -> Data (CF -0.40)
  Combined CF      : {'AI': -0.5, 'Data': -0.25, 'Security': -0.5, 'Web': 0.75}
  Model confidence : Web at 0.79  | probabilities {'AI': 0.03, 'Data': 0.1, 'Security': 0.07, 'Web': 0.79}
  Rules vs model   : AGREE
  Suggested courses: UXDesign, WebEngineering

Student ST165  ->  RECOMMENDED TRACK: Web   (true best track: Web)
  Facts from graph : hasStanding high_cgpa, interestedIn Web, strongIn maths, strongIn programming, weakIn networks
  Rule R1: strongIn maths -> AI (CF +0.50)
  Rule R2: strongIn programming -> AI (CF +0.40)
  Rule R3: strongIn maths -> Data (CF +0.50)
  Rule R6: strongIn programming -> Security (CF +0.20)
  Rule R8: strongIn programming -> Web (CF +0.40)
  Rule R12: interestedIn Web -> Web (CF +0.50)
  Rule R13: weakIn networks -> Security (CF -0.50)
  Combined CF      : {'AI': 0.7, 'Data': 0.5, 'Security': -0.37, 'Web': 0.7}
  Model confidence : Web at 0.59  | probabilities {'AI': 0.21, 'Data': 0.2, 'Security': 0.01, 'Web': 0.59}
  Rules vs model   : DISAGREE (rules AI, model Web)
  Suggested courses: UXDesign, WebEngineering

Student ST168  ->  RECOMMENDED TRACK: Web   (true best track: Data)
  Facts from graph : interestedIn Data, strongIn design, strongIn networks, weakIn maths, weakIn programming
  Rule R4: strongIn design -> Data (CF +0.20)
  Rule R5: strongIn networks -> Security (CF +0.60)
  Rule R7: strongIn design -> Web (CF +0.50)
  Rule R10: interestedIn Data -> Data (CF +0.50)
  Rule R14: weakIn maths -> AI (CF -0.50)
  Rule R16: weakIn maths -> Data (CF -0.40)
  Combined CF      : {'AI': -0.5, 'Data': 0.33, 'Security': 0.6, 'Web': 0.5}
  Model confidence : Web at 0.48  | probabilities {'AI': 0.01, 'Data': 0.44, 'Security': 0.08, 'Web': 0.48}
  Rules vs model   : DISAGREE (rules Security, model Web)
  Suggested courses: UXDesign, WebEngineering

=== Failure case analysis ===
Hybrid wrong on 6 of 40 cases.
Student ST163  ->  RECOMMENDED TRACK: Security   (true best track: Data)
  Facts from graph : interestedIn Security, weakIn programming
  Rule R11: interestedIn Security -> Security (CF +0.50)
  Combined CF      : {'AI': 0.0, 'Data': 0.0, 'Security': 0.5, 'Web': 0.0}
  Model confidence : Security at 0.58  | probabilities {'AI': 0.06, 'Data': 0.16, 'Security': 0.58, 'Web': 0.19}
  Rules vs model   : AGREE
  Suggested courses: CryptoBasics, EthicalHacking

  Why it failed: the true track 'Data' received rule CF 0.00 and model probability 0.16, while the chosen 'Security' scored CF 0.50 / probability 0.58.
  The labels were generated with 8% random noise and an interest bonus, so part of such errors is
  label noise; the rules also use hard 70 / 55 mark thresholds, so students near a threshold are misjudged.

System diagram saved: system_diagram.png | results saved: results_table.csv

=== Limitations and ethics ===
- Data are SYNTHETIC; results show the method works, not that it suits real students.
- Must NOT be used to deny a student a course, rank students, or make any admission decision.
- Before real use: real, representative data; expert (faculty) review of every rule and CF;
  fairness checks across groups; a way for students to see the explanation and override the advice.

```

Files produced: `results_table.csv`, `system_diagram.png`
