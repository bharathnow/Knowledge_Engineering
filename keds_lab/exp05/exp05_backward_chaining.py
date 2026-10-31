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
