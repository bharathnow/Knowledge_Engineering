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
