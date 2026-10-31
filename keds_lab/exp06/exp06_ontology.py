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
