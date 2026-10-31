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
