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
