<div align="center">

# 🧠 Hybrid AI & Knowledge Engineering Framework

**A Neuro-Symbolic AI Reasoning Engine integrating Semantic Knowledge Graphs (RDFLib), Rule-Based Expert Systems with Certainty Factors (CF), and Machine Learning (Random Forests).**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![RDFLib](https://img.shields.io/badge/RDFLib-Knowledge_Graphs-FF6F00.svg)](https://rdflib.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📌 Architecture Overview

This framework implements a **neuro-symbolic reasoning pipeline** that bridges deterministic symbolic knowledge with probabilistic statistical learning to make explainable, high-confidence recommendations.

```
+---------------------------------------------------------------+
|                       Input Features                          |
|             (Student Performance, Domain Profiles)           |
+-------------------------------+-------------------------------+
                                |
        +-----------------------+-----------------------+
        |                                               |
        v                                               v
+-------------------------------+               +---------------+
|     Knowledge Graph (RDFLib)  |               |  Random Forest|
|       Domain Triples & Rules  |               |   Classifier  |
+---------------+---------------+               +-------+-------+
                |                                       |
                v                                       v
+-------------------------------+               +---------------+
|    Rule Inference Engine      |               | Class Prob-   |
|   (Certainty Factors: -1 to 1)|               | abilities     |
+---------------+---------------+               +-------+-------+
                |                                       |
                +-----------------------+---------------+
                                        |
                                        v
                        +---------------+---------------+
                        |    Hybrid Combiner & Veto     |
                        |   Score = 0.5*CF + 0.5*Prob   |
                        +---------------+---------------+
                                        |
                                        v
                        +---------------+---------------+
                        | Explainable Recommendation &  |
                        | Audit Trail Generation        |
                        +-------------------------------+
```

---

## 📊 Benchmark Results

Evaluated on held-out test distributions comparing standalone baselines vs the hybrid reasoning engine:

| Reasoning Model | Accuracy | Macro F1 | Decision Explainability |
| :--- | :---: | :---: | :---: |
| **Statistical Baseline** | 32.5% | 0.123 | ❌ None |
| **Symbolic Rule Engine (CF)** | 72.5% | 0.725 | ✅ Full Rule Trace |
| **Machine Learning (Random Forest)** | 80.0% | 0.798 | ⚠️ Feature Importance Only |
| **Hybrid Neuro-Symbolic System** | **85.0%** | **0.855** | **✅ Full Neuro-Symbolic Audit** |

> **Key Finding**: In disagreement cases between symbolic rules and ML probabilities, the hybrid veto and certainty combiner successfully resolved 77.8% of conflicts correctly.

---

## 🔬 Core Components & Laboratory Suite

The repository contains modular implementations across 10 structured experiments:
- **`exp01` to `exp03`**: Knowledge Representation, Semantic Triples, and RDF Ontology construction.
- **`exp04` to `exp06`**: Rule-based inferencing, forward/backward chaining, and Certainty Factor propagation.
- **`exp07` to `exp09`**: Supervised learning integration, hyperparameter tuning, and cross-validation pipelines.
- **`exp10`**: End-to-end Neuro-Symbolic integration combining Knowledge Graphs + Expert Rules + Random Forest with explainable audit generation.

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/bharathnow/Knowledge_Engineering.git
cd Knowledge_Engineering

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install rdflib scikit-learn pandas numpy matplotlib
```

### 2. Run All Experiments
```bash
cd keds_lab
bash run_all.sh
```

Or run the full hybrid integration:
```bash
python exp10/hybrid_system.py
```

---

<div align="center">
  <sub>Developed by <a href="https://github.com/bharathnow">Mothukuri Bharath</a></sub>
</div>