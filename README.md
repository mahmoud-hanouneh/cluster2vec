# Cluster2Vec

Graph Neural Networks.
**Research Question:** can _cluster membership_ replace or improve the random walk as the definition of a node's context?

---

## Motivation

Node2Vec is a shallow embedding model: it samples node sequences with second-order random walks and feeds them to a Skip-gram objective, exactly like Word2Vec does with sentences. This works well, but the context it produces is **local and stochastic**. It mostly recovers nearby community structure and says little about where a node sits in the global topology. Struc2Vec pushed in a different direction by building context from _structural roles_ rather than proximity.

Here we keep the walk, but we tell it about **clusters**. Before walking, we group the nodes into communities. The walk then knows which cluster each node belongs to, and we can push it to stay inside its own cluster or to move out of it. Later on, we want to use a whole hierarchy of clusters instead of just one grouping.

---

## The biased-walk engine

Three independent factors multiply into one unnormalised transition weight:

```
weight(t, v → x) = α_pq(t, x) · c(v, x) · m(t, x)
```

| factor       | meaning                                                                                                                      | knobs           |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `α_pq(t, x)` | Node2Vec second-order search bias: `1/p` to return to `t`, `1` to a neighbour of `t`, `1/q` outward                          | `p`, `q`        |
| `c(v, x)`    | cluster bias: `alpha` if `x` is in `v`'s cluster, `beta` if it is outside                                                    | `alpha`, `beta` |
| `m(t, x)`    | anti-return: if the walk _just crossed a border_, multiply by `r` every neighbour that lies back in the cluster it came from | `r`             |

<!-- The fix has to work on whole clusters, not single nodes. That is what `m` does: right after a crossing, it multiplies the weight of _every_ neighbour still sitting in the old cluster by `r`. A small `r` (example, 0.0001) makes going back very unlikely for one step, which is enough to let the walk go further to B. Setting `r = 1` turns the factor off. -->

The engine is exact by construction:

- `alpha = beta = 1`, `r = 1` → collapses to plain Node2Vec
- `p = q = 1`, `r = 1` → collapses to the flat first-order cluster walk

<!-- (the _Correctness_ section in the code). -->

The walk-free variant needs none of these knobs: a node's context is the nodes that share a cluster with, across a multi-scale hierarchy, with finer shared levels weighted more strongly.

---

## Repository layout

```
src/
  base/
    node2vec_baseline.py           plain Node2Vec q-sweep baseline on Planetoid (Cora / CiteSeer / PubMed)
  biased_walks/                    Approach 1 — cluster-biased Node2Vec walk (alpha, beta, anti-return r)
    cluster2vec_hetroph_graph.py     Roman-empire (heterophilous); Louvain; sampled reduction check; walk diagnostics
    cluster2vec_homoph_graph.py      Amazon Photo (homophilous); ARI screen; stay vs leave bias
  walk_free/                       Approach 2 — walk-free cluster co-membership (multi-scale hierarchy)
    cluster2vec_hetro_graph.py       Roman-empire; structural vs community co-membership
    cluster2vec_homo_graph.py        Amazon Photo; structural vs community co-membership
  helpers/
    clustering_methods_test.py       ARI: Louvain vs Spectral vs Structural clustering
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Approach 1 — cluster-biased walk
python src/biased_walks/cluster2vec_hetroph_graph.py   # Roman-empire (heterophilous)
python src/biased_walks/cluster2vec_homoph_graph.py    # Amazon Photo (homophilous)

# Approach 2 — walk-free co-membership
python src/walk_free/cluster2vec_hetro_graph.py        # Roman-empire (heterophilous)
python src/walk_free/cluster2vec_homo_graph.py         # Amazon Photo (homophilous)

# Clustering ARI screen (which clustering aligns with the labels)
python src/helpers/clustering_methods_test.py

# Planetoid Node2Vec baseline sweep
python src/base/node2vec_baseline.py --dataset Cora


# The biased-walk scripts run these configurations (same engine, different knobs):

# | config                  | p   | q   | alpha | beta | r     |
# | ----------------------- | --- | --- | ----- | ---- | ----- |
# | DeepWalk                | 1.0 | 1.0 | 1.0   | 1.0  | 1.0   |
# | Node2Vec                | 1.0 | 0.5 | 1.0   | 1.0  | 1.0   |
# | Cluster-N2V             | 1.0 | 0.5 | 0.5   | 2.0  | 1.0   |
# | Cluster-N2V + no-return | 1.0 | 0.5 | 0.5   | 2.0  | 0.001 |

---

## Evaluation

# Walks → gensim Skip-gram (128 dims, window 10, negative sampling) → frozen embeddings → logistic regression on standardised features, scored on each of the 10 official splits and reported as mean ± std.

# Alongside accuracy, `walk_diagnostics()` reports what the walk actually did:

# - **crossings per walk** — how often it leaves its cluster at all
# - **re-cross rate** — how often a crossing is immediately reversed (the oscillation metric)
# - **coverage** — distinct nodes visited per walk

# These are important because accuracy alone cannot tell whether the bias changed the walk's behaviour or just added noise.

# <!-- ## Correctness

# Both scripts verify the reductions rather than assuming them:

# - `walk_biased_node2vec.py` — **exhaustive** over every `(t, v)` state on Karate, for several `(p, q)` and `(alpha, beta)` pairs
# - `cluster2vec.py` — **sampled** over 400 random states on Roman-empire (exhaustive is infeasible at 22k nodes)

# Both pass at max transition-probability difference < 1e-12.

# `cluster2vec.py` also prints the **ARI between the Louvain partition and the 18 ground-truth labels**, which is the honest diagnostic for whether the cluster bias is even pointing at the right target. -->

# ---

## References

- Mikolov et al., _Efficient Estimation of Word Representations in Vector Space_, 2013
- Grover & Leskovec, _node2vec: Scalable Feature Learning for Networks_, 2016
- Ribeiro et al., _struc2vec: Learning Node Representations from Structural Identity_, 2017
```
