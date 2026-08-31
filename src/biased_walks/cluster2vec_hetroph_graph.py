"""
Cluster2Vec: cluster-aware Node2Vec + anti-return on Roman-empire
(a heterophilous graph: ~22.7k nodes, 18 syntactic-role classes).

"""

import time, os
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 7



# Data: Pytorch-Gemoetric loader & direct-.npz fallback          

def load_roman_empire():
    try:
        from torch_geometric.datasets import HeterophilousGraphDataset
        d = HeterophilousGraphDataset(root="data/heterophilous", name="Roman-empire")[0]
        edges = d.edge_index.t().cpu().numpy()
        y = d.y.cpu().numpy()
        tr = d.train_mask.t().cpu().numpy()   # PyG stores (N, splits) -> (splits, N)
        te = d.test_mask.t().cpu().numpy()
        return edges, y, tr, te
    except Exception:
        path = "./helpers/roman_empire.npz"
        if not os.path.exists(path):
            import urllib.request
            urllib.request.urlretrieve(
                "https://github.com/yandex-research/heterophilous-graphs/"
                "raw/main/data/roman_empire.npz", path)
        d = np.load(path)
        return d["edges"], d["node_labels"], d["train_masks"], d["test_masks"]

# edge list, the per-node labels (18 roles) and 10 train/test split masks
edges, y, train_masks, test_masks = load_roman_empire()
n = len(y) # length of nodes 
n_classes = len(np.unique(y))
majority = np.bincount(y).max() / n
print(f"Roman-empire: {n} nodes, {len(edges)} edges, {n_classes} classes")
print(f"majority-class baseline accuracy = {majority:.3f}\n")

# sparse adjacency (lists + sets); cluster as a plain list for fast lookups
# building the graph as adjaceny lists 
G = nx.Graph()
G.add_nodes_from(range(n))
G.add_edges_from(map(tuple, edges))
adj = [list(G.neighbors(i)) for i in range(n)]
adj_set = [set(a) for a in adj]


# Clustering (Louvain) - Not best option for Roman-Empire, to be changed.
comms = nx.community.louvain_communities(G, seed=SEED)
cluster = [0] * n
for cid, c in enumerate(comms):
    for u in c:
        cluster[u] = cid
clu_ari = adjusted_rand_score(y, cluster)
print(f"Louvain: {len(comms)} communities; alignment with the 18 labels "
      f"ARI = {clu_ari:+.3f}\n")




# Clean transition rules (used by a sampled reduction check)

# plain node2vec (p/q factor)
def node2vec_probs(t, v, p, q):
    nbrs = adj[v]; w = []
    for x in nbrs:
        if t is None:            a = 1.0
        elif x == t:             a = 1.0 / p
        elif x in adj_set[t]:    a = 1.0
        else:                    a = 1.0 / q
        w.append(a)
    s = sum(w); return nbrs, [wi / s for wi in w]

# plain cluster bias (alpha & beta)
def first_order_cluster_probs(v, alpha, beta):
    nbrs = adj[v]
    w = [alpha if cluster[x] == cluster[v] else beta for x in nbrs]
    s = sum(w); return nbrs, [wi / s for wi in w]

# the full-engine - for each neighbor, multiplies node2vec × cluster × anti-return, then normalizes to probabilities.
def cluster_node2vec_probs(t, v, p, q, alpha, beta, r):
    nbrs = adj[v]; cv = cluster[v]
    crossed = (t is not None) and (cluster[t] != cv)
    ct = cluster[t] if t is not None else -1
    w = []
    for x in nbrs:
        cx = cluster[x]
        wi = alpha if cx == cv else beta
        if t is not None:
            if x == t:            wi /= p
            elif x in adj_set[t]: pass
            else:                 wi /= q
            if crossed and cx == ct: wi *= r
        w.append(wi)
    s = sum(w); return nbrs, [wi / s for wi in w]

# sampled reduction check 
## a quick correctness test on 400 random walk states, confirming the full engine collapses  to plain node2vec (when α=β=1, r=1) and to the plain cluster walk (when p=q=1, r=1)
rng = np.random.default_rng(SEED)
states = []
for _ in range(400):
    v = int(rng.integers(n))
    t = int(rng.choice(adj[v])) if adj[v] else None
    states.append((t, v))
def sdiff(a, b):
    return max(max(abs(pa - pb) for pa, pb in zip(a(t, v)[1], b(t, v)[1]))
               for t, v in states)
d1 = sdiff(lambda t, v: cluster_node2vec_probs(t, v, 0.5, 2.0, 1.0, 1.0, 1.0),
           lambda t, v: node2vec_probs(t, v, 0.5, 2.0))
d2 = sdiff(lambda t, v: cluster_node2vec_probs(t, v, 1.0, 1.0, 0.5, 2.0, 1.0),
           lambda t, v: first_order_cluster_probs(v, 0.5, 2.0))
print(f"Sampled reduction check (400 states): "
      f"vs Node2Vec {d1:.1e}, vs flat-cluster {d2:.1e}  "
      f"{'PASS' if max(d1, d2) < 1e-12 else 'FAIL'}\n")



# Fast walk generation (pure-Python hot loop for speed)
## From every node, takes steps by sampling from cluster_node2vec_probs, producing the node sequences that serve as "context."
import random
def generate_walks(p, q, alpha, beta, r, num_walks=10, walk_len=40, seed=0):
    rnd = random.Random(seed)
    inv_p, inv_q = 1.0 / p, 1.0 / q
    walks = []
    order = list(range(n))
    for _ in range(num_walks):
        rnd.shuffle(order)
        for start in order:
            walk = [start]
            for _ in range(walk_len - 1):
                v = walk[-1]; nbrs = adj[v]
                if not nbrs:
                    break
                cv = cluster[v]
                t = walk[-2] if len(walk) > 1 else None
                if t is None:
                    ws = [alpha if cluster[x] == cv else beta for x in nbrs]
                else:
                    ct = cluster[t]; Nt = adj_set[t]; crossed = ct != cv
                    ws = []
                    for x in nbrs:
                        cx = cluster[x]
                        wi = alpha if cx == cv else beta
                        if x == t:       wi *= inv_p
                        elif x in Nt:    pass
                        else:            wi *= inv_q
                        if crossed and cx == ct: wi *= r
                        ws.append(wi)
                tot = 0.0
                for wi in ws: tot += wi
                rv = rnd.random() * tot
                acc = 0.0
                for i in range(len(ws)):
                    acc += ws[i]
                    if acc >= rv:
                        walk.append(nbrs[i]); break
            walks.append(walk)
    return walks

# measures the oscillation: crossings per walk, how often a crossing is immediately reversed, and how many distinct nodes each walk covers.
def walk_diagnostics(walks):
    tot_cross = tot_recross = 0; coverage = []
    for w in walks:
        cl = [cluster[v] for v in w]
        for i in range(1, len(cl)):
            if cl[i] != cl[i - 1]:
                tot_cross += 1
                if i + 1 < len(cl) and cl[i + 1] == cl[i - 1]:
                    tot_recross += 1
        coverage.append(len(set(w)))
    rr = tot_recross / tot_cross if tot_cross else 0.0
    return tot_cross / len(walks), rr, float(np.mean(coverage))


# Embedding (gensim) + evaluation over the dataset's splits
## returns one 128-dim vector per node. The "embedding" pillar.
def embed(walks, dim=128, window=10, epochs=1, seed=0):
    sents = [[str(v) for v in w] for w in walks]
    m = Word2Vec(sents, vector_size=dim, window=window, min_count=0, sg=1,
                 negative=5, workers=4, epochs=epochs, seed=seed)
    return np.array([m.wv[str(i)] for i in range(n)])

# for each of the 10 splits trains a logistic-regression classifier on the train nodes and scores accuracy on the test nodes, returning mean ± std. 
def evaluate(X):
    Xs = StandardScaler().fit_transform(X)
    accs = []
    for s in range(train_masks.shape[0]):
        tr, te = train_masks[s], test_masks[s]
        clf = LogisticRegression(max_iter=300, n_jobs=-1)
        clf.fit(Xs[tr], y[tr])
        accs.append(accuracy_score(y[te], clf.predict(Xs[te])))
    return float(np.mean(accs)), float(np.std(accs))


# compare the four steps
configs = [
    ("DeepWalk",                dict(p=1.0, q=1.0, alpha=1.0, beta=1.0, r=1.0)),
    ("Node2Vec (q=0.5)",        dict(p=1.0, q=0.5, alpha=1.0, beta=1.0, r=1.0)),
    ("Cluster-N2V",             dict(p=1.0, q=0.5, alpha=1, beta=10, r=1.0)),
    ("Cluster-N2V + no-return", dict(p=1.0, q=0.5, alpha=1, beta=10, r=0.001)),
]

print("Running configs (walks -> gensim -> node classification over 10 splits):\n")
rows = []
for name, kw in configs:
    t0 = time.time()
    walks = generate_walks(**kw, seed=SEED)
    tw = time.time() - t0
    diag = walk_diagnostics(walks)
    X = embed(walks, seed=SEED)
    mean, std = evaluate(X)
    rows.append((name, diag, mean, std))
    print(f"  {name:26s} acc = {mean:.3f} +/- {std:.3f}   "
          f"[recross {diag[1]:.0%}, cover {diag[2]:.0f}, walks {tw:.0f}s]")

print("\nSummary (test accuracy on Roman-empire, mean over 10 splits):")
print(f"  majority baseline           {majority:.3f}")
for name, diag, mean, std in rows:
    print(f"  {name:26s} {mean:.3f} +/- {std:.3f}")

# bar chart
fig, ax = plt.subplots(figsize=(8, 4.5))
names = [r[0] for r in rows]; means = [r[2] for r in rows]; stds = [r[3] for r in rows]
colors = ["#7f8c8d", "#2c6fbb", "#c0392b", "#27ae60"]
ax.bar(range(len(names)), means, yerr=stds, color=colors, capsize=4,
       edgecolor="white", linewidth=1)
ax.axhline(majority, ls="--", c="gray", lw=1, label=f"majority ({majority:.2f})")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
ax.set_ylabel("test accuracy"); ax.set_ylim(0, max(means) * 1.25)
ax.set_title("Roman-empire (heterophilous): node classification", fontsize=12)
ax.legend(fontsize=8)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.005, f"{m:.3f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig("roman_empire_results.png", dpi=130, bbox_inches="tight")
print("\nsaved figure -> roman_empire_results.png")