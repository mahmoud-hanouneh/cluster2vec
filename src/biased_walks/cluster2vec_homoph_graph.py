import time, random
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 7


def load_amazon_photo():
    try:
        from torch_geometric.datasets import Amazon
        d = Amazon(root="data/amazon", name="Photo")[0]
        edges_raw = d.edge_index.t().cpu().numpy()
        y_all = d.y.cpu().numpy()
        n_all = len(y_all)
    except Exception:
        print("Error loading the Data ...")

    G = nx.Graph(); G.add_nodes_from(range(n_all))
    G.add_edges_from(map(tuple, edges_raw))
    G.remove_edges_from(nx.selfloop_edges(G))
    lcc = sorted(max(nx.connected_components(G), key=len))
    mapping = {old: new for new, old in enumerate(lcc)}
    Gl = nx.relabel_nodes(G.subgraph(lcc).copy(), mapping)
    y = np.array([y_all[o] for o in lcc])
    edges = np.array(list(Gl.edges()))
    return edges, y

edges, y = load_amazon_photo()
n = len(y); K = len(np.unique(y))
G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(map(tuple, edges))
adj = [list(G.neighbors(i)) for i in range(n)]
adj_set = [set(a) for a in adj]
majority = np.bincount(y).max() / n
homophily = sum(1 for u, v in G.edges() if y[u] == y[v]) / G.number_of_edges()
print(f"Amazon Photo (LCC): {n} nodes, {G.number_of_edges()} edges, {K} classes, "
      f"mean deg {2*G.number_of_edges()/n:.1f}")
print(f"edge homophily = {homophily:.3f}   majority baseline = {majority:.3f}\n")



# Clustering + ARI screen (which clustering carries label signal?)    
comms = nx.community.louvain_communities(G, seed=SEED)
cluster = [0] * n
for cid, c in enumerate(comms):
    for u in c:
        cluster[u] = cid

def structural_labels():
    deg = np.array([dd for _, dd in G.degree()])
    tri = nx.triangles(G); clu = nx.clustering(G)
    F = np.zeros((n, 7))
    for i in range(n):
        nd = [deg[j] for j in adj[i]]
        F[i, 0] = deg[i]
        if nd:
            F[i, 1] = np.mean(nd); F[i, 2] = np.std(nd); F[i, 3] = min(nd); F[i, 4] = max(nd)
        F[i, 5] = tri[i]; F[i, 6] = clu[i]
    return KMeans(K, n_init=10, random_state=SEED).fit_predict(StandardScaler().fit_transform(F))

print("ARI screen (label alignment):")
print(f"  Louvain    ({len(comms)} communities): ARI = {adjusted_rand_score(y, cluster):+.4f}")
print(f"  Structural (k={K}):            ARI = {adjusted_rand_score(y, structural_labels()):+.4f}")
print("  -> Louvain is well-aligned here (homophilous); it drives the bias.\n")



# Random splits: 20 labelled nodes per class for train, rest for test. Because the Amazon Photos has no official splits.
def make_splits(n_per_class=20, n_splits=10, seed=SEED):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n_splits):
        tr = np.zeros(n, bool)
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            tr[rng.choice(idx, min(n_per_class, len(idx)), replace=False)] = True
        out.append((tr, ~tr))
    return out
SPLITS = make_splits()



# Walk engine + diagnostics + gensim embedding + evaluation
def generate_walks(p, q, alpha, beta, r, num_walks=10, walk_len=40, seed=0):
    rnd = random.Random(seed); inv_p, inv_q = 1.0 / p, 1.0 / q
    walks = []; order = list(range(n))
    for _ in range(num_walks):
        rnd.shuffle(order)
        for start in order:
            walk = [start]
            for _ in range(walk_len - 1):
                v = walk[-1]; nbrs = adj[v]
                if not nbrs:
                    break
                cv = cluster[v]; t = walk[-2] if len(walk) > 1 else None
                if t is None:
                    ws = [alpha if cluster[x] == cv else beta for x in nbrs]
                else:
                    ct = cluster[t]; Nt = adj_set[t]; crossed = ct != cv; ws = []
                    for x in nbrs:
                        cx = cluster[x]; wi = alpha if cx == cv else beta
                        if x == t: wi *= inv_p
                        elif x in Nt: pass
                        else: wi *= inv_q
                        if crossed and cx == ct: wi *= r
                        ws.append(wi)
                tot = 0.0
                for wi in ws: tot += wi
                rv = rnd.random() * tot; acc = 0.0
                for i in range(len(ws)):
                    acc += ws[i]
                    if acc >= rv:
                        walk.append(nbrs[i]); break
            walks.append(walk)
    return walks

def walk_diagnostics(walks):
    tc = tr = 0; cov = []
    for w in walks:
        cl = [cluster[v] for v in w]
        for i in range(1, len(cl)):
            if cl[i] != cl[i - 1]:
                tc += 1
                if i + 1 < len(cl) and cl[i + 1] == cl[i - 1]: tr += 1
        cov.append(len(set(w)))
    return tc / len(walks), (tr / tc if tc else 0.0), float(np.mean(cov))

def embed(walks, dim=128, window=10, epochs=1, seed=0):
    from gensim.models import Word2Vec
    sents = [[str(v) for v in w] for w in walks]
    m = Word2Vec(sents, vector_size=dim, window=window, min_count=0, sg=1,
                 negative=5, workers=4, epochs=epochs, seed=seed)
    return np.array([m.wv[str(i)] for i in range(n)])

def evaluate(X):
    Xs = StandardScaler().fit_transform(X); accs = []
    for tr, te in SPLITS:
        clf = LogisticRegression(max_iter=300).fit(Xs[tr], y[tr])
        accs.append(accuracy_score(y[te], clf.predict(Xs[te])))
    return float(np.mean(accs)), float(np.std(accs))



#   Configurations. bias direction
#   beta > alpha = "Means leave the cluster"  
#   beta < alpha = "Means stay in the cluster"

## changing the engine params from here ##
configs = [
    ("DeepWalk",                 dict(p=1.0, q=1.0, alpha=1.0, beta=1.0, r=1.0)),
    ("Node2Vec (q=0.5)",         dict(p=1.0, q=0.5, alpha=1.0, beta=1.0, r=1.0)),
    ("Cluster-N2V leave (b/a=4)",dict(p=1.0, q=0.5, alpha=0.5, beta=2.0, r=1.0)),
    ("+ no-return leave",        dict(p=1.0, q=0.5, alpha=0.5, beta=2.0, r=0.001)),
    ("Cluster stay (b/a=0.25)",  dict(p=1.0, q=0.5, alpha=2.0, beta=0.5, r=1.0)),
    ("Cluster stay (b/a=0.1)",   dict(p=1.0, q=0.5, alpha=3.0, beta=0.3, r=1.0)),
]

print("Node classification on Amazon Photo (mean over 10 random splits):\n")
rows = []
for name, kw in configs:
    t0 = time.time(); walks = generate_walks(**kw, seed=SEED)
    diag = walk_diagnostics(walks); mean, std = evaluate(embed(walks, seed=SEED))
    rows.append((name, mean, std, diag))
    print(f"  {name:28s} acc = {mean:.3f} +/- {std:.3f}   "
          f"[recross {diag[1]:.0%}, cross/walk {diag[0]:.1f}, {time.time()-t0:.0f}s]")

n2v = [m for nm, m, s, d in rows if nm == "Node2Vec (q=0.5)"][0]
print(f"\n  majority baseline            {majority:.3f}")
print(f"  Node2Vec reference           {n2v:.3f}")

# chart
colors = ["#95a5a6", "#2c6fbb", "#c0392b", "#e07b39", "#27ae60", "#16a085"]
labels = [nm.replace(" (", "\n(").replace(" leave", "\nleave").replace(" stay", "\nstay")
          for nm, _, _, _ in rows]
means = [m for _, m, _, _ in rows]; stds = [s for _, _, s, _ in rows]
fig, ax = plt.subplots(figsize=(10, 4.9))
ax.bar(range(len(rows)), means, yerr=stds, color=colors, capsize=4, edgecolor="white", linewidth=1)
ax.axhline(n2v, ls=":", c="#2c6fbb", lw=1.2, alpha=0.8, label=f"Node2Vec ({n2v:.3f})")
ax.axhline(majority, ls="--", c="gray", lw=1.1, label=f"majority ({majority:.3f})")
ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("test accuracy (10 random splits)"); ax.set_ylim(0.25, 0.90)
ax.set_title(f"Amazon Photo (homophilous, Louvain-label ARI="
             f"{adjusted_rand_score(y, cluster):.2f}): Cluster2Vec vs Node2Vec", fontsize=10.5)
ax.legend(fontsize=8, loc="lower center")
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.006, f"{m:.3f}", ha="center", fontsize=8.5)
fig.tight_layout()
fig.savefig("amazon_photo_results.png", dpi=130, bbox_inches="tight")
print("\nsaved figure -> amazon_photo_results.png")