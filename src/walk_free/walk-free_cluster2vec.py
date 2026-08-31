import os, time, random, bisect
import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 7


# Data                                                                

def load_roman_empire():
    try:
        from torch_geometric.datasets import HeterophilousGraphDataset
        d = HeterophilousGraphDataset(root="data/heterophilous", name="Roman-empire")[0]
        return d.edge_index.t().cpu().numpy(), d.y.cpu().numpy(), \
               d.train_mask.t().cpu().numpy(), d.test_mask.t().cpu().numpy()
    except Exception:
        p = "roman_empire.npz"
        if not os.path.exists(p):
            import urllib.request
            urllib.request.urlretrieve("https://github.com/yandex-research/"
                "heterophilous-graphs/raw/main/data/roman_empire.npz", p)
        d = np.load(p)
        return d["edges"], d["node_labels"], d["train_masks"], d["test_masks"]


edges, y, train_masks, test_masks = load_roman_empire()
n = len(y); K = len(np.unique(y)); majority = np.bincount(y).max() / n
G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(map(tuple, edges))
adj = [list(G.neighbors(i)) for i in range(n)]
print(f"Roman-empire: {n} nodes, {G.number_of_edges()} edges, {K} classes, "
      f"majority {majority:.3f}\n")


# Two hierarchies                                                     
def structural_features():
    deg = np.array([dd for _, dd in G.degree()])
    tri = nx.triangles(G); clu = nx.clustering(G)
    F = np.zeros((n, 7))
    for i in range(n):
        nd = [deg[j] for j in adj[i]]
        F[i, 0] = deg[i]
        if nd:
            F[i, 1] = np.mean(nd); F[i, 2] = np.std(nd); F[i, 3] = min(nd); F[i, 4] = max(nd)
        F[i, 5] = tri[i]; F[i, 6] = clu[i]
    return StandardScaler().fit_transform(F)

def structural_hierarchy(level_ks=(256, 64, 16, 4), seed=SEED):
    """Nested hierarchy: k-means for the finest level, then Ward on the fine
    centroids to derive the coarser (nested) levels. Returns memb[level][node]."""
    Fs = structural_features()
    Kfine = max(level_ks)
    km = KMeans(Kfine, n_init=5, random_state=seed).fit(Fs)
    fine = km.labels_
    Z = linkage(km.cluster_centers_, method="ward")
    memb = []
    for k in level_ks:
        memb.append(fine.copy() if k >= Kfine
                    else fcluster(Z, t=k, criterion="maxclust")[fine])
    return memb

def community_hierarchy(seed=SEED):
    comms = nx.community.louvain_communities(G, seed=seed)
    lab = np.zeros(n, int)
    for cid, c in enumerate(comms):
        for u in c:
            lab[u] = cid
    return [lab]     # single flat level


# Co-membership sampling -> Skip-gram pairs (as 2-token sentences)     
def sample_sentences(memb, S=80, decay=0.5, seed=SEED):
    rnd = random.Random(seed); L = len(memb)
    members = []                                   # per level: cluster -> [nodes]
    for lab in memb:
        dd = {}
        for node in range(n):
            dd.setdefault(int(lab[node]), []).append(node)
        members.append(dd)
    w = np.array([decay ** l for l in range(L)]); w /= w.sum()   # fine emphasised
    cum = np.cumsum(w).tolist()
    sents = []
    for v in range(n):
        for _ in range(S):
            l = bisect.bisect_left(cum, rnd.random() * cum[-1]); l = min(l, L - 1)
            mem = members[l][int(memb[l][v])]
            if len(mem) < 2:
                continue
            c = mem[rnd.randrange(len(mem))]
            if c == v:
                c = mem[rnd.randrange(len(mem))]
                if c == v:
                    continue
            sents.append([str(v), str(c)])
    return sents

def embed(sentences, dim=128, epochs=5, seed=SEED):
    from gensim.models import Word2Vec
    m = Word2Vec(sentences, vector_size=dim, window=1, min_count=0, sg=1,
                 negative=5, workers=4, epochs=epochs, seed=seed)
    return np.array([m.wv[str(i)] if str(i) in m.wv else np.zeros(dim)
                     for i in range(n)])

def evaluate(X):
    Xs = StandardScaler().fit_transform(X); accs = []
    for s in range(train_masks.shape[0]):
        tr, te = train_masks[s], test_masks[s]
        clf = LogisticRegression(max_iter=300).fit(Xs[tr], y[tr])
        accs.append(accuracy_score(y[te], clf.predict(Xs[te])))
    return float(np.mean(accs)), float(np.std(accs))


# Run: structural (multi-scale) vs community (Louvain) co-membership   
runs = [
    ("Cluster2Net (structural)", structural_hierarchy()),
    ("Cluster2Net (community)",  community_hierarchy()),
]
NODE2VEC = 0.169   # established Option-1 baseline on Roman-empire (same eval)

print("Walk-free node classification on Roman-empire (mean over 10 splits):\n")
rows = []
for name, memb in runs:
    t0 = time.time()
    ari = adjusted_rand_score(y, memb[0])
    sents = sample_sentences(memb)
    mean, std = evaluate(embed(sents))
    rows.append((name, mean, std, ari))
    print(f"  {name:28s} acc = {mean:.3f} +/- {std:.3f}   "
          f"[finest-level ARI {ari:+.3f}, {len(sents)} pairs, {time.time()-t0:.0f}s]")

print(f"\n  majority baseline            {majority:.3f}")
print(f"  Node2Vec (walk-based)        {NODE2VEC:.3f}")

fig, ax = plt.subplots(figsize=(8, 4.7))
names = [r[0].replace(" (", "\n(") for r in rows] + ["Node2Vec\n(walk-based)"]
means = [r[1] for r in rows] + [NODE2VEC]
stds = [r[2] for r in rows] + [0.003]
colors = ["#591576", "#682a23", "#093260"]
ax.bar(range(len(means)), means, yerr=stds, color=colors, capsize=4, edgecolor="white", linewidth=1)
ax.axhline(NODE2VEC, ls=":", c="#2c6fbb", lw=1.2, alpha=0.8, label=f"Node2Vec ({NODE2VEC:.3f})")
ax.axhline(majority, ls="--", c="gray", lw=1.1, label=f"majority ({majority:.3f})")
ax.set_xticks(range(len(means))); ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylabel("test accuracy (10 splits)"); ax.set_ylim(0.13, max(means) * 1.2)
ax.set_title("Option 2 on Roman-empire: walk-free cluster co-membership", fontsize=11)
ax.legend(fontsize=8, loc="upper right")
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.004, f"{m:.3f}", ha="center", fontsize=9)
fig.tight_layout(); fig.savefig("cluster2net_roman_empire.png", dpi=130, bbox_inches="tight")
print("\nsaved figure -> cluster2net_roman_empire.png")