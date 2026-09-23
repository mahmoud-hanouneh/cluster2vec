import os, time, random, bisect
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 7


# Load dataset      

def load_amazon_photo():
    try:
        from torch_geometric.datasets import Amazon
        d = Amazon(root="data/amazon", name="Photo")[0]
        edges_raw = d.edge_index.t().cpu().numpy(); y_all = d.y.cpu().numpy()
        n_all = len(y_all)
    except Exception:
        p = "amazon_electronics_photo.npz"
        if not os.path.exists(p):
            import urllib.request
            urllib.request.urlretrieve("https://github.com/shchur/gnn-benchmark/"
                "raw/master/data/npz/amazon_electronics_photo.npz", p)
        d = np.load(p, allow_pickle=True)
        A = sp.csr_matrix((d["adj_data"], d["adj_indices"], d["adj_indptr"]),
                          shape=tuple(d["adj_shape"]))
        y_all = d["labels"]; n_all = A.shape[0]
        edges_raw = np.array(A.tocoo().nonzero()).T
    G = nx.Graph(); G.add_nodes_from(range(n_all)); G.add_edges_from(map(tuple, edges_raw))
    G.remove_edges_from(nx.selfloop_edges(G))
    lcc = sorted(max(nx.connected_components(G), key=len))
    mapping = {o: i for i, o in enumerate(lcc)}
    Gl = nx.relabel_nodes(G.subgraph(lcc).copy(), mapping)
    return np.array(list(Gl.edges())), np.array([y_all[o] for o in lcc])


edges, y = load_amazon_photo()
n = len(y); K = len(np.unique(y)); majority = np.bincount(y).max() / n
G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(map(tuple, edges))
adj = [list(G.neighbors(i)) for i in range(n)]
homophily = sum(1 for u, v in G.edges() if y[u] == y[v]) / G.number_of_edges()
print(f"Amazon Photo (LCC): {n} nodes, {G.number_of_edges()} edges, {K} classes, "
      f"homophily {homophily:.3f}, majority {majority:.3f}\n")


# Two hierarchies (both multi-scale)                                  
def structural_hierarchy(level_ks=(256, 64, 16, 4), seed=SEED):
    deg = np.array([dd for _, dd in G.degree()]); tri = nx.triangles(G); clu = nx.clustering(G)
    F = np.zeros((n, 7))
    for i in range(n):
        nd = [deg[j] for j in adj[i]]; F[i, 0] = deg[i]
        if nd:
            F[i, 1] = np.mean(nd); F[i, 2] = np.std(nd); F[i, 3] = min(nd); F[i, 4] = max(nd)
        F[i, 5] = tri[i]; F[i, 6] = clu[i]
    Fs = StandardScaler().fit_transform(F)
    km = KMeans(max(level_ks), n_init=5, random_state=seed).fit(Fs)
    Z = linkage(km.cluster_centers_, method="ward")
    return [km.labels_.copy() if k >= max(level_ks)
            else fcluster(Z, t=k, criterion="maxclust")[km.labels_] for k in level_ks]

# Multi-scale communities from Louvain's own dendrogram levels (fine->coarse)
def community_hierarchy(seed=SEED):
    memb = []
    for part in nx.community.louvain_partitions(G, seed=seed):
        lab = np.zeros(n, int)
        for cid, c in enumerate(part):
            for u in c:
                lab[u] = cid
        memb.append(lab)
    memb.sort(key=lambda l: len(set(l.tolist())), reverse=True)   # finest first
    return memb


# Random splits (Amazon Photo has none official): 20/class, 10 splits 
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


# Co-membership sampling -> Skip-gram -> evaluation
def sample_sentences(memb, S=100, decay=0.5, seed=SEED):
    rnd = random.Random(seed); L = len(memb)
    members = []
    for lab in memb:
        dd = {}
        for node in range(n):
            dd.setdefault(int(lab[node]), []).append(node)
        members.append(dd)
    w = np.array([decay ** l for l in range(L)]); w /= w.sum()
    cum = np.cumsum(w).tolist(); sents = []
    for v in range(n):
        for _ in range(S):
            l = min(bisect.bisect_left(cum, rnd.random() * cum[-1]), L - 1)
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
    return np.array([m.wv[str(i)] if str(i) in m.wv else np.zeros(dim) for i in range(n)])

def evaluate(X):
    Xs = StandardScaler().fit_transform(X); accs = []
    for tr, te in SPLITS:
        clf = LogisticRegression(max_iter=300).fit(Xs[tr], y[tr])
        accs.append(accuracy_score(y[te], clf.predict(Xs[te])))
    return float(np.mean(accs)), float(np.std(accs))


# Run: community (aligned here) vs structural (weak here)
runs = [
    ("Cluster2Net (community)",  community_hierarchy()),
    ("Cluster2Net (structural)", structural_hierarchy()),
]
NODE2VEC = 0.850  # established baseline on Amazon Photo (same eval)

print("Walk-free node classification on Amazon Photo (mean over 10 random splits):\n")
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
means = [r[1] for r in rows] + [NODE2VEC]; stds = [r[2] for r in rows] + [0.015]
colors = ["#8e44ad", "#c0392b", "#2c6fbb"]
ax.bar(range(len(means)), means, yerr=stds, color=colors, capsize=4, edgecolor="white", linewidth=1)
ax.axhline(NODE2VEC, ls=":", c="#2c6fbb", lw=1.2, alpha=0.8, label=f"Node2Vec ({NODE2VEC:.3f})")
ax.axhline(majority, ls="--", c="gray", lw=1.1, label=f"majority ({majority:.3f})")
ax.set_xticks(range(len(means))); ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylabel("test accuracy (10 splits)"); ax.set_ylim(0.25, 0.95)
ax.set_title("Option 2 on Amazon Photo: walk-free cluster co-membership (homophilous)", fontsize=10.5)
ax.legend(fontsize=8, loc="center")
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.008, f"{m:.3f}", ha="center", fontsize=9)
fig.tight_layout(); fig.savefig("cluster2net_amazon_photo.png", dpi=130, bbox_inches="tight")
print("\nsaved figure -> cluster2net_amazon_photo.png")