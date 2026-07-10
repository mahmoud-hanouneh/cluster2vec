import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 7

# Graph, labels, flat clustering  (Louvain)

G = nx.karate_club_graph()
nodes = sorted(G.nodes())
n = len(nodes)
labels = np.array([0 if G.nodes[u]["club"] == "Mr. Hi" else 1 for u in nodes])
A = nx.to_numpy_array(G, nodelist=nodes)
deg = A.sum(1)
neighbors = [np.nonzero(A[i])[0] for i in range(n)]

comms = nx.community.louvain_communities(G, seed=SEED)
cluster = np.empty(n, dtype=int)
for cid, community in enumerate(comms):
    for u in community:
        cluster[u] = cid
print(f"Karate: {n} nodes, {G.number_of_edges()} edges, "
      f"{len(comms)} Louvain communities\n")


##### Three transition rules  ######                

 # 1) - Standard independant Node2Vec second-order transition (no clusters)                                     
def node2vec_probs(t, v, p, q):
    nbrs = neighbors[v]
    w = np.ones(len(nbrs))
    if t is not None:
        for i, x in enumerate(nbrs):
            if x == t:
                w[i] = 1.0 / p
            elif A[t, x] > 0:
                w[i] = 1.0
            else:
                w[i] = 1.0 / q
    return nbrs, w / w.sum()

# 2) - Independent, the flat first-order cluster bias
def first_order_cluster_probs(v, alpha, beta):
    nbrs = neighbors[v]
    w = np.array([alpha if cluster[x] == cluster[v] else beta for x in nbrs],
                 dtype=float)
    return nbrs, w / w.sum()

# 3) - The general engine: node2vec search bias x cluster bias
def cluster_node2vec_probs(t, v, p, q, alpha, beta):
    nbrs = neighbors[v]
    w = np.empty(len(nbrs))
    for i, x in enumerate(nbrs):
        if t is None:
            a = 1.0
        elif x == t:
            a = 1.0 / p
        elif A[t, x] > 0:
            a = 1.0
        else:
            a = 1.0 / q
        c = alpha if cluster[x] == cluster[v] else beta
        w[i] = a * c
    return nbrs, w / w.sum()



##### The two exact reduction tests #####

def all_states():
    for v in range(n): # at v coming from t
        yield None, v
        for t in neighbors[v]:
            yield int(t), v

def max_diff(rule_a, rule_b):
    d = 0.0
    for t, v in all_states():
        _, pa = rule_a(t, v)
        _, pb = rule_b(t, v)
        d = max(d, np.max(np.abs(pa - pb)))
    return d

print("Reduction tests (max difference in transition probability):")
for (p, q) in [(1.0, 1.0), (0.5, 2.0), (4.0, 0.25)]:
    d1 = max_diff(lambda t, v: cluster_node2vec_probs(t, v, p, q, 1.0, 1.0),
                  lambda t, v: node2vec_probs(t, v, p, q))
    print(f"  Test 1  alpha=beta=1, p={p}, q={q}: "
          f"cluster-N2V vs Node2Vec         diff = {d1:.2e}  "
          f"{'PASS' if d1 < 1e-12 else 'FAIL'}")

for (alpha, beta) in [(0.5, 2.0), (0.2, 3.0)]:
    d2 = max_diff(lambda t, v: cluster_node2vec_probs(t, v, 1.0, 1.0, alpha, beta),
                  lambda t, v: first_order_cluster_probs(v, alpha, beta))
    print(f"  Test 2  p=q=1, alpha={alpha}, beta={beta}: "
          f"cluster-N2V vs first-order walk  diff = {d2:.2e}  "
          f"{'PASS' if d2 < 1e-12 else 'FAIL'}")
print()



##### See the bias concretely on one node #####

# Pick a node that has neighbors in both its own and other clusters.
demo = next(v for v in range(n)
            if len({cluster[x] for x in neighbors[v]} | {cluster[v]}) > 1)
nbrs, plain = node2vec_probs(None, demo, 1.0, 0.5)
_, biased = cluster_node2vec_probs(None, demo, 1.0, 0.5, 0.5, 2.0)
print(f"Transition from node {demo} (cluster {cluster[demo]}), first step:")
for x, pp, bb in zip(nbrs, plain, biased):
    tag = "same" if cluster[x] == cluster[demo] else "DIFF"
    print(f"   -> {x:2d} ({tag} cluster {cluster[x]}):  plain {pp:.3f}   biased {bb:.3f}")
print()



##### Walks -> skip-gram (shared trainer, stabilized) #####                          

def generate_walks(p, q, alpha, beta, num_walks=20, walk_len=40, seed=0):
    rng = np.random.default_rng(seed)
    walks = []
    for _ in range(num_walks):
        for start in range(n):
            walk = [start]
            while len(walk) < walk_len:
                v = walk[-1]
                t = walk[-2] if len(walk) > 1 else None
                nbrs, probs = cluster_node2vec_probs(t, v, p, q, alpha, beta)
                walk.append(int(rng.choice(nbrs, p=probs)))
            walks.append(walk)
    return walks

def walk_pairs(walks, window=5):
    pl = []
    for walk in walks:
        for pos, c in enumerate(walk):
            lo, hi = max(0, pos - window), min(len(walk), pos + window + 1)
            for k in range(lo, hi):
                if k != pos:
                    pl.append((c, walk[k]))
    return np.array(pl)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def train_sgns(pairs, neg_dist, dim=16, steps=4000, batch=256,
               n_neg=5, lr=0.5, seed=0):
    rng = np.random.default_rng(seed)
    Vin = (rng.random((n, dim)) - 0.5) / dim
    Vout = (rng.random((n, dim)) - 0.5) / dim
    prob = np.full(len(pairs), 1.0 / len(pairs))
    for _ in range(steps):
        pick = rng.choice(len(pairs), size=batch, p=prob)
        c, o = pairs[pick, 0], pairs[pick, 1]
        negs = rng.choice(n, size=(batch, n_neg), p=neg_dist)
        vc, uo, un = Vin[c], Vout[o], Vout[negs]
        gpos = sigmoid(np.sum(vc * uo, 1)) - 1.0
        gneg = sigmoid(np.einsum("bd,bkd->bk", vc, un))
        grad_vc = gpos[:, None] * uo + np.einsum("bk,bkd->bd", gneg, un)
        grad_uo = gpos[:, None] * vc
        grad_un = gneg[:, :, None] * vc[:, None, :]
        gVin = np.zeros_like(Vin); gVout = np.zeros_like(Vout)
        np.add.at(gVin, c, grad_vc)
        np.add.at(gVout, o, grad_uo)
        np.add.at(gVout, negs.reshape(-1), grad_un.reshape(-1, dim))
        gVin /= batch; gVout /= batch
        np.clip(gVin, -5, 5, out=gVin); np.clip(gVout, -5, 5, out=gVout)
        Vin -= lr * gVin; Vout -= lr * gVout
    return Vin

neg_dist = deg ** 0.75
neg_dist /= neg_dist.sum()

def embed(p, q, alpha, beta):
    walks = generate_walks(p, q, alpha, beta, seed=SEED)
    return train_sgns(walk_pairs(walks), neg_dist, seed=SEED)


##### Three configurations, same engine, different knobs #####

configs = [
    ("DeepWalk",          dict(p=1.0, q=1.0, alpha=1.0, beta=1.0)),
    ("Node2Vec (q=0.5)",  dict(p=1.0, q=0.5, alpha=1.0, beta=1.0)),
    ("Cluster-N2V",       dict(p=1.0, q=0.5, alpha=0.5, beta=2.0)),
]

def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

def evaluate(X):
    km = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit_predict(X)
    ari = adjusted_rand_score(labels, km)
    preds = np.empty(n, dtype=int)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        preds[i] = KNeighborsClassifier(n_neighbors=5).fit(
            X[m], labels[m]).predict(X[i:i+1])[0]
    return ari, accuracy_score(labels, preds)

print("End-to-end on Karate (recovering the two factions):")
results = []
for name, kw in configs:
    X = unit(embed(**kw))
    ari, acc = evaluate(X)
    results.append((name, X, ari, acc))
    print(f"  {name:18s}  k-means ARI = {ari:+.3f}   LOO-kNN acc = {acc:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, X, ari, acc) in zip(axes, results):
    P2 = PCA(n_components=2, random_state=SEED).fit_transform(X)
    for lab, color in [(0, "#2c6fbb"), (1, "#c0392b")]:
        m = labels == lab
        ax.scatter(P2[m, 0], P2[m, 1], c=color, s=80, edgecolors="white",
                   linewidths=1.1, label=f"faction {lab}")
    for i in range(n):
        ax.annotate(str(nodes[i]), (P2[i, 0], P2[i, 1]), fontsize=6.5,
                    ha="center", va="center", color="white")
    ax.set_title(f"{name}\nARI={ari:+.2f}  acc={acc:.2f}", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=8)
fig.suptitle("Cluster-aware Node2Vec engine on Karate (2D PCA)", fontsize=13)
fig.tight_layout()
fig.savefig("karate_step1.png", dpi=130, bbox_inches="tight")
print("\nsaved figure -> karate_step1.png")