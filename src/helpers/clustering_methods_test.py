# Testing different clustering methods on Roman-Empire Graph 
## 

import os, numpy as np, networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import spectral_embedding
from sklearn.metrics import adjusted_rand_score
SEED = 7
d = np.load("./src/helpers/roman_empire.npz"); edges, y = d["edges"], d["node_labels"]; n = len(y)
G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(map(tuple, edges))
K = len(np.unique(y))
def ari(lab): return adjusted_rand_score(y, lab)
print(f"Roman-empire: {n} nodes, {K} classes. ARI screen (higher = more label signal):\n")

# 1) Louvain: community method
if os.path.exists("cluster.npy"):
    louv = np.load("cluster.npy")
else:
    comms = nx.community.louvain_communities(G, seed=SEED); louv = np.zeros(n, int)
    for cid, c in enumerate(comms):
        for u in c: louv[u] = cid
print(f"  Louvain      (community, {len(set(louv.tolist()))} clusters):  ARI = {ari(louv):+.4f}")

#  2) Spectral: community/proximity method
A = nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr", dtype=float)
A.indices = A.indices.astype(np.int32); A.indptr = A.indptr.astype(np.int32)
Xspec = spectral_embedding(A, n_components=K, eigen_solver="arpack", random_state=SEED)
spec = KMeans(K, n_init=10, random_state=SEED).fit_predict(Xspec)
print(f"  Spectral     (community, k={K}):          ARI = {ari(spec):+.4f}")

# 3) Structural-role features
deg = np.array([dd for _, dd in G.degree()])
tri = nx.triangles(G); clu = nx.clustering(G)
F = np.zeros((n, 7))
for i in range(n):
    nd = [deg[j] for j in G.neighbors(i)]
    F[i, 0] = deg[i]
    if nd:
        F[i, 1] = np.mean(nd); F[i, 2] = np.std(nd); F[i, 3] = min(nd); F[i, 4] = max(nd)
    F[i, 5] = tri[i]; F[i, 6] = clu[i]
Fs = StandardScaler().fit_transform(F)
for k in [K, 40, 100]:
    st = KMeans(k, n_init=10, random_state=SEED).fit_predict(Fs)
    print(f"  Structural   (role-based, k={k:>3}):        ARI = {ari(st):+.4f}")