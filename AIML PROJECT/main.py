# ✅ Spectral Clustering for Social Media User Communities
# Complete Project Code for VS Code ✅
# Prathi - AIML Project

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

# ---------------------------------------
# 1️⃣ Load Dataset
# ---------------------------------------
edges = pd.read_csv("data/facebook_combined.txt", sep=" ", header=None, names=["source", "target"])
print("✅ Dataset Loaded Successfully!")
print(edges.head())

# ---------------------------------------
# 2️⃣ Create Graph from Edge List
# ---------------------------------------
G = nx.from_pandas_edgelist(edges, source="source", target="target")
print("Total Users (Nodes):", G.number_of_nodes())
print("Total Connections (Edges):", G.number_of_edges())

# ---------------------------------------
# 3️⃣ Spectral Clustering on Full Graph
# ---------------------------------------
adj_matrix = nx.to_numpy_array(G)
n_clusters = 3  # You can change cluster value
model = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
labels = model.fit_predict(adj_matrix)

print("\nCluster Labels Assigned to Each User:")
print(labels)

# ---------------------------------------
# 4️⃣ Subgraph for Visualization (Better Clarity)
# ---------------------------------------
sub_nodes = list(G.nodes())[:200]  # Visualize 200 users
H = G.subgraph(sub_nodes)

adj_matrix_sub = nx.to_numpy_array(H)
model_sub = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
labels_sub = model_sub.fit_predict(adj_matrix_sub)

print("\n✅ Visualization Subgraph Created with 200 Users")

# ---------------------------------------
# 5️⃣ Community Visualization
# ---------------------------------------
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(H, seed=42, k=0.15)  # Better spacing

nx.draw_networkx_nodes(H, pos, node_size=80, node_color=labels_sub, cmap='turbo')
nx.draw_networkx_edges(H, pos, alpha=0.3)

plt.title("Spectral Clustering - Facebook User Communities (Visual Subgraph)")
plt.axis("off")
plt.show()

# ---------------------------------------
# 6️⃣ Display Community Statistics
# ---------------------------------------
unique, counts = np.unique(labels_sub, return_counts=True)
print("\n📊 Community Size Stats:")
for u, c in zip(unique, counts):
    print(f"Community {u}: {c} users")

print("\n🎯 Project Completed Successfully!")
