import torch
import dgl
from dgl.nn.pytorch.utils import LabelPropagation
import networkx as nx
import matplotlib.pyplot as plt

label_propagation = LabelPropagation(k=10,            # iterations
                                     alpha=0.5,       # weight param for balancing between updates and initial labels
                                     clamp=False,     # whether to clamp the labels to [0,1]
                                     normalize=True)  # whether to apply row-normalization (sum of row is 1)

labels = [0, 2, 1, 3, 0]
g = dgl.graph((
    [0, 1, 2, 3],
    [2, 3, 1, 0]),
    num_nodes=5)

labels = torch.tensor(labels).long()   # (N, 1) N: number of nodes, 1: label
mask = torch.tensor([0, 1, 1, 1, 0]).bool()     # (N, 1) 0: unlabeled nodes, 1: labeled nodes

G = dgl.to_networkx(g)
plt.figure(figsize=[15, 7])
nx.draw(G, with_labels=True)

new_labels = label_propagation(g, labels, mask)  # returns (N,D) N: number of nodes, D: number of labels (percentages)

for i in range(0, len(labels)):
    print(f"Node %s:" % i, labels[i])

print(new_labels)

plt.show()

