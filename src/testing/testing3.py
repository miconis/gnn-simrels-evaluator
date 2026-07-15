import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from src.dgl_graph.dataset import PubmedSubgraph
import numpy as np
from src.utils.config import (
    DATASET_URL,
    DATA_DIR,
    ORCIDS_PATH,
    RAW_DIR,
    SIMRELS_FOR_TESTING_PATH,
    SIMRELS_SCORES_PATH,
)


def plot_results(results):
    """
    results: dict {threshold: {metric_name: value, ...}, ...}
    Automatically plot all the metrics in the map
    """
    # sort thresholds
    ths = sorted(results.keys())

    # dynamic metrics
    metric_names = list(results[ths[0]].keys())

    plt.figure(figsize=(10, 6))

    # plot di ogni metrica
    for metric in metric_names:
        values = [results[t][metric] for t in ths]
        plt.plot(ths, values, marker='o', label=metric)

    plt.xlabel("Threshold")
    plt.ylabel("%")
    plt.title("Metrics vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()


def prune_simrels(simrels: torch.Tensor, scores: torch.Tensor, th: float):
    mask = scores >= th
    simrels = simrels[mask]
    return simrels


def connected_components_from_simrels(num_nodes: int, simrels: torch.Tensor):
    """
    num_nodes: number of nodes
    simrels: Tensor [M,2] (src, dst)
    scores: Tensor [M,2] (score of simrels)
    th:  (cut threshold)
    returns: Tensor [num_nodes]  (id connected component)
    """
    if simrels.numel() == 0:
        return torch.arange(num_nodes, dtype=torch.int64)

    # DSU / Union-Find
    parent = torch.arange(num_nodes, dtype=torch.int64)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra
    # union for each simrel
    for src, dst in simrels.tolist():
        union(src, dst)
    for i in range(num_nodes):
        parent[i] = find(i)
    # ids normalizations
    _, comp_ids = torch.unique(parent, return_inverse=True)
    return comp_ids


def pairwise_metrics(orcids, groups):

    true_labels = orcids.squeeze().cpu().numpy()
    pred_labels = groups.squeeze().cpu().numpy()

    # computation of true pairs
    true_pairs = 0
    for c in np.unique(true_labels):
        idx = np.where(true_labels == c)[0]
        if len(idx) > 1:  # pairs can exist if the class has at least 2 elements
            true_pairs += len(idx) * (len(idx) - 1) // 2  # all possible number of pairs

    pred_pairs = []
    for g in np.unique(pred_labels):
        idx = np.where(pred_labels == g)[0]
        if len(idx) > 1:  # pairs can exist if the class has at least 2 elements
            for i in range(len(idx)):
                for j in range(i+1, len(idx)):
                    pred_pairs.append((idx[i], idx[j]))
    pred_pairs_set = set(pred_pairs)

    TP = 0
    for i, j in pred_pairs_set:
        if true_labels[i] == true_labels[j]:
            TP += 1

    FP = len(pred_pairs_set) - TP
    FN = true_pairs - TP

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return precision, recall, f1


def compute_statistics(simrels, correct_simrels_mask, orcids, groups):

    # true_labels = orcids.squeeze().cpu().numpy()
    # pred_labels = groups.squeeze().cpu().numpy()
    groups = groups.squeeze()

    simrels_accuracy = correct_simrels_mask.float().mean().item()

    # groups with a single element
    singletons = count_singletons(groups) / len(groups)

    # clusters metrics
    # ari = adjusted_rand_score(true_labels, pred_labels)  # Adjusted Rand Index (-1 to 1)
    # nmi = normalized_mutual_info_score(true_labels, pred_labels)  # Normalized Mutual Information (0 to 1)
    # fmi = fowlkes_mallows_score(true_labels, pred_labels)  # Fowlkes Mallows Index (0 to 1)
    precision, recall, f1 = pairwise_metrics(orcids, groups)

    return dict(simrels_accuracy=simrels_accuracy,
                singletons_percentage=singletons,
                # ARI=ari,
                # NMI=nmi,
                # FMI=fmi,
                precision=precision,
                recall=recall,
                F1=f1)


def compute_group_pairwise_metrics(orcids, groups):
    orc = orcids.cpu().numpy()
    grp = groups.cpu().numpy()

    # True pairs (ground truth)
    true_pairs = 0
    for c in np.unique(orc):
        idx = np.where(orc == c)[0]
        if len(idx) > 1:
            true_pairs += len(idx) * (len(idx) - 1) // 2

    # Predicted pairs
    pred_pairs = []
    for g in np.unique(grp):
        idx = np.where(grp == g)[0]
        if len(idx) > 1:
            # tutte le coppie (i,j) nel cluster predetto
            for i in range(len(idx)):
                for j in range(i+1, len(idx)):
                    pred_pairs.append((idx[i], idx[j]))

    pred_pairs_set = set(pred_pairs)

    TP = 0
    for i, j in pred_pairs_set:
        if orc[i] == orc[j]:
            TP += 1

    FP = len(pred_pairs_set) - TP
    FN = true_pairs - TP

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return precision, recall, f1


def get_non_singleton_mask(groups):
    unique_ids, counts = groups.unique(return_counts=True)
    non_singleton_cluster_ids = unique_ids[counts > 1]
    return torch.isin(groups, non_singleton_cluster_ids)


def count_singletons(groups):
    uniques, counts = groups.unique(return_counts=True)
    singletons = (counts == 1).sum().item()
    return singletons


dataset = PubmedSubgraph(dataset_name="Pubmed Subgraph",
                         url=DATASET_URL,
                         raw_dir=RAW_DIR,
                         save_dir=DATA_DIR)

# GRAPH
orcids_graph = dataset.get_orcids_graph()
# SIMILARITY RELATIONS
simrels = torch.load(SIMRELS_FOR_TESTING_PATH)
correct_simrels_mask = orcids_graph.has_edges_between(simrels[:, 0], simrels[:, 1]).long()
scores = torch.load(SIMRELS_SCORES_PATH)
# NODES
orcids = torch.load(ORCIDS_PATH)
groups = connected_components_from_simrels(len(orcids), simrels)

print("Singleton ORCIDs: ", count_singletons(orcids))
print("Singleton groups: ", count_singletons(groups))
print("Number of authors: ", len(orcids))
print("Number of simrels; ", len(simrels))
print("Number of groups: ", torch.unique(groups).numel())

results = {}
results_excluding_singleton_orcids = {}
for th in [i * 0.1 for i in range(11)]:
    pruned_simrels = prune_simrels(simrels, scores, th)
    correct_simrels_mask = orcids_graph.has_edges_between(pruned_simrels[:, 0], pruned_simrels[:, 1]).long()
    groups = connected_components_from_simrels(len(orcids), pruned_simrels)
    non_singleton_groups_mask = get_non_singleton_mask(groups)
    # results[th] = compute_statistics(pruned_simrels, correct_simrels_mask, orcids[non_singleton_groups_mask], groups[non_singleton_groups_mask])
    results[th] = compute_statistics(pruned_simrels, correct_simrels_mask, orcids, groups)

plot_results(results)
