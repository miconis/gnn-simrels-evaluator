import hashlib
import json

import dgl
from unidecode import unidecode
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm
import torch.nn.functional as F

from src.utils.models import *


def create_author_id(fullname, pub_id):
    return "00|author______::" + hashlib.md5(pub_id.encode("utf-8") + fullname.encode("utf-8")).hexdigest()


def lnfi(author):
    try:
        return (unidecode(author['surname'] + author['name'][0])).lower().strip()
    except:
        return ""


def is_author_wellformed(author):
    return author['name'] != "" and author['surname'] != ""


def replace_char(s, pos, c):
    s = list(s)
    s[pos] = c
    return "".join(s)


def save_rdd(rdd, path):
    rdd.map(lambda x: json.dumps(x, ensure_ascii=False)).saveAsTextFile(path=path, compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")


def construct_negative_graph(graph, k):
    src, dst = graph.edges()

    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(), (len(src) * k,))
    return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())


def binary_cross_entropy_with_logits(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])

    return F.binary_cross_entropy_with_logits(scores, labels)


def binary_cross_entropy(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    )

    return F.binary_cross_entropy(scores, labels.float())


def margin_loss(pos_score, neg_score):
    n_edges = pos_score.shape[0]
    return (1 - pos_score.view(n_edges, -1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()


def contrastive_loss(pos_score, neg_score, margin: float = 1.0):
    # force the positive scores to stay close to 0 and the negative to be higher than the positives for more than the margin
    scores = torch.cat([pos_score, neg_score])  # euclidean distances
    labels = torch.cat([torch.zeros(pos_score.shape[0]), torch.ones(neg_score.shape[0])])

    loss = (1 - labels) * torch.pow(scores, 2) + (labels) * torch.pow(torch.clamp(margin - scores, min=0.0), 2)
    loss = torch.mean(loss)
    return loss


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().numpy()
    labels = torch.cat(
        [torch.zeros(pos_score.shape[0]), torch.ones(neg_score.shape[0])]
    ).numpy()
    return metrics.roc_auc_score(labels, scores)


def compute_acc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    correct = (scores.round() == labels).sum().item()
    return correct / scores.shape[0] * 100


def save_checkpoint(model, optimizer, epoch, logdir):
    delete_model(f"{logdir}{model.__class__.__name__}-epoch*.pth")
    model_path = f"{logdir}{model.__class__.__name__}-epoch{epoch}.ckpt.pth"

    torch.save({
        "optimizer": optimizer.state_dict(),
        "model": model.state_dict()
    }, model_path)
    return model_path


def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def delete_model(pattern):
    for f in glob.glob(pattern):
        os.remove(f)


def dbscan_plot(X, dbscan):
    dbscan_labels = dbscan.labels_
    unique_labels = set(dbscan_labels)
    core_samples_mask = np.zeros_like(dbscan_labels, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]
        class_member_mask = dbscan_labels == k
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6
        )

    n_clusters_ = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise_ = list(dbscan_labels).count(-1)
    plt.title(f"{n_clusters_} blocks and {n_noise_} noise points")
    plt.show()


def compute_reduction_ratio(labels):
    n_entities = labels.shape[0]  # number of entities
    p = n_entities*(n_entities-1)/2  # number of pair-wise comparisons when no blocking is applied

    blocks_counts = labels.unique(return_counts=True)[1]
    blocks_pairs = blocks_counts.apply_(lambda x: (x*(x-1))/2)
    pb = torch.sum(blocks_pairs)  # number of pair-wise comparisons when blocking is applied

    return 1 - (pb/p)


def compute_number_of_blocks(labels):
    return labels.unique(return_counts=True)[0].size(dim=0)


def plot_block_stats(labels):
    block_stats = labels.unique(return_counts=True)
    block_counts = block_stats[1]
    block_size_stats = block_counts.unique(return_counts=True)
    x = block_size_stats[0].tolist()
    y = block_size_stats[1].tolist()
    plt.bar(x,y,align="center")
    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.show()


def compute_singleton_block(labels):
    block_stats = labels.unique(return_counts=True)
    block_counts = block_stats[1]
    block_size_stats = block_counts.unique(return_counts=True)
    singleton_stat_index = (block_size_stats[0] == 1).nonzero(as_tuple=True)[0]
    n_singleton = block_size_stats[1][singleton_stat_index]
    return 0 if n_singleton.size(dim=0) == 0 else n_singleton[0]


def print_blocking_stats(X, labels, labels_pred, stats_name):
    print(f"***** {stats_name} STATISTICS *****")
    n_blocks = compute_number_of_blocks(labels_pred)
    n_singleton_blocks = compute_singleton_block(labels_pred)
    homogeneity = metrics.homogeneity_score(labels, labels_pred)
    completeness = metrics.completeness_score(labels, labels_pred)
    rand_index = metrics.rand_score(labels, labels_pred)
    adjusted_rand_index = metrics.adjusted_rand_score(labels, labels_pred)
    vmeasure = metrics.v_measure_score(labels, labels_pred)
    normalized_mutual_information = metrics.normalized_mutual_info_score(labels, labels_pred)
    print(f"Number of blocks: {n_blocks} ({n_singleton_blocks} singleton)")
    print(f"Homogeneity: {homogeneity}")  # blocks contain only data of the same class
    print(f"Completeness: {completeness}")  # member of a class are contained in the same block
    print(f"V-measure: {vmeasure}")  # harmonic mean between homogeneity and completeness
    print(f"RI: {rand_index}")  # Rand Index (number of agreeing pairs against total number of pairs)
    print(f"ARI: {adjusted_rand_index}")  # Adjusted Rand Index (Rand Index adjusted for chance)
    print(f"NMI: {normalized_mutual_information}")  # Normalized Mutual Information ()
    print(f"==================================================")


def print_authors_stats(labels):
    print("***** AUTHORS STATISTICS *****")
    n_authors = compute_number_of_blocks(labels)
    n_raw_authors = labels.size(dim=0)
    n_singleton_authors = compute_singleton_block(labels)
    print(f"Number of authors: {n_authors} ({n_singleton_authors} singleton; {n_authors - n_singleton_authors} groups)")
    print(f"Number of raw authors: {n_raw_authors}")
    print(f"==================================================")


def transitive_closure(neighbors):
    seen = set()
    labels = torch.zeros(len(neighbors), dtype=torch.int)
    for src in tqdm(range(0, len(neighbors))):
        if src not in seen:
            connected_group, seen = get_connected_group(src, seen, neighbors)
            labels[connected_group] = src
    return labels


def get_connected_group(src, seen, neighbors):
    result = []
    nodes = set([src])
    while nodes:
        node = nodes.pop()
        seen.add(node)
        nodes.update(n for n in neighbors[node] if n not in seen)
        result.append(node)
    return result, seen


def conf_matrix_metrics(conf_matrix):
    accuracy = (conf_matrix[1, 1] + conf_matrix[0, 0]) / (conf_matrix[0, 0] + conf_matrix[0, 1] + conf_matrix[1, 0] + conf_matrix[1, 1])
    tpr = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])  # recall
    tnr = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    fnr = conf_matrix[1, 0] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    fpr = conf_matrix[0, 1] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
    balanced_accuracy = (tpr + tnr) / 2
    f1_score = 2*conf_matrix[1, 1] / (2*conf_matrix[1, 1] + conf_matrix[0, 1] + conf_matrix[1, 0])
    return f"Accuracy (A): {accuracy}\nBalanced Accuracy (BA): {balanced_accuracy}\nTrue Positive Rate (TPR) - Recall: {tpr}\nTrue Negative Rate (TNR): {tnr}\nFalse Negative Rate (FNR): {fnr}\nFalse Positive Rate (FPR): {fpr}\nPrecision: {precision}\nF1-Score: {f1_score}"


def plot_confusion_matrix(conf_matrix):
    conf_matrix = conf_matrix.numpy().tolist()
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(2):
        for j in range(2):
            ax.text(x=j, y=i, s=conf_matrix[i][j], va="center", ha="center")

    plt.xlabel("Predicted", fontsize=18)
    plt.ylabel("Label", fontsize=18)
    plt.show()
    return fig