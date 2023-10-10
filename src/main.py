import random

import dgl
import numpy as np
from src.dgl_graph.dataset import PubmedSubgraph
from src.utils.models import *
from sklearn.metrics import roc_auc_score
import itertools

random.seed(1234)
np.random.seed(1234)
num_links = 4000000

dataset = PubmedSubgraph(dataset_name="Pubmed Subgraph",
                         subgraph_base_path="../dataset/processed_pubmed_subgraph",
                         num_links=num_links,
                         raw_dir="../dataset/pubmed_subgraph/raw",
                         save_dir="../dataset/pubmed_subgraph/processed")


g = dataset.get_graph()[0]
neg_links = dataset.get_neg_links()

# CREATE POSITIVE AND NEGATIVE EDGES
# extract the potentially equates graph (to take positive edges)
g_pe = dgl.edge_type_subgraph(g, etypes=['potentially_equates'])
train_size = int(num_links * 0.8)
test_size = num_links - train_size
# positive edges (random from potentially equivalent graph)
pos_u, pos_v = g_pe.edges()
eids = np.random.permutation(np.arange(g_pe.num_edges()))  # vector of random positions
train_pos_u, train_pos_v = pos_u[eids[:train_size]], pos_v[eids[:train_size]]
test_pos_u, test_pos_v = pos_u[eids[train_size:num_links]], pos_v[eids[train_size:num_links]]
# negative edges (random from the nodes with different orcids)
neg_u, neg_v = neg_links[0], neg_links[1]
test_neg_u, test_neg_v = neg_u[:test_size], neg_v[:test_size]
train_neg_u, train_neg_v = neg_u[test_size:], neg_v[test_size:]
train_g = dgl.remove_edges(g_pe, eids[train_size:])  # create train graph removing test edges

# CONSTRUCT POSITIVE GRAPH AND NEGATIVE GRAPH
train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g_pe.num_nodes())
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g_pe.num_nodes())

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g_pe.num_nodes())
test_neg_g = dgl.graph((test_neg_u, test_neg_u), num_nodes=g_pe.num_nodes())

print("Training Graph:", train_g)
print("TRAINING EDGES")
print("Positive Train Edges:", len(train_pos_u))
print("Positive Test Edges:", len(test_pos_u))
print("Negative Train Edges:", len(train_neg_u))
print("Negative Test Edges:", len(test_neg_u))


# TRAINING LOOP
input_features = train_g.ndata["feat"].shape[1]
hidden_feats = 16
learning_rate = 0.01
n_epochs = 10

model = GraphSAGE(in_feats=train_g.ndata["feat"].shape[1], h_feats=hidden_feats)
pred = MLPPredictor(h_feats=hidden_feats)
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=learning_rate)


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])  # concatenate inputs
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])  # 1 positive, 0 negative
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)


all_logits = []
for e in range(n_epochs):
    # compute nodes representations with GraphSAGE
    h = model(train_g, train_g.ndata["feat"])
    # predict positives and negatives separately
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    # compute the loss
    loss = compute_loss(pos_score, neg_score)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("In epoch {}, loss: {}".format(e, loss))


# CHECK RESULTS
with torch.no_grad():
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h)
    print("AUC", compute_auc(pos_score, neg_score))
