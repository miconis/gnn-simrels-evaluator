import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch.conv import SAGEConv

torch.manual_seed(42)


# NODE EMBEDDINGS
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, h_feats, "mean")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


# METAPATH AGGREGATION
class MetapathAttention(nn.Module):
    def __init__(self, in_feats):
        super(MetapathAttention, self).__init__()
        self.linear1 = nn.Linear(in_features=in_feats, out_features=1)
        self.linear2 = nn.Linear(in_features=in_feats, out_features=1)
        self.linear3 = nn.Linear(in_features=in_feats, out_features=1)
        self.linear4 = nn.Linear(in_features=in_feats, out_features=1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, node_embeddings1, node_embeddings2, node_embeddings3, node_embeddings4):
        l1 = self.linear1(node_embeddings1)
        l2 = self.linear2(node_embeddings2)
        l3 = self.linear3(node_embeddings3)
        l4 = self.linear4(node_embeddings4)
        return self.softmax(torch.stack((l1, l2, l3, l4), dim=0))


class GraphSAGE4WeightedMetapathMLPEdgeScorer(nn.Module):
    def __init__(self, in_feats, h_feats, potentially_equates_graph, colleague_graph, citation_graph, collaboration_graph):
        super(GraphSAGE4WeightedMetapathMLPEdgeScorer, self).__init__()

        #graphs
        self.potentially_equates_graph = potentially_equates_graph
        self.colleague_graph = colleague_graph
        self.citation_graph = citation_graph
        self.collaboration_graph = collaboration_graph

        # node embedding layers
        self.sage1 = GraphSAGE(in_feats=in_feats, h_feats=h_feats)
        self.sage2 = GraphSAGE(in_feats=in_feats, h_feats=h_feats)
        self.sage3 = GraphSAGE(in_feats=in_feats, h_feats=h_feats)
        self.sage4 = GraphSAGE(in_feats=in_feats, h_feats=h_feats)

        # metapath attention
        self.metapath_attention = MetapathAttention(in_feats=h_feats)

        # edge classification layers
        self.linear1 = nn.Linear(in_features=h_feats * 2, out_features=h_feats)
        self.linear2 = nn.Linear(in_features=h_feats, out_features=1)

    def compute_loss(self, pos_scores, neg_scores):
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat(
            [torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])]
        )
        return F.binary_cross_entropy(scores, labels.float())

    def compute_accuracy(self, pos_scores, neg_scores):
        scores = torch.cat([pos_scores, neg_scores]).detach()
        labels = torch.cat([torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])])
        correct = (scores.round() == labels).sum().item()
        return correct / scores.shape[0] * 100

    def compute_node_embeddings(self, node_features):
        h1 = self.sage1(self.potentially_equates_graph, node_features)
        h2 = self.sage2(self.colleague_graph, node_features)
        h3 = self.sage3(self.citation_graph, node_features)
        h4 = self.sage4(self.collaboration_graph, node_features)

        weights = self.metapath_attention(h1, h2, h3, h4)
        node_embeddings_stack = torch.stack((h1, h2, h3, h4), dim=0)
        node_embeddings_stack = node_embeddings_stack * weights

        node_embeddings = torch.sum(node_embeddings_stack, dim=0)
        return node_embeddings

    def mlp_edge_predictor(self, edges):
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        h = self.linear1(h)
        h = F.relu(h)
        h = self.linear2(h)
        return {"score": F.sigmoid(h).squeeze(1)}

    def forward(self, simrels_graph, node_features):
        with simrels_graph.local_scope():
            simrels_graph.ndata["h"] = self.compute_node_embeddings(node_features)
            simrels_graph.apply_edges(self.mlp_edge_predictor)
            return simrels_graph.edata["score"]
