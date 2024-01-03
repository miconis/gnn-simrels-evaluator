import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch.conv import SAGEConv
import dgl.function as fn


# EDGE PREDICTORS
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            return g.edata["score"][:, 0]


class EuclideanPredictor(nn.Module):
    def euclidean_distance(self, edges):
        dist = F.pairwise_distance(edges.src["h"], edges.dst["h"])
        return {"score": dist}
    def forward(self, g, h):
        g.ndata["h"] = h
        g.apply_edges(self.euclidean_distance)
        return g.edata["score"]


class EdgeScorer(nn.Module):
    def __init__(self, n_feats):
        super(EdgeScorer, self).__init__()
        self.linear1 = nn.Linear(n_feats * 2, 1)

    def apply_edges(self, edges):
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        h = self.linear1(h)
        return {"score": F.sigmoid(h).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]


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
    def __init__(self, m_feats):
        super(MetapathAttention, self).__init__()
        self.linear1 = nn.Linear(in_features=m_feats, out_features=1)
        self.linear2 = nn.Linear(in_features=m_feats, out_features=1)
        self.linear3 = nn.Linear(in_features=m_feats, out_features=1)
        self.linear4 = nn.Linear(in_features=m_feats, out_features=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, node_embeddings1, node_embeddings2, node_embeddings3, node_embeddings4):
        l1 = self.linear1(node_embeddings1)
        l2 = self.linear2(node_embeddings2)
        l3 = self.linear3(node_embeddings3)
        l4 = self.linear4(node_embeddings4)
        return self.softmax(torch.cat((l1, l2, l3, l4), dim=1))


class MetapathGating(nn.Module):
    def __init__(self):
        super(MetapathGating, self).__init__()
        self.glu1 = nn.GLU()
        self.glu2 = nn.GLU()
        self.glu3 = nn.GLU()
        self.glu4 = nn.GLU()

    def forward(self, node_embeddings1, node_embeddings2, node_embeddings3, node_embeddings4):
        g1 = self.glu1(node_embeddings1)
        g2 = self.glu2(node_embeddings2)
        g3 = self.glu3(node_embeddings3)
        g4 = self.glu4(node_embeddings4)
        return g1, g2, g3, g4


class MetapathAggregation(nn.Module):
    def __init__(self, m_feats):
        super(MetapathAggregation, self).__init__()
        self.gating = MetapathGating()
        self.attention = MetapathAttention(m_feats)

    def forward(self, node_embeddings1, node_embeddings2, node_embeddings3, node_embeddings4):
        g1, g2, g3, g4 = self.gating(node_embeddings1, node_embeddings2, node_embeddings3, node_embeddings4)
        weigths = self.attention(node_embeddings1, node_embeddings2, node_embeddings3, node_embeddings4)
        return (torch.mul(g1, weigths[:, 0].reshape(-1, 1)) +
                torch.mul(g2, weigths[:, 1].reshape(-1, 1)) +
                torch.mul(g3, weigths[:, 2].reshape(-1, 1)) +
                torch.mul(g4, weigths[:, 3].reshape(-1, 1)))


class GS3Agg(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GS3Agg, self).__init__()
        self.sage1 = GraphSAGE(in_feats=in_feats, h_feats=h_feats)
        self.sage2 = GraphSAGE(in_feats=in_feats, h_feats=h_feats)
        self.sage3 = GraphSAGE(in_feats=in_feats, h_feats=h_feats)
        self.sage4 = GraphSAGE(in_feats=in_feats, h_feats=h_feats)

        self.metapath_aggregator = MetapathAggregation(m_feats=h_feats)

    def forward(self, graph1, graph2, graph3, graph4, node_features):
        h1 = self.sage1(graph1, node_features)
        h2 = self.sage2(graph2, node_features)
        h3 = self.sage3(graph3, node_features)
        h4 = self.sage4(graph4, node_features)

        node_embeddings = self.metapath_aggregator(h1, h2, h3, h4)

        return node_embeddings


class GS3LSTM(nn.Module):
    def __init__(self, in_feats, h_feats, num_layers, dropout):
        super(GS3LSTM, self).__init__()
        self.dropout = dropout
        self.sage1 = GraphSAGE(in_feats=in_feats, h_feats=h_feats)
        self.sage2 = GraphSAGE(in_feats=in_feats, h_feats=h_feats)
        self.sage3 = GraphSAGE(in_feats=in_feats, h_feats=h_feats)
        self.sage4 = GraphSAGE(in_feats=in_feats, h_feats=h_feats)

        self.lstm = nn.LSTM(input_size=h_feats, hidden_size=h_feats, num_layers=num_layers, batch_first=False)

    def forward(self, graph1, graph2, graph3, graph4, node_features):
        h1 = self.sage1(graph1, node_features)
        h2 = self.sage2(graph2, node_features)
        h3 = self.sage3(graph3, node_features)
        h4 = self.sage4(graph4, node_features)

        hidden_global = torch.stack((h1,h2,h3,h4), dim=0)

        hidden_global, (last_hidden_lstm, _) = self.lstm(hidden_global)
        node_embeddings = F.dropout(last_hidden_lstm[-1, :, :], p=self.dropout, training=self.training)

        return node_embeddings
