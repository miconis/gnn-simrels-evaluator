import torch
from pytorch_metric_learning import losses
from models import *
from utility import *

# il problema è che se uso la loss così generando il grafo a caso, ottengo che la distanza euclidea di quelli con lo stesso orcid sia più bassa della distanza euclidea degli altri presi a caso
pos_scores = torch.FloatTensor([6,6])
neg_scores = torch.FloatTensor([202,202,1,2,3,4,5])

acc = compute_auc(pos_scores, neg_scores)
print(acc)

loss = contrastive_loss(pos_scores, neg_scores, margin=2.0)
print(loss)

exit()
a = torch.FloatTensor(
   [[5, 5],
    [5, 3],
    [3, 5],
    [6, 4],
    [3, 7]])

a = torch.FloatTensor([[1,2,3],[0,5,6]])

# print(a[a[:,0].sort()[1]])

print(torch.FloatTensor([1,2,3,4,5,6,7])[0:2])

exit()

scores = torch.FloatTensor([0.4,0.6,0.3])
labels = torch.FloatTensor([0,1,0])

a = (scores.round() == labels).sum().item()

print(a)
print(a.type())
print(a.long())

exit()
node_embeddings1 = torch.FloatTensor([[110,111,112,113], [120,121,122,123], [130,131,132,133]])
node_embeddings2 = torch.FloatTensor([[210,211,212,213], [220,221,222,223], [230,231,232,233]])
node_embeddings3 = torch.FloatTensor([[310,311,312,313], [320,321,322,323], [330,331,332,333]])

mpa = MetapathAggregation(m_feats=4)

res = mpa(node_embeddings1, node_embeddings2, node_embeddings3)

print(res)

exit()

def criterion(x1, x2, label, margin: float = 1.0):
    """
    Contrastive loss
    """
    dist = torch.nn.functional.pairwise_distance(x1,x2)

    loss = (1 - label) * torch.pow(dist, 2) + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    loss = torch.mean(loss)
    return loss


x1 = torch.FloatTensor([0,0,0])
x2 = torch.FloatTensor([100,0,0])

print(criterion(x1, x2, 1, 101))

x1 = torch.FloatTensor([[0,0,0], [1,1,1], [2,2,2]])
x2 = torch.FloatTensor([[0,0,0], [1,1,1], [2,2,2]])
label = torch.FloatTensor([0,0,0])

print(criterion(x1, x2, label))
exit()

from utility import *

t = torch.FloatTensor([[0,0,0], [1,1,1], [2,2,2]])

print(F.normalize(t, dim=1))
exit()

t1 = torch.FloatTensor([0,1,2])

print(F.cosine_similarity(torch.FloatTensor([0, 0, 1]), torch.FloatTensor([0,0,100]), dim=0))
print(F.cosine_similarity(torch.FloatTensor([0, 0, 1]), torch.FloatTensor([0,0,1]), dim=0))

exit()

print(t.unsqueeze(1))

print(t.unsqueeze(0))

s = F.cosine_similarity(t.unsqueeze(1), t.unsqueeze(0), dim=2)
print(s)
exit()
class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        # z_i = F.normalize(emb_i, dim=1)
        # z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([emb_i, emb_j], dim=0)
        # similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        similarity_matrix = F.pairwise_distance(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


pos_src = torch.IntTensor([[0,1,2], [0,1,2]])
pos_dst = torch.IntTensor([[0,1,2], [0,1,2]])

neg_src = torch.IntTensor([[0,0,0], [100,100,100]])
neg_dst = torch.IntTensor([[100,100,100], [0,0,0]])

pos_score = F.pairwise_distance(pos_src, pos_dst)
neg_score = F.pairwise_distance(neg_src, neg_dst)

pos_score = pos_score / neg_score.max()
neg_score = neg_score / neg_score.max()
print(pos_score)
print(neg_score)

loss = binary_cross_entropy(pos_score, neg_score)
# loss = binary_cross_entropy_with_logits(pos_score, neg_score)
print(loss)

loss_func = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
loss = loss_func

print(loss)

exit()

def transitive_closure(neighbors):
    seen = set()
    labels = torch.zeros(len(neighbors), dtype=torch.int)
    for src in range(0, len(neighbors)):
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


# graph = [[0,1,2,3], [1], [2,1], [3,4,5], [4,3,5], [5,3,4,7], [6, 8], [7], [8,9], [9]]
graph = [[0], [1], [2], [3], [4], [5]]

components = transitive_closure(graph)
print(components)
