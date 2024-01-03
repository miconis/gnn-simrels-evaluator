import itertools
from datetime import datetime
import random
from src.dgl_graph.dataset import PubmedSubgraph
from src.utils.utility import *
import warnings
from src.utils.config import *
import numpy as np
from copy import deepcopy
import torch
import faiss
from sklearn.cluster import DBSCAN
import json

warnings.filterwarnings("ignore")
random.seed(1234)
np.random.seed(1234)
os.environ["DGLBACKEND"] = "pytorch"
min_valid_loss = np.inf
embeddings_model_path = "./log/models/GS3Agg/GS3Agg-epoch353.ckpt.pth"
current_date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

dataset = PubmedSubgraph(dataset_name="Pubmed Subgraph",
                         subgraph_base_path="../dataset/processed_pubmed_subgraph",
                         neg_etype='equates',
                         raw_dir="../dataset/pubmed_subgraph/raw",
                         save_dir="../dataset/pubmed_subgraph/processed")

sc = dataset.get_sc()

# collect support tensors (for computations)
orcid_labels = dataset.get_labels()
key_labels = dataset.get_keys()
indexes = (dataset.get_labels_block_size_mask() > 1).nonzero(as_tuple=True)  # to chose which element to take into account
malformed_index = (dataset.get_wellformed_mask() == 0).nonzero(as_tuple=True)
dirty_key_labels = deepcopy(key_labels)
dirty_key_labels[malformed_index] = torch.IntTensor([i for i in range(-malformed_index[0].shape[0], 0)])  # put malformed elements in different groups

# compute statistics for authors
print_authors_stats(orcid_labels[indexes])

# compute statistics for a random blocking
print_blocking_stats(None, orcid_labels[indexes], torch.randint(0, 150000, (orcid_labels.size(dim=0),))[indexes], "RANDOM BLOCKING")

# compute statistics for perfect LNFI blocking
print_blocking_stats(None, orcid_labels[indexes], key_labels[indexes], "PERFECT BLOCKING")

# compute statistics for real-case LNFI blocking (missing keys)
print_blocking_stats(None, orcid_labels[indexes], dirty_key_labels[indexes], "REAL-CASE BLOCKING")

full_graph = dataset.get_graph()[0]

potentially_equates_graph, colleague_graph, citation_graph, collaboration_graph = dataset.get_node_embeddings_graphs()

node_features = full_graph.ndata["feat"]["author"]

embeddings_model = GS3Agg(768, 100)
predictor = EuclideanPredictor()
optimizer = torch.optim.Adam(itertools.chain(embeddings_model.parameters(), predictor.parameters()), lr=LEARNING_RATE)

load_checkpoint(embeddings_model, optimizer, embeddings_model_path)

# create embeddings
embeddings_model.eval()
# node_embeddings = embeddings_model(g, node_features).detach().numpy()
node_embeddings = embeddings_model(potentially_equates_graph, colleague_graph, citation_graph, collaboration_graph, node_features).detach()

# FAISS BLOCKING
nfeatures = node_embeddings.shape[1]

# just to check size of groups
labels_size = torch.load("../dataset/pubmed_subgraph/processed/label_block_size_mask.pt")
print(labels_size[0:5])

# PRODUCT QUANTIZATION
# ncentroids = 5  # number of cells to split the full index
# m = 25  # number of subquantizers (nfeatures % m = 0) -> higher is more accurate but more memory expensive
# quantizer = faiss.IndexFlatL2(nfeatures)
# code_size = 8  # typically a power of 2 between 8 and 64
# index = faiss.IndexIVFPQ(quantizer, nfeatures, ncentroids, m, code_size)
# index.train(node_embeddings)  # train the model to get centroids
# INDEX FLAT
index = faiss.IndexFlatL2(nfeatures)

index.add(node_embeddings)  # add the embeddings to the index

# evaluation
author_index = 0
k = 20 # number of nearest neighbors
labels = torch.load("../dataset/pubmed_subgraph/processed/author_labels.pt")
correct_indexes = (labels == labels[author_index]).nonzero()
print("Correct indexes: ", correct_indexes.reshape(correct_indexes.shape[1], -1).tolist()[0])
authors_rdd = sc.textFile("../dataset/pubmed_subgraph/raw/authors").map(eval).map(lambda x: (json.loads(x[0])['key'], x[1]))  # (key, index)
D, I = index.search(node_embeddings[author_index:author_index+1], k)
close_author_index = I[0]
print("Inferred indexes: ", close_author_index)
close_authors_labels = authors_rdd.filter(lambda x: x[1] in close_author_index).map(lambda x: x[0]).collect()
print(close_authors_labels)

exit()

gnn_labels = transitive_closure(I)

print_blocking_stats(None, orcid_labels, gnn_labels, "GNN-FAISS-TRANSITIVE BLOCKING")
