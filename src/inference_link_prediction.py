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
from dgl.data.utils import save_graphs, load_graphs


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
spark = dataset.get_spark()

full_graph = dataset.get_graph()[0]
simrels_graph = dataset.get_simrels_graph()
orcids_graph = dataset.get_orcids_graph()
correct_simrel_mask = orcids_graph.has_edges_between(simrels_graph.edges()[0], simrels_graph.edges()[1]).long()

print("Number of simrels: ", simrels_graph.num_edges())
print("Correct: ", correct_simrel_mask.sum().item())
print("Wrong: ", correct_simrel_mask.shape[0] - correct_simrel_mask.sum().item())

potentially_equates_graph, colleague_graph, citation_graph, collaboration_graph = dataset.get_node_embeddings_graphs()
node_features = full_graph.ndata["feat"]["author"]

embeddings_model = GS3Agg(768, 100)
predictor = EuclideanPredictor()
optimizer = torch.optim.Adam(itertools.chain(embeddings_model.parameters(), predictor.parameters()), lr=LEARNING_RATE)

load_checkpoint(embeddings_model, optimizer, embeddings_model_path)

# create embeddings
embeddings_model.eval()
predictor.eval()
node_embeddings = embeddings_model(potentially_equates_graph, colleague_graph, citation_graph, collaboration_graph, node_features).detach()

# to check how correct relations are distributed
scores = predictor(orcids_graph, node_embeddings)
res = torch.cat((scores.reshape(-1, 1), orcids_graph.has_edges_between(orcids_graph.edges()[0], orcids_graph.edges()[1]).long().reshape(-1, 1)), dim=1)

# to check simrels
# scores = predictor(simrels_graph, node_embeddings)
# res = torch.cat((scores.reshape(-1, 1), correct_simrel_mask.reshape(-1, 1)), dim=1)
res = res[res[:, 0].sort()[1]]

x = res[:, 0].reshape(1, -1).numpy()
y = res[:, 1].reshape(1, -1).numpy()

plt.scatter(x, y, s=0.01)

plt.title("Customized Scatter Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.show()

exit()

wrong_indexes = (scores < 0.0001).nonzero(as_tuple=True)

simrels = simrels_graph.edges()

wrong_src = simrels[0][wrong_indexes].tolist()
wrong_dst = simrels[1][wrong_indexes].tolist()

# basta controllare se esiste tra gli orcid, se non esiste è sbagliato

authors_rdd = sc.textFile("../dataset/pubmed_subgraph/raw/authors").map(eval).map(lambda x: (json.loads(x[0]), x[1]))  # (key, index)

for i in range(0, len(wrong_src)):
    source_author = authors_rdd.filter(lambda x: x[1] == wrong_src[i]).map(lambda x: x[0]).take(1)[0]
    target_author = authors_rdd.filter(lambda x: x[1] == wrong_dst[i]).map(lambda x: x[0]).take(1)[0]
    print("wrong couple ", i+1)
    print(source_author['key'], source_author['orcid'])
    print(target_author['key'], target_author['orcid'])

