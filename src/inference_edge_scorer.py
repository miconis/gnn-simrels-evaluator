from datetime import datetime
import random
from src.dgl_graph.dataset import PubmedSubgraph
from src.utils.utility import *
import warnings
from src.utils.config import *
import numpy as np
import torch
from torchmetrics.classification import BinaryConfusionMatrix


warnings.filterwarnings("ignore")
random.seed(1234)
np.random.seed(1234)
os.environ["DGLBACKEND"] = "pytorch"
min_valid_loss = np.inf
model_path = "./log/models/GraphSAGE4WeightedMetapathMLPEdgeScorer/05-01-2024_17-33-43/GraphSAGE4WeightedMetapathMLPEdgeScorer-epoch51.ckpt.pth"
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

# extract homogeneous graphs
potentially_equates_graph, colleague_graph, citation_graph, collaboration_graph = dataset.get_node_embeddings_graphs()
node_features = full_graph.ndata["feat"]["author"]

model = GraphSAGE4WeightedMetapathMLPEdgeScorer(in_feats=768,
                                                h_feats=100,
                                                potentially_equates_graph=potentially_equates_graph,
                                                colleague_graph=colleague_graph,
                                                citation_graph=citation_graph,
                                                collaboration_graph=collaboration_graph)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

load_checkpoint(model, optimizer, model_path)

model.eval()
edge_scores = model(simrels_graph, node_features)

bcm = BinaryConfusionMatrix(threshold=0.5)

confusion_matrix = bcm(edge_scores, correct_simrel_mask)

print(conf_matrix_metrics(confusion_matrix))

plot_confusion_matrix(confusion_matrix)

authors_rdd = sc.textFile("../dataset/pubmed_subgraph/raw/authors").map(eval).map(lambda x: (json.loads(x[0])['orcid'], x[1]))  # (orcid, index)

print("Example of inferred correct edges")
correct_edges = (edge_scores > 0.99).nonzero(as_tuple=False)[0:5].squeeze(1).tolist()
for i in correct_edges:
    source_index = simrels_graph.edges()[0][i]
    target_index = simrels_graph.edges()[1][i]
    source = authors_rdd.filter(lambda x: x[1] == source_index).map(lambda x: x[0]).take(1)[0]
    target = authors_rdd.filter(lambda x: x[1] == target_index).map(lambda x: x[0]).take(1)[0]
    print(source, " ---> ", target)


print("Example of inferred wrong edges")
wrong_edges = (edge_scores < 0.00001).nonzero(as_tuple=False)[5:10].squeeze(1).tolist()
for i in wrong_edges:
    source_index = simrels_graph.edges()[0][i]
    target_index = simrels_graph.edges()[1][i]
    source = authors_rdd.filter(lambda x: x[1] == source_index).map(lambda x: x[0]).take(1)[0]
    target = authors_rdd.filter(lambda x: x[1] == target_index).map(lambda x: x[0]).take(1)[0]
    print(source, " ---> ", target)
