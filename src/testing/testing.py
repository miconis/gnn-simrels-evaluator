from src.dgl_graph.dataset import PubmedSubgraph
import json
import torch
import os

from src.utils.config import (
    AUTHORS_SIMREL_DIR,
    BEST_MODEL_PATH,
    DATASET_URL,
    DATA_DIR,
    H_FEATURES,
    INPUT_FEATURES,
    LEARNING_RATE,
    RAW_DIR,
    SIMRELS_FOR_TESTING_PATH,
    SIMRELS_SCORES_PATH,
)
from src.utils.models import AttentiveGraphSAGE4
from src.utils.utility import load_checkpoint

os.environ.pop("SPARK_HOME", None) # to prevent conflicts

dataset = PubmedSubgraph(dataset_name="Pubmed Subgraph",
                         url=DATASET_URL,
                         raw_dir=RAW_DIR,
                         save_dir=DATA_DIR)

sc = dataset.get_sc()
spark = dataset.get_spark()

full_graph = dataset.get_graph()
simrels_tensor = torch.load(SIMRELS_FOR_TESTING_PATH)
simrels_graph = dataset.get_simrels_graph()
simrels_graph.remove_edges(torch.arange(simrels_graph.num_edges()))
simrels_graph.add_edges(simrels_tensor[:, 0], simrels_tensor[:, 1])

model_path = BEST_MODEL_PATH

orcids_graph = dataset.get_orcids_graph()
correct_simrel_mask = orcids_graph.has_edges_between(simrels_graph.edges()[0], simrels_graph.edges()[1]).long()

print("Number of simrels: ", simrels_graph.num_edges())
print("Correct: ", correct_simrel_mask.sum().item())
print("Wrong: ", correct_simrel_mask.shape[0] - correct_simrel_mask.sum().item())

# extract homogeneous graphs
potentially_equates_graph, colleague_graph, citation_graph, collaboration_graph = dataset.get_node_embeddings_graphs()
node_features = full_graph.ndata["feat"]["author"]

model = AttentiveGraphSAGE4(in_feats=INPUT_FEATURES,
                            h_feats=H_FEATURES,
                            potentially_equates_graph=potentially_equates_graph,
                            colleague_graph=colleague_graph,
                            citation_graph=citation_graph,
                            collaboration_graph=collaboration_graph)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

load_checkpoint(model, optimizer, model_path)

model.eval()
edge_scores = model(simrels_graph, node_features)

os.makedirs(os.path.dirname(SIMRELS_SCORES_PATH), exist_ok=True)
torch.save(edge_scores, SIMRELS_SCORES_PATH)

# authors = sc.textFile(RAW_DIR + "/authors").map(eval).map(lambda x: (json.loads(x[0]), x[1])).map(lambda x: (x[0]['id'], x[1]))  # (id, index)
# simrels = spark.read.load(AUTHORS_SIMREL_DIR).rdd.map(lambda x: (x['source'], x['target']))
#
# simrels = simrels.join(authors)
# simrels = simrels.map(lambda x: (x[1][0], x[1][1]))
# simrels = simrels.join(authors)
# simrels = simrels.map(lambda x: [x[1][0], x[1][1]])
#
# simrels_tensor = torch.LongTensor(simrels.collect())
# torch.save(simrels_tensor, SIMRELS_FOR_TESTING_PATH)
#
# print(simrels.take(1))
# print(simrels.count())
