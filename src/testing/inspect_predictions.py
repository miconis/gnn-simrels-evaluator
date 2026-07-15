import random
import warnings
from datetime import datetime

from torchmetrics.classification import BinaryConfusionMatrix

from src.dataset import PubmedSubgraph
from src.model import *
from src.utils.config import *
from src.utils.utility import *

warnings.filterwarnings("ignore")
random.seed(SEED)
np.random.seed(SEED)
os.environ["DGLBACKEND"] = "pytorch"
min_valid_loss = np.inf
current_date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
number_of_examples = 3
correct_threshold = 0.99
wrong_threshold = 0.00001


dataset = PubmedSubgraph(dataset_name="Pubmed Subgraph",
                         url=DATASET_URL,
                         raw_dir=RAW_DIR,
                         save_dir=DATA_DIR)

sc = dataset.get_sc()
spark = dataset.get_spark()

full_graph = dataset.get_graph()
simrels_graph = dataset.get_simrels_graph()
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

load_checkpoint(model, optimizer, BEST_MODEL_PATH)

model.eval()
edge_scores = model(simrels_graph, node_features)

bcm = BinaryConfusionMatrix(threshold=CUT_THRESHOLD)

confusion_matrix = bcm(edge_scores, correct_simrel_mask)

print(conf_matrix_metrics(confusion_matrix))

plot_confusion_matrix(confusion_matrix)

authors_rdd = sc.textFile(RAW_DIR + "/authors").map(eval).map(lambda x: (format_author_for_print(x[0]), x[1]))  # (orcid, index)

print("Example of inferred correct edges")
correct_edges = (edge_scores > correct_threshold).nonzero(as_tuple=False)[0:number_of_examples].squeeze(1).tolist()
for i in correct_edges:
    source_index = simrels_graph.edges()[0][i]
    target_index = simrels_graph.edges()[1][i]
    source = authors_rdd.filter(lambda x: x[1] == source_index).map(lambda x: x[0]).take(1)[0]
    target = authors_rdd.filter(lambda x: x[1] == target_index).map(lambda x: x[0]).take(1)[0]
    print(source, " ---> ", target)


print("Example of inferred wrong edges")
wrong_edges = (edge_scores < wrong_threshold).nonzero(as_tuple=False)[0:number_of_examples].squeeze(1).tolist()
for i in wrong_edges:
    source_index = simrels_graph.edges()[0][i]
    target_index = simrels_graph.edges()[1][i]
    source = authors_rdd.filter(lambda x: x[1] == source_index).map(lambda x: x[0]).take(1)[0]
    target = authors_rdd.filter(lambda x: x[1] == target_index).map(lambda x: x[0]).take(1)[0]
    print(source, " ---> ", target)
