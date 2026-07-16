import random
import warnings
from src.dataset import PubmedSubgraph
from src.model import AttentiveGraphSAGE4
from src.utils.config import *
from src.utils.functions import *
os.environ.pop("SPARK_HOME", None)  # Prevent conflicts with PySpark's bundled Spark.

# EVALUATION PARAMETERS
THRESHOLDS = tuple(index / 10 for index in range(11))


warnings.filterwarnings("ignore")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

dataset = PubmedSubgraph(dataset_name="Pubmed Subgraph",
                         url=DATASET_URL,
                         raw_dir=RAW_DIR,
                         save_dir=DATA_DIR)

# collect simrels to be analyzed
full_graph = dataset.get_graph()
simrels_graph = dataset.get_dedup_graph()
simrels = torch.stack(
    simrels_graph.edges(etype=("author", "similar_for_dedup", "author")),
    dim=1
)

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
with torch.no_grad():
    scores = model(simrels_graph, node_features)
orcid_labels = build_orcid_labels(
    dataset.get_sc(),
    RAW_DIR,
    dataset.get_graph().num_nodes("author"),
)

print("Number of authors:", len(orcid_labels))
print("Number of similarity relations:", len(simrels))
print(
    "Singleton ORCIDs:",
    compute_singleton_block(orcid_labels).item() / len(orcid_labels),
)

results = {}
for threshold in THRESHOLDS:
    pruned_simrels = prune_similarity_relations(simrels, scores, threshold)
    correct_mask = dataset.get_orcids_graph().has_edges_between(
        pruned_simrels[:, 0],
        pruned_simrels[:, 1],
    ).long()
    groups = connected_components_from_simrels(len(orcid_labels), pruned_simrels)
    results[threshold] = compute_threshold_statistics(
        correct_mask,
        orcid_labels,
        groups,
    )
    print(
        f"threshold={threshold:.1f} "
        f"edges={len(pruned_simrels)} "
        f"F1={results[threshold]['F1']:.4f}"
    )

plot_threshold_results(results)
