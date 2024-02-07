import copy
import random
import warnings
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from src.dgl_graph.dataset import PubmedSubgraph
from src.utils.config import *
from src.utils.utility import *
from torchmetrics.classification import BinaryConfusionMatrix

warnings.filterwarnings("ignore")
random.seed(1234)
np.random.seed(1234)
os.environ["DGLBACKEND"] = "pytorch"
min_valid_loss = np.inf
best_model_path = "./log/models/GraphSAGE4WeightedMetapathMLPEdgeScorer/04-01-2024_19-34-03/GraphSAGE4WeightedMetapathMLPEdgeScorer-epoch400.ckpt.pth"
load_saved_model = True
current_date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

dataset = PubmedSubgraph(dataset_name="Pubmed Subgraph",
                         subgraph_base_path="../dataset/processed_pubmed_subgraph",
                         raw_dir="../dataset/pubmed_subgraph/raw",
                         save_dir="../dataset/pubmed_subgraph/processed")

full_graph = dataset.get_graph()[0]

(train_pos_graph,
 train_neg_graph,
 valid_pos_graph,
 valid_neg_graph,
 test_pos_graph,
 test_neg_graph) = dataset.get_simrel_splittings(ratios=RATIOS)

(potentially_equates_graph,
 colleague_graph,
 citation_graph,
 collaboration_graph) = dataset.get_node_embeddings_graphs()

node_features = full_graph.ndata["feat"]["author"]

model = GraphSAGE4WeightedMetapathMLPEdgeScorer(in_feats=768,
                                                h_feats=100,
                                                potentially_equates_graph=potentially_equates_graph,
                                                colleague_graph=colleague_graph,
                                                citation_graph=citation_graph,
                                                collaboration_graph=collaboration_graph)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss = binary_cross_entropy

print("Starting training process")
log_dir = f"./log/models/{model.__class__.__name__}/"
train_writer = SummaryWriter(log_dir=f"{log_dir}{current_date}/training")
valid_writer = SummaryWriter(log_dir=f"{log_dir}{current_date}/validation")
test_writer = SummaryWriter(log_dir=f"{log_dir}{current_date}/testing")

# start server (in gnn-entity-blocking directory): tensorboard --logdir ./log

# TRAINING
# uncomment when resume training
# load_checkpoint(model, optimizer, best_model_path)
# min_valid_loss = 0.3014
counter = EARLY_STOPPING
for e in range(1, EPOCHS+1):
    model.train()
    pos_score = model(train_pos_graph, node_features)
    neg_score = model(train_neg_graph, node_features)

    train_acc = model.compute_accuracy(pos_score, neg_score)
    train_loss = model.compute_loss(pos_score, neg_score)
    train_writer.add_scalar("Loss", train_loss, e)
    train_writer.add_scalar("Accuracy", train_acc, e)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        valid_pos_score = model(valid_pos_graph, node_features)
        valid_neg_score = model(valid_neg_graph, node_features)

        valid_loss = model.compute_loss(valid_pos_score, valid_neg_score)
        valid_acc = model.compute_accuracy(valid_pos_score, valid_neg_score)
        valid_writer.add_scalar("Loss", valid_loss, e)
        valid_writer.add_scalar("Accuracy", valid_acc, e)
        print(f"In epoch {e:03d} - train loss: {train_loss:.4f} - train acc: {train_acc:.4f} - valid loss: {valid_loss:.4f} - valid acc: {valid_acc:.4f}")

    if min_valid_loss >= valid_loss:
        counter = EARLY_STOPPING
        min_valid_loss = valid_loss
        best_model = copy.deepcopy(model)
        best_model_path = save_checkpoint(model, optimizer, e, log_dir)
    counter -= 1
    if counter <= 0:
        print("Early stopping!")
        break

# EVALUATION
print(f"Best model location: {best_model_path}")
load_checkpoint(model, optimizer, best_model_path)
model.eval()
test_pos_score = model(test_pos_graph, node_features)
test_neg_score = model(test_neg_graph, node_features)

test_loss = model.compute_loss(test_pos_score, test_neg_score)
test_acc = model.compute_accuracy(test_pos_score, test_neg_score)

bcm = BinaryConfusionMatrix(threshold=0.5)
confusion_matrix = bcm(torch.cat([test_pos_score, test_neg_score]), torch.cat([torch.ones(test_pos_score.shape[0]), torch.zeros(test_neg_score.shape[0])]))
test_writer.add_figure(f"Confusion matrix", plot_confusion_matrix(confusion_matrix))

train_writer.close()
valid_writer.close()
test_writer.close()
