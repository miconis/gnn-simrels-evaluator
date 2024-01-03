import itertools
from datetime import datetime
import random
import numpy as np
from src.dgl_graph.dataset import PubmedSubgraph
from src.utils.models import *
from src.utils.utility import *
import warnings
from src.utils.config import *
import copy
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")
random.seed(1234)
np.random.seed(1234)
os.environ["DGLBACKEND"] = "pytorch"
min_valid_loss = np.inf
best_embeddings_model_path = "./log/models/GraphSAGE/GraphSAGE-epoch188.ckpt.pth"
load_saved_model = True
current_date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

dataset = PubmedSubgraph(dataset_name="Pubmed Subgraph",
                         subgraph_base_path="../dataset/processed_pubmed_subgraph",
                         neg_etype='equates',
                         raw_dir="../dataset/pubmed_subgraph/raw",
                         save_dir="../dataset/pubmed_subgraph/processed")

full_graph = dataset.get_graph()[0]

train_pos_graph, train_neg_graph, test_pos_graph, test_neg_graph = dataset.get_simrel_splittings(train_ratio=TRAIN_RATIO)
potentially_equates_graph, colleague_graph, citation_graph, collaboration_graph = dataset.get_node_embeddings_graphs()

node_features = full_graph.ndata["feat"]["author"]

embeddings_model = GS3LSTM(in_feats=768, h_feats=100, num_layers=2, dropout=0.2)
predictor = EdgeScorer(n_feats=100)
optimizer = torch.optim.Adam(itertools.chain(embeddings_model.parameters(), predictor.parameters()), lr=LEARNING_RATE)
loss = binary_cross_entropy_with_logits

print("Starting training process")
log_dir = f"./log/models/{embeddings_model.__class__.__name__}/"
train_writer = SummaryWriter(log_dir=f"{log_dir}{current_date}/training")
test_writer = SummaryWriter(log_dir=f"{log_dir}{current_date}/testing")
# start server (in gnn-entity-blocking directory): tensorboard --logdir ./log
counter = EARLY_STOPPING
for e in range(1, EPOCHS+1):
    embeddings_model.train()
    predictor.train()
    node_embeddings = embeddings_model(potentially_equates_graph, colleague_graph, citation_graph, collaboration_graph, node_features)

    pos_score = predictor(train_pos_graph, node_embeddings)
    neg_score = predictor(train_neg_graph, node_embeddings)

    train_acc = compute_acc(pos_score, neg_score)
    train_loss = loss(pos_score, neg_score)
    train_writer.add_scalar("Loss", train_loss, e)
    train_writer.add_scalar("Accuracy", train_acc, e)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    embeddings_model.eval()
    predictor.eval()
    with torch.no_grad():
        test_pos_score = predictor(test_pos_graph, node_embeddings)
        test_neg_score = predictor(test_neg_graph, node_embeddings)
        test_loss = loss(test_pos_score, test_neg_score)
        test_acc = compute_acc(test_pos_score, test_neg_score)
        test_writer.add_scalar("Loss", test_loss, e)
        test_writer.add_scalar("Accuracy", test_acc, e)
        print(f"In epoch {e:03d} - loss: {train_loss:.4f} - train acc: {train_acc:.4f} - test loss: {test_loss:.4f} - test acc: {test_acc:.4f}")

    if min_valid_loss >= test_loss:
        counter = EARLY_STOPPING
        min_valid_loss = test_loss
        best_embeddings_model = copy.deepcopy(embeddings_model)
        best_embeddings_model_path = save_checkpoint(embeddings_model, optimizer, e, log_dir)
    counter -= 1
    if counter <= 0:
        print("Early stopping!")
        break
train_writer.close()
test_writer.close()
