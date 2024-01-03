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

train_pos_graph, train_neg_graph, test_pos_graph, test_neg_graph = dataset.get_link_prediction_graphs(pos_etype="equates", train_ratio=TRAIN_RATIO)
potentially_equates_graph, colleague_graph, citation_graph, collaboration_graph = dataset.get_node_embeddings_graphs()

node_features = full_graph.ndata["feat"]["author"]

embeddings_model = GS3Agg(in_feats=768, h_feats=100)
predictor = EuclideanPredictor()  # to be used to have closer embeddings for similar nodes
optimizer = torch.optim.Adam(itertools.chain(embeddings_model.parameters(), predictor.parameters()), lr=LEARNING_RATE)
loss = contrastive_loss

print("Starting training process")
log_dir = f"./log/models/{embeddings_model.__class__.__name__}/"
train_writer = SummaryWriter(log_dir=f"{log_dir}{current_date}/training")
test_writer = SummaryWriter(log_dir=f"{log_dir}{current_date}/testing")
# start server (in gnn-entity-blocking directory): tensorboard --logdir ./log
counter = EARLY_STOPPING
for e in range(EPOCHS):
    embeddings_model.train()
    predictor.train()
    node_embeddings = embeddings_model(potentially_equates_graph, colleague_graph, citation_graph, collaboration_graph, node_features)

    train_neg_graph = construct_negative_graph(train_pos_graph, 5)  # generated every time as regularization

    pos_score = predictor(train_pos_graph, node_embeddings)
    neg_score = predictor(train_neg_graph, node_embeddings)

    train_auc = compute_auc(pos_score, neg_score)
    train_loss = loss(pos_score, neg_score)
    train_writer.add_scalar("Loss", train_loss, e)
    train_writer.add_scalar("Area Under Curve", train_auc, e)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    embeddings_model.eval()
    predictor.eval()
    with torch.no_grad():
        test_node_embeddings = embeddings_model(potentially_equates_graph, colleague_graph, citation_graph, collaboration_graph, node_features)
        test_pos_score = predictor(test_pos_graph, test_node_embeddings)
        test_neg_score = predictor(test_neg_graph, test_node_embeddings)
        test_loss = loss(test_pos_score, test_neg_score)
        test_auc = compute_auc(test_pos_score, test_neg_score)
        test_writer.add_scalar("Loss", test_loss, e)
        test_writer.add_scalar("Area Under Curve", test_auc, e)
        print(f"In epoch {e:03d} - loss: {train_loss:.4f} - train auc: {train_auc:.4f} - test loss: {test_loss:.4f} - test auc: {test_auc:.4f}")

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
