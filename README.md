This tool is a GNN architecture for the evaluation of similarity relationships drawn by a disambiguation algorithm. The evaluation is performed in terms of percentage (0% bad similarity relationship, 100% correct similarity relationships) of similarity relationships created by the FDup framework (see https://peerj.com/articles/cs-1058/). The code presents the use case of Author Name Disambiguation.
The input dataset contains publications and authors as extracted from the OpenAIRE Graph (https://graph.openaire.eu/). The publications have been processed using a custom LDA model on the abstract and Authors have been extracted by each one of them to create a new entity with attributes inherited by the publication itself (e.g. a new author is identified by the publication identifier, the LDA topics vector of his publication, and the co-authors in the same publication). The deduplication of such authors is based on a preliminary LNFI (Last Name First Initial) clustering stage to limit the number of comparisons, followed by a decision tree on their attributes. Two authors are considered to be equivalent if they share at least 2 co-authors and/or they have a cosine similarity between the LDA topic vectors greater than 0.5. Once the similarity relationships have been drawn, a pre-trained BERT model was used to extract 768-sized feature vectors from the abstracts.

The code presents a promising architecture based on a metapath attention module, 4 GraphSAGE-based node embedding methods to extract embeddings by 4 different metapath graphs, and a final edge scorer based on two linear layers. The best model results in ~88% of accuracy on the test set.

The entire code in this release has been developed using PyTorch (https://pytorch.org/) and the DGL library for Graph Neural Networks (https://www.dgl.ai/), while results of the experiments have been visualized using Tensorboard (https://www.tensorflow.org/tensorboard).

## Installation
```bash
pip install -r requirements.txt
```
## Running
```bash
cd src
python main.py 
```
