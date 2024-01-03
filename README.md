Training methodology for Link Prediction (negative sampling): https://docs.dgl.ai/en/1.1.x/guide/training-link.html
- comparing the scores between nodes connected by an edge against the scores between an arbitrary pair of nodes
- given an edge (u,v) the training encourage its score to be higher than (u,v')
- loss function used to obtain this behaviour: cross-entropy loss, bpr loss, margin loss

## Installation
```bash
pip install -r requirements.txt
```
## Running
```bash
cd src
python main.py 
```