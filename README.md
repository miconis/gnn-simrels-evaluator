# GNN Similarity Relationships Evaluator

## Description

GNN Similarity Relationships Evaluator is a research prototype for assessing candidate similarity relationships produced by an entity-deduplication system. The included use case is author name disambiguation: given two author records that a traditional deduplication workflow considers similar, the model estimates the probability that the relationship is correct.

The project represents authors, publications, and their relations as a heterogeneous graph. It combines four two-layer GraphSAGE encoders, one for each author-centric graph view, with a metapath-attention module. A small multilayer perceptron then scores each candidate author-author edge between `0` (incorrect relationship) and `1` (correct relationship).

The repository includes dataset downloading and preprocessing, model training, a pretrained checkpoint, threshold evaluation, and utilities for inspecting individual predictions. The original experiments obtained approximately 88% test accuracy; results can vary with the environment, data split, and training run.

## Dataset information

The dataset is a subgraph derived from the OpenAIRE Graph and is distributed through [Zenodo record 10593022](https://zenodo.org/records/10593022). Its records describe publications and author occurrences extracted from publications. Candidate similarity relationships originate from an FDup-based disambiguation workflow.

The raw archive contains the following Spark-readable datasets:

| Dataset | Meaning                                                                                         |
| --- |-------------------------------------------------------------------------------------------------|
| `authors` | Author records, node indices, identifiers such as ORCID, and author features                    |
| `publications` | Publication records, node indices, and 768-dimensional BERT features                            |
| `cites_rels` | Publication-to-publication citation edges                                                       |
| `coprojected_rels` | Publication-to-publication co-projection edges                                                  |
| `collaborates_rels` | Author-to-author collaboration edges                                                            |
| `writes_rels` | Author-to-publication authorship edges                                                          |
| `potentiallyequivalent_rels` | Candidate author-equivalence edges                                                              |
| `equivalent_rels` | Ground-truth author-equivalence edges                                                           |
| `simrels` | Similarity relationships used for training and testing                                          |
| `simrels_dedup` | Similarity relationships produced by the deduplication workflow with a realistic configuration |

During preprocessing, external record identifiers are mapped to graph node indices. The resulting DGL heterogeneous graph contains `author` and `publication` nodes and the following edge types: `cites`, `coprojected`, `collaborates`, `writes`, `is_written_by`, `potentially_equates`, `equates`, `similar`, and `similar_for_dedup`.

The first dataset initialization downloads and extracts the archive into `data/pubmed_subgraph/raw/`, builds the graph, and stores it as `data/pubmed_subgraph/processed/pubmed_graph.dgl`. Later initializations load the processed graph directly. Raw data are downloaded again only if the expected raw datasets are missing; extraction paths are validated before files are copied.

Consult the Zenodo record for the dataset's version, attribution requirements, and license.

## Code information

```text
.
├── README.md
├── requirements.txt
├── models/
│   └── AttentiveGraphSAGE4-epoch691.ckpt.pth  # pretrained model
└── src/
    ├── dataset.py                             # where the data is loaded and processed
    ├── model.py                               # where the model is defined
    ├── training.py                            # training, validation, testing, and checkpoints of the model
    ├── testing/
    │   ├── evaluate_thresholds.py             # threshold and deduplication-group evaluation
    │   └── inspect_predictions.py             # metrics and examples of scored edges
    └── utils/
        ├── config.py                          # where parameters are defined
        └── functions.py                       # shared data, metric, plotting, and checkpoint helpers
```

All configured paths are derived from the repository root in `src/utils/config.py`.

## Requirements

The project requires:

- Python 3.11 or a compatible Python 3 version;
- a Java runtime available through `java`, as required by PySpark;
- sufficient memory for local Spark preprocessing (the current Spark driver configuration reserves up to 15 GB);
- PyTorch and TensorBoard;
- the packages pinned in `requirements.txt`, including DGL, PySpark, NumPy, scikit-learn, Matplotlib, tqdm, Unidecode, and TorchMetrics.

Create an isolated environment and install the dependencies from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch tensorboard
python -m pip install -r requirements.txt
```

PyTorch installation can depend on the operating system and CPU/GPU backend. If the generic command is unsuitable, select the appropriate installation command from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/), then install `requirements.txt`.

## Usage instructions

Run commands from the repository root. The current scripts use both package-qualified and source-root imports, so add both locations to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

### Load or create the dataset

Instantiating `PubmedSubgraph` is sufficient to download, preprocess, cache, or load the dataset:

```python
from src.dataset import PubmedSubgraph
from src.utils.config import DATASET_URL, DATA_DIR, RAW_DIR

dataset = PubmedSubgraph(
    dataset_name="Pubmed Subgraph",
    url=DATASET_URL,
    raw_dir=RAW_DIR,
    save_dir=DATA_DIR,
)

graph = dataset.get_graph()
print(graph)
```

Download, extraction, and copying progress bars are displayed on the first run. To rebuild the processed graph, remove `data/pubmed_subgraph/processed/pubmed_graph.dgl` or instantiate the dataset with `force_reload=True`. Existing raw files are reused when complete.

### Configure an experiment

Edit `src/utils/config.py` before running an experiment. It defines dataset and output paths, input and hidden dimensions, random seed, number of epochs, learning rate, early-stopping patience, train/validation/test ratios, and classification threshold.

### Train the model

```bash
python src/training.py
```

Training uses balanced positive and negative candidate edges, writes the best checkpoint under `runs/`, and stores training, validation, and testing logs for TensorBoard. Start TensorBoard with:

```bash
tensorboard --logdir runs
```

### Evaluate deduplication thresholds

```bash
python -m src.testing.evaluate_thresholds
```

This script loads the checkpoint configured by `BEST_MODEL_PATH`, scores `similar_for_dedup` edges, evaluates thresholds from 0.0 to 1.0, forms author groups through connected components, prints the F1 score for each threshold, and plots edge accuracy, singleton fraction, pairwise precision, recall, and F1.

To change the tested thresholds, edit `THRESHOLDS` at the beginning of `src/testing/evaluate_thresholds.py`.

### Inspect predictions

```bash
python -m src.testing.inspect_predictions
```

This script evaluates candidate `similar` edges against the `equates` ground truth, displays confusion-matrix metrics, and prints examples assigned very high or very low scores. The number of examples and score cutoffs are configurable near the beginning of the file.

## Methodology

The implementation follows these stages:

1. **Raw-data acquisition.** `PubmedSubgraph` checks for all required raw datasets, downloads the configured ZIP or TAR archive when necessary, validates its contents, and extracts it with progress reporting.
2. **Graph construction.** PySpark reads the raw datasets, resolves record identifiers to contiguous indices, builds node-feature tensors, and creates a DGL heterogeneous graph.
3. **Graph-view extraction.** Four homogeneous author graphs are derived from direct or metapath relationships: potentially equivalent authors, co-projected publications, citation-linked publications, and direct collaboration. Self-loops are added to each view.
4. **Node representation learning.** A separate two-layer mean-aggregating GraphSAGE encoder produces author embeddings for each graph view.
5. **Metapath attention.** Learned attention weights combine the four embeddings into one representation per author.
6. **Edge scoring.** Source and target representations are concatenated and passed through two linear layers and a sigmoid to estimate relationship correctness.
7. **Training and validation.** Candidate `similar` edges are labelled through the `equates` graph, balanced by class, and split according to the configured 60/20/20 ratios. The model uses binary cross-entropy, AdamW, validation-based checkpointing, and early stopping.
8. **Deduplication evaluation.** Scores below a selected threshold are removed from `similar_for_dedup`; connected components become predicted author groups and are compared with ORCID-derived groups using pairwise precision, recall, F1, singleton fraction, and surviving-edge accuracy.

Random seeds and the main hyperparameters are centralized in `src/utils/config.py` to support reproducible experiments.

## Citations

If this repository or its dataset is used in research, cite the dataset record and the FDup work on which the candidate-generation workflow is based:

- Dataset archive: [GNN Similarity Relationships Evaluator dataset, Zenodo record 10593022](https://zenodo.org/records/10593022).
- Michele De Bonis, Paolo Manghi, and Claudio Atzori. “FDup: a framework for general-purpose and efficient entity deduplication of record collections.” *PeerJ Computer Science* 8:e1058, 2022. [https://doi.org/10.7717/peerj-cs.1058](https://doi.org/10.7717/peerj-cs.1058).
- [OpenAIRE Graph](https://graph.openaire.eu/), the source graph from which the use-case data were derived.