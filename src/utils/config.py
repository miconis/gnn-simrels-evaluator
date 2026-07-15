"""Module containing all the config."""
from pathlib import Path

# PROJECT DIRECTORIES
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_ROOT: Path = PROJECT_ROOT / "data"
MODELS_ROOT: Path = PROJECT_ROOT / "models"
RUNS_ROOT: Path = PROJECT_ROOT / "runs"

# DATASET PARAMETERS
DATASET_URL: str = "https://zenodo.org/api/records/10593022/files/subgraph_similarity_relationships.zip/content"
RAW_DIR: str = str(DATA_ROOT / "pubmed_subgraph" / "raw")
DATA_DIR: str = str(DATA_ROOT / "pubmed_subgraph" / "processed")
AUTHORS_SIMREL_DIR: str = str(DATA_ROOT / "authors_simrel")

# MODEL PARAMETERS
BEST_MODEL_PATH: str = str(MODELS_ROOT / "pretrained" / "AttentiveGraphSAGE4-epoch691.ckpt.pth")
INPUT_FEATURES: int = 768
H_FEATURES: int = 100

# EVALUATION ARTIFACTS
SIMRELS_FOR_TESTING_PATH: str = str(DATA_ROOT / "simrels_for_testing.pt")
SIMRELS_SCORES_PATH: str = str(DATA_ROOT / "simrels_scores.pt")
ORCIDS_PATH: str = str(DATA_ROOT / "orcids.pt")

# TRAINING PARAMETERS
LOG_DIR: str = str(RUNS_ROOT)
SEED: int = 1234
EPOCHS: int = 800
LEARNING_RATE: float = 0.001
EARLY_STOPPING: int = 40
TRAIN_RATIO: float = 0.6
VALID_RATIO: float = 0.2
TEST_RATIO: float = 0.2
MARGIN: float = 2.0  # the distance between positive and negative samples should be higher than this threshold
CUT_THRESHOLD: float = 0.5
