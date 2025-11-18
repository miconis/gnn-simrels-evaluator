"""Module containing all the config."""
# parameters
EPOCHS: int = 800
LEARNING_RATE: float = 0.001
EARLY_STOPPING: int = 40
TRAIN_RATIO: float = 0.6
VALID_RATIO: float = 0.2
TEST_RATIO: float = 0.2
MARGIN: float = 2.0  # the distance between positive and negative samples should be higher than this threshold
