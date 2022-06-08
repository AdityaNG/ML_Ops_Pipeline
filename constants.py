import os

DATASET_DIR = "data/dataset/"

MODEL_TRAINING_SETTINGS = "data/model_training_settings.pkl"
MODEL_VALIDATION_SETTINGS = "data/model_validation_settings.pkl"

MODEL_CHECKPOINTS = "data/model_checkpoints"

ANALYSIS_DIR = "data/model_analysis_results"

# Hight and width of the images
IMAGE_SIZE = 32
# 3 channels, Red, Green and Blue
CHANNELS = 3
# Number of epochs
#NUM_EPOCH = 350
NUM_EPOCH = 1
# learning rate
LEARN_RATE = 1.0e-4

SLEEP_TIME = 1

os.makedirs(DATASET_DIR, exist_ok=True)