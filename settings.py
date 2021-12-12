import torch

# Data
DATA_PATH = "data/flickr30/archive"
BATCH_SIZE = 3
GENERATED_IMAGES_PATH = "generated/"

# Training
MODEL_PATH = "models/final-model-1.0/"
EPOCHS = 6
TENSORBOARD_LOGS = "logs/latest-logs"
ENTROPY_SCAN_FILE = "dataset-scans/dataset_scan_results.json"
METRICS_FILE = "metrics/calculated_metrics.json"
ENTROPY_SCAN_CHANNELS = ("r", "g", "b", "grayscale")

# Adam optimizer
lr = 0.0002
beta1 = 0.9
beta2 = 0.999

# hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
