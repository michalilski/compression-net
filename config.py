import torch

# Data
data_path = "flickr30/archive"
model_path = "training-run/"
batch_size = 3
image_size = 256
channels = 3
scan_filename = "dataset_scan_results.json"

# Transforms:
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# Training:
train_data_part = 0.8
lr = 0.0002
epochs = 1
tensorboard_runs = "runs/log-extended-loss"

# Beta1 hyperparam for Adam optimizers
beta1 = 0.9

#hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

