import torch

# Data
data_path = "flickr30/archive/flickr30k_images"
model_path = "model-big-images/"
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
epochs = 5
tensorboard_runs = "runs/compression-net"

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

#hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

