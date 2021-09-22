# Data
data_path = "kodak"
model_path = "model-flickr/"
batch_size = 4
image_size = 256
channels = 3
scan_filename = "dataset_scan_results.json"

# Transforms:
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# Training:
train_data_part = 0.1
lr = 0.0002
epochs = 5

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

