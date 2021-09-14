# Data
data_path = "flickr30/archive/flickr30k_images/"
model_path = "model-flickr/"
batch_size = 4
image_size = 256
channels = 3

# Transforms:
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# Training:
train_data_part = 0.75
lr = 0.0002
epochs = 5

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
