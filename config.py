#Data
data_path = "data/"
model_path = "model-vgg-color/"
batch_size = 4
image_size = 256
channels = 3

#Generator:
#input
gi = 100
#features
gf = 64

#Discriminator:
#features
df = 32

#Transforms:
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

#Training:
train_data_part = 0.6
num_epochs = 5
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
