#Data
data_path = "data/"
model_path = "model/"
batch_size = 10
image_size = 64
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
mean = (0.5, 0.5, 0.5)
std = (0.24703223,  0.24348513 , 0.26158784)

#Training:
train_data_part = 0.6
num_epochs = 5
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5