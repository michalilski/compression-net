from dataloader import ImageDataLoader
from transforms import ImageTransform
import matplotlib.pyplot as plt
from model2 import Generator, Encoder
import torch
from config import model_path


def main():
    device = 'cuda'

    dataloader = ImageDataLoader().test_loader
    image_transform = ImageTransform()
    iterator = iter(dataloader)
    raw_batch = iterator.next()
    data = raw_batch[0].to(device)
    image = raw_batch[0][0]
    original_image = image_transform.denormalize(image)

    encoder = Encoder().to(device)
    generator = Generator().to(device)

    encoder.load_state_dict(torch.load(model_path + 'encoder.pth'))
    generator.load_state_dict(torch.load(model_path + 'generator.pth'))

    encoder.eval()
    generator.eval()

    encoded = encoder(data)
    output = generator(encoded)
    image_tensor = output[0].cpu()
    image = image_tensor.detach()

    generated_image = image_transform.denormalize(image)

    _, grid = plt.subplots(1,2)
    grid[0].imshow(original_image.permute(1,2,0))
    grid[1].imshow(generated_image.permute(1,2,0))
    plt.show()

if __name__ == '__main__':
    main()