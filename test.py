from utils.entropy_manager import EntropyManager
import matplotlib.pyplot as plt
import torch

from config import model_path
from dataloader import ImageDataLoader
from model import Encoder, Generator
from transforms import ImageTransform


def main():
    device = "cuda"
    entropy_manager = EntropyManager()

    test_loader = ImageDataLoader().test_loader
    train_loader = ImageDataLoader().train_loader
    image_transform = ImageTransform()
    iterator = iter(test_loader)
    raw_batch = iterator.next()
    data = raw_batch[0].to(device)
    image = raw_batch[0][0]
    original_image = image_transform.denormalize(image)

    encoder = Encoder().to(device)
    generator = Generator().to(device)

    encoder.load_state_dict(torch.load(model_path + "encoder.pth"))
    generator.load_state_dict(torch.load(model_path + "generator.pth"))

    encoder.eval()
    generator.eval()
    print(data.size())
    encoded = encoder(data)
    print(encoded.size())
    output = generator(encoded)
    print(output.size())
    image_tensor = output[0].cpu()
    image = image_tensor.detach()

    generated_image = image_transform.denormalize(image)
    print(entropy_manager.calculate_image_entropy(original_image))
    print(entropy_manager.calculate_image_entropy(generated_image))
    #print(entropy_manager.plot_entropy_distribution(test_loader))

    _, grid = plt.subplots(1, 2)
    grid[0].imshow(original_image.permute(1, 2, 0))
    grid[1].imshow(generated_image.permute(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    main()
