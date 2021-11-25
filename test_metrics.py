from dataloader import ImageDataLoader
from utils.metrics_manager import MetricsManager
import matplotlib.pyplot as plt


def main():
    results = MetricsManager(ImageDataLoader().test_loader(test_batch=1)).run()
    red_entropy_results = sorted(results, key=lambda x: x.entropy["b"])
    plt.plot(
        [x.entropy["b"] for x in red_entropy_results],
        [x.mse_loss for x in red_entropy_results],
    )
    plt.show()


if __name__ == "__main__":
    main()
