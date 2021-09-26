from config import scan_filename
import matplotlib.pyplot as plt
import json


def plot_data(data : dict):
    figure, axs = plt.subplots(2, 2)
    figure.tight_layout(pad=3.0)
    figure.suptitle("Dataset entropies distribution")

    keys = ("r", "g", "b", "grayscale")
    channels = {key:[] for key in keys}
    for entropy in data:
        for key in keys:
            channels[key].append(entropy[key])

    axs[0, 0].hist(channels['r'], bins=100)
    axs[0, 0].set_title("Red")
    axs[0, 1].hist(channels['g'], bins=100)
    axs[0, 1].set_title("Green")
    axs[1, 0].hist(channels['b'], bins=100)
    axs[1, 0].set_title("Blue")
    axs[1, 1].hist(channels['grayscale'], bins=100)
    axs[1, 1].set_title("Grayscale")
    plt.show()
    


def main():
    with open(scan_filename, "r") as file:
        raw_data = file.read()

    data = json.loads(raw_data)
    plot_data(data["entropies"])


if __name__ == "__main__":
    main()