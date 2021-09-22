import matplotlib.pyplot as plt
from config import scan_filename
import json
 
def load_scan_results():
    with open(scan_filename, "r") as file:
        results = file.read()
    return json.loads(results)


def results():
    keys = ("r", "g", "b", "grayscale")
    data = load_scan_results()
    listed_entropies = {
        key: [
            entropy[key] 
            for entropy in data["entropies"]
        ] for key in keys
    }
    plt.hist(listed_entropies['r'])
    plt.show()

if __name__ == "__main__":
    results()