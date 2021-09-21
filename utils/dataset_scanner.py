import dataclasses
from utils.entropy_manager import EntropyManager
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import List
import json
from contextlib import contextmanager
from traceback import format_exc

@dataclass
class ScanResult():
    entropies: List[dict]
    average_entropies: dict


class ScanResultEncoder(json.JSONEncoder):
    def default(self, scan):
        if dataclasses.is_dataclass(scan):
            return dataclasses.asdict(scan)
        return super().default(scan)


class DatasetScanner():
    def __init__(self, dataloader: DataLoader):
        self.entropy_manager = EntropyManager()
        self.dataloader = dataloader
    
    @contextmanager
    def run_status(self):
        print("Dataset scan started...")
        try:
            yield
        except Exception as err:
            print("Scan failed!")
            print(format_exc())
            print(f"[Scanner Error] {err}")
        else:
            print("Scan finished successfully!")


    def scan_dataset(self):
        with self.run_status():
            entropies = self.entropy_manager.calculate_dataset_entropy(
                self.dataloader
            )
            scan_result = ScanResult(
                entropies=entropies,
                average_entropies=self._average_entropies(entropies),
            )
            with open("dataset_scan_results.json", "w+") as file:
                file.write(
                    json.dumps(scan_result, cls=ScanResultEncoder)
                )
    
    def _average_entropies(self, entropies: List[dict]) -> dict:
        keys = ("r", "g", "b", "grayscale")
        total = {key:0 for key in keys}
        for entropy in entropies:
            for key in keys:
                total[key] += entropy[key]
        return total

    



    