import dataclasses
import json
import logging
import os.path
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List

from torch.utils.data import DataLoader

from settings import ENTROPY_SCAN_FILE
from utils.entropy_manager import EntropyManager

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    entropy: List[dict]
    average_entropy: dict


class ScanResultEncoder(json.JSONEncoder):
    def default(self, scan):
        if dataclasses.is_dataclass(scan):
            return dataclasses.asdict(scan)
        return super().default(scan)


class DatasetScanner:
    def __init__(self, dataloader: DataLoader):
        self.entropy_manager = EntropyManager()
        self.dataloader = dataloader

    @contextmanager
    def run_status(self):
        logger.info("Dataset scan started...")
        try:
            yield
        except Exception as err:
            logger.error(f"[Scanner Error] {err}")
        else:
            logger.info("Scan finished successfully!")

    def scan_dataset(self):
        with self.run_status():
            entropy = self.entropy_manager.calculate_dataset_entropy(self.dataloader)
            scan_result = ScanResult(
                entropy=entropy, average_entropy=self._average_entropy(entropy),
            )
            if not os.path.exists(os.path.dirname(ENTROPY_SCAN_FILE)):
                os.makedirs(os.path.dirname(ENTROPY_SCAN_FILE))
            with open(ENTROPY_SCAN_FILE, "w+") as file:
                file.write(json.dumps(scan_result, cls=ScanResultEncoder))

    def _average_entropy(self, entropy: List[dict]) -> dict:
        keys = ("r", "g", "b", "grayscale")
        total = {key: 0 for key in keys}
        for value in entropy:
            for key in keys:
                total[key] += value[key]
        for key in keys:
            total[key] /= len(entropy)
        return total
