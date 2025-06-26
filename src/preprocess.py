import sys
sys.path.append("./src")

from dataset import OurDataset
from annotations import Annotations

if __name__ == "__main__":
    ds = OurDataset(
        csv_dirpath="./data/original_csv", 
        save_dirpath="./notebooks/data",
        initial_cleaning=True
    )
    Annotations.aggregate_rationales(
        dirpath="./data/annotations",
    )