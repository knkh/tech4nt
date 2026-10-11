"""Train the existing PatchTST using the shared fair-validation protocol.

Run from `book_files` or adjust DATA_DIR to the folder containing X.npy,
Y.npy, and meta.json.  Results are intentionally written to a new directory
and do not overwrite the current production checkpoint.
"""

import json
from pathlib import Path
import os
import numpy as np

from fair_validation_training import set_reproducible_seed, train_and_test
from patchtst_model import PatchTST

DATA_DIR = os.path.expanduser("~/Documents/my_project/clone_files/data/part_2/demo_data/dataset_v01r1")
OUT_DIR = os.path.expanduser("~/Documents/my_project/clone_files/data/part_2/demo_data/ptst_cm_demo_validation")



def main():
    # Seed before constructing the model so its initial weights are repeatable.
    set_reproducible_seed()
    with (Path(DATA_DIR) / "meta.json").open() as handle:
        lookback = int(json.load(handle)["lookback"])
    feature_count = int(np.load(f"{DATA_DIR}/X.npy", mmap_mode="r").shape[2])
    train_and_test(
        model=PatchTST(
            c_in=feature_count,
            patch_len=8,
            stride=4,
            max_len=lookback,
            num_layers=2,
            num_heads=2,
            d_model=64,
            d_ff=128,
            dropout=0.1,
            n_classes=3,
        ),
        model_name="PatchTST",
        data_dir=DATA_DIR,
        out_dir=OUT_DIR,
        optimizer_kind="adamw",
    )


if __name__ == "__main__":
    main()
