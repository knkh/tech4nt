"""Train the existing TCN using the shared fair-validation protocol.

Run from `book_files` or adjust DATA_DIR to the folder containing X.npy,
Y.npy, and meta.json.  Results are intentionally written to a new directory
and do not overwrite the current production checkpoint.
"""
import os

from fair_validation_training import set_reproducible_seed, train_and_test
from tcn_model import TCN

DATA_DIR = os.path.expanduser("~/Documents/my_project/clone_files/data/part_2/demo_data/dataset_v01r1")
OUT_DIR = os.path.expanduser("~/Documents/my_project/clone_files/data/part_2/demo_data/tcn_cm_demo_validation")


def main():
    # Seed before constructing the model so its initial weights are repeatable.
    set_reproducible_seed()
    train_and_test(
        model=TCN(
            num_features=load_feature_count(),
            num_classes=3,
            channels=(32, 64, 64, 64, 64),
            kernel_size=4,
            dropout=0.2,
        ),
        model_name="TCN",
        data_dir=DATA_DIR,
        out_dir=OUT_DIR,
        optimizer_kind="adam",
    )


def load_feature_count() -> int:
    """Read only the array shape; scaling and all model evaluation stay shared."""
    import numpy as np
    return int(np.load(f"{DATA_DIR}/X.npy", mmap_mode="r").shape[2])


if __name__ == "__main__":
    main()
