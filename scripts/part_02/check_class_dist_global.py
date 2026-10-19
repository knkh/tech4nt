
'''
USAGE: run from my_project

python diagnostic_tools/check_class_dist_global.py

'''
# check_class_dist_global.py

import os
import glob
import pandas as pd

ROOT = "demo_data/data_labels_v01_round1"
all_dfs = []

# search all csv files recursively
files = glob.glob(os.path.join(ROOT, "**", "*.csv"), recursive=True)

print("Total files found:", len(files))

for f in files:
    try:
        df = pd.read_csv(f)

        if "label" in df.columns:
            all_dfs.append(df[["label"]])

    except Exception as e:
        print("Skipping:", f, e)

# combine all files
df_all = pd.concat(all_dfs, ignore_index=True)

print("\nTotal rows:", len(df_all))

print("\nLabel distribution:")
print(df_all["label"].value_counts(normalize=True))




