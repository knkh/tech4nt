import os, glob, json
import numpy as np
import pandas as pd

LOOKBACK = 64
LABEL_ROOT = "demo_data/data_labels_v01_round1"
FEATURE_ROOT = "demo_data/data_feats_v01"
OUT_DIR = "demo_data/dataset_v01r1"

def build():
    X_all, Y_all, instrument_all, timestamp_all = [], [], [], []
    feature_list = None
    counts = {}

    for instr in sorted(os.listdir(FEATURE_ROOT)):
        fd, ld = os.path.join(FEATURE_ROOT, instr), os.path.join(LABEL_ROOT, instr)
        if not (os.path.isdir(fd) and os.path.isdir(ld)): continue

        ff = sorted(glob.glob(os.path.join(fd, "*.csv")))
        lf = sorted(glob.glob(os.path.join(ld, "*.csv")))
        fm = {os.path.basename(f)[:10]: f for f in ff}
        lm = {os.path.basename(f)[:10]: f for f in lf}
        common = sorted(set(fm) & set(lm))

        fs, ls = [], []
        for d in common:
            f = pd.read_csv(fm[d], parse_dates=["ts"])
            l = pd.read_csv(lm[d], parse_dates=["ts"])
            if not f.empty and not l.empty:
                fs.append(f); ls.append(l)
        if not fs: continue

        f_all = pd.concat(fs, ignore_index=True)
        l_all = pd.concat(ls, ignore_index=True)

        if feature_list is None:
            feature_list = [c for c in f_all.columns
                            if c not in ("ts","open","high","low","close","volume")]

        df = (f_all[["ts"] + feature_list]
              .merge(l_all[["ts","label"]], on="ts")
              .sort_values("ts").dropna().reset_index(drop=True))
        if len(df) < LOOKBACK: continue

        delta = df["ts"].diff()
        median_delta = delta.median()
        if pd.isna(median_delta): continue
        gap_idx = np.flatnonzero((delta > median_delta * 1.5).to_numpy())

        x = df[feature_list].to_numpy(np.float32)
        y = df["label"].to_numpy(np.int64)
        ts = df["ts"].to_numpy()

        n = 0
        for i in range(LOOKBACK - 1, len(x)):
            start = i - LOOKBACK + 2
            if np.any((gap_idx >= start) & (gap_idx < i + 1)): continue
            X_all.append(x[i-LOOKBACK+1:i+1])
            Y_all.append(y[i])
            instrument_all.append(instr)
            timestamp_all.append(ts[i])
            n += 1
        counts[instr] = n
        print(f"{instr:<25} windows={n:,}")

    if not X_all: raise RuntimeError("No valid windows generated.")

    X_all = np.stack(X_all).astype(np.float32)
    Y_all = np.asarray(Y_all, np.int64)
    instrument_all = np.asarray(instrument_all)
    timestamp_all = np.asarray(timestamp_all, dtype="datetime64[ns]")

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(f"{OUT_DIR}/X.npy", X_all)
    np.save(f"{OUT_DIR}/Y.npy", Y_all)
    np.save(f"{OUT_DIR}/instrument.npy", instrument_all)
    np.save(f"{OUT_DIR}/timestamp.npy", timestamp_all)
    with open(f"{OUT_DIR}/meta.json","w") as f:
        json.dump({"features":feature_list,"lookback":LOOKBACK,
                   "n_samples":len(X_all),"instrument_counts":counts}, f, indent=2)

    print("✅ UMAP dataset:", X_all.shape)

if __name__ == "__main__":
    build()
