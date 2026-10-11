


# build_windowed_dataset.py
# BAR-MINUTES = 1

import os
import glob
import json
import numpy as np
import pandas as pd


LOOKBACK = 64

LABEL_ROOT = "data_labels_v01_round1"

FEATURE_ROOT = "data_feats_vo1"

OUT_DIR = "dataset_v01_round1"

BAR_MINUTES = 1
EXPECTED_DELTA = pd.Timedelta(minutes=BAR_MINUTES)


def build():

    X_all = []
    Y_all = []

    feature_list = None

    instruments = sorted(os.listdir(FEATURE_ROOT))

    for instr in instruments:

        feat_dir = os.path.join(FEATURE_ROOT, instr)
        lab_dir = os.path.join(LABEL_ROOT, instr)

        if not os.path.isdir(feat_dir):
            continue

        if not os.path.isdir(lab_dir):
            continue

        feat_files = sorted(
            glob.glob(os.path.join(feat_dir, "*.csv"))
        )

        lab_files = sorted(
            glob.glob(os.path.join(lab_dir, "*.csv"))
        )

        feat_map = {
            os.path.basename(f)[:10]: f
            for f in feat_files
        }

        lab_map = {
            os.path.basename(f)[:10]: f
            for f in lab_files
        }

        common_dates = sorted(
            set(feat_map) & set(lab_map)
        )

        if not common_dates:
            continue

        print(f"\n========== {instr} ==========")

        # ------------------------------------------------------------
        # Read and combine all matching daily files
        # ------------------------------------------------------------

        instr_feats = []
        instr_labs = []

        for d in common_dates:

            f_df = pd.read_csv(
                feat_map[d],
                parse_dates=["ts"]
            )

            l_df = pd.read_csv(
                lab_map[d],
                parse_dates=["ts"]
            )

            if f_df.empty or l_df.empty:
                continue

            instr_feats.append(f_df)
            instr_labs.append(l_df)

        if not instr_feats:
            continue

        df_f_all = pd.concat(
            instr_feats,
            ignore_index=True
        )

        df_l_all = pd.concat(
            instr_labs,
            ignore_index=True
        )

        # ------------------------------------------------------------
        # Determine feature columns
        # ------------------------------------------------------------

        if feature_list is None:

            feature_list = [
                c for c in df_f_all.columns
                if c not in (
                    "ts",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "label",
                )
            ]

        # ------------------------------------------------------------
        # Feature + label alignment
        #
        # INNER JOIN means:
        #   feature without label -> excluded
        #   label without feature -> excluded
        #
        # This naturally removes cold-start labels for the first day.
        # ------------------------------------------------------------

        df = df_f_all[
            ["ts"] + feature_list
        ].merge(
            df_l_all[
                ["ts", "label"]
            ],
            on="ts",
            how="inner"
        )

        df = (
            df
            .sort_values("ts")
            .drop_duplicates("ts")
            .reset_index(drop=True)
        )

        if df.empty:
            continue

        # ------------------------------------------------------------
        # Remove rows containing invalid feature/label values
        # ------------------------------------------------------------

        df = df.dropna(
            subset=feature_list + ["label"]
        ).reset_index(drop=True)

        if df.empty:
            continue

        # ------------------------------------------------------------
        # Generate windows separately for each trading day.
        #
        # IMPORTANT:
        # A model window must NEVER cross an overnight/weekend gap.
        # ------------------------------------------------------------

        df["session_date"] = df["ts"].dt.date

        instrument_windows = 0

        for session_date, session in df.groupby(
            "session_date",
            sort=True
        ):

            session = (
                session
                .sort_values("ts")
                .reset_index(drop=True)
            )

            if len(session) < LOOKBACK:
                continue

            # --------------------------------------------------------
            # Check continuity inside the trading session.
            #
            # A valid 3-minute sequence must have:
            #
            # 03:45
            # 03:48
            # 03:51
            # ...
            # --------------------------------------------------------

            deltas = session["ts"].diff()

            # Start of session is always considered continuous.
            continuous = (
                deltas.isna()
                | (deltas == EXPECTED_DELTA)
            )

            # --------------------------------------------------------
            # Create windows.
            #
            # We only allow a window if ALL 64 rows are continuous.
            # --------------------------------------------------------

            X = session[
                feature_list
            ].values.astype(np.float32)

            Y = session[
                "label"
            ].values.astype(np.int64)

            for i in range(
                LOOKBACK - 1,
                len(session)
            ):

                start = i - LOOKBACK + 1

                # Every transition inside the window must be
                # exactly one 3-minute bar apart.
                if not continuous.iloc[
                    start + 1 : i + 1
                ].all():

                    continue

                X_all.append(
                    X[start:i + 1]
                )

                Y_all.append(
                    Y[i]
                )

                instrument_windows += 1

        print(
            f"  Matching days: {len(common_dates)}"
        )

        print(
            f"  Windows generated: {instrument_windows}"
        )

    # ------------------------------------------------------------
    # Final dataset
    # ------------------------------------------------------------

    if not X_all:

        print(
            "\n❌ No valid windows were generated!"
        )

        return

    X_all = np.stack(X_all)
    Y_all = np.asarray(
        Y_all,
        dtype=np.int64
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    os.makedirs(
        OUT_DIR,
        exist_ok=True
    )

    np.save(
        os.path.join(OUT_DIR, "X.npy"),
        X_all
    )

    np.save(
        os.path.join(OUT_DIR, "Y.npy"),
        Y_all
    )

    meta = {
        "features": feature_list,
        "lookback": LOOKBACK,
        "bar_minutes": BAR_MINUTES,
        "n_samples": int(len(Y_all)),
        "n_features": int(len(feature_list)),
    }

    with open(
        os.path.join(OUT_DIR, "meta.json"),
        "w"
    ) as f:

        json.dump(
            meta,
            f,
            indent=2
        )

    print(
        "\n✅ Structurally sound windowed dataset built:"
    )

    print(
        f"   X shape: {X_all.shape}"
    )

    print(
        f"   Y shape: {Y_all.shape}"
    )

    print(
        "\nLabel distribution:"
    )

    print(
        pd.Series(Y_all)
        .value_counts(
            normalize=True
        )
        .sort_index()
    )


if __name__ == "__main__":
    build()