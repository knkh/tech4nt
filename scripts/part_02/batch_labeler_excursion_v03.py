
'''
'''
'''

USAGE: 
python batch_labeler_excursion_v03.py \
  --raw-root data_ohlcv_sample \
  --label-root data_labels_excur_v03 \
  --script labeler_excursion_v03.py \
  --start_date 2026-01-01 \
  --end_date 2026-08-31 \
  --horizon 12 \
  --mfe-threshold-atr 1.2 \
  --mae-threshold-atr 0.7 \
  --end-ret-threshold-atr 0.2 \
  --persistence-threshold 0.55
  
  
  
  
'''




# batch_labels_structural.py

import os
import sys
import glob
import subprocess
import argparse
from datetime import datetime, date
import re


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def is_today_filename(path):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path))
    if not m:
        return False
    file_date = datetime.strptime(m.group(1), "%Y-%m-%d").date()
    return file_date == date.today()


def find_instruments(raw_root):
    return [
        name for name in os.listdir(raw_root)
        if os.path.isdir(os.path.join(raw_root, name))
    ]

def extract_date_from_filename(path):
    """
    Assumes filename contains YYYY-MM-DD
    """
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d").date()


def main():
    p = argparse.ArgumentParser(description="Batch target b4 stop label generator")

    p.add_argument("--raw-root", required=True,
                   help="Root folder with per-instrument raw data")

    p.add_argument("--label-root", required=True,
                   help="Where labeled files will be saved")

    '''
    mfe-threshold-atr 1.2 \
  --mae-threshold-atr 0.7 \
  --end-ret-threshold-atr 0.2 \
  --persistence-threshold 0.55
    '''
    
    p.add_argument("--script", required=True,
                   help="Script to run (feature or label generator)")
    p.add_argument("--start_date", required=True,
                   help="YYYY-MM-DD")
    p.add_argument("--end_date", required=True,
                   help="YYYY-MM-DD")
    p.add_argument("--mfe-threshold-atr", type=float, default=1.2)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--mae-threshold-atr", type=float, default=0.7)
    p.add_argument("--end-ret-threshold-atr", type=float, default=0.20)
    p.add_argument("--persistence-threshold", type=float, default=0.55)
    
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing label files")

    args = p.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    raw_root = os.path.abspath(os.path.expanduser(args.raw_root))
    label_root = os.path.abspath(os.path.expanduser(args.label_root))

    instruments = find_instruments(raw_root)

    if not instruments:
        print("❌ No instruments found in", raw_root)
        sys.exit(1)

    for instr in instruments:
        print(f"\n=== {instr} ===")

        raw_dir = os.path.join(raw_root, instr)
        out_dir = os.path.join(label_root, instr)
        ensure_dir(out_dir)

        raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
        if not raw_files:
            continue

        for raw in raw_files:

            file_date = extract_date_from_filename(raw)
            if file_date is None:
                continue

            if not (start_date <= file_date <= end_date):
                continue

            basename = os.path.splitext(os.path.basename(raw))[0]
            out_file = os.path.join(out_dir, f"{basename}_labeled.csv")

            if os.path.exists(out_file) and not args.force:
                continue

            cmd = [
                sys.executable, args.script,
                "--infile", raw,
                "--outfile", out_file,
                "--mfe-threshold-atr", str(args.mfe_threshold_atr),
                "--mae-threshold-atr", str(args.mae_threshold_atr)
                
                
            ]

            print("🏷️ Generating labels:", basename)

            r = subprocess.run(cmd)

            if r.returncode != 0:
                print("❌ ERROR running labeler for", raw)
                sys.exit(1)

    print("\n✅ Batch labeling completed successfully.")


if __name__ == "__main__":
    main()




'''
confirm-bars = 3 means:

For each signal bar, the labeler looks at the next 3 future bars and checks 
whether the MACD spread stays strong enough and grows enough during those 
3 bars.

From your code, this line is the key:

window = future_bars.iloc[:confirm_bars].copy()
So with --confirm-bars 3:

it takes the first 3 bars from the future window
computes macd_spread on those 3 bars
normalizes by ATR
checks:
are all 3 bars above spread-threshold-atr for UP?
or all 3 below -spread-threshold-atr for DOWN?
did the spread grow enough from bar 1 to bar 3?
So in plain language:

confirm-bars = 3 means “the move must stay confirmed for 3 consecutive future bars”
What it does not mean:

it does not use all 12 horizon bars for MACD confirmation
horizon = 12 is only the total future room available
confirm-bars = 3 is the shorter sub-window used to decide the MACD label
Effect of changing it:

smaller confirm-bars like 2:
labels fire more easily
more noisy
larger confirm-bars like 4 or 5:
labels are stricter
fewer labels
better confirmation, but slower and sparser
With your setup:

confirm-bars = 3
spread-threshold-atr = 0.03
growth-threshold-atr = 0.02
this means:

“Label UP only if, over the next 3 bars, MACD spread remains above 0.03 ATR on 
every bar and expands by at least 0.02 ATR from the first to the third bar.”
'''