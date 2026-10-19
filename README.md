

# Tech4NT: Machine Learning Scripts & Datasets

This repository contains the source code, Python scripts, and datasets accompanying the book. 

## 📁  `tech4nt` GitHub repository Structure

tech4nt/
│
├── README.md
│
├── data/
│   ├── part1_ml/
│   │   └── diabetes_clean.csv
│   │
│   └── part2_trading/
│       └── sample_ohlcv/
│           ├── instrument_01.csv
│           ├── instrument_02.csv
│           └── ...
│
├── scripts/
│   ├── part1/
│   │   ├── ml_basic.py
│   │   └── ...
│   │
│   └── part2/
│       ├── collect_ohlcv.py
│       ├── feature_engineering.py
│       ├── label_generation.py
│       └── ...
│
├── requirements.txt
│
└── .gitignore

⚙️ Broker/API example

collect_ohlcv_kite.py is an example implementation using the Kite Connect API. It is provided to demonstrate the mechanics of collecting and constructing 1-minute OHLCV data. The book does not recommend or endorse any particular broker.



📊 Datasets
Datasets are located inside the data/ directory. You can also read raw datasets directly in Python using pandas:
import pandas as pd

url = "[https://raw.githubusercontent.com/knkh/tech4nt/main/data/diabetes_clean.csv](https://raw.githubusercontent.com/knkh/tech4nt/main/data/diabetes_clean.csv)"
df = pd.read_csv(url)

For illustration, demo dataset and saved models can be used in the executor.
use it for dry run only. 

📊 instruments to be traded:

1. cash and future, the saved models may be used
2. option trading, collect option data separately. 
Preferably, nifty option trading is a good starting point.
Collect nifty call and put options together with strike price 
at or near nifty cash price. 
Remember: nifty cash is a very good estimator of the current 
nifty future for option trading.