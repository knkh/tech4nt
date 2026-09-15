

📊 Datasets
Datasets are located inside the data/ directory. You can also read raw datasets directly in Python using pandas:
import pandas as pd

url = "[https://raw.githubusercontent.com/knkh/tech4nt/main/data/diabetes_clean.csv](https://raw.githubusercontent.com/knkh/tech4nt/main/data/diabetes_clean.csv)"
df = pd.read_csv(url)


---

### 2. `requirements.txt` Format and Details

The `requirements.txt` file lists all third-party Python packages your project depends on. Each line contains a package name, optionally followed by an exact or minimum version number.

#### Recommended `requirements.txt` Content

Based on your project scripts, create a file named `requirements.txt` in your root folder with the following lines[span_0](start_span)[span_0](end_span):

```text
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0

Key Rules for requirements.txt:
⚬	Plain Text: Do not use Markdown formatting or bullet points inside requirements.txt.
⚬	Standard Package Names: Use the official package names as recognized by PyPI (e.g., scikit-learn, not sklearn).
⚬	Version Specifiers:
⚬	pandas: Installs any version of pandas.
⚬	pandas>=2.0.0: Installs version 2.0.0 or higher.
⚬	pandas==2.2.1: Pinpoints the exact version to ensure strict reproducibility across different computers.