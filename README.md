# 🛡️ PhishGuard AI — Explainable Phishing Detection and Risk Analysis

An interactive cybersecurity dashboard that uses machine learning and explainable AI
to detect phishing websites from pre-extracted website features.

---

## Features

- **Dataset analysis** — class distribution, statistics, missing-value and duplicate checks
- **Two ML models** — Logistic Regression vs. Random Forest, trained and compared side-by-side
- **Full evaluation** — accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, FP/FN counts
- **Explainable AI** — global feature importance (Random Forest) and permutation importance (Logistic Regression)
- **Record-level prediction** — select any dataset record, see the model prediction and per-feature explanation
- **CSV downloads** — export the dataset or individual prediction explanations
- **Dark cybersecurity theme** — professional Streamlit dashboard

---

## Technology Stack

| Component | Library |
|---|---|
| Dashboard | Streamlit ≥ 1.32 |
| Data | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Auto-download | ucimlrepo |
| Language | Python 3.9+ |

---

## Dataset

**UCI Phishing Websites Dataset (ID 327)**  
URL: <https://archive.ics.uci.edu/dataset/327/phishing>  
License: Creative Commons Attribution 4.0 (CC BY 4.0)  
Citation: Mohammad, R. M., Thabtah, F., & McCluskey, L. (2014).

The dataset contains ~11,055 website records with 30 pre-extracted features
(URL characteristics, SSL status, domain age, etc.) and a binary target label.

### Automatic download

The application will attempt to download the dataset automatically on first run
using the `ucimlrepo` package (included in `requirements.txt`).

### Manual download

1. Visit <https://archive.ics.uci.edu/dataset/327/phishing>
2. Click **Download** to get `phishing.zip`
3. Extract the archive and locate the `.arff` or data file inside
4. Save it as `data/phishing.csv` with the 30 feature columns and `Result` column

Or run this one-liner after activating the virtual environment:

```powershell
python -c "from ucimlrepo import fetch_ucirepo; import pandas as pd; d=fetch_ucirepo(id=327); df=pd.concat([d.data.features,d.data.targets],axis=1); df.to_csv('data/phishing.csv',index=False)"
```

---

## Project Structure

```
PhishGuard-AI/
├── app.py                              # All Python application code
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── YourName_PhisGuardAI_ProjectReport.docx  # Project report
├── data/
│   └── phishing.csv                    # Dataset (download separately)
└── screenshots/                        # Application screenshots
```

---

## Setup Instructions (Windows / VS Code)

### 1. Prerequisites

- Python 3.9 or later — download from <https://www.python.org/downloads/>
- Git (optional) — download from <https://git-scm.com/>

### 2. Open the project in VS Code

Open the `PhishGuard-AI` folder in VS Code using **File → Open Folder**.

Open the integrated terminal with **Terminal → New Terminal** (PowerShell).

### 3. Create a virtual environment

```powershell
python -m venv .venv
```

### 4. Activate the virtual environment

```powershell
.venv\Scripts\Activate.ps1
```

> If you see an execution policy error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then try again.

You should see `(.venv)` at the start of your prompt.

### 5. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

This may take a few minutes on first run.

### 6. (Optional) Place the dataset manually

If the automatic download does not work, place `phishing.csv` in the `data/` folder.
See the **Dataset** section above for download instructions.

### 7. Launch the application

```powershell
streamlit run app.py
```

Streamlit will print a local URL such as:

```
Local URL: http://localhost:8501
```

Open that URL in your web browser.

---

## Usage

1. **Dataset Analysis tab** — explore the loaded data, class balance, and statistics
2. **Model Evaluation tab** — compare Logistic Regression and Random Forest performance
3. **Explainable AI tab** — view global feature importance charts and tables
4. **Record Analysis tab** — use the slider to pick a record; see the prediction and explanation
5. **About tab** — read about the project, dataset, and limitations

---

## Testing the Application

After launching with `streamlit run app.py`:

| Check | Expected result |
|---|---|
| Dashboard loads | Title "PhishGuard AI" visible; 4 metric cards populated |
| Dataset tab | Class distribution chart and statistics table shown |
| Model tab | Two sets of metrics and confusion matrices displayed |
| XAI tab | Feature importance bar charts rendered |
| Record Analysis | Slider works; prediction and explanation update |
| Download buttons | CSV files download successfully |

---

## Limitations

1. The dataset contains pre-extracted features, not raw URLs — raw URL scanning is not supported
2. Model trained on 2014 data; phishing techniques have evolved since then
3. Feature importance shows global averages; individual predictions may differ
4. For educational purposes only — do not use as a sole production security tool
5. No real-time threat intelligence is included

---

## References

- Mohammad, R. M., Thabtah, F., & McCluskey, L. (2014). Predicting phishing websites
  based on self-structuring neural network. *Neural Computing and Applications*, 25(2), 443–458.
- UCI ML Repository: <https://archive.ics.uci.edu/dataset/327/phishing>
- Scikit-learn documentation: <https://scikit-learn.org/>
- Streamlit documentation: <https://docs.streamlit.io/>
