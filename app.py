"""
PhishGuard AI — Explainable Phishing Detection and Risk Analysis
================================================================
A Streamlit-based cybersecurity dashboard that uses machine learning
to detect phishing websites from the UCI Phishing Websites dataset.

Dataset: UCI ML Repository – Phishing Websites (ID 327)
Source : https://archive.ics.uci.edu/dataset/327/phishing
License: Creative Commons Attribution 4.0 (CC BY 4.0)

All Python application code is kept in this single file as required.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import io
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
DATA_DIR = "data"
DATASET_FILENAME = "phishing.csv"
DATASET_PATH = os.path.join(DATA_DIR, DATASET_FILENAME)

# UCI dataset column names — lowercase as exported by ucimlrepo (30 features + 1 target)
# Source: https://archive.ics.uci.edu/dataset/327/phishing
FEATURE_COLUMNS = [
    "having_ip_address", "url_length", "shortining_service",
    "having_at_symbol", "double_slash_redirecting", "prefix_suffix",
    "having_sub_domain", "sslfinal_state", "domain_registration_length",
    "favicon", "port", "https_token", "request_url", "url_of_anchor",
    "links_in_tags", "sfh", "submitting_to_email", "abnormal_url",
    "redirect", "on_mouseover", "rightclick", "popupwindow", "iframe",
    "age_of_domain", "dnsrecord", "web_traffic", "page_rank",
    "google_index", "links_pointing_to_page", "statistical_report",
]
TARGET_COLUMN = "result"

# Human-readable feature descriptions (keyed by lowercase column name)
FEATURE_DESCRIPTIONS = {
    "having_ip_address":          "Uses IP address instead of domain name",
    "url_length":                 "Length of the URL string",
    "shortining_service":         "Uses URL shortening service",
    "having_at_symbol":           "Contains @ symbol in URL",
    "double_slash_redirecting":   "Contains // redirect in URL path",
    "prefix_suffix":              "Has hyphen prefix/suffix in domain",
    "having_sub_domain":          "Number of subdomains present",
    "sslfinal_state":             "SSL certificate validity/presence",
    "domain_registration_length": "Domain registration duration",
    "favicon":                    "Favicon loaded from external domain",
    "port":                       "Non-standard port in URL",
    "https_token":                "HTTPS token appears in domain part",
    "request_url":                "Percentage of external request URLs",
    "url_of_anchor":              "Percentage of anchor hrefs pointing outside",
    "links_in_tags":              "Links in <meta>, <script>, <link> tags",
    "sfh":                        "Server Form Handler destination",
    "submitting_to_email":        "Form submits data to email",
    "abnormal_url":               "URL doesn't match domain WHOIS",
    "redirect":                   "Number of redirects",
    "on_mouseover":               "Status bar changed via onMouseOver",
    "rightclick":                 "Right-click disabled",
    "popupwindow":                "Pop-up window with text field",
    "iframe":                     "Invisible iframe used",
    "age_of_domain":              "Age of registered domain",
    "dnsrecord":                  "DNS record found for domain",
    "web_traffic":                "Alexa traffic rank",
    "page_rank":                  "Google PageRank",
    "google_index":               "Whether page is indexed by Google",
    "links_pointing_to_page":     "Number of external links to page",
    "statistical_report":         "Host/IP flagged in statistical reports",
}

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark cybersecurity theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---- base ---- */
html, body, [class*="css"] {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background-color: #0d1117;
    color: #c9d1d9;
}
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }

/* ---- headings ---- */
h1 { color: #58a6ff !important; font-weight: 700; }
h2, h3 { color: #79c0ff !important; }
h4, h5, h6 { color: #cae8ff !important; }

/* ---- metric cards ---- */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
}
[data-testid="metric-container"] label { color: #8b949e !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #58a6ff !important; font-size: 2rem !important;
}

/* ---- dataframes ---- */
[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 6px; }

/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: #8b949e; }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #58a6ff; border-bottom: 2px solid #58a6ff;
}

/* ---- buttons ---- */
.stButton > button {
    background: #238636; color: #fff; border: none;
    border-radius: 6px; padding: 8px 20px;
}
.stButton > button:hover { background: #2ea043; }

/* ---- expander ---- */
.streamlit-expanderHeader { color: #58a6ff !important; }

/* ---- risk badge ---- */
.risk-high   { background:#da3633; color:#fff; padding:6px 16px; border-radius:20px; font-weight:700; }
.risk-medium { background:#d29922; color:#fff; padding:6px 16px; border-radius:20px; font-weight:700; }
.risk-low    { background:#238636; color:#fff; padding:6px 16px; border-radius:20px; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# SECTION 1: DATA LOADING & VALIDATION
# ===========================================================================

def build_download_instructions() -> str:
    """Return step-by-step instructions for downloading the UCI dataset."""
    return (
        "**How to download the dataset manually:**\n\n"
        "1. Visit https://archive.ics.uci.edu/dataset/327/phishing\n"
        "2. Click the **Download** button to get `phishing.zip`.\n"
        "3. Extract the archive — look for the `.arff` or `.csv` file inside.\n"
        "4. The ARFF file can be opened as plain text; rename/convert it "
        "to `phishing.csv` keeping the 30 feature columns and the `Result` column.\n"
        "5. Place `phishing.csv` inside the `data/` folder of this project.\n\n"
        "Alternatively, run this one-liner in your terminal "
        "(requires `ucimlrepo`):\n"
        "```\npip install ucimlrepo\n"
        "python -c \""
        "from ucimlrepo import fetch_ucirepo; "
        "d=fetch_ucirepo(id=327); "
        "import pandas as pd; "
        "df=pd.concat([d.data.features, d.data.targets],axis=1); "
        "df.to_csv('data/phishing.csv',index=False)\"\n```"
    )


@st.cache_data(show_spinner="Loading dataset…")
def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the phishing dataset from *path*.
    Supports:
      - CSV with a header row (ucimlrepo export, lowercase column names)
      - Headerless ARFF-derived CSV (31 numeric columns)
    Returns a cleaned DataFrame with integer binary labels:
        1 = phishing, 0 = legitimate.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at '{path}'.")

    # ---- attempt header-aware read first ----
    raw = pd.read_csv(path, na_values=["?", "", " "])

    # Normalise all column names to lowercase and strip whitespace
    raw.columns = [c.strip().lower() for c in raw.columns]

    # If header read looks valid (target column present), use it directly
    if TARGET_COLUMN not in raw.columns:
        # Fall back: headerless ARFF/CSV with 31 numeric columns
        raw = pd.read_csv(path, header=None, na_values=["?", "", " "])
        if raw.shape[1] == 31:
            raw.columns = FEATURE_COLUMNS + [TARGET_COLUMN]
        elif raw.shape[1] == 32:
            raw = raw.iloc[:, 1:]
            raw.columns = FEATURE_COLUMNS + [TARGET_COLUMN]
        else:
            raise ValueError(
                f"Unrecognised column count ({raw.shape[1]}) and no header row found. "
                "Expected a CSV with a header row (ucimlrepo export) or 31 numeric columns. "
                "Please verify the dataset file."
            )

    # ---- convert all columns to numeric ----
    for col in raw.columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # ---- validate and map target labels ----
    unique_vals = set(raw[TARGET_COLUMN].dropna().unique().astype(int))
    if unique_vals.issubset({-1, 0, 1}):
        # Original UCI encoding: -1 = phishing, 0 = suspicious, 1 = legitimate
        # Map → binary: phishing=1, legitimate=0
        raw[TARGET_COLUMN] = raw[TARGET_COLUMN].map({-1: 1, 0: 1, 1: 0})
    elif unique_vals.issubset({0, 1}):
        pass  # already binary; assume 1 = phishing
    else:
        raise ValueError(
            f"Unexpected target values: {unique_vals}. "
            "Expected -1/0/1 (UCI original) or 0/1 (pre-mapped)."
        )

    # ---- drop rows with missing target ----
    raw = raw.dropna(subset=[TARGET_COLUMN])
    raw[TARGET_COLUMN] = raw[TARGET_COLUMN].astype(int)

    # ---- validate feature columns are all present ----
    missing_cols = [c for c in FEATURE_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    return raw.reset_index(drop=True)


def try_auto_download() -> bool:
    """
    Attempt to fetch the dataset via the `ucimlrepo` package.
    Returns True if successful, False otherwise.
    """
    try:
        from ucimlrepo import fetch_ucirepo  # type: ignore
        with st.spinner("Attempting automatic download from UCI ML Repository…"):
            ds = fetch_ucirepo(id=327)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(DATASET_PATH, index=False)
        return True
    except Exception:
        return False


# ===========================================================================
# SECTION 2: PREPROCESSING
# ===========================================================================

@st.cache_data(show_spinner="Preprocessing data…")
def preprocess(df: pd.DataFrame):
    """
    Validate, clean, and split the dataset.
    Returns X_train, X_test, y_train, y_test, df_clean.
    """
    df = df.copy()

    # Fill missing feature values with column median
    for col in FEATURE_COLUMNS:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Drop exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test, df


# ===========================================================================
# SECTION 3: MODEL TRAINING & EVALUATION
# ===========================================================================

@st.cache_resource(show_spinner="Training models…")
def train_models(X_train: np.ndarray, y_train: np.ndarray):
    """
    Train Logistic Regression and Random Forest pipelines.
    Caching avoids unnecessary retraining on every widget interaction.
    """
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        )),
    ])
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            n_jobs=-1, class_weight="balanced"
        )),
    ])
    lr_pipe.fit(X_train, y_train)
    rf_pipe.fit(X_train, y_train)
    return lr_pipe, rf_pipe


def evaluate_model(pipe, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Return a dictionary of evaluation metrics for a fitted pipeline."""
    y_pred = pipe.predict(X_test)
    y_prob = (
        pipe.predict_proba(X_test)[:, 1]
        if hasattr(pipe, "predict_proba")
        else None
    )
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        "confusion_matrix": cm,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


# ===========================================================================
# SECTION 4: EXPLAINABLE AI
# ===========================================================================

@st.cache_data(show_spinner="Computing feature importance…")
def compute_rf_feature_importance(_rf_pipe) -> pd.DataFrame:
    """
    Extract built-in Random Forest feature importances.
    Returns a DataFrame sorted by importance descending.
    """
    importances = _rf_pipe.named_steps["clf"].feature_importances_
    df_imp = pd.DataFrame({
        "Feature":     FEATURE_COLUMNS,
        "Importance":  importances,
        "Description": [FEATURE_DESCRIPTIONS.get(f, "") for f in FEATURE_COLUMNS],
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    return df_imp


@st.cache_data(show_spinner="Computing permutation importance…")
def compute_permutation_importance(_lr_pipe, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """
    Compute permutation-based feature importance for Logistic Regression.
    More reliable than coefficient magnitudes for mixed-scale features.
    """
    result = permutation_importance(
        _lr_pipe, X_test, y_test,
        n_repeats=10, random_state=RANDOM_STATE, scoring="f1", n_jobs=-1
    )
    df_imp = pd.DataFrame({
        "Feature":            FEATURE_COLUMNS,
        "Importance":         result.importances_mean,
        "Importance_Std":     result.importances_std,
        "Description":        [FEATURE_DESCRIPTIONS.get(f, "") for f in FEATURE_COLUMNS],
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    return df_imp


def explain_single_prediction(pipe, row: np.ndarray, rf_importance: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a simple per-feature explanation for one prediction.
    Uses the feature values and global importance ranking to highlight
    the top factors that influenced the prediction.
    NOTE: This is an approximation based on global importance, not
    a per-instance SHAP explanation. It identifies which globally
    important features have 'suspicious' values (≠ 1) for this record.
    """
    top_features = rf_importance.head(10)["Feature"].tolist()
    feature_vals = dict(zip(FEATURE_COLUMNS, row.flatten()))

    rows = []
    for feat in top_features:
        val = feature_vals.get(feat, 0)
        # In the UCI dataset: +1 is generally 'safe', -1 is 'suspicious'
        suspicious = val == -1
        rows.append({
            "Feature":      feat,
            "Value":        val,
            "Suspicious":   suspicious,
            "Description":  FEATURE_DESCRIPTIONS.get(feat, ""),
            "Importance":   float(rf_importance[rf_importance["Feature"] == feat]["Importance"].values[0]),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# SECTION 5: CHART HELPERS
# ===========================================================================

DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
BORDER    = "#30363d"
BLUE      = "#58a6ff"
GREEN     = "#3fb950"
RED       = "#f85149"
ORANGE    = "#d29922"
TEXT_CLR  = "#c9d1d9"


def _fig_style(fig: plt.Figure, ax):
    """Apply consistent dark theme styling to a matplotlib figure."""
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(BLUE)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)


def plot_class_distribution(df: pd.DataFrame) -> plt.Figure:
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["Legitimate (0)", "Phishing (1)"]
    colors = [GREEN, RED]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor=BORDER, linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f"{val:,}", ha="center", va="bottom", color=TEXT_CLR, fontsize=10
        )
    ax.set_title("Class Distribution", fontsize=12, pad=10)
    ax.set_ylabel("Record Count")
    _fig_style(fig, ax)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> plt.Figure:
    labels = np.array([["TN\n(Legit→Legit)", "FP\n(Legit→Phish)"],
                        ["FN\n(Phish→Legit)", "TP\n(Phish→Phish)"]])
    annot = np.array([[f"{cm[0,0]}\n{labels[0,0]}", f"{cm[0,1]}\n{labels[0,1]}"],
                       [f"{cm[1,0]}\n{labels[1,0]}", f"{cm[1,1]}\n{labels[1,1]}"]])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=annot, fmt="", cmap="Blues",
        linewidths=0.5, linecolor=BORDER,
        xticklabels=["Predicted: Legit", "Predicted: Phish"],
        yticklabels=["Actual: Legit", "Actual: Phish"],
        ax=ax, cbar=False,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=11, pad=10)
    _fig_style(fig, ax)
    fig.tight_layout()
    return fig


def plot_roc_curves(lr_metrics: dict, rf_metrics: dict, y_test: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    for metrics, name, color in [
        (lr_metrics, "Logistic Regression", BLUE),
        (rf_metrics, "Random Forest", GREEN),
    ]:
        if metrics["y_prob"] is not None:
            fpr, tpr, _ = roc_curve(y_test, metrics["y_prob"])
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{name} (AUC={metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], ":", color=BORDER, lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontsize=12, pad=10)
    ax.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT_CLR, fontsize=9)
    _fig_style(fig, ax)
    fig.tight_layout()
    return fig


def plot_feature_importance(df_imp: pd.DataFrame, title: str, top_n: int = 15) -> plt.Figure:
    top = df_imp.head(top_n).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [BLUE if i >= len(top) - 5 else "#3d5a80" for i in range(len(top))]
    ax.barh(top["Feature"], top["Importance"], color=colors, edgecolor=BORDER, linewidth=0.5)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Importance Score")
    _fig_style(fig, ax)
    fig.tight_layout()
    return fig


def plot_metric_comparison(lr_metrics: dict, rf_metrics: dict) -> go.Figure:
    metric_names = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    metric_keys  = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    lr_vals = [lr_metrics.get(k, 0) or 0 for k in metric_keys]
    rf_vals = [rf_metrics.get(k, 0) or 0 for k in metric_keys]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Logistic Regression", x=metric_names, y=lr_vals,
        marker_color=BLUE, text=[f"{v:.3f}" for v in lr_vals],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Random Forest", x=metric_names, y=rf_vals,
        marker_color=GREEN, text=[f"{v:.3f}" for v in rf_vals],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group", title="Model Performance Comparison",
        paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_CLR, size=11),
        legend=dict(bgcolor=PANEL_BG, bordercolor=BORDER),
        yaxis=dict(range=[0, 1.12], gridcolor=BORDER),
        xaxis=dict(gridcolor=BORDER),
        height=420,
    )
    return fig


# ===========================================================================
# SECTION 6: DASHBOARD — individual tab renderers
# ===========================================================================

def render_dataset_tab(df: pd.DataFrame):
    st.subheader("📂 Dataset Analysis")

    # ---- basic stats ----
    n_rows, n_feats = len(df), len(FEATURE_COLUMNS)
    n_legit   = int((df[TARGET_COLUMN] == 0).sum())
    n_phish   = int((df[TARGET_COLUMN] == 1).sum())
    n_missing = int(df[FEATURE_COLUMNS].isnull().sum().sum())
    n_dupes   = int(df.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",   f"{n_rows:,}")
    c2.metric("Features",        f"{n_feats}")
    c3.metric("Missing Values",  f"{n_missing}")
    c4.metric("Duplicate Rows",  f"{n_dupes}")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("#### Class Distribution")
        st.pyplot(plot_class_distribution(df))
        st.caption(
            f"**Legitimate (0):** {n_legit:,} ({n_legit/n_rows*100:.1f}%)  |  "
            f"**Phishing (1):** {n_phish:,} ({n_phish/n_rows*100:.1f}%)"
        )
    with col_right:
        st.markdown("#### Dataset Statistics")
        stats = df[FEATURE_COLUMNS].describe().T[["mean","std","min","max"]]
        stats = stats.round(3)
        st.dataframe(stats, height=320, use_container_width=True)

    st.markdown("#### Dataset Preview (first 100 rows)")
    st.dataframe(df.head(100), use_container_width=True)

    # ---- download ----
    csv_buf = io.BytesIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download full dataset (CSV)",
        data=csv_buf.getvalue(),
        file_name="phishing_dataset.csv",
        mime="text/csv",
    )

    with st.expander("📖 Feature Descriptions"):
        desc_df = pd.DataFrame(
            [(f, d) for f, d in FEATURE_DESCRIPTIONS.items()],
            columns=["Feature", "Description"]
        )
        st.dataframe(desc_df, use_container_width=True)

    with st.expander("ℹ️ About the Dataset"):
        st.markdown("""
**Source:** UCI Machine Learning Repository — Phishing Websites (ID 327)  
**URL:** https://archive.ics.uci.edu/dataset/327/phishing  
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)  
**Citation:** Mohammad, R. M., Thabtah, F., & McCluskey, L. (2014).

The dataset contains **11,055 website records** with 30 pre-extracted features 
(e.g. URL characteristics, SSL status, domain age) and a binary target label  
(`-1 = phishing`, `1 = legitimate` in the original encoding, mapped to `1/0` here).  
Features are already encoded as categorical integers (-1, 0, 1) so no raw URLs 
are present in the dataset.
        """)


def render_model_tab(lr_metrics: dict, rf_metrics: dict,
                     y_test: np.ndarray, best_name: str):
    st.subheader("🤖 Model Evaluation")

    st.markdown("""
> **Why accuracy alone is not enough for phishing detection:**  
> A model that labels everything as "legitimate" on an 80/20 split would achieve 80% accuracy  
> but would miss every phishing site. **Recall** (sensitivity) on the phishing class is critical —  
> a **False Negative** (missed phishing site) is far more dangerous than a **False Positive**  
> (flagging a safe site). We therefore prioritise **F1-score** and **Recall** as primary metrics.
    """)

    # ---- comparison chart ----
    st.plotly_chart(plot_metric_comparison(lr_metrics, rf_metrics),
                    use_container_width=True)

    # ---- side-by-side metrics ----
    col1, col2 = st.columns(2)
    for col, metrics, name in [(col1, lr_metrics, "Logistic Regression"),
                                (col2, rf_metrics, "Random Forest")]:
        with col:
            tag = " ✅ (selected)" if name == best_name else ""
            st.markdown(f"#### {name}{tag}")
            m1, m2 = st.columns(2)
            m1.metric("Accuracy",  f"{metrics['accuracy']:.4f}")
            m2.metric("Precision", f"{metrics['precision']:.4f}")
            m3, m4 = st.columns(2)
            m3.metric("Recall",    f"{metrics['recall']:.4f}")
            m4.metric("F1-score",  f"{metrics['f1']:.4f}")
            if metrics["roc_auc"]:
                st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            st.markdown(
                f"**FP:** {metrics['FP']}  |  **FN:** {metrics['FN']}  |  "
                f"**TP:** {metrics['TP']}  |  **TN:** {metrics['TN']}"
            )
            st.pyplot(plot_confusion_matrix(metrics["confusion_matrix"], name))

    # ---- ROC curves ----
    st.markdown("#### ROC Curves")
    st.pyplot(plot_roc_curves(lr_metrics, rf_metrics, y_test))

    # ---- model selection rationale ----
    with st.expander("📌 Model Selection Rationale"):
        st.markdown(f"""
The **{best_name}** was selected as the final model based on the highest F1-score  
on the held-out test set, which balances precision and recall for the phishing class.

For phishing detection:
- **False Negatives (phishing labelled as legitimate)** are the most dangerous error —  
  they allow phishing sites to go undetected.
- **False Positives (legitimate labelled as phishing)** cause inconvenience but not harm.
- A model with higher recall at acceptable precision is therefore preferred.
        """)


def render_xai_tab(df_imp_rf: pd.DataFrame, df_imp_lr: pd.DataFrame):
    st.subheader("🔍 Explainable AI — Feature Importance")

    st.info(
        "**Limitation notice:** The feature importance values shown here are global "
        "averages across the entire dataset. They indicate which features the model "
        "relies on most, but do not prove causation and do not guarantee any individual "
        "prediction is correct."
    )

    tab_rf, tab_lr = st.tabs(["Random Forest (built-in)", "Logistic Regression (permutation)"])

    with tab_rf:
        st.markdown(
            "Random Forest feature importance is based on the mean decrease in "
            "impurity (Gini importance) across all trees."
        )
        st.pyplot(plot_feature_importance(df_imp_rf, "RF Feature Importance — Top 15"))
        top15 = df_imp_rf.head(15)[["Feature","Importance","Description"]].copy()
        top15["Importance"] = top15["Importance"].round(4)
        st.dataframe(top15, use_container_width=True)

    with tab_lr:
        st.markdown(
            "Permutation importance measures how much the F1-score drops when each "
            "feature's values are randomly shuffled (averaged over 10 repeats)."
        )
        st.pyplot(plot_feature_importance(df_imp_lr, "LR Permutation Importance — Top 15"))
        top15_lr = df_imp_lr.head(15)[["Feature","Importance","Importance_Std","Description"]].copy()
        top15_lr["Importance"]     = top15_lr["Importance"].round(4)
        top15_lr["Importance_Std"] = top15_lr["Importance_Std"].round(4)
        st.dataframe(top15_lr, use_container_width=True)


def render_prediction_tab(df: pd.DataFrame, best_pipe,
                           best_name: str, df_imp_rf: pd.DataFrame):
    st.subheader("🎯 Record Analysis & Prediction")

    st.markdown("""
The UCI Phishing Websites dataset contains **pre-extracted website features** 
(not raw URLs). The prediction interface below lets you select any record from 
the dataset and see the model's classification alongside the top contributing features.

> ⚠️ **Disclaimer:** This tool is for educational purposes only. Predictions can be 
> incorrect. Do not rely solely on this tool to determine whether a website is safe.
    """)

    # ---- record selector ----
    n = len(df)
    record_idx = st.slider(
        "Select a dataset record to analyse",
        min_value=0, max_value=n - 1, value=0, step=1,
        help="Move the slider to choose any record from the dataset."
    )

    record = df.iloc[record_idx]
    actual_label = int(record[TARGET_COLUMN])
    row = record[FEATURE_COLUMNS].values.reshape(1, -1)

    # ---- prediction ----
    pred_label   = int(best_pipe.predict(row)[0])
    pred_prob    = (
        float(best_pipe.predict_proba(row)[0][1])
        if hasattr(best_pipe, "predict_proba") else None
    )

    # ---- display ----
    col_res, col_act = st.columns(2)
    with col_res:
        st.markdown("#### Prediction")
        if pred_label == 1:
            st.markdown('<span class="risk-high">🚨 PHISHING</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="risk-low">✅ LEGITIMATE</span>', unsafe_allow_html=True)
        if pred_prob is not None:
            st.metric("Phishing Probability", f"{pred_prob:.1%}")

    with col_act:
        st.markdown("#### Actual Label")
        if actual_label == 1:
            st.markdown('<span class="risk-high">🚨 PHISHING</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="risk-low">✅ LEGITIMATE</span>', unsafe_allow_html=True)
        correct = pred_label == actual_label
        st.markdown(f"**Prediction correct:** {'✅ Yes' if correct else '❌ No'}")

    # ---- explanation ----
    st.markdown("#### Top Contributing Features")
    explanation = explain_single_prediction(best_pipe, row, df_imp_rf)

    for _, erow in explanation.iterrows():
        icon = "🔴" if erow["Suspicious"] else "🟢"
        label = "Suspicious" if erow["Suspicious"] else "Normal"
        st.markdown(
            f"{icon} **{erow['Feature']}** = `{int(erow['Value'])}` — "
            f"*{erow['Description']}* — {label}"
        )

    with st.expander("📋 Full Feature Values for this Record"):
        feat_df = pd.DataFrame({
            "Feature":     FEATURE_COLUMNS,
            "Value":       record[FEATURE_COLUMNS].values.astype(int),
            "Description": [FEATURE_DESCRIPTIONS.get(f, "") for f in FEATURE_COLUMNS],
        })
        st.dataframe(feat_df, use_container_width=True)

    # ---- download explanation ----
    csv_exp = explanation.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download explanation (CSV)",
        data=csv_exp,
        file_name=f"explanation_record_{record_idx}.csv",
        mime="text/csv",
    )


def render_about_tab():
    st.subheader("ℹ️ About PhishGuard AI")

    st.markdown("""
## Project Overview
**PhishGuard AI** is an explainable machine-learning application for phishing website 
detection. It uses the UCI Phishing Websites dataset to train and compare two classifiers 
(Logistic Regression and Random Forest), visualises evaluation metrics, and provides 
feature-level explanations for individual predictions.

## Technology Stack
| Component | Library |
|-----------|---------|
| Dashboard & UI | Streamlit |
| Data manipulation | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Visualisations | Matplotlib, Seaborn, Plotly |
| Explainability | Scikit-learn permutation importance, RF feature importance |
| Language | Python 3.9+ |

## Dataset
- **Name:** UCI Phishing Websites Dataset (ID 327)
- **Source:** https://archive.ics.uci.edu/dataset/327/phishing
- **License:** CC BY 4.0
- **Records:** ~11,055 websites; 30 pre-extracted features + binary target
- **Citation:** Mohammad, R.M., Thabtah, F., & McCluskey, L. (2014)

## Limitations
1. The dataset contains **pre-extracted features**, not raw URLs. Real-time URL 
   scanning would require a separate feature-extraction pipeline.
2. Feature importance shows global averages — individual predictions may differ.
3. The model was trained on a 2014 dataset; phishing techniques evolve over time.
4. This tool is for **educational purposes only**. It does not guarantee detection 
   of all phishing sites and should not replace professional security tools.
5. No real-time threat intelligence is used.

## Responsible Use
- Do not use this tool as the sole security mechanism for any production system.
- Always combine automated tools with human review and established security practices.
- The application never transmits entered data to external services.
    """)


# ===========================================================================
# SECTION 7: SIDEBAR
# ===========================================================================

def render_sidebar(best_name: str, rf_f1: float, lr_f1: float):
    with st.sidebar:
        st.markdown("## 🛡️ PhishGuard AI")
        st.caption("Explainable Phishing Detection")
        st.markdown("---")
        st.markdown("### 🏆 Selected Model")
        st.success(f"**{best_name}**")
        st.markdown("### 📊 Quick Metrics")
        st.markdown(f"- LR F1-score:  `{lr_f1:.4f}`")
        st.markdown(f"- RF F1-score:  `{rf_f1:.4f}`")
        st.markdown("---")
        st.markdown("### 📁 Dataset")
        st.markdown(f"Path: `{DATASET_PATH}`")
        st.markdown("---")
        st.caption(
            "Dataset: UCI Phishing Websites  \n"
            "License: CC BY 4.0  \n"
            "Version: 1.0"
        )


# ===========================================================================
# SECTION 8: MAIN APPLICATION
# ===========================================================================

def main():
    # ---- header ----
    st.markdown("""
<div style="padding:20px 0 10px 0;">
  <h1 style="margin-bottom:2px;">🛡️ PhishGuard AI</h1>
  <p style="color:#8b949e; font-size:1.1rem; margin-top:0;">
    Explainable Phishing Detection and Risk Analysis
  </p>
</div>
""", unsafe_allow_html=True)

    # ====================================================================
    # STEP 1: Load dataset
    # ====================================================================
    if not os.path.exists(DATASET_PATH):
        st.error(
            f"**Dataset not found** at `{DATASET_PATH}`.\n\n"
            "Attempting automatic download…"
        )
        downloaded = try_auto_download()
        if not downloaded:
            st.warning("Automatic download failed. Please follow the manual instructions below.")
            st.markdown(build_download_instructions())
            st.stop()
        else:
            st.success("Dataset downloaded successfully! Reloading…")
            st.rerun()

    try:
        df_raw = load_dataset(DATASET_PATH)
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"**Error loading dataset:** {exc}")
        st.markdown(build_download_instructions())
        st.stop()

    # ====================================================================
    # STEP 2: Preprocess
    # ====================================================================
    X_train, X_test, y_train, y_test, df = preprocess(df_raw)

    # ====================================================================
    # STEP 3: Train models
    # ====================================================================
    lr_pipe, rf_pipe = train_models(X_train, y_train)

    # ====================================================================
    # STEP 4: Evaluate
    # ====================================================================
    lr_metrics = evaluate_model(lr_pipe, X_test, y_test)
    rf_metrics = evaluate_model(rf_pipe, X_test, y_test)

    # Select best model by F1-score
    if rf_metrics["f1"] >= lr_metrics["f1"]:
        best_name, best_pipe, best_metrics = "Random Forest", rf_pipe, rf_metrics
    else:
        best_name, best_pipe, best_metrics = "Logistic Regression", lr_pipe, lr_metrics

    # ====================================================================
    # STEP 5: Feature importance
    # ====================================================================
    df_imp_rf = compute_rf_feature_importance(rf_pipe)
    df_imp_lr = compute_permutation_importance(lr_pipe, X_test, y_test)

    # ====================================================================
    # STEP 6: Top-level metric cards
    # ====================================================================
    n_total  = len(df)
    n_legit  = int((df[TARGET_COLUMN] == 0).sum())
    n_phish  = int((df[TARGET_COLUMN] == 1).sum())
    best_f1  = best_metrics["f1"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📊 Total Records",       f"{n_total:,}")
    m2.metric("✅ Legitimate Records",  f"{n_legit:,}")
    m3.metric("🚨 Phishing Records",    f"{n_phish:,}")
    m4.metric(f"🏆 {best_name} F1",     f"{best_f1:.4f}")

    st.markdown("---")

    # ====================================================================
    # STEP 7: Sidebar
    # ====================================================================
    render_sidebar(best_name, rf_metrics["f1"], lr_metrics["f1"])

    # ====================================================================
    # STEP 8: Tabbed dashboard
    # ====================================================================
    tabs = st.tabs([
        "📂 Dataset Analysis",
        "🤖 Model Evaluation",
        "🔍 Explainable AI",
        "🎯 Record Analysis",
        "ℹ️ About",
    ])

    with tabs[0]:
        render_dataset_tab(df)

    with tabs[1]:
        render_model_tab(lr_metrics, rf_metrics, y_test, best_name)

    with tabs[2]:
        render_xai_tab(df_imp_rf, df_imp_lr)

    with tabs[3]:
        render_prediction_tab(df, best_pipe, best_name, df_imp_rf)

    with tabs[4]:
        render_about_tab()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
