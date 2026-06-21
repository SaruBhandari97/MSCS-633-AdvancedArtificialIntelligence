"""
Fraud Detection with a Deep AutoEncoder (PyOD)
==============================================

Course   : MSCS-633 Advanced Artificial Intelligence
Author   : Saru Bhandari(SaruBhandari97)

This script builds an unsupervised fraud-detection model on the anonymized
Kaggle credit-card transactions dataset using the AutoEncoder detector from
PyOD (https://pyod.readthedocs.io).

How an AutoEncoder finds fraud
------------------------------
An AutoEncoder is a neural network that is trained to copy its input to its
output through a narrow "bottleneck" layer. Because the network only has
enough capacity to learn the dominant patterns in the data, it reconstructs
ordinary (legitimate) transactions well and unusual (fraudulent) ones poorly.
The per-row reconstruction error therefore acts as an anomaly score: the
larger the error, the more likely the transaction is fraudulent.

Dataset
-------
Download `creditcard.csv` from Kaggle and place it next to this file:
    https://www.kaggle.com/datasets/whenamancodes/fraud-detection

Columns: Time, V1..V28 (PCA components), Amount, Class (0 = genuine, 1 = fraud).

Usage
-----
    python fraud_detection_autoencoder.py --data creditcard.csv

Outputs the evaluation metrics to the console and saves the figures referenced
in the report into the ./outputs directory.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend so the script runs headless
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)

from pyod.models.auto_encoder import AutoEncoder

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.30
OUTPUT_DIR = "outputs"

# Reproducibility: fix the seeds for NumPy so repeated runs are comparable.
np.random.seed(RANDOM_STATE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fraud_detection")


# --------------------------------------------------------------------------- #
# Data loading and preprocessing
# --------------------------------------------------------------------------- #
def load_data(path: str) -> pd.DataFrame:
    """Load the credit-card transactions CSV into a DataFrame.

    Args:
        path: Path to ``creditcard.csv``.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the dataset cannot be found at ``path``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Download the dataset from Kaggle "
            "(whenamancodes/fraud-detection) and place creditcard.csv beside "
            "this script."
        )
    df = pd.read_csv(path)
    logger.info("Loaded %d transactions with %d columns.", df.shape[0], df.shape[1])
    fraud = int(df["Class"].sum())
    logger.info(
        "Class balance: %d genuine / %d fraud (%.3f%% fraud).",
        len(df) - fraud,
        fraud,
        100.0 * fraud / len(df),
    )
    return df


def preprocess(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Split the data and standardize the features.

    The ``Time`` and ``Amount`` columns are on very different scales from the
    anonymized V1..V28 PCA components, so every feature is standardized to zero
    mean and unit variance. The scaler is fit on the training split only to
    avoid leaking information from the test split.

    Args:
        df: Raw dataset including the ``Class`` label.

    Returns:
        X_train, X_test, y_train, y_test, contamination
        where ``contamination`` is the observed fraud rate in the training
        split, used by PyOD to place the decision threshold.
    """
    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].values
    y = df["Class"].values

    # Stratify so the tiny fraud class is represented in both splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    contamination = float(np.mean(y_train))
    contamination = min(max(contamination, 1e-4), 0.5)  # keep within valid range
    logger.info("Estimated contamination (fraud rate): %.5f", contamination)

    return X_train, X_test, y_train, y_test, contamination


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(contamination: float) -> AutoEncoder:
    """Create a PyOD AutoEncoder configured for this dataset.

    The encoder compresses the ~30 input features down to an 8-unit bottleneck
    (30 -> 32 -> 16 -> 8) and the decoder mirrors that back out. Batch
    normalization and dropout regularize the network so it generalizes to the
    dominant "genuine transaction" pattern instead of memorizing the data.

    Args:
        contamination: Expected proportion of anomalies, used to set the
            score threshold that converts anomaly scores into 0/1 labels.

    Returns:
        An unfitted ``AutoEncoder`` instance.
    """
    return AutoEncoder(
        hidden_neuron_list=[32, 16, 8],  # encoder shape; decoder mirrors it
        hidden_activation_name="relu",
        batch_norm=True,
        dropout_rate=0.2,
        epoch_num=30,
        batch_size=256,
        lr=1e-3,
        contamination=contamination,
        random_state=RANDOM_STATE,
        verbose=1,
    )


def train_model(model: AutoEncoder, X_train: np.ndarray) -> AutoEncoder:
    """Fit the AutoEncoder on the (mostly genuine) training data.

    This is unsupervised: the labels are never shown to the model. The network
    simply learns to reconstruct the bulk pattern of normal spending behavior.
    """
    logger.info("Training AutoEncoder for %d epochs...", model.epoch_num)
    model.fit(X_train)
    logger.info("Training complete.")
    return model


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(model: AutoEncoder, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Score the test set and print fraud-detection metrics.

    Because fraud is rare (~0.17%), plain accuracy is misleading, so we report
    ROC-AUC and Average Precision (area under the precision-recall curve), which
    are the standard metrics for imbalanced anomaly detection.

    Returns:
        A dictionary of the computed metrics plus the raw scores and labels,
        which the plotting helpers consume.
    """
    # decision_function returns the reconstruction-error anomaly score.
    scores = model.decision_function(X_test)
    # predict applies the contamination threshold -> 0 (genuine) / 1 (fraud).
    y_pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, scores)
    avg_precision = average_precision_score(y_test, scores)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["Genuine", "Fraud"], digits=4
    )

    logger.info("ROC-AUC            : %.4f", roc_auc)
    logger.info("Average Precision  : %.4f", avg_precision)
    logger.info("Confusion matrix (rows=true, cols=pred):\n%s", cm)
    logger.info("Classification report:\n%s", report)

    return {
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "confusion_matrix": cm,
        "report": report,
        "scores": scores,
        "y_test": y_test,
        "y_pred": y_pred,
    }


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def make_plots(results: dict, output_dir: str = OUTPUT_DIR) -> str:
    """Save a 2x2 figure summarizing the experiment.

    Panels: reconstruction-error distribution, ROC curve, precision-recall
    curve, and the confusion matrix. Returns the saved file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    scores = results["scores"]
    y_test = results["y_test"]
    cm = results["confusion_matrix"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (1) Reconstruction-error distribution by true class.
    ax = axes[0, 0]
    ax.hist(
        scores[y_test == 0], bins=60, alpha=0.7, label="Genuine", color="#2a9d8f",
        density=True,
    )
    ax.hist(
        scores[y_test == 1], bins=60, alpha=0.7, label="Fraud", color="#e76f51",
        density=True,
    )
    ax.set_title("Reconstruction Error Distribution")
    ax.set_xlabel("Anomaly score (reconstruction error)")
    ax.set_ylabel("Density")
    ax.legend()

    # (2) ROC curve.
    ax = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_test, scores)
    ax.plot(fpr, tpr, color="#264653", lw=2,
            label=f"AUC = {results['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    # (3) Precision-Recall curve (more informative under heavy imbalance).
    ax = axes[1, 0]
    precision, recall, _ = precision_recall_curve(y_test, scores)
    ax.plot(recall, precision, color="#e9c46a", lw=2,
            label=f"AP = {results['avg_precision']:.4f}")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right")

    # (4) Confusion matrix.
    ax = axes[1, 1]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Genuine", "Fraud"])
    ax.set_yticklabels(["Genuine", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=12,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "fraud_detection_results.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_pipeline(df: pd.DataFrame) -> dict:
    """Run preprocess -> train -> evaluate -> plot on a loaded DataFrame."""
    X_train, X_test, y_train, y_test, contamination = preprocess(df)
    model = build_model(contamination)
    model = train_model(model, X_train)
    results = evaluate(model, X_test, y_test)
    make_plots(results)
    return results


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="AutoEncoder fraud detection (PyOD).")
    parser.add_argument(
        "--data",
        default="creditcard.csv",
        help="Path to the Kaggle creditcard.csv file.",
    )
    args = parser.parse_args()

    df = load_data(args.data)
    run_pipeline(df)
    logger.info("Done. See the ./%s directory for the saved figure.", OUTPUT_DIR)


if __name__ == "__main__":
    main()
