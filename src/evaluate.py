"""
evaluate.py
-----------
Evaluation utilities: metrics, confusion matrix, ROC-AUC plots, feature importance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)


def model_metrics(true, pred, prob, title="Model", labels=None, plot=False):
    """
    Compute a comprehensive set of binary classification metrics.

    Args:
        true: Ground truth labels
        pred: Predicted labels
        prob: Predicted probabilities for the positive class
        title (str): Column name in the returned DataFrame
        labels (list): [negative_label, positive_label]
        plot (bool): Whether to display confusion matrix + ROC curve

    Returns:
        pd.DataFrame: Metrics table
    """
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    fpr, tpr, _ = roc_curve(true, prob)
    labels = labels or ["0", "1"]

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt="g",
                    cmap="YlGnBu", ax=ax[0],
                    xticklabels=labels, yticklabels=labels)
        ax[0].set(title="Confusion Matrix", xlabel="Predicted", ylabel="True")

        ax[1].plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(true, prob):.3f}")
        ax[1].plot([0, 1], [0, 1], "k--", lw=1)
        ax[1].set(title="ROC Curve", xlabel="FPR", ylabel="TPR")
        ax[1].legend(loc="lower right")
        plt.tight_layout()
        plt.show()

    metrics = {
        "Accuracy": (tp + tn) / (tp + fp + tn + fn),
        "Misclassification Rate": 1 - (tp + tn) / (tp + fp + tn + fn),
        "Sensitivity (Recall)": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "Precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "True Positive": tp,
        "False Positive": fp,
        "False Negative": fn,
        "True Negative": tn,
        "F1-Score": f1_score(true, pred),
        "AUC Score": roc_auc_score(true, prob),
        "Matthews Correlation": matthews_corrcoef(true, pred),
    }

    return pd.DataFrame.from_dict(metrics, orient="index", columns=[title])


def plot_feature_importance(features_df, title, top_n=20, color="steelblue"):
    """
    Bar chart of top N most important features.

    Args:
        features_df (pd.DataFrame): Must have 'feature' and 'feature_importance' columns.
        title (str): Chart title.
        top_n (int): Number of features to display.
        color (str): Bar color.
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x="feature_importance",
        y="feature",
        data=features_df.head(top_n),
        color=color,
    )
    plt.title(title, fontsize=16)
    plt.xlabel("Feature Importance", fontsize=12)
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def show_lr_top_features(vectorizer, clf, n=20):
    """
    Display top positive/negative logistic regression coefficients.

    Args:
        vectorizer: Fitted TfidfVectorizer
        clf: Trained LogisticRegression
        n (int): Number of top features per direction

    Returns:
        pd.DataFrame: Full coefficient-feature table sorted descending.
    """
    feature_names = vectorizer.get_feature_names_out()
    coefs = sorted(zip(clf.coef_[0], feature_names))

    top_pos = coefs[:-(n + 1):-1]
    top_neg = coefs[:n]

    df_pos = pd.DataFrame(top_pos, columns=["positive_coeff", "pos_feature"])
    df_neg = pd.DataFrame(top_neg, columns=["negative_coeff", "neg_feature"])
    display_df = df_pos.join(df_neg)
    print(display_df.to_string())

    return pd.DataFrame(coefs, columns=["feature_importance", "feature"]).sort_values(
        "feature_importance", ascending=False
    )


def combine_metrics(*metric_dfs):
    """Concatenate multiple model metric DataFrames side by side."""
    combined = pd.concat(metric_dfs, axis=1, join="inner")
    return combined.reset_index().rename(columns={"index": "metric"})
