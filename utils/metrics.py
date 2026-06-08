import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def compute_metrics(y_true, y_pred, class_names):

    cm = confusion_matrix(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        zero_division=0
    )

    rows = []

    for i, cls in enumerate(class_names):

        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP

        specificity = TN / (TN + FP + 1e-12)

        rows.append([
            cls,
            precision[i],
            recall[i],
            f1[i],
            specificity
        ])

    df = pd.DataFrame(
        rows,
        columns=["Class", "Precision", "Recall", "F1", "Specificity"]
    )

    # macro averages
    macro = {
        "Precision": np.mean(precision),
        "Recall": np.mean(recall),
        "F1": np.mean(f1),
        "Specificity": np.mean([r[4] for r in rows])
    }

    return df, macro, cm