import torch
import numpy as np

import matplotlib.pyplot as plt

from typing import List, Dict, Optional, Sequence
import os


def calc_accuracy(y_pred, y_true) -> float:
    return float((y_pred == y_true).sum() / len(y_true))

def calc_precision(y_pred, y_true, y_class_num: int) -> float | List[float]:
    if y_class_num == 2:
        tp = (y_pred * y_true).sum()
        fp = (y_pred * (1 - y_true)).sum()
        return float(tp / (tp + fp)) if tp + fp != 0 else 0
    else:
        precisions = []
        for i in range(y_class_num):
            tp = ((y_pred == i) * (y_true == i)).sum()
            fp = ((y_pred == i) * (y_true != i)).sum()
            precisions.append(float(tp / (tp + fp)) if tp + fp != 0 else 0)
        return precisions

def calc_recall(y_pred, y_true, y_class_num: int) -> List[float]:
    if y_class_num == 2:
        tp = (y_pred * y_true).sum()
        fn = ((1 - y_pred) * y_true).sum()
        return float(tp / (tp + fn)) if tp + fn != 0 else 0
    else:
        recalls = []
        for i in range(y_class_num):
            tp = ((y_pred == i) * (y_true == i)).sum()
            fn = ((y_pred != i) * (y_true == i)).sum()
            recalls.append(float(tp / (tp + fn)) if tp + fn != 0 else 0)
        return recalls
    
def calc_f1_score(y_pred, y_true, y_class_num: int) -> List[float]:
    precisions = calc_precision(y_pred, y_true, y_class_num)
    recalls = calc_recall(y_pred, y_true, y_class_num)
    f1_scores = []
    for precision, recall in zip(precisions, recalls):
        f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall != 0 else 0)
    return f1_scores

def calc_f1_score_by_precision_recall(precision: float | Sequence[float], recall: float | Sequence[float]) -> float | List[float]:
    if isinstance(precision, float) and isinstance(recall, float):
        return 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    elif isinstance(precision, Sequence) and isinstance(recall, Sequence):
        f1_scores = []
        for p, r in zip(precision, recall):
            f1_scores.append(2 * p * r / (p + r) if p + r != 0 else 0)
        return f1_scores
    raise ValueError("Precision and Recall must be either float or Sequence of floats.")

def get_ROC_points(y_prob: torch.Tensor, y_true: torch.Tensor, interval: float = 0.001, get_f1_scores: bool = False):
    fprs = []
    tprs = []
    f1_scores = []
    thresholds = torch.arange(1.0, -interval, -interval)
    if get_f1_scores:
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).float() if threshold != 1.0 else torch.zeros_like(y_prob)
            tp = (y_pred * y_true).sum()
            fn = ((1 - y_pred) * y_true).sum()
            fp = (y_pred * (1 - y_true)).sum()
            tn = ((1 - y_pred) * (1 - y_true)).sum()
            tpr = tp / (tp + fn) if tp + fn != 0 else 0
            fpr = fp / (fp + tn) if fp + tn != 0 else 0
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn != 0 else 0
            tprs.append(tpr)
            fprs.append(fpr)
            f1_scores.append(f1)
        return np.array(fprs), np.array(tprs), thresholds.numpy(), f1_scores
    else:
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).float() if threshold != 1.0 else torch.zeros_like(y_prob)
            tp = (y_pred * y_true).sum()
            fn = ((1 - y_pred) * y_true).sum()
            fp = (y_pred * (1 - y_true)).sum()
            tn = ((1 - y_pred) * (1 - y_true)).sum()
            tpr = tp / (tp + fn) if tp + fn != 0 else 0
            fpr = fp / (fp + tn) if fp + tn != 0 else 0
            tprs.append(tpr)
            fprs.append(fpr)
        return np.array(fprs), np.array(tprs), thresholds.numpy()

def draw_ROC_curve(fprs, tprs, save_path: str):
    # Draw ROC curve
    plt.figure()
    plt.plot(fprs, tprs)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    # shade auc
    plt.fill_between(fprs, tprs, alpha=0.25)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def get_all_classification_metrics(y_prob: torch.Tensor, 
                                   y_pred: torch.Tensor,
                                   y_true: torch.Tensor, 
                                   y_class_num: int, 
                                   result_save_dir: Optional[str] = None, 
                                   save_roc_name: Optional[str] = "roc_curve") -> Dict[str, float | List[float]]:
    results = {}
    results["acc"] = calc_accuracy(y_pred, y_true)
    results["precision"] = calc_precision(y_pred, y_true, y_class_num)
    results["recall"] = calc_recall(y_pred, y_true, y_class_num)
    results["f1_score"] = calc_f1_score_by_precision_recall(results["precision"], results["recall"])
    if result_save_dir is not None:
        if y_class_num == 2:
            # Get TPR & FPR
            y_prob_1 = y_prob[:, 1].reshape(-1, 1)
            fpr, tpr, ths = get_ROC_points(y_prob_1, y_true)
            np.savez(os.path.join(result_save_dir, save_roc_name + ".npz"), fpr=fpr, tpr=tpr, thresholds=ths)

            # Calculate AUC
            auc = np.trapz(tpr, fpr)
            results["auc"] = float(auc)

            draw_ROC_curve(fpr, tpr, os.path.join(result_save_dir, save_roc_name + ".svg"))
    return results