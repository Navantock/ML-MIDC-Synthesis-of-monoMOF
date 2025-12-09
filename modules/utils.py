from sklearn.metrics import make_scorer, f1_score, fbeta_score, precision_score, recall_score
from sklearn.metrics import f1_score


def _f1_class_i(y_true, y_pred, class_i):
    return f1_score(y_true, y_pred, average=None, labels=[class_i], zero_division=0)[0]

def _fbeta_class_i(y_true, y_pred, class_i, beta=1.0):
    return fbeta_score(y_true, y_pred, beta=beta, average=None, labels=[class_i], zero_division=0)[0]

def _precision_class_i(y_true, y_pred, class_i):
    return precision_score(y_true, y_pred, average=None, labels=[class_i], zero_division=0)[0]

def _recall_class_i(y_true, y_pred, class_i):
    return recall_score(y_true, y_pred, average=None, labels=[class_i], zero_division=0)[0]

def get_class_i_scorer(class_i, score: str, **score_func_kwargs):
    if score == "f1":
        return make_scorer(_f1_class_i, class_i=class_i)
    elif score == "fbeta":
        return make_scorer(_fbeta_class_i, class_i=class_i, **score_func_kwargs)
    elif score == "precision":
        return make_scorer(_precision_class_i, class_i=class_i)
    elif score == "recall":
        return make_scorer(_recall_class_i, class_i=class_i)
    else:
        raise ValueError(f"Unknown score: {score}. Supported scores are 'f1', 'precision', and 'recall'.")
    