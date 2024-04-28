from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, multilabel_confusion_matrix, matthews_corrcoef
import numpy as np


def getMetrics(y_true, y_pred):
    result = {
        'acc': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'specificity': specificity_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }

    return result


def specificity_score(y_true, y_pred):
    MCM = multilabel_confusion_matrix(y_true, y_pred)
    tn_sum = MCM[:, 0, 0]
    fp_sum = MCM[:, 0, 1]

    tp_sum = MCM[:, 1, 1]
    fn_sum = MCM[:, 1, 0]

    Condition_negative = tn_sum + fp_sum + 1e-6

    Specificity = tn_sum / Condition_negative
    macro_specificity = np.average(Specificity, weights=None)

    # micro_specificity = np.sum(tn_sum) / np.sum(tn_sum+fp_sum)

    return macro_specificity


if __name__ == '__main__':
    y_true = [1, 0, 0, 1, 1, 1, 0, 1, 0]
    y_pred = [0, 0, 1, 1, 0, 1, 0, 1, 1]
