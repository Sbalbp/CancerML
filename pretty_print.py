from __future__ import division
from functools import reduce

import numpy as np

def print_variable(name, value):
    var_str = '-'*100 + '\n|' + ' '*98 + '|\n|' + ('%s: %.04f' % (name, value)).rstrip('0').rstrip('.').center(98, ' ') + '|\n|' + ' '*98 + '|\n' + '-'*100 + '\n\n'

    print(var_str)

def print_stats(labels, values, spacing = 1):
    # Accuracy stats
    vvalues = np.asarray(values, dtype = np.float)
    real = np.sum(vvalues, 1)
    pred = np.sum(vvalues, 0)
    asum = np.sum(vvalues)

    acc_str = '--- Accuracy ---\n\n'

    acc_str += 'Total accuracy: %s\n\n' % ('%.04f' % (np.sum(np.diagonal(vvalues)) / asum)).rstrip('0').rstrip('.')

    for i, label in enumerate(labels):
        acc_str += 'Class %s(%d):\n\n' % (label, real[i])

        acc_str += '\tRecall:    %s\n' % ('%.04f' % (values[i][i] / real[i]) if real[i] > 0 else '-' ).rstrip('0').rstrip('.')
        acc_str += '\tPrecision: %s\n\n' % ('%.04f' % (values[i][i] / pred[i]) if pred[i] > 0 else '-' ).rstrip('0').rstrip('.')

    # Confusion matrix
    cols = labels[:]
    pvalues = values[:]

    cols.insert(0, 'Real\\Pred')
    col_w = spacing*2 + len(reduce(lambda x, y: x if len(x) >= len(y) else y, cols))
    n_cols = len(cols)
    pvalues.insert(0, labels)

    confusion_matrix_str = '--- Confusion Matrix ---\n\n'

    confusion_matrix_str += '-' * (col_w*n_cols+n_cols+1) + '\n'

    for i, row in enumerate(cols):
        line = '|%s' % row.center(col_w)
        for value in pvalues[i]:
            line = '%s|%s' % (line, str(value).center(col_w))
        line = '%s|\n' % line
        confusion_matrix_str += line
        if i == 0 or i == len(cols)-1:
            confusion_matrix_str += '-' * (col_w*n_cols+n_cols+1) + '\n'

    print(acc_str + confusion_matrix_str)
