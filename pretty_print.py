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

def print_results_summary(summary):
    data_size = 26
    res_line = '|'
    data_line = {}

    resolutions = summary.keys()
    for i, resolution in enumerate(sorted(resolutions, key=len)):
        res_line += resolution.center(data_size) + '|'

        scores = summary[resolution].keys()
        for score in scores:
            if not score in data_line.keys():
                data_line[score] = '|' + score.center(19) + '|'

            mean = np.mean(summary[resolution][score])
            std = np.std(summary[resolution][score])

            data_line[score] += ('%s +/- %s' % (str(mean), str(std))).center(data_size) + '|'

            if i == len(resolutions)-1:
                data_line[score] += '\n'

    res_line += '\n'

    sep_h_line = ' '*20 + '-'*(len(res_line)-1) + '\n'
    res_line = ' '*20 + res_line
    sep_line = '-'*(len(res_line)-1) + '\n'

    final_line = sep_h_line + res_line + sep_line
    for score in data_line.keys():
        final_line += data_line[score] + sep_line

    print(final_line)

def print_results_training(results):
    data_size = 11

    sums = {}
    table_str = {}

    folds = results.keys()
    fold_line = '|'
    res_line = '|'
    trn_tst_line = {}

    previous_len = 1
    for i, fold in enumerate(folds):
        resolutions = results[fold].keys()
        for j, resolution in enumerate(sorted(resolutions, key=len)):
            res_line += resolution.center(data_size) + '|'
            for types in results[fold][resolution].keys():
                if not types in sums.keys():
                    sums[types] = {}
                if not resolution in sums[types].keys():
                    sums[types][resolution] = {}

                if not types in trn_tst_line.keys():
                    trn_tst_line[types] = {}

                scores = results[fold][resolution][types].keys()
                for score in scores:
                    if not score in sums[types][resolution]:
                        sums[types][resolution][score] = [float(results[fold][resolution][types][score])]
                    else:
                        sums[types][resolution][score] += [float(results[fold][resolution][types][score])]

                    if not score in trn_tst_line[types].keys():
                        trn_tst_line[types][score] = '|' + score.center(19) + '|'

                    trn_tst_line[types][score] += results[fold][resolution][types][score].center(data_size) + '|'
                    if i == len(folds)-1 and j == len(resolutions)-1:
                        trn_tst_line[types][score] += '\n'


        current_len = len(res_line) - previous_len
        previous_len = len(res_line)
        fold_line += fold.center(current_len-1) + '|'
    fold_line += '\n'
    res_line += '\n'

    sep_h_line = ' '*20 + '-'*(len(fold_line)-1) + '\n'
    fold_line = ' '*20 + fold_line
    res_line = ' '*20 + res_line
    sep_line = '-'*(len(fold_line)-1) + '\n'

    for types in trn_tst_line.keys():
        table_str[types] = '\n%s\n\n' % types
        table_str[types] += sep_h_line + fold_line + sep_h_line + res_line
        for score in trn_tst_line[types].keys():
            table_str[types] += sep_line + trn_tst_line[types][score]
        table_str[types] += sep_line

        print(table_str[types])
        print_results_summary(sums[types])

    #trn_table_str += sep_h_line + fold_line + sep_h_line + res_line + sep_line + trn_tst_line['train']['image score'] + sep_line + trn_tst_line['train']['patient score'] + sep_line
    #tst_table_str += sep_h_line + fold_line + sep_h_line + res_line + sep_line + trn_tst_line['test']['image score'] + sep_line + trn_tst_line['test']['patient score'] + sep_line

    """
    print(trn_table_str)
    print_results_summary(sums[])
    print(tst_table_str)
    print_results_summary(sums)
    """
    
