import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_train_test_from_csv(csv_file):
    column_names = ['train img score','train patient score','test img score','test patient score']

    cancer_res = pd.read_csv(csv_file, header = None)
    cancer_res = cancer_res.loc[:, 0:(3 if cancer_res.shape[1] > 3 else 1)]
    cancer_res.columns = column_names[0:cancer_res.shape[1]]

    fig, ax = plt.subplots()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.6, 1)
    for column in cancer_res.columns:
        fmt = ('g' if 'train' in column else 'y') + (':' if 'patient' in column else '-')
        plt.plot(cancer_res[column], fmt, label = column)
    ax.legend(loc = 4);
    plt.savefig(csv_file[:-3] + 'png')

def plot_in_dirs(src_dir):
    for route, subdirs, files in os.walk(src_dir):
        for file in files:
            if len(file) >= 7 and file[-7:] == 'log.csv':
                print('Creating plot for %s/%s' % (route, file))
                plot_train_test_from_csv(route + '/' + file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser( description = 'Plotting utilities for the cancer dataset' )
    parser.add_argument( '-s', dest = 'source', metavar = 'Source directory', required = True, help = 'Location of the log files' )

    args = parser.parse_args()

    source_dir = args.source

    plot_in_dirs(source_dir)
