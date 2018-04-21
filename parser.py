import os
import sys
import re
import argparse

import pretty_print

REGEX = {
    'image score': 'image score: (.*?)\s',
    'patient score': 'patient score: (.*?)\s'
}

def parse_scores_file(filename, regex_dict):
    scores = {'train': {}, 'test': {}}

    with open(filename) as results_file:
        results = results_file.read()

    for regex in regex_dict:
        regexp = re.compile(regex_dict[regex])
        f_all = regexp.findall(results)
        scores['train'][regex] = f_all[0]
        scores['test'][regex] = f_all[1]

    return scores

def parse_from_dir(src_dir):
    scores = {}

    for route, subdirs, files in os.walk(src_dir):
        for filename in files:
            if 'results.txt' in filename:
                fold = re.search('(fold\d)', route).group(1)
                resolution = re.search('(\d\d\d?X)', route).group(1)

                if not filename in scores:
                    scores[filename] = {}
                if not fold in scores[filename]:
                    scores[filename][fold] = {}
                scores[filename][fold][resolution] = parse_scores_file('%s/%s' % (route, filename), REGEX)

    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser( description = 'Parsing utilities for the cancer results' )
    parser.add_argument( '-s', dest = 'source', metavar = 'Source directory', default = 'output/folds_rescale_350x230', help = 'Root location of the result files' )

    args = parser.parse_args()

    source_dir = args.source

    results = parse_from_dir(source_dir)

    experiments = results.keys()

    orig_stdout = sys.stdout
    for experiment in experiments:
        with open('%s/%sresults_summary.txt' % (source_dir, experiment[:-11]), 'w') as f:
            sys.stdout = f
            pretty_print.print_results_training(results[experiment])
    sys.stdout = orig_stdout


