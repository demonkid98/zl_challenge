from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import re
import os
import argparse
import logging
import glob
import time
import datetime


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_npz', required=True)
    parser.add_argument('--ref_csv', required=True)
    parser.add_argument('--out_fname', required=True)
    parser.add_argument('--k', default=1, type=int)
    parser.add_argument('--nb_picks', default=3, type=int)
    parser.add_argument('--log_freq', default=1000, type=int)
    parser.add_argument('--log_level', default='INFO')
    args = parser.parse_args()


    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=args.log_level)
    logging.info('Arguments: %s', args)

    now = datetime.datetime.now()
    st = time.time()

    with np.load(args.input_npz) as data:
        ids_test = data['id']
        results = data['results']

    df_ref = pd.read_csv(args.ref_csv)
    df_results = pd.DataFrame(data={'id': ids_test, 'predicted': ''}).set_index('id')
    nb_classes = df_ref['category'].max() + 1

    for i in range(ids_test.shape[0]):
        if (i + 1) % args.log_freq == 0:
            logging.info('%d/%d %.2fs', i + 1, ids_test.shape[0], time.time() - st)

        uid = ids_test[i]
        refs = results[i, :args.k]

        predicted = np.random.permutation(nb_classes)[:args.nb_picks]
        if len(refs) > 0:
            # df_ref.at[uid, 'predicted'] = ' '.np.random.permutation(nb_classes)[:args.k]
            klasses = df_ref[df_ref['id'].isin(refs)]['category'].values

            values, counts = np.unique(klasses, return_counts=True)
            sorted_keys = np.argsort(counts)[::-1]
            for k in range(min(sorted_keys.shape[0], args.nb_picks)):
                predicted[k] = values[sorted_keys[k]]

        df_results.at[uid, 'predicted'] = ' '.join([str(v) for v in predicted])

    df_results.reset_index().to_csv(args.out_fname, index=False, columns=['id', 'predicted'])
    logging.info('File logging to %s', args.out_fname)
