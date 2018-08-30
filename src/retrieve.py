from __future__ import absolute_import, division, print_function

import os
import glob
import re
import argparse
import logging
import time
import datetime

import numpy as np
import pandas as pd

import faiss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feats_index_npz')
    parser.add_argument('--feats_query_npz')
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--k', type=int, default=64)
    parser.add_argument('--M', type=int, default=2**8)
    parser.add_argument('--nlist', type=long, default=2**13)
    parser.add_argument('--index_choice', choices=['flat', 'pq'], default='flat')
    parser.add_argument('--out_fname')

    args = parser.parse_args()

    with np.load(args.feats_index_npz) as data:
        ids_index = data['id']
        feats_index = data['descriptor']

    with np.load(args.feats_query_npz) as data:
        ids_query = data['id']
        feats_query = data['descriptor']

    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)
    logging.info('Arguments: %s', args)

    d = feats_index.shape[1]
    logging.info('index size=%d, query size=%d', ids_index.shape[0], ids_query.shape[0])
    logging.info('d = %d, k = %d', d, args.k)

    stn = time.time()
    if args.norm:
        feats_index = feats_index / np.linalg.norm(feats_index, axis=1, keepdims=True)
        feats_query = feats_query / np.linalg.norm(feats_query, axis=1, keepdims=True)
        logging.info('Feats normalization took %.2f', (time.time() - stn))

    nbits = 2**3

    if args.index_choice == 'flat':
        index = faiss.IndexFlatL2(d)
    elif args.index_choice == 'pq':
        index = faiss.IndexPQ(d, args.M, nbits)
        stt = time.time()
        index.train(feats_index)
        logging.info('Train took %.2f', (time.time() - stt))

    sti = time.time()
    index.add(feats_index)
    logging.info('Index took %.2f', (time.time() - sti))

    sts = time.time()
    D, I = index.search(feats_query[:args.limit] if args.limit > 0 else feats_query, args.k)
    logging.info('Search took %.2f', (time.time() - sts))

    results = np.empty_like(I, dtype='S8')
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            results[i, j] = ids_index[I[i, j]]

    np.savez(args.out_fname, id=ids_query, results=results, distances=D)
    logging.info('File logging to %s', args.out_fname)
