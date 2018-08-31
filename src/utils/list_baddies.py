from __future__ import absolute_import, division, print_function

import re
import os
import argparse
import logging
import glob

from PIL import Image

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', default='../data/TrainVal/*/*.jpg')
    parser.add_argument('--out_pat', default='.*(TrainVal/.*)')
    parser.add_argument('--out_fname', default='/tmp/TrainVal_baddies.txt')

    args = parser.parse_args()
    logging.info('Arguments: %s', args)

    files = glob.glob(args.pattern)
    logging.info('%d files', len(files))

    pat = re.compile(args.out_pat)
    bads = []
    for f in files:
        try:
            Image.open(f)
        except IOError:
            bads.append(re.sub(pat, r'\1', f))

    with open(args.out_fname, 'w') as of:
        for bad in bads:
            of.write(bad)
            of.write('\n')
    logging.info('Baddies logged to %s', args.out_fname)
