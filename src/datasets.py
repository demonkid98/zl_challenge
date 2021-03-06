from __future__ import absolute_import, division, print_function

import os
from torch.utils import data
from PIL import Image
import logging
import glob


class ZaloLandmarkDataset(data.Dataset):
    '''
        records: data-frame with 2 columns: id and category
    '''
    def __init__(self, root_dir, records, transform=None):
        self.root_dir = root_dir
        self.records = records
        self.transform = transform

    def __len__(self):
        return self.records.shape[0]

    def __getitem__(self, idx):
        record = self.records.iloc[idx]
        img_name = os.path.join(self.root_dir, str(record['category']), '{}.jpg'.format(record['id']))
        try:
            image = Image.open(img_name).convert('RGB')
            if self.transform:
                image = self.transform(image)

        except IOError:
            logging.error('IO error %s', img_name)
            return None

        return image, record['id'], record['category']

class ZaloUnlabeledLandmarkDataset(data.Dataset):
    '''
        ids: array of ids
    '''
    def __init__(self, root_dir, pattern='*.jpg', transform=None):
        self.ext = pattern.split('.')[-1]
        files = glob.glob(os.path.join(root_dir, pattern))

        ids = {}
        for f in files:
            uid = f.split('/')[-1].replace('.{}'.format(self.ext), '')
            ids[uid] = f

        self.root_dir = root_dir
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        uid = self.ids.keys()[idx]
        img_name = self.ids[uid]
        try:
            image = Image.open(img_name).convert('RGB')
            if self.transform:
                image = self.transform(image)

        except IOError:
            logging.error('IO error %s', img_name)
            return None

        return image, uid
