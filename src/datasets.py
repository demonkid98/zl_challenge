from __future__ import absolute_import, division, print_function

import os
from torch.utils import data
from PIL import Image



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
            image = Image.open(img_name)
        except IOError:
            return None

        if self.transform:
            image = self.transform(image)

        return image, record['id'], record['category']
