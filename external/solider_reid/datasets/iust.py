# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp
import pandas as pd
from config import cfg
from .bases import BaseImageDataset
from collections import defaultdict
import pickle
class IUSTPersonReID(BaseImageDataset):
    
    dataset_dir = ''
    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(IUSTPersonReID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin

        train = self._process_dir(self.train_dir, Filter=cfg.DATASETS.TRAINING_FILTER,relabel=True)
        query = self._process_dir(self.query_dir,Filter=cfg.DATASETS.QUERY_FILTER,relabel=False)
        gallery = self._process_dir(self.gallery_dir,Filter=cfg.DATASETS.GALLERY_FILTER,relabel=False)

        if verbose:
            print("=> IUSTPersonReID-Dataset loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, Filter=False, relabel=False):
        if Filter:
           df = pd.read_csv(cfg.DATASETS.CSV_PATH)
           valid_ids = set(df['master_id'].astype(str))          
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')
    
        pid_container = set()
        filtered_img_paths = []
    
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())  # Fixed '*' to '_'
            if pid == -1: 
                continue  # junk images are just ignored
        
            if Filter:
                if str(pid) in valid_ids:
                    pid_container.add(pid)
                    filtered_img_paths.append(img_path)
            else:
                pid_container.add(pid)
    
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []   
    
        if Filter:
            img_paths = filtered_img_paths   	
    
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
        
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 1))
    
        return dataset
    
