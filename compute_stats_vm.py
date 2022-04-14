
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import deepdish as dd
import pprint
import cv2

from datasets.kitti import KITTI
from datasets.kitti import ALL_VEHICLES, CARS_ONLY, PEDESTRIANS_ONLY, CYCLISTS_ONLY

from pixor_targets import PIXORTargets

DS_DIR = os.path.expanduser('/comm_dat/morteza/KITTI/training')

kitti = KITTI(DS_DIR, CARS_ONLY)

train_ids = kitti.get_ids('train')
val_ids = kitti.get_ids('val')
micro_ids = kitti.get_ids('micro')

target_encoder = PIXORTargets(shape=(200, 175), P_WIDTH=70, P_HEIGHT=80, P_DEPTH=3.5)

sin, cos = [], []
yaw = []
w, l, h  = [], [], []
alt = []

def generate_offset_stats_2d( box):
        # Get 4 corners of the rectangle in BEV and transform to feature coordinate frame
        target_map_test = np.zeros((94, 311), dtype=np.uint8)
        r = np.ceil(((box.w / 4) * (box.h / 4)) / (target_map_test.shape[0] * target_map_test.shape[1]) * 100)
        cv2.circle(target_map_test, (int(box.cx / 4), int(box.cy / 4)), int((box.h / 4) / 2), (255, 255, 255), -1, lineType=cv2.LINE_AA)
        x_off, y_off = [], []
        # Generate corresponding geometry map for pts in rectangle
        for i in range(target_map_test.shape[0]): 
            for j in range(target_map_test.shape[1]):
               if target_map_test[i,j] >0:
                  y_off.append((box.cy/4)-j)          
                  x_off.append((box.cx/4)-i)
        return x_off, y_off
x_off, y_off, z_off = [], [], []
i = 0


for id in train_ids:
    boxes = kitti.get_boxes_2D(id)
    for box in boxes:
        w.append(np.log(box.w))
        h.append(np.log(box.h))
         
        x, y = generate_offset_stats_2d(box)
        
        x_off += x
        y_off += y
        
    if i % 1000 == 0:
        print('finished {0}'.format(i))
    i += 1
    
stats = {
    'mean': {
        'log_w': np.mean(w),
        'log_h': np.mean(h),
        'dx': np.mean(x_off),
        'dy': np.mean(y_off),
    },
    'std': {
        'log_w': np.std(w),
        'log_h': np.std(h),
        'dx': np.std(x_off),
        'dy': np.std(y_off),
    },
}

dd.io.save('kitti_stats/statsv2.h5', stats)

