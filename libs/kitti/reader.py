import os

import imagesize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import cv2

from .core import Box2D, Box3D, transform, project
from .utils import fov_filter, box_filter

# Constants
IMG_WIDTH, IMG_HEIGHT = 1242, 375
KITTI_COLUMN_NAMES = ['Type', 'Truncated', 'Occluded', 'Alpha',
                      'X1', 'Y1', 'X2', 'Y2',
                      '3D_H', '3D_W', '3D_L',
                      '3D_X', '3D_Y', '3D_Z',
                      'Rot_Y']
GT_COLOR = (0.0, 1.0, 0.0)
PRED_COLOR = (1.0, 0.0, 0.0)
BOX_CONNECTIONS = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [4, 0], [5, 1], [6, 2], [7, 3], [1, 6], [2, 5]]

# Dataset path
KITTI_DIR = 'D:/Datasets/KITTI/training' if os.name == 'nt' else os.path.expanduser('~/datasets/KITTI/training/')

# Commonly used class dicts
CARS_ONLY = {'Car': ['Car']}
PEDESTRIANS_ONLY = {'Pedestrian': ['Pedestrian']}
SMALL_OBJECTS = {'Pedestrian': ['Pedestrian'], 'Cyclist': ['Cyclist']}
ALL_VEHICLES = {'Car': ['Car'], 'Van': ['Van'], 'Truck': ['Truck'], 'Tram': ['Tram'], 'Misc': ['Misc']}
ALL_OBJECTS = {'Car': ['Car'], 'Van': ['Van'], 'Truck': ['Truck'],
               'Pedestrian': ['Pedestrian'], 'Person_sitting': ['Person_sitting'], 'Cyclist': ['Cyclist'],
               'Tram': ['Tram'], 'Misc': ['Misc']}


class KITTI:
    def __init__(self, ds_dir=KITTI_DIR, class_dict=CARS_ONLY):
        self.ds_dir = ds_dir
        self.img2_dir = os.path.join(ds_dir, 'image_2')
        self.label_dir = os.path.join(ds_dir, 'label_2')
        self.velo_dir = os.path.join(ds_dir, 'velodyne')
        self.calib_dir = os.path.join(ds_dir, 'calib')

        self.class_to_group = {}
        for group, classes in class_dict.items():
            for cls in classes:
                self.class_to_group[cls] = group

        # # Cache labels
        # self.boxes_3D, self.boxes_2D, self.calibs = {}, {}, {}
        # for t in tqdm(self.get_ids('micro' if is_micro else 'trainval')):
        #     self.boxes_3D[t] = self.get_boxes_3D(t)
        #     self.boxes_2D[t] = self.get_boxes_2D(t)
        #     self.calibs[t] = self.get_calib(t)

    def get_image(self, t):
        return get_image(os.path.join(self.img2_dir, t + ".png"))

    def get_boxes_2D(self, t):
        return get_boxes_2D(path=os.path.join(self.label_dir, t + ".txt"),
                            img_path=os.path.join(self.img2_dir, t + ".png"),
                            class_to_group=self.class_to_group)

    def get_boxes_3D(self, t):
        return get_boxes_3D(path=os.path.join(self.label_dir, t + ".txt"),
                            class_to_group=self.class_to_group)

    def get_velo(self, t, workspace_lim=((-40, 40), (-1, 2.5), (0, 70)), use_fov_filter=True):
        pts, reflectance = get_velo(path=os.path.join(self.velo_dir, t + '.bin'),
                                    calib_path=os.path.join(self.calib_dir, t + '.txt'),
                                    workspace_lim=workspace_lim,
                                    use_fov_filter=use_fov_filter)
        return pts, reflectance

    def get_calib(self, t):
        return get_calib(os.path.join(self.calib_dir, t + '.txt'))

    def get_ids(self, subset):
        assert subset in ['train', 'val', 'micro', 'trainval']
        return [line.rstrip('\n') for line in open(os.path.join(self.ds_dir, 'subsets', subset + '.txt'))]


def get_image(path):
    img = Image.open(path).resize((IMG_WIDTH, IMG_HEIGHT))
    return np.asarray(img, dtype=np.float32) / 255.0


def get_boxes_2D(path, img_path, class_to_group):
    # Load labels from txt file
    df = pd.read_csv(path, names=KITTI_COLUMN_NAMES, header=None, delim_whitespace=True)

    # Get image shape
    w, h = imagesize.get(img_path)

    # Convert label to list of Box2D
    gt_boxes_2D = []
    for _, row in df.iterrows():
        if row['Type'] in class_to_group:
            gt_boxes_2D += [Box2D((row['X1'] * IMG_WIDTH / w, row['Y1'] * IMG_HEIGHT / h,
                                   row['X2'] * IMG_WIDTH / w, row['Y2'] * IMG_HEIGHT / h),
                                  mode=Box2D.CORNER_CORNER, cls=class_to_group[row['Type']])]

    return gt_boxes_2D


def get_boxes_3D(path, class_to_group):
    # Load labels from txt file
    df = pd.read_csv(path, names=KITTI_COLUMN_NAMES, header=None, delim_whitespace=True)

    # Convert label to list of Box3D
    gt_boxes_3D = []
    for _, row in df.iterrows():
        if row['Type'] in class_to_group:
            gt_boxes_3D += [Box3D(row['3D_H'], row['3D_W'], row['3D_L'],
                                  row['3D_X'], row['3D_Y'], row['3D_Z'],
                                  row['Rot_Y'], cls=class_to_group[row['Type']])]
    return gt_boxes_3D


# Returns velo pts and reflectance in rectified camera coordinates
def get_velo(path, calib_path, workspace_lim=((-40, 40), (-1, 2.5), (0, 70)), use_fov_filter=True):
    velo = np.fromfile(path, dtype=np.float32).reshape((-1, 4)).T
    pts = velo[0:3]
    reflectance = velo[3:]

    # Transform points from velo coordinates to rectified camera coordinates
    V2C, R0, P2 = get_calib(calib_path)
    pts = transform(np.dot(R0, V2C), pts)

    # Remove points out of workspace
    pts, reflectance = box_filter(pts, workspace_lim, decorations=reflectance)

    # Remove points not projecting onto the image plane
    if use_fov_filter:
        pts, reflectance = fov_filter(pts, P=P2, img_size=(IMG_HEIGHT, IMG_WIDTH), decorations=reflectance)

    return pts, reflectance


def get_calib(path):
    # Read file
    with open(path, 'r') as f:
        for line in f:
            line = line.split()
            # Skip if line is empty
            if len(line) == 0:
                continue
            # Load required matrices only
            matrix_name = line[0][:-1]
            if matrix_name == 'Tr_velo_to_cam':
                V2C = np.array([float(i) for i in line[1:]]).reshape(3, 4)  # Read from file
                V2C = np.insert(V2C, 3, values=[0, 0, 0, 1], axis=0)  # Add bottom row
            elif matrix_name == 'R0_rect':
                R0 = np.array([float(i) for i in line[1:]]).reshape(3, 3)  # Read from file
                R0 = np.insert(R0, 3, values=0, axis=1)  # Pad with zeros on the right
                R0 = np.insert(R0, 3, values=[0, 0, 0, 1], axis=0)  # Add bottom row
            elif matrix_name == 'P2':
                P2 = np.array([float(i) for i in line[1:]]).reshape(3, 4)
                P2 = np.insert(P2, 3, values=[0, 0, 0, 1], axis=0)  # Add bottom row

    return V2C, R0, P2


def range_view(img, P2=None, gt_boxes=None, pred_boxes=None, scale=1.0, title='', ax=None, save_path=None):
    img = np.copy(img)  # Clone

    def draw_boxes_2D(boxes, color):
        for box in boxes:
            cv2.rectangle(img, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, 1)

    def draw_boxes_3D(boxes, color):
        for box in boxes:
            corners = project(P2, box.get_corners()).astype(np.int32)
            for start, end in BOX_CONNECTIONS:
                x1, y1 = corners[:, start]
                x2, y2 = corners[:, end]
                cv2.line(img, (x1, y1), (x2, y2), color, 1)

    if gt_boxes is not None and len(gt_boxes) > 0:
        if isinstance(gt_boxes[0], Box2D):
            draw_boxes_2D(gt_boxes, GT_COLOR)
        elif isinstance(gt_boxes[0], Box3D):
            draw_boxes_3D(gt_boxes, GT_COLOR)

    if save_path is not None:
        if save_path == '#RAW':
            return img
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, img * 255.0)
            return img
    else:
        fig = None
        if ax is None:
            fig, ax = plt.subplots()
            fig.set_size_inches(scale * 8, scale * 3)

        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.tight_layout()

        if fig is not None:
            plt.show()


def bev(pts=None, gt_boxes=None, pred_boxes=None, scale=1.0, title='', ax=None, save_path=None):
    pass


def open3d(pts=None):
    pass
