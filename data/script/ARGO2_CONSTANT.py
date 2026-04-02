"""
Public constant accessible to all files
"""

import numpy as np

# metrics with two return values
METRIC = ['iou_3d', 'giou_3d']
FAST_METRIC = ['giou_3d', 'giou_bev']

# category name(str) <-> category label(int)
CLASS_SEG_TO_STR_CLASS = {
    'BICYCLE': 0,
    'BUS': 1,
    'REGULAR_VEHICLE': 2,
    'MOTORCYCLE': 3,
    'PEDESTRIAN': 4,
    'VEHICULAR_TRAILER': 5,
    'TRUCK': 6,
}
CLASS_STR_TO_SEG_CLASS = {
    0: 'BICYCLE',
    1: 'BUS',
    2: 'REGULAR_VEHICLE',
    3: 'MOTORCYCLE',
    4: 'PEDESTRIAN',
    5: 'VEHICULAR_TRAILER',
    6: 'TRUCK',
}

# math
PI, TWO_PI = np.pi, 2 * np.pi

# init EKFP for different non-linear motion model
CTRA_INIT_EFKP = {
    # [x, y, z, w, l, h, v, a, theta, omega]
    'BUS': [10, 10, 10, 10, 10, 10, 10, 10, 1000, 10],
    'REGULAR_VEHICLE': [4, 4, 4, 4, 4, 4, 1000, 4, 1, 0.1],
    'VEHICULAR_TRAILER': [10, 10, 10, 10, 10, 10, 10, 10, 1000, 10],
    'TRUCK': [10, 10, 10, 10, 10, 10, 10, 10, 1000, 10],
    'PEDESTRIAN': [10, 10, 10, 10, 10, 10, 10, 10, 1000, 10]
}
BIC_INIT_EKFP = {
    # [x, y, z, w, l, h, v, a, theta, sigma]
    'BICYCLE': [10, 10, 10, 10, 10, 10, 10000, 10, 10, 10],
    'MOTORCYCLE': [4, 4, 4, 4, 4, 4, 100, 4, 4, 1],
}