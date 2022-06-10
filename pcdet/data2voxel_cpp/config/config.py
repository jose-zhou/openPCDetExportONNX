CLASSES = ['car', 'bus', 'truck', 'pedestrian', 'bimo']
# The 3D box defines the range of point clouds that will be processed.
# 0, -40, -1.5, 70.4, 40, 2.5
RANGE = {'X_MIN': 0,
         'X_MAX': 70.4,
         'Y_MIN': -40,
         'Y_MAX': 40,
         'Z_MIN': -1.5,
         'Z_MAX': 2.5}
# 0.2m*0.2m*0.2m for each voxel.
VOXEL_SIZE = [0.2, 0.2, 0.05]
BATCH_SIZE = 1
#MODEL_PATH = "model/livoxmodel"
# MODEL_PATH = "/home/cvte/WSK/virtualenviroment/detection_tracking/catkin_ws/src/perception_wsk/scripts/livox_detection/model/livoxmodel"

OVERLAP = 0

GPU_INDEX = 0
NMS_THRESHOLD = 0.1
BOX_THRESHOLD = 0.6
