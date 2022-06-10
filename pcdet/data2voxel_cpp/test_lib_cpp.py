import lib_cpp
import numpy as np
import config.config as cfg


DX = cfg.VOXEL_SIZE[0]
DY = cfg.VOXEL_SIZE[1]
DZ = cfg.VOXEL_SIZE[2]

X_MIN = cfg.RANGE['X_MIN']
X_MAX = cfg.RANGE['X_MAX']

Y_MIN = cfg.RANGE['Y_MIN']
Y_MAX = cfg.RANGE['Y_MAX']

Z_MIN = cfg.RANGE['Z_MIN']
Z_MAX = cfg.RANGE['Z_MAX']

overlap = cfg.OVERLAP # 11.2
# the size of voxel
HEIGHT = round((X_MAX - X_MIN + 2*overlap) / DX)
WIDTH = round((Y_MAX - Y_MIN) / DY)
CHANNELS = round((Z_MAX - Z_MIN) / DZ)


file_path = './data/cvte02_20211231121530_livox_front_00105001.bin'
points_list = np.fromfile(file_path, dtype=np.float32, count=-1).reshape([-1, 4])

vox = lib_cpp.data2voxel(points_list, HEIGHT, WIDTH, CHANNELS, X_MIN, Y_MIN, Z_MIN, X_MAX, 
                Y_MAX, Z_MAX, DX, DY, DZ, overlap)

for i in vox:
    for ii in i:
        for ss in ii:
            if(ss > 0):
                print(ss)           

print(vox.shape)