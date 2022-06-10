import numpy as np


def get_objects_from_label(label_file):
    '''

    '''
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects


def cls_type_to_id(cls_type):
    '''
        arg:  将type专程ID
            输入: type 名称
            输出: ID
    '''
    # 根据类别数做相应的添加
    # type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4, 'Heavy_Truck': 5, 'Light_Truck': 6}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Object3d(object):
    '''
        function: 该类装载单帧 数据 的标注信息 放在self的属性里
        arg: 

    '''
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        
        self.cls_type = label[0]                                            # 类别名称
        self.cls_id = cls_type_to_id(self.cls_type)     # 类别序号   {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4, 'Heavy_Truck': 5, 'Light_Truck': 6}

        # self.truncation = float(label[1])                        
        # self.occlusion = float(label[2])                            # 遮挡程度   0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        # self.alpha = float(label[3])
        # self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        
        # 3D box的高宽长
        self.h = float(label[1])
        self.w = float(label[2])
        self.l = float(label[3])
        # 3D box的高宽长  基于雷达坐标系的位置 和 朝向 弧度表示   绕Z轴旋转与x轴夹角   顺时针负  逆时针正
        self.loc = np.array((float(label[4]), float(label[5]), float(label[6])), dtype=np.float32)
        # self.dis_to_cam = np.linalg.norm(self.loc)
        self.dis_to_lida = np.linalg.norm(self.loc)                 #计算到坐标系原点的距离
        # self.rz = float(label[14])
        self.rz = float(label[7])

        # self.score = float(label[15]) if label.__len__() == 16 else -1.0
        # self.level_str = None
        # self.level = self.get_kitti_obj_level()

    def get_kitti_obj_level(self):
        '''
            function:kittii数据集中获取目标物体  被感知的难以程度 
                                结合2D图片遮挡度  可见程度等来判定
                                注：自建数据集中无用
            args:
        '''
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 0  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 2  # Hard
        else:
            self.level_str = 'UnKnown'
            return -1

    def generate_corners3d(self):
        """
           function:   generate corners3d representation for this object  
            arg:
                return corners_3d: (8, 3) corners of box3d in camera coord  
                注:  不一定是camera坐标系   标注信息是基于哪个坐标系描述的  这个框也就基于哪个坐标系进行描述

            注意:
                l  > dx
                h > dy
                w > dz
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.rz), 0, np.sin(self.rz)],
                      [0, 1, 0],
                      [-np.sin(self.rz), 0, np.cos(self.rz)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.loc
        return corners3d

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s rz: %.3f' \
                     % (self.cls_type, self.truncation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                        self.loc, self.rz)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.truncation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.loc[0], self.loc[1], self.loc[2],
                       self.rz)
        return kitti_str
