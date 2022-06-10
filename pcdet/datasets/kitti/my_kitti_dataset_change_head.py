from cgi import test
import copy, os
import pickle
from joblib import PrintTime
from pcdet.data2voxel_cpp import lib_cpp


import numpy as np
from skimage import io

# 导入数据处理的基本库（eg: DatasetTemplate）   和数据处理工具库（自己编码的功能库）
from . import kitti_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, my_object3d_kitti
from ..dataset import DatasetTemplate


class KittiDataset(DatasetTemplate):
    '''
    args:
        继承pytorch Dataset类完成数据的加载 处理 和读取相关工作 
        必须重写__init__()   __len__() 以及 __getitem__()
    '''
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        # 索引到split的字符串  train  或者是 test
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]

        # 确定数据的读取路径root_path输入为空时从数据配置文件中读取
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
       
        # 获取数据集加载索引文件（.txt）的路径
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        # 获取索引数据的list
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
    

        # Debug：查看加载的索引是否正确
        # print('Debug：查看加载的索引是否正确', self.sample_id_list)

        # 如果中间数据文件已创建 运行加载相应信息
        self.kitti_infos = []
        self.include_kitti_data(self.mode)
        self.point_range = self.dataset_cfg['POINT_CLOUD_RANGE']
        self.voxel_size =  self.dataset_cfg['DATA_PROCESSOR'][2]['VOXEL_SIZE']

    def include_kitti_data(self, mode):
        '''
        arg:
            加载指定数据集（测试集/训练集/验证集）的中间信息文件(.pkl文件)
        '''
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')
        kitti_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(kitti_infos)))

    def set_split(self, split):
        '''
        arg:
            重新初始化数据集对象
            确认  待加载数据的  list  存放于 self.sample_id_list中
        '''
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        '''
        arg:
            加载点云数据  返回np格式的点云数据  shape(-1, 4)
        '''
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    

    def get_label(self, idx):
        '''
            function: 获取单帧 标注信息存放到 一个Object3d对象中
            arg: 获取标注信息
                 input: idx  标注文件文件名
                 returm: 
        '''
        label_file = self.root_split_path / 'label' / ('%s.txt' % idx)

        if not os.path.exists(label_file):
            print('Debug: 文件不存在', label_file)
        assert label_file.exists()
        return my_object3d_kitti.get_objects_from_label(label_file)

    def get_depth_map(self, idx):
        """
        注:   对自建数据集无用
        Loads depth map for a sample
        Args:
            idx: str, Sample index
        Returns:
            depth: (H, W), Depth map
        """
        depth_file = self.root_split_path / 'depth_2' / ('%s.png' % idx)
        assert depth_file.exists()
        depth = io.imread(depth_file)
        depth = depth.astype(np.float32)
        depth /= 256.0
        return depth

    def get_calib(self, idx):
        '''
            注:  对自建数据集无用
        '''
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_road_plane(self, idx):
        '''
            注:  对自建数据集无用
        '''
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
            注:  对自建数据集无用
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=1, has_label=True, count_inside_pts=True, sample_id_list=None):
        '''
            function:  获取数据集中间格式文件所需信息   处理对应数据集（test/train/val）所有的信息 并返回
           
            arg:

            注:  为了便于调试  处理线程数  4改为 1  num_workers=1

        '''
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            '''
                function:  处理单帧点云的信息

                arg:
                    input:  sample_idx  待处理的文件名
            '''
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info


            # 自建数据集不需要图像信息  标定信息
            # image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            # info['image'] = image_info
            # calib = self.get_calib(sample_idx)

            # P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            # R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            # R0_4x4[3, 3] = 1.
            # R0_4x4[:3, :3] = calib.R0
            # V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            # calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            # info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                # annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                # annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                # annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                # annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.w, obj.h] for obj in obj_list])                                   # lhw(lidar) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)       #位置
                annotations['rotation_y'] = np.array([obj.rz for obj in obj_list])                                                                  #绕z轴旋转量
                # annotations['score'] = np.array([obj.score for obj in obj_list])
                # annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                
                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])                 # 除了DontCare  的目标物   单帧中 的目标个数
                num_gt = len(annotations['name'])                                                                                                           #  单帧中所有目标的个数

                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)                                      # 标志出哪些是有效目标   如 index = [0, 1, 2, 3, 4, -1, -1, -1 ]

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                # loc_lidar = calib.rect_to_lidar(loc)
                loc_lidar =loc
                l, w, h = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                # loc_lidar[:, 2] += h[:, 0] / 2                #雷达坐标系下  标注点位置信息本身便是  3D框中心位置无需移动
                # gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)   #无需对角度信息做调整
                # print('Debug:输出加载信息查看正确与否')
                # print('sample_inex', sample_idx )
                # print('h w l location rotate', h[0], w[0], l[0],loc_lidar[0], rots[0])
                # print('number object:' ,h.shape)
                
                
                # gt_boxes_lidar  >  [x, y, z, dx, dy, dz]
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, rots[..., np.newaxis]], axis=1)   # 根据 求8顶点的函数来看  dx, dy, dz 分别对应 l h w
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                

                # 计算目标框内部点的个数
                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    # calib = self.get_calib(sample_idx)
                    # pts_rect = calib.lidar_to_rect(points[:, 0:3])
                    # fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    # pts_fov = points[fov_flag]

                    pts_fov = points
                    # 计算目标框8顶点的位置信息
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)

                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)    #初始化  标注目标个数 的  箱子 bin  后期计算每个  目标物中的 点的个数

                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt
                
                info['annos'] = annotations
            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        '''
            function:  重新加载训练集 info并增加几纬度的信息'kitti_dbinfos_%s.pkl'，  存储gt box 内目标框内部  的点gt_database/

        '''
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('kitti_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        # 每个  k 为一帧的数据
        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            # difficulty = annos['difficulty']
            # bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            
            # 这里可能存在问题
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    # db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                    #            'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                    #            'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}

                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        function: 根据预测信息生成kitti格式的存储形式 
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor   pred_dicts是一个batch的predict
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            '''
                function:  直译  获取预测模板， 翻译：依据预测框个数生成相应的空 dict来存放 预测内容

                arg:
                    num_sample:  预测目标的个数
                return:
                    ret_dict:  装载预测结果的  *空*  容器
            '''
            ret_dict = {
                'name': np.zeros(num_samples), 
                'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 
                'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 
                'boxes_lidar': np.zeros([num_samples, 7])
                # 'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                # 'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                # 'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                # 'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                # 'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            '''
                arg:
                    box_dict:  一个sample的预测数据 

                    batch_index：  对应帧的 frame_id
                return:
                    pred_dict:  一个以键值对形式的容器，存储了单帧点云的预测内容
            '''
            # [N]
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            # [N, 7]
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            # [N]
            pred_labels = box_dict['pred_labels'].cpu().numpy()

            # pred_scores.shape[0]   frame中 预测框的个数
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            # calib = batch_dict['calib'][batch_index]
            # image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            # pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            # pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
            #     pred_boxes_camera, calib, image_shape=image_shape
            # )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            # pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            # pred_dict['bbox'] = pred_boxes_img
            # pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            # pred_dict['location'] = pred_boxes_camera[:, 0:3]
            # pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['dimensions'] = pred_boxes[:, 3:6]    # 长宽高
            pred_dict['location'] = pred_boxes[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            # 获取batch中的 frame_id
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        '''
            注：暂时未修改
        '''
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import my_eval as kitti_eval
        # from .kitti_object_eval_python import eval_park as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        return len(self.kitti_infos)

    def __getitem__(self, index):
        '''
            function:  自建数据集类别的dataloader  在运行迭代器时  调用该函数取出数据
        '''
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        

        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])    #['points']
        # print('get_item_list', get_item_list)    #get_item_list ['points']
       
        input_dict = {
            'frame_id': sample_idx,
            # 'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            # gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            # gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
            gt_boxes_lidar = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            # if "gt_boxes2d" in get_item_list:
            #     input_dict['gt_boxes2d'] = annos["bbox"]

            # road_plane = self.get_road_plane(sample_idx)
            
            # if road_plane is not None:
            #     input_dict['road_plane'] = road_plane

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            # if self.dataset_cfg.FOV_POINTS_ONLY:
            #     pts_rect = calib.lidar_to_rect(points[:, 0:3])
            #     fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            #     points = points[fov_flag]
            input_dict['points'] = points

        # if "images" in get_item_list:
        #     input_dict['images'] = self.get_image(sample_idx)

        # if "depth_maps" in get_item_list:
        #     input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        # if "calib_matricies" in get_item_list:
        #     input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

        # 将输入数据进一步处理  形成训练数据   
        # print('Debug: 运行self.prepare_data前', input_dict)
        data_dict = self.prepare_data(data_dict=input_dict)
                # 添加'spatial_features' 属性/数据

        HEIGHT = round((self.point_range[3] - self.point_range[0])/self.voxel_size[0])
        WIDTH = round((self.point_range[4] - self.point_range[1])/self.voxel_size[1])
        CHANNELS =  round((self.point_range[5] - self.point_range[2])/self.voxel_size[2])
        spatial_features = lib_cpp.data2voxel(data_dict['points'], HEIGHT, WIDTH, CHANNELS,self.point_range[0], self.point_range[1], 
        self.point_range[2], self.point_range[3],  self.point_range[4], self.point_range[5], self.voxel_size[0], self.voxel_size[1], self.voxel_size[2], 0)
        
        print('J note: spatial_features shape is', spatial_features.shape)
        # print('voxel de size is ', self.voxel_size)
        # print('point_range is', self.point_range)
        # print('spatial_features shape is', spatial_features.transpose((2, 1, 0)).shape)
        # data_dict['spatial_features' ] = spatial_features.transpose((2, 1, 0))
        # [channel, w, h]
        data_dict['spatial_features' ] = spatial_features.transpose((2, 1, 0))
        print('J note:np.unique(spatial_features)', np.unique(spatial_features))
        # print(np.sum(spatial_features.flatten() > 0))
        # print('!!!!!!!!!!!!!!!!')


        # print('Debug: 运行self.prepare_data后', data_dict['gt_boxes'].shape)

        # data_dict['image_shape'] = img_shape
        return data_dict


def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=1):
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split, test_split = 'train', 'val', 'test'

    train_filename = save_path / ('kitti_infos_%s.pkl' % train_split)
    val_filename = save_path / ('kitti_infos_%s.pkl' % val_split)
    test_filename = save_path /('kitti_infos_%s.pkl' % test_split)
    trainval_filename = save_path / 'kitti_infos_trainval.pkl'
    

    print('---------------Start to generate data infos---------------')

    # 相当于重新初始化对象
    dataset.set_split(train_split)
    # 1  根据初始化的数据对象  获取 训练集标注文档内的标注信息 ，并存储成中间格式的标注信息文档
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('Kitti info train file is saved to %s' % train_filename)

    # 2  根据初始化的数据对象  获取 验证集标注文档内的标注信息 ，并存储成中间格式的标注信息文档
    # dataset.set_split(val_split)
    # kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=True)
    # with open(val_filename, 'wb') as f:
    #     pickle.dump(kitti_infos_val, f)
    # print('Kitti info val file is saved to %s' % val_filename)

    # with open(trainval_filename, 'wb') as f:
    #     pickle.dump(kitti_infos_train + kitti_infos_val, f)
    # print('Kitti info trainval file is saved to %s' % trainval_filename)

    # 2  根据初始化的数据对象  获取 测试集标注文档内的标注信息 ，并存储成中间格式的标注信息文档
    dataset.set_split('test')
    kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    
    with open(test_filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)
    print('Kitti info test file is saved to %s' % test_filename)

    #4  打印 训练集、测试集  的帧数和目标框数
    print('Debug:  训练集帧数目', len(kitti_infos_train))
    a = 0
    for index in kitti_infos_train:
        a += len(index['annos']['rotation_y'])
    print('Debug:  训练集目标物个数', a)

     # 打印 测试集的帧数 和 目标框数
    print('Debug:  测试集帧数目', len(kitti_infos_test))
    a = 0
    for index in kitti_infos_test:
        a += len(index['annos']['rotation_y'])
    print('Debug:  测试集目标物个数', a)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist',  'Van', 'Heavy_Truck', 'Light_Truck', 'Tricycle',  'Small_Bus',  'Big_Bus', 'Ying_Er_Che'],
            data_path=ROOT_DIR / 'data' / 'mykitti',
            save_path=ROOT_DIR / 'data' / 'mykitti'
        )
