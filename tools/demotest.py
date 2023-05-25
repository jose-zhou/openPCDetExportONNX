import argparse
import glob
from pathlib import Path
import os
import time
try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch,sys
abPath = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(abPath)

print(sys.path)
from pcdet.data2voxel_cpp import lib_cpp
# import lib_cpp


from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
# from pcdet.datasets.kitti.kitti_park_dataset import KittiParkDataset
from tools.visual_utils.open3d_vis_utils import draw_box
import time




class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
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
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        # label_file_list =
        data_file_list.sort()
        for i in data_file_list:
            print("data_list: "+ str(i))
      
        self.sample_file_list = data_file_list
        self.total_time = 0
        self.total_frame_number = 0

    def __len__(self):
        return len(self.sample_file_list)

    def getRunTime(self):
        return  len(self.sample_file_list), self.total_frame_number, self.total_time

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 8)[:, 0:4]
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        
        data_dict = self.prepare_data(data_dict=input_dict)
        # print('J note data_dict pointcloud',data_dict['points'].shape)




        pointcloud_range = self.dataset_cfg.POINT_CLOUD_RANGE
        print('self.dataset_cfg.POINT_CLOUD_RANGE', pointcloud_range)
        voxel_size = self.dataset_cfg.DATA_PROCESSOR[2].VOXEL_SIZE
        
        HEIGHT = round((pointcloud_range[3] - pointcloud_range[0])/voxel_size[0])
        WIDTH = round((pointcloud_range[4] - pointcloud_range[1])/voxel_size[1])
        CHANNELS =  round((pointcloud_range[5] - pointcloud_range[2])/voxel_size[2])
        time_start = time.time()
        spatial_features = lib_cpp.data2voxel(data_dict['points'], HEIGHT, WIDTH, CHANNELS,pointcloud_range[0], pointcloud_range[1], 
                                                                                        pointcloud_range[2], pointcloud_range[3],  pointcloud_range[4], pointcloud_range[5], 
                                                                                        voxel_size[0], voxel_size[1], voxel_size[2], 0)
        np.mean(spatial_features)
        print(spatial_features.shape)
        data_dict['spatial_features' ] = spatial_features.transpose((2, 1, 0))
        # a = 100
        time_end = time.time()
        self.total_time += (time_end - time_start)
        self.total_frame_number += 1
        
        return data_dict


datapath = 'data/mykitti/testing/velodyne/1663139164.978024244.bin'
cfg_ifle = 'tools/cfgs/kitti_models/my_centerpoint_yolo_down4.yaml'
ckpt = 'output/kitti_models/my_centerpoint_yolo_down4/default/ckpt/checkpoint_epoch_80.pth'

datapath = os.path.join(abPath, datapath)
cfg_ifle = os.path.join(abPath, cfg_ifle)
ckpt = os.path.join(abPath, ckpt)


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=cfg_ifle,
                        help='specify the config for demo')
                        
    parser.add_argument('--data_path', type=str, default=datapath, 
                        help='specify the point cloud data file or directory')
   
       
    parser.add_argument('--ckpt', type=str, default= ckpt, 
                        help='specify the pretrained model')
    
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )

    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    display = True
    with torch.no_grad():
        total_time = 0
        total_num = 0

        if(display):
            vis = open3d.visualization.Visualizer()
            vis.create_window()
            vis.get_render_option().point_size = 2.0
            vis.get_render_option().background_color = np.zeros(3)
            axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            vis.add_geometry(axis_pcd)
            pts = open3d.geometry.PointCloud()
            vis.add_geometry(pts)
            to_reset = True

        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            torch.cuda.synchronize()
            start_time = time.time()

            # 导出onnx示例
            # spatial_features  =torch.FloatTensor(1, 64, 320, 280).cuda()
            # torch.onnx.export(model, spatial_features, onnx_path, input_names = ['modelInput'], output_names = ['modelOutput'])
            

            pred_dicts, _ = model.forward(data_dict)
            torch.cuda.synchronize()
            end_time = time.time()
            
            

            total_time = end_time - start_time + total_time
            total_num += 1
            if (display):
                # for ind in pred_dicts[0]['pred_labels']:
                #     print(cfg['CLASS_NAMES'][ind - 1])
                # print('pred_labels: ', pred_dicts[0]['pred_boxes'])
                # print('J note !! ', pred_dicts[0]['pred_labels'])
                print(pred_dicts[0]['pred_boxes'])
                V.draw_scenes(
                    points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'] 
                )

                points = data_dict['points'][:, 1:].cpu().numpy()
                ref_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()

                # pts.points = open3d.utility.Vector3dVector(points[:, :3])
                # vis.add_geometry(pts)
                # pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
                # # draw boxes
                # vis = draw_box(vis, ref_boxes, (0, 1, 0), pred_dicts[0]['pred_labels'], pred_dicts[0]['pred_scores'])
                # time.sleep(0.5)         
                # vis.poll_events()
                # vis.clear_geometries()
                # vis.update_renderer()
                
        real_frame_number, cacu_frame_num, totalTime = demo_dataset.getRunTime()
        print('J note data 2voxel time is ', real_frame_number, cacu_frame_num, totalTime, totalTime/ cacu_frame_num)
        print("per pointcloud predict time: " + str(total_time / total_num))
    logger.info('Demo done.')

if __name__ == '__main__':
    main()
