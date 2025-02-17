import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils


class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False):
        '''
            use_bias = true
        '''
        super().__init__()
        self.sep_head_dict = sep_head_dict
        # print('J note sep_head_dict', sep_head_dict)

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                # print('J note: fc is', fc)
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            # 这一句将子模块在该类内进行注册
            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            # ret_dict means            result  dict 
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class CenterHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()
        # print('J note: centerpoint input', input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size)
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        # 这是编码用的
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.class_names = class_names
        
        
        # 只创建了 一个head 
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []
        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            # 给类别标序号  从0开始
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        #  输入输出通道分别为 512 和 64
        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            nn.BatchNorm2d(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(),
        )

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        # print('J noted, self.class_names_each_head', self.class_names_each_head)
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            # 将head_dict中的'hm'输出通道与类别数对齐    
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=self.model_cfg.SHARED_CONV_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
                )
            )
        self.predict_boxes_when_training = predict_boxes_when_training
        # 存放推理结果  （最初始的 推理结果）
        self.forward_ret_dict = {}
        self.build_losses()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        #gt_box shape  [19, 8]]
        #  

        # heatmap size [10, 200, 180]
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        # print('J note assign target input gt_box shape is', gt_boxes.shape) 

        # print('J note: assign target heatmap resolution is:', heatmap.shape)
        
        # ret_boxes shape is  [500, 8]
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))

        inds   = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        # print('J note:  coord_x  is ', coord_x)
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        # print('J note:  clamp coord_x  is ', coord_x)
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        # print('J note: center.shape', center.shape)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        # 由目标框影射到featuremap后的dx dy来计算生成heatmap时的gaussian radius
        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        # 对每个目标框做操作
        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            # 目标框类别ID
            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

            # 把偏移量计算出来
            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

        return heatmap, ret_boxes, inds, mask

    def assign_targets(self, gt_boxes, feature_map_size=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)     其中M为一个batch的标注数据中 最大标注框的个数   M 理解成Max      [2, 29, 8]    其中29不是定值
            feature_map_size: [H, W]                                                                                                                                [300, 200]
            feature_map_stride: 8   来自Centerpoint模型配置文件  对应 FEATURE_MAP_STRIDE 参数值  这些信息直接在model_cfg.TARGET_ASSIGNER_CONFIG中拿
        Returns:
            all_targets_dict = {
                'box_cls_labels': cls_labels,               # (4，321408）
                'box_reg_targets': bbox_targets,     # (4，321408，7）
                'reg_weights': reg_weights                 # (4，321408）
            }

        """
        # print('J note feature_map_size input is', feature_map_size)
        # 这里进行了以此翻转  不知道为什么要翻转
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]    [200, 180] ==> [180, 200]
        # print('J note feature_map_size input change', feature_map_size)
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        # gt_boxes.shape   [2, 29, 8]    其中29不是定值
        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': []
        }

        # 不知道为什么加了一个 'bg'  类别
        all_names = np.array(['bg', *self.class_names])
        # print('J note: all_names:', self.class_names_each_head)
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list = [], [], [], []
            for bs_idx in range(batch_size):
                # 依次取出batch中标注数据
                cur_gt_boxes = gt_boxes[bs_idx]
                # 当前帧标注数据的目标对应的类别
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]
                # print('J note: gt_class_names', gt_class_names)

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                # print('J note: gt_boxes_single_head', gt_boxes_single_head[0].shape)

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                # print('J note: gt_boxes_single_head', feature_map_size)
                heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        tb_dict = {}
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            loss += hm_loss + loc_loss
            # print('J note hm_loss and loc_loss', hm_loss, loc_loss)
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def generate_predicted_boxes(self, batch_size, pred_dicts):
        # 获取后处理参数
        post_process_cfg = self.model_cfg.POST_PROCESSING
        print('post_process_cfg.POST_CENTER_LIMIT_RANGE', post_process_cfg.POST_CENTER_LIMIT_RANGE)
        # 中心点的限制范围[0, -20, -1, 36, 20, 3]
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        # 以batchsize 1 为例  
        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm']
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            # print('J temp batch_center_z', batch_center_z.shape)
            # assign target时 lwh 取了 log
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            # print('J temp batch_rot_cos', batch_rot_cos.shape)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)

            # 当前模型不执行
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

            print('J note post progress pointCloud range is', self.point_cloud_range)

            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),  #false
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )
            # print('J note final_pred_dicts', len(final_pred_dicts))

            for k, final_dict in enumerate(final_pred_dicts):
                # 没啥用应该是筛选预测类别中的label的
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                
                
                # 如果没做非极大值抑制  就做一个非极大值抑制
                if post_process_cfg.NMS_CONFIG.NMS_TYPE != 'circle_nms':
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )

                    final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                    final_dict['pred_scores'] = selected_scores
                    final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def forward(self, data_dict):
        # spatial_features_2d.shape is : 
        # spatial_features_2d = data_dict['spatial_features_2d']
        spatial_features_2d = data_dict
        # print('J note: CenterHead input feature map shape', spatial_features_2d.shape)
        # 经过共享卷积层后  尺寸变为 [1, , , ]
        x = self.shared_conv(spatial_features_2d)
        # print('J note: after share cov  shape is', x.shape)
        

        #  预测结果存在这里边   有哪些结果?    如下记录
        #  1 预测了 center                      (x, y)              [1, 2, 300, 200]
        #  2 预测了 center_Z                    (z)                 [1, 1, 300, 200]
        #  4 预测了 dim                         (Dx, Dy, Dz)        [1, 3, 300, 200]
        #  5 预测了 rot                         ()                  [1, 2, 300, 200]
        #  6 预测了 hm(heat map?)     (10个类别)                     [1, 10, 300, 200]
        pred_dicts = []
        # separateHead只有一个头
        for head in self.heads_list:
            pred_dicts.append(head(x))

        # print('J note: predict result shape', len(pred_dicts))  1 
        # for inde in pred_dicts:
        #     for sds in inde.values():
        #         print(sds.shape)
        # if self.training:
        #     # print('J note:  should be None', data_dict.get('spatial_features_2d_strides', None))   None
        #     # feature_map_size  is [150, 250]  就是feature_map输入的高和宽
            
        #     # 训练阶段  此函数内做了GT 编码  
        #     target_dict = self.assign_targets(
        #         data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],
        #         feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
        #     )
        #     self.forward_ret_dict['target_dicts'] = target_dict

        # self.forward_ret_dict['pred_dicts'] = pred_dicts
        # print('J note type!!!', target_dict.keys())
        # print('J note ai!!!!!!!', pred_dicts[0].keys())

        # 查看解码后的数据
        # target_decode = self.generate_predicted_boxes(
        #         data_dict['batch_size'], target_dict
        #     )
        # print('J data annotate',  data_dict['gt_boxes'])
        # print('J data target_decode', target_decode)

        # print('J note pred_dicts size is', pred_dicts[0].keys())
        result = []
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm']
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim']
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            
            result.append(batch_hm)
            result.append(batch_center)
            result.append(batch_center_z)
            result.append(batch_dim)
            result.append(batch_rot_cos)
            result.append(batch_rot_sin)
        # 导出onnx时直接返回
        return result
