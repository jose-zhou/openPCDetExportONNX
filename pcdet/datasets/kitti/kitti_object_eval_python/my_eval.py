import io as sysio
from matplotlib.pyplot import box

import numba
import numpy as np

from .rotate_iou import rotate_iou_gpu_eval


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()             # 将得分的一维数组 升序排列，如[1,2,3,4]
    scores = scores[::-1]        # 再将得分数组降序排列
    current_recall = 0  
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall)) and (i < (len(scores) - 1))):
            continue
    
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


#这个函数是处理一帧的数据, current_class是5个类别中的其中一类
def clean_data(gt_anno, dt_anno, current_class, difficulty):
    
    '''
        print("____________clean_data() args:________________")
        print('current_class  :  ',current_class)
        print('difficulty : ',difficulty)
            ____________clean_data() args:________________
            current_class  :   0
            difficulty :  0
    '''

    CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist', 'Van', 'Heavy_Truck', 'Light_Truck', 'Tricycle', 'Small_Bus',  'Big_Bus', 'Ying_Er_Che']

    ignored_gt, ignored_dt =  [], []

    # 这一句的作用是：将current_class中对应的类别的名字找出来，
    # 如0 对应 vehicle。方法.lower()的意思是将字符串中的大写全部变成小写
    current_cls_name = CLASS_NAMES[current_class].lower()

    '''
        print("________________current_cls_name________________")
        print(current_cls_name)
        #得到的是：vehicle
    '''

    # 获取当前帧中物体object的个数
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0

    #对num_gt中每一个物体object：
    for i in range(num_gt):

        #获取这个物体的name，并小写
        gt_name = gt_anno["name"][i].lower()

        valid_class = -1

        # 如果该物体正好是 需要处理的当前的object，将valid_class值为 1
        if (gt_name == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        
        ignore = False
        if valid_class == 1 and not ignore:
            # 如果 为有效的物体， 且该物体object不忽略，
            # 则ignored_gt上该值为0，有效的物体数num_valid_gt+1
            ignored_gt.append(0)
            num_valid_gt += 1
        else:
            ignored_gt.append(-1)

    #对num_dt中每一个物体object：
    for i in range(num_dt):
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1

        if valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)
    

    return num_valid_gt, ignored_gt, ignored_dt


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1):
    '''
    args:
        boxes   GT box   一批  好多个sample的GT box

        qboxes   对应的预测Box
    '''
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou



def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lider.
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:

                # iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                #     boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))
                iw = (min(boxes[i, 2] + boxes[i, 5] / 2, qboxes[j, 2] + qboxes[j, 5] / 2) - max(
                    boxes[i, 2] - boxes[i, 5]/2, qboxes[j, 2] - qboxes[j, 5]/2))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    # # 返回 x z 平面上鸟瞰图的相交长度
    # rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
    #                            qboxes[:, [0, 2, 3, 5, 6]], 2)
     # 返回 x y 平面上鸟瞰图的相交长度
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 1, 3, 4, 6]],
                               qboxes[:, [0, 1, 3, 4, 6]], 2)
    # 计算3D框的交并比  并非如函数所述只能计算camera下的教并比
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba.jit(nopython=True)

def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas

    dt_scores = dt_datas[:, -1]   #获取预测的得分情况
    #dt_scores = dt_datas

    assigned_detection = [False] * det_size # 存储是否每个检测都分配给了一个gt。
    ignored_threshold = [False] * det_size    # 如果检测分数低于阈值，则存储数组
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0

    thresholds = np.zeros((gt_size,))
    thresh_idx = 0
    delta = np.zeros((gt_size, ))
    delta_idx = 0

    for i in range(gt_size):
        if ignored_gt[i] == -1:
            #如果不是当前class，如vehicle类别，
            # 则跳过当前循环，继续判断下一个类别
            continue

        det_idx = -1            #! 储存对此gt存储的最佳检测的idx
        valid_detection = NO_DETECTION      
        max_overlap = 0
        assigned_ignored_det = False

        # 遍历det中的所有数据，找到一个与真实值最高得分的框
        for j in range(det_size):
            # 如果该数据 无效，则跳过继续判断
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue

            # 获取 overlaps 中相应的数值
            overlap = overlaps[j, i]
            # 获取这个预测框的得分 
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap) and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                # 不存在该类别，： ignored_det[j] == 1
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            # 如果没有找到，valid_detection还等于 NO_DETECTION，
            # 且真实框确实属于vehicle类别，则fn+1
            fn += 1
        elif ((valid_detection != NO_DETECTION) and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            # 这种情况不存在：ignored_gt[i] == 1
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            # 这种情况是检测出来了，且是正确的
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            
            assigned_detection[det_idx] = True
    
    
    if compute_fp:
        #遍历验证det中的每一个：
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        fp -= nstuff

        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


#@numba.jit(nopython=True)
def compute_statistics_jit1(
                           overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):


    det_size = dt_datas.shape[0]
    gt_size = gt_datas

    dt_scores = dt_datas  #获取预测的得分情况
    #dt_scores = dt_datas

    assigned_detection = [False] * det_size # 存储是否每个检测都分配给了一个gt。
    ignored_threshold = [False] * det_size    # 如果检测分数低于阈值，则存储数组
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0

    thresholds = np.zeros((gt_size,))
    thresh_idx = 0
    delta = np.zeros((gt_size, ))
    delta_idx = 0

    for i in range(gt_size):
        if ignored_gt[i] == -1:
            #如果不是当前class，如vehicle类别，
            # 则跳过当前循环，继续判断下一个类别
            continue

        det_idx = -1            #! 储存对此gt存储的最佳检测的idx
        valid_detection = NO_DETECTION      
        max_overlap = 0
        assigned_ignored_det = False

        # 遍历det中的所有数据，找到一个与真实值最高得分的框
        for j in range(det_size):
            # 如果该数据 无效，则跳过继续判断
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue

            # 获取 overlaps 中相应的数值
            overlap = overlaps[j, i]
            # 获取这个预测框的得分 
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap) and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                # 不存在该类别，： ignored_det[j] == 1
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            # 如果没有找到，valid_detection还等于 NO_DETECTION，
            # 且真实框确实属于vehicle类别，则fn+1
            fn += 1
        elif ((valid_detection != NO_DETECTION) and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            # 这种情况不存在：ignored_gt[i] == 1
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            # 这种情况是检测出来了，且是正确的
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            
            assigned_detection[det_idx] = True
    
    
    if compute_fp:
        #遍历验证det中的每一个：
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        fp -= nstuff

        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if same_part == 0:
        return [num]

    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


#@numba.jit(nopython=True)
def fused_compute_statistics(
                             overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             gt_datas,
                             dt_datas,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             compute_aos=False):

    
    gt_num = 0
    dt_num = 0
    # 传入的数据是10帧数据，分10次进行运行
    for i in range(gt_nums.shape[0]):            
        for t,thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num+dt_nums[i],gt_num:gt_num+gt_nums[i]]
            gt_data = gt_datas[i]
            dt_data = dt_datas[i]
            ignored_gt = ignored_gts[i]
            ignored_det = ignored_dets[i]

            tp,fp,fn,similarity, _ = compute_statistics_jit1(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos)

            pr[t,0]+=tp
            pr[t,1]+=fp
            pr[t,2]+=fn
            if similarity !=-1:
                pr[t,3]+=similarity
            
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=5):
    """fast iou algorithm. this function can be used independently to
    do result analysis. Must be used in CAMERA coordinate system.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
    """
    #如果长度不相等，直接报错
    assert len(gt_annos) == len(dt_annos)

    #  计算每一帧中包含物体的个数，组成一个列表[164,121,152...]   长度为测试集sample的个数> 帧数
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)

    num_examples = len(gt_annos)
    # 把测试样本index num_parts个一组进行划分  最后一组不足num_parts个 没关系split_parts[num_parts, num_parts, num_parts.....]
    split_parts = get_split_parts(num_examples, num_parts)

    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        #  分组取预测数据dict和 预测信息的dict
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]

        if metric == 0:
            continue

        elif metric == 1:
            loc = np.concatenate(
                [a["location"][:, [0, 1]] for a in gt_annos_part], 0)
            # dims的顺序为  l w h
            dims = np.concatenate(
                [a["dimensions"][:, [0, 1]] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            # gt_boxes  [x y z l w h rz]
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            
            loc = np.concatenate(
                [a["location"][:, [0, 1]] for a in dt_annos_part], 0)
            
            # 此处推理部分的dims是lwh
            dims = np.concatenate(
                [a["dimensions"][:, [0, 1]] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            # tmp = np.copy(dims[:, 2])
            # dims[:, 2] = dims[:, 1]
            # dims[:, 1] = tmp
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        else:
            raise ValueError("unknown metric")
        
        # 最终是数据集的b/n个部分的iou矩阵的列表
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0

    for j, num_part in enumerate(split_parts):         # 遍历每一个部分
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


#参数difficulty是int类型，为0,1,2
def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    
    #数据初始化
    gt_datas_list = []
    dt_datas_list = []
    ignored_gts, ignored_dets = [], []
    total_num_valid_gt = 0

    # 对于每一帧的数据进行操作
    for i in range(len(gt_annos)):
        
        #得到的是参数，当前帧的这个类别的 有效物体数，和有效物体的索引列表
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        num_valid_gt, ignored_gt, ignored_det  = rets
        # 将每一帧的ignored_gt数据类型进行转换为numpy格式，再添加到ignored_gts
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        total_num_valid_gt += num_valid_gt

        gt_datas_num = len(gt_annos[i]["name"])
        gt_datas_list.append(gt_datas_num)

        #dt_datas_score = dt_annos[i]["score"]
        dt_datas_score = dt_annos[i]["score"][..., np.newaxis]
        dt_datas_list.append(dt_datas_score)

    return (
                    gt_datas_list,  #存放的是 每一帧物体的个数
                    dt_datas_list,  #存放的是每一帧 不同物体的得分的情况，是（N,1）
                    ignored_gts, ignored_dets,   #存在
                    total_num_valid_gt                 #存在
                    )               


def eval_class(gt_annos,
               dt_annos,
               current_classes,
               difficultys,
               metric,
               min_overlaps,
               compute_aos=False,
               num_parts=5):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
        Args:
            gt_annos: dict, must from get_label_annos() in kitti_common.py
            dt_annos: dict, must from get_label_annos() in kitti_common.py
            current_classes: list of int, 0: car, 1: pedestrian, 2: cyclist
            difficultys: list of int. eval difficulty, 0: easy, 1: normal, 2: hard
            metric: eval type. 0: bbox, 1: bev, 2: 3d
            min_overlaps: float, min overlap. format: [num_overlap, metric, class].
            num_parts: int. a parameter for fast calculate algorithm

        Returns:
            dict of recall, precision and aos
        
            kitti中的数据字典里的键是：{
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': []
                'score'    # 对于验证集的数据，kitti中有score这一列的数字
            }
            min_overlaps:
                                    # (2, 3, num_classes) 其中:
                                    # 2 表示阈值为中等或者容易
                                    # 3 表示表示不同的指标 (bbox, bev, 3d), 
                                    # num_classes用于每个类的阈值

            参数difficultys:[0, 1, 2],<class 'list'>
    """

    # 保证测试集 预测结果的sample帧数与测试集的GT数据帧数是相同的 ！！！
    assert len(gt_annos) == len(dt_annos)
    # 验证集中帧的总数是 num_examples:51
    num_examples = len(gt_annos)
    #得到的split_parts是一个list的类型，num_parts=5,
    # 意思是将51分为5部分，经过一下函数得到的是：split_parts：[10,10,10,10,10,1]
    split_parts = get_split_parts(num_examples, num_parts)
    #计算iou
    #rets = calculate_iou_partly(gt_annos,dt_annos, metric, num_parts)
    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets

    N_SAMPLE_PTS = 41

    #获取min_overlaps的各个的维度，得到的是(2, 3, 5)
    # 获取当前类别的个数num_class：5，难度的个数为3
    num_minoverlap = len(min_overlaps)            #得到长度为2
    num_class = len(current_classes)
    num_difficulty = len(difficultys)

    #初始化precision，recall，aos
    precision = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])

    #每个类别：
    for m, current_class in enumerate(current_classes):
        # 每个难度：
        for l, difficulty in enumerate(difficultys):
            #参数difficulty是int类型，为0,1,2
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, total_num_valid_gt) = rets
            
            # 运行两次，首先进行中等难度的总体设置，然后进行简单设置。
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                thresholdss = []
                # 循环浏览数据集中的图像。因此一次只显示一张图片。
                for i in range(len(gt_annos)):
                    rets = compute_statistics_jit(
                        overlaps[i],     # 单个图像的iou值b/n gt和dt
                        gt_datas_list[i],       # 是一个数，表示当前帧中的物体个数
                        dt_datas_list[i],       # N x 1阵列，表示的是预测得到的N个物体的得分情况
                        ignored_gts[i],         # 长度N数组，-1、0
                        ignored_dets[i],        # 长度N数组，-1、0
                        metric,                             # 0, 1, 或 2 (bbox, bev, 3d)
                        min_overlap=min_overlap,         # 浮动最小IOU阈值为正
                        thresh=0.0,                 # 忽略得分低于此值的dt。
                        compute_fp=False)

                    tp, fp, fn, similarity, thresholds = rets
                    thresholdss += thresholds.tolist()
            
                #一维数组，记录匹配的dts分数，将list转为np格式
                thresholdss = np.array(thresholdss)

                # total_num_valid_gt是51帧数据里，vehicle出现的总个数
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)

                # thresholds是 N_SAMPLE_PTS长度的一维数组，记录分数，递减，表示阈值
                # 储存有关gt/dt框的信息（是否忽略，fn，tn，fp）
                pr = np.zeros([len(thresholds), 4])

                idx = 0
                for j,num_part in enumerate(split_parts):

                    gt_datas_part = np.array(gt_datas_list[idx:idx+num_part])
                    dt_datas_part = np.array(dt_datas_list[idx:idx+num_part])
                    ignored_dets_part = np.array(ignored_dets[idx:idx+num_part])
                    ignored_gts_part = np.array(ignored_gts[idx:idx+num_part])

                    # 再将各部分数据融合
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx+num_part],
                        total_dt_num[idx:idx+num_part],
                        gt_datas_part,
                        dt_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds
                    )
                    idx += num_part

                #计算recall和precision
                for i in range(len(thresholds)):
                    recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])

                # 返回各自序列的最值
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = np.max(precision[m, l, k, i:], axis=-1)
                    recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)
    ret_dict = {
        "recall": recall,             # [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]  
        "precision": precision,          # RECALLING RECALL的顺序，因此精度降低
        "orientation": aos,
    }
    return ret_dict


def get_mAP(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


def get_mAP_R40(prec):
    sums = 0
    for i in range(1, prec.shape[-1]):
        sums = sums + prec[..., i]
    return sums / 40 * 100


#打印结果的函数
def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


# 该函数是实现计算和评估的具体的函数
def do_eval(gt_annos,
            dt_annos,
            current_classes,
            min_overlaps,
            compute_aos=False,
            PR_detail_dict=None):
    '''
    function:  计算评价指标

    args :
            gt_annos:   整个测试集的GT标注字典

            dt_annos:  测试集的预测字典

            current_classes:   [0, 1, 2, 3 ...] 类别对应序号

            min_overlaps:   不同困难度 不同类别的一些个 IOU阈值  其size 是[num_minoverlap, metric, num_classes]  其中metric应该可以理解为困难度的size 3

            compute_aos:    目前不知道aos是个啥玩意儿

    '''

    # min_overlaps: [num_minoverlap, metric, num_class]
    #     #由上面得到的min_overlaps的形状是（2,3,5），
    # 这个是每个类别的IOU达到这个阈值时判断是否预测正确

    difficultys = [0, 1, 2]

    #metric: eval type. 0: bbox, 1: bev, 2: 3d 3:ano  这个函数可以做三个指标的评估  不同指标评估时 参数metric传入不同的数值
    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 1, min_overlaps)

    mAP_bev = get_mAP(ret["precision"])
    mAP_bev_R40 = get_mAP_R40(ret["precision"])

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2,min_overlaps)
    mAP_3d = get_mAP(ret["precision"])
    mAP_3d_R40 = get_mAP_R40(ret["precision"])

    #return mAP_bbox, mAP_bev, mAP_3d, mAP_aos, mAP_bbox_R40, mAP_bev_R40, mAP_3d_R40, mAP_aos_R40
    return  mAP_bev, mAP_3d, mAP_bev_R40, mAP_3d_R40


#定义函数，实现
def do_coco_style_eval(gt_annos, dt_annos, current_classes, overlap_ranges,
                       compute_aos):
    
    # overlap_ranges: [range, metric, num_class]
    #overlap_ranges的形状是：（3,3,5）
    min_overlaps = np.zeros([10, *overlap_ranges.shape[1:]])
    for i in range(overlap_ranges.shape[1]):
        for j in range(overlap_ranges.shape[2]):
            min_overlaps[:, i, j] = np.linspace(*overlap_ranges[:, i, j])
    #由上面得到的min_overlaps的形状是（10,3,5），
    # 这个是每个类别的IOU达到这个阈值时判断是否预测正确

    print("________________________overlap_ranges:__________________________")
    print(overlap_ranges)
    print(type(overlap_ranges))
    print(overlap_ranges.shape)

    print("________________________min_overlaps:__________________________")
    print(min_overlaps)
    print(type(min_overlaps))
    print(min_overlaps.shape)
    
    
    mAP_bev, mAP_3d, mAP_bev_R40, mAP_3d_R40 = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos)
    
    # ret: [num_class, num_diff, num_minoverlap]
    #mAP_bbox = mAP_bbox.mean(-1)
    mAP_bev = mAP_bev.mean(-1)
    mAP_3d = mAP_3d.mean(-1)
    
    return  mAP_bev, mAP_3d, mAP_bev_R40, mAP_3d_R40


#传入的参数current_classes 是五个类别：
#['vehicle', 'big_vehicle', 'pedestrian', 'bicycle', 'cone']
def get_coco_eval_result1(gt_annos, dt_annos, current_classes):
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Heavy_Truck',
        5: 'Light_Truck',
        6: 'Tricycle', 
        7: 'Small_Bus',  
        8: 'Big_Bus', 
        9: 'Ying_Er_Che'
    }
    class_to_range = {
        0: [0.5, 0.95, 10],
        1: [0.25, 0.7, 10],
        2: [0.25, 0.7, 10],
        3: [0.5, 0.95, 10],
        4: [0.25, 0.7, 10],
        5: [0.25, 0.7, 10],
        6: [0.25, 0.7, 10],
        7: [0.25, 0.7, 10],
        8: [0.25, 0.7, 10],
        9: [0.25, 0.7, 10],
    }
    #定义由名字变成类别，如0,1,2之类的字典
    name_to_class = {v: n for n, v in class_to_name.items()}

    #如果传入的current_classes 不是list或者tuple，则转为列表list格式
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    #定义一个空列表，如果current_classes中每一类为str类型，则存入相应的类别号
    #如当前判断的vehicle，big_vehicle，pedestrian，bicycle,cone,则current_classes_int=[ 0,1,2,3,4]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    #现在current_classes = [0,1,2,3,4],#当前的类别变成了含有数字的列表

    #定义一个三维的全零数组，形状是(3,3,5),有5个类别
    overlap_ranges = np.zeros([3, 3, len(current_classes)])
    for i, curcls in enumerate(current_classes):
        overlap_ranges[:, :, i] = np.array(
            class_to_range[curcls])[:, np.newaxis]
    # [:, np.newaxis]的作用是将class_to_range中每一值的行向量转化为列向量，并赋值为overlap_ranges中
    # 由此得到了，overlap_ranges = [[[0.5,0.25,0.25,0.5,0.25],[],[]],[[]]...]
    
    print("________________________overlap_ranges:__________________________")
    print(overlap_ranges)
    print(type(overlap_ranges))
    print(overlap_ranges.shape)

    result = ''
    
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        ##判断dt_annos中每一帧中有几个物体objects，如果不为零
        #接着判断，第一个值是否为-10，如果不是，将计算aos的标志compute_aos设为true
        if anno['name'].shape[0] != 0:
            if anno['box_rotation'][0][2] != -10:
                compute_aos = True
            break
    
    #调用函数，计算各个指标的结果
    # 函数的返回值是mAP_bev, mAP_3d, mAP_bev_R40, mAP_3d_R40
    mAPbev, mAP3d,mAP_bev_R40, mAP_3d_R40= do_coco_style_eval(gt_annos, dt_annos, current_classes, overlap_ranges, compute_aos)
    mAP_bev_R40, mAP_3d_R40=mAP_bev_R40, mAP_3d_R40
    
    print('_____________mAPbev________________')
    print(mAPbev)
    print('_____________mAP3d________________')
    print(mAP3d)

    #打印输出，，，，current_classes = [0,1,2,3,4],#当前的类别变成了含有数字的列表
    #循环当前的每个类别
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]

        #由class_to_range[curcls]得到的是一个类别的[0.5,0.95,10]的数组，
        # 再[[0, 2, 1]]，得到的是[0.5,10,0.95]，即顺序反了一下
        o_range = np.array(class_to_range[curcls])[[0, 2, 1]]
        #下面计算得到，= （0.95-0.5）/（10-1）= 0.9/9 =0.01,则o_range=[0.5,0.01,0.95]
        o_range[1] = (o_range[2] - o_range[0]) / (o_range[1] - 1)

        result += print_str((f"{class_to_name[curcls]} "
                             "coco AP@{:.2f}:{:.2f}:{:.2f}:".format(*o_range)))

        result += print_str((f"bev  AP:{mAPbev[j, 0]:.2f}, "
                             f"{mAPbev[j, 1]:.2f}, "
                             f"{mAPbev[j, 2]:.2f}"))
        result += print_str((f"3d   AP:{mAP3d[j, 0]:.2f}, "
                             f"{mAP3d[j, 1]:.2f}, "
                             f"{mAP3d[j, 2]:.2f}"))


    print("_________________result:___________________")
    print(result)
    return result



def get_official_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
    '''
        function: 使用kitti的官方评价准则计算评价结果
        
        arg:
            gt_annos:     测试集所有帧的标注信息字典  包括 {name, roatation_z,  gt_boxes_lidar....}

            dt_annos:  模型预测出的所有帧的信息字典 包括{........如上}信息

            current_classes:  数据集的类别信息

        return:

    '''
    overlap_0_7 = np.array([
                                                        [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7], 
                                                        [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
                                                        [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
                                                    ])
    overlap_0_5 = np.array([
                                                        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 
                                                        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                                        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                                                    ])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 3, 5]
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Heavy_Truck',
        5: 'Light_Truck',
        6: 'Tricycle', 
        7: 'Small_Bus',  
        8: 'Big_Bus', 
        9: 'Ying_Er_Che'

    }
    # 将类别名和索引  对应起来 便于 从类别 索引对应序号
    name_to_class = {v: n for n, v in class_to_name.items()}

    # 确保 current_classes是 list类型
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    
    #定义一个空列表，如果current_classes中每一类为str类型，则存入相应的类别号
    #如当前判断的Car，Pedestrian，Cyclist，则current_classes_int=[ 0,1,2]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    #当前的类别变成了含有数字的列表   current_classes=[ 0,1,2,3 ....]
    current_classes = current_classes_int

    #下面一行的作用：min_overlaps[:,:,[0,1,2,3,4]]，
    # 取min_overlaps的前5列，其中 5 为所用模型类别数量
    # min_overlaps.shape 是：（ 2 , 3, 5）
    min_overlaps = min_overlaps[:, :, current_classes]
    
    result = ''
    # check whether name is valid
    compute_aos = False

    #调用函数，计算各个值，4个指标
    mAP_bev, mAP_3d, mAP_bev_R40, mAP_3d_R40 = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, PR_detail_dict=PR_detail_dict)

    #将结果打印并返回
    ret_dict = {}
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        for i in range(min_overlaps.shape[0]):
            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
            
            result += print_str((f"bev  AP:{mAP_bev[j, 0, i]:.4f}, "
                                 f"{mAP_bev[j, 1, i]:.4f}, "
                                 f"{mAP_bev[j, 2, i]:.4f}"))
            result += print_str((f"3d   AP:{mAP_3d[j, 0, i]:.4f}, "
                                 f"{mAP_3d[j, 1, i]:.4f}, "
                                 f"{mAP_3d[j, 2, i]:.4f}"))

            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP_R40@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))

            result += print_str((f"bev  AP:{mAP_bev_R40[j, 0, i]:.4f}, "
                                 f"{mAP_bev_R40[j, 1, i]:.4f}, "
                                 f"{mAP_bev_R40[j, 2, i]:.4f}"))
            result += print_str((f"3d   AP:{mAP_3d_R40[j, 0, i]:.4f}, "
                                 f"{mAP_3d_R40[j, 1, i]:.4f}, "
                                 f"{mAP_3d_R40[j, 2, i]:.4f}"))

            if i == 0:
                ret_dict['%s_3d/easy_R40' % class_to_name[curcls]] = mAP_3d_R40[j, 0, 0]
                ret_dict['%s_3d/moderate_R40' % class_to_name[curcls]] = mAP_3d_R40[j, 1, 0]
                ret_dict['%s_3d/hard_R40' % class_to_name[curcls]] = mAP_3d_R40[j, 2, 0]
                ret_dict['%s_bev/easy_R40' % class_to_name[curcls]] = mAP_bev_R40[j, 0, 0]
                ret_dict['%s_bev/moderate_R40' % class_to_name[curcls]] = mAP_bev_R40[j, 1, 0]
                ret_dict['%s_bev/hard_R40' % class_to_name[curcls]] = mAP_bev_R40[j, 2, 0]

    return result, ret_dict
