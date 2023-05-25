import numpy as np
import torch
import torch.nn as nn


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS                  #  [5, 5]
            layer_strides = self.model_cfg.LAYER_STRIDES           #  [1, 2]
            num_filters = self.model_cfg.NUM_FILTERS                   # [128, 256]
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS    # [256, 256]
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES                          # [1, 2]
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)      # 2
        c_in_list = [input_channels, *num_filters[:-1]]
        
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),

                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        # 统计1的个数 和 > 0的个数
        # print(np.sum(spatial_features.cpu().numpy().flatten() > -1))
        # print(np.sum(spatial_features.cpu().numpy().flatten() == 1))
        # print('test 02 spatial_features.shape is',spatial_features.shape)
        ups = []
        ret_dict = {}
        # x = spatial_features.to(torch.float32)
        x = spatial_features
        # print('J note spatial 2 d type', x.dtype)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x
        # print('J note: after backbone_2d feature shape is:',  x.shape)

        return data_dict

    def test(self):
        print('testtest!!!!!!')

class res_imitateyolo(nn.Module):
    '''
        说明：此类为残差网络组件，执行后尺寸减半  channel由实例化时的outchannel决定
                    1 3*3卷积尺寸不变 通道指定 outchannel
                    2 2*2max_pooling尺寸减半通道不变
                    3 按res_num循环执行res_num次残差网络子模块

    '''
    def __init__(self, inputChannel, outchannel, res_num):
        super(res_imitateyolo, self).__init__()
        self.res_num = res_num
        self.conv1 = nn.Sequential(
            nn.Conv2d(inputChannel, outchannel, kernel_size=3,padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU()
        )
        
        self.maxpool2d = nn.MaxPool2d(kernel_size=2)
        self.re_bloc = self.res_bloc(outchannel)
        

    def res_bloc(self, channel):
        # print('J note channel', channel)
        res = nn.Sequential(
            nn.Conv2d( int(channel), int(channel/2), kernel_size=1),
            nn.BatchNorm2d(int(channel/2)),
            nn.ReLU(),
           
            nn.Conv2d(int(channel/2), channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        return res
    
    def forward(self, input_data):
        '''
        '''
        input_data = self.conv1(input_data)
        input_data = self.maxpool2d(input_data)
        for i in range(self.res_num):
            shortCut = input_data
            input_data = self.re_bloc(input_data)
            input_data = input_data + shortCut
        return input_data

    
class ImitateYoloBackbone(nn.Module):
    '''
        说明：该组件为模仿yolo特征提取组件所写，用于测试修改backbone后的AP以及推理速度
        设计为 (后期设计)
                        一个共享卷积 尺寸不变做特征融合
                        接一个convBlock（带残差项）尺寸减半
                        在接一个identiblock（带残差 带上采样）融合多尺度信息
        当前模块：
                        直接把吴博的2D特征提取拿过来
    '''
    def __init__(self, model_cfg, input_channels):
        super(ImitateYoloBackbone, self).__init__()
        self.model_cfg = model_cfg
        print('J note backbone2D input channel', input_channels)
        
        #  [64, 128, 256, 512]  残差模块的输出channel
        self.res_channels = self.model_cfg.resChannels

        #  根在残差模块后 卷积的输出channel convChannels: [512, 1024, 512, 256]
        self.conv_channels = self.model_cfg.convChannels

        # 对特征图做一次尺度不变的特征融合 inputChannel, h, w > 64, h, w 
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, self.res_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.res_channels[0]),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.res_channels[0], self.res_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.res_channels[0]),
            nn.ReLU()
        )
        
        # 残差块 1                                                        64, h, w  >  64, h/2, w/2 
        self.res_yolo1 = self.make_res_bloc(res_bloc_class=res_imitateyolo, inputChannel= self.res_channels[0], 
                                        outchannel= self.res_channels[0], res_num= 1)

        # 残差块 2                                                        64, h/2, w/2  >  128, h/4, w/4
        self.res_yolo2 = self.make_res_bloc(res_bloc_class=res_imitateyolo, inputChannel= self.res_channels[0], 
                                        outchannel= self.res_channels[1], res_num= 2)

        # 残差块 3                                                        128, h/4, w/4  >  256, h/8, w/8 
        self.res_yolo3 = self.make_res_bloc(res_bloc_class=res_imitateyolo, inputChannel= self.res_channels[1], 
                                        outchannel= self.res_channels[2], res_num= 4)
        
        # 残差块 3                                                        256, h/8, w/8  >  512, h/16, w/16
        # self.res_yolo4 = self.make_res_bloc(res_bloc_class=res_imitateyolo, inputChannel= self.res_channels[2], 
        #                                 outchannel= self.res_channels[3], res_num= 6)


        # slelf.convChannels: [512, 1024, 512, 256]
        # 几个卷积后  256, h/8, w/8
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels= self.res_channels[2], out_channels= self.conv_channels[0], kernel_size=1),
            nn.BatchNorm2d(self.conv_channels[0]),
            nn.ReLU()
        )
        
        self.conv_2  = nn.Sequential(
            nn.Conv2d(in_channels= self.conv_channels[0], out_channels= self.conv_channels[1], kernel_size= 3, padding= 1 ),
            nn.BatchNorm2d(self.conv_channels[1]),
            nn.ReLU()
        )
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels= self.conv_channels[1], out_channels= self.conv_channels[2],kernel_size= 1 ),
            nn.BatchNorm2d(self.conv_channels[2]),
            nn.ReLU()
        )
        
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels= self.conv_channels[2], out_channels= self.conv_channels[3],kernel_size= 1 ),
            nn.BatchNorm2d(self.conv_channels[3]),
            nn.ReLU()
        )
                
        #  deConv
        # 256, h/4, w/4    forward时要与self.res_yolo3结果相加
        self.deConv =  nn.Sequential(
            nn.ConvTranspose2d(in_channels= self.conv_channels[3], out_channels= self.conv_channels[3], 
                                                        kernel_size=2, stride= 2, bias=False),
            nn.BatchNorm2d(self.conv_channels[3]),
            nn.ReLU()
        )
                             

        # 相加后的尺寸仍为256, h/8, w/8
        # 再做几次卷积
        #  256, h/4, w/4
        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_channels= self.res_channels[1] + self.conv_channels[3], out_channels= self.conv_channels[3], kernel_size=1),
            nn.BatchNorm2d(self.conv_channels[3]),
            nn.ReLU()
        )
        # 512, h/8, w/8
        self.conv_6 = nn.Sequential(
            nn.Conv2d(in_channels=  self.conv_channels[3], out_channels= 2 * self.conv_channels[3], kernel_size=3, padding= 1),
            nn.BatchNorm2d(2 * self.conv_channels[3],),
            nn.ReLU()
        )
        self.conv_7 = nn.Sequential(
            nn.Conv2d(in_channels= 2 * self.conv_channels[3], out_channels= self.conv_channels[3], kernel_size= 1),
            nn.BatchNorm2d(self.conv_channels[3]),
            nn.ReLU()
        )
        self.conv_8 = nn.Sequential(
            nn.Conv2d(in_channels= self.conv_channels[3], out_channels= self.conv_channels[3], kernel_size=3, padding= 1),
            nn.BatchNorm2d(self.conv_channels[3]),
            nn.ReLU()
        )
        # self.conv_9 = nn.Sequential(
        #     nn.Conv2d(in_channels= self.conv_channels[3], out_channels= 256, kernel_size=3, padding= 1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU()
        # )

        # 这个是该模块的输出channel
        self.num_bev_features = self.conv_channels[3]

    def make_res_bloc(self, res_bloc_class, inputChannel, outchannel, res_num):
        layers = []
        layers.append(res_bloc_class(inputChannel, outchannel, res_num))
        return nn.Sequential(*layers)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict
        # 64 scale = 1
        x = self.conv1(spatial_features)
        x = self.conv2(x)
        # 64 scale = 1/2
        x = self.res_yolo1(x)
        # 128 scale = 1/4
        res_1_part_4 = self.res_yolo2(x)
        # 256  scale = 1/8
        x = self.res_yolo3(res_1_part_4)
        # 512  scale = 1/16
        # x = self.res_yolo4(res_1_part_8)
        # print('J after res_yolo4', x.shape)

        # 512 scale = 1/8
        x = self.conv_1(x)
        # 1024 scale = 1/8
        x = self.conv_2(x)
        # 512 scale = 1/8
        x = self.conv_3(x)
        # 256 scale = 1/8
        x = self.conv_4(x)
        # print('J note after conv_4', x.shape)
        
        # 256 scale = 1/4
        x = self.deConv(x)
        # 512 scale = 1/8
        # print('J note, cat shape01', x.shape, res_1_part_4.shape)
        x = torch.cat((x, res_1_part_4), dim=1)

        # 256 scale = 1/8
        x  =self.conv_5(x)
        # 512 scale = 1/8
        x  =self.conv_6(x)
        # 256 scale = 1/8
        x  =self.conv_7(x)
        # 256 scale = 1/8
        x  =self.conv_8(x)
        # 23 scale = 1/8
        # x  =self.conv_9(x)

        # data_dict['spatial_features_2d'] = x
        # print('J note: after backbone_2d feature shape is:',  x.shape)

        return x