import torch.nn as nn
import torch
import torch.nn.functional as F


# 18和34层的block
# 只有两层卷积
# 下采样层（残差链接）看是否传入
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # 第一层BN
        self.bn1 = nn.BatchNorm2d(out_channel)
        # 第一层ReLu
        self.relu = nn.ReLU()

        # 第二层卷积
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        # 第二层BN
        self.bn2 = nn.BatchNorm2d(out_channel)

        # 残差链接
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # 判断是否需要下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        # 第一层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二层
        out = self.conv2(out)
        out = self.bn2(out)

        # 残差链接
        out += identity
        out = self.relu(out)

        return out


# 50和101层的block
# 有三层
# 下采样层（残差链接）看是否传入
class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    # 深度因子
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        # 第一层，卷积核为1 * 1 * width
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)

        # 第二层, 卷积核为3 * 3 * width
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)

        # 第三层，卷积核为1 * 1 * out_channel*self.expansion,
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # 下采样层
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        # 第一层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二层
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 第三层
        out = self.conv3(out)
        out = self.bn3(out)

        # 残差链接
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.layer_out_channels = [256, 512, 1024, 2048]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        # 下采样层
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        # 第一块，都有下采样层，输出通道都是输入通道的四倍
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        # 输入通道翻四倍
        self.in_channel = channel * block.expansion

        # 第二块到最后一块，没有下采样层，输出通道和输入通道一样
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

    def forward_layer(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)

        return features


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


class UPerNet(nn.Module):
    def __init__(self, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256,512,1024,2048), fpn_dim=512):
        super(UPerNet, self).__init__()
        # 是否使用softmax
        self.use_softmax = use_softmax

        # ============================PPM Module====================================
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            # we use the feature map size instead of input image size, so down_scale = 1.0
            # ROIPooling
            # 以1，2，3，6为核大小的池化
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(output_size=(scale, scale)))
            # 1*1 -> bn -> relu
            self.ppm_conv.append(nn.Sequential(
                # 1*1降维
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                # bn
                nn.BatchNorm2d(512),
                # relu
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        # 融合所有特征
        # 3*3 -> bn -> relu
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # ============================FPN Module=====================================
        self.fpn_in = []
        # 256,512,1024
        for fpn_inplane in fpn_inplanes[:-1]: # skip the top layer
            self.fpn_in.append(nn.Sequential(
                # 1*1
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                # bn
                nn.BatchNorm2d(fpn_dim),
                # relu
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        # 256,512,1024
        for i in range(len(fpn_inplanes) - 1): # skip the top layer
            # 3*3 -> bn -> relu
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        # 1*1降维，4个特征层融合
        self.conv_fusion = conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1)


    def forward(self, conv_out, seg_size=None):

        # output_dict = {k: None for k in output_switch.keys()}

        # 获取resnet最后一层输出
        conv5 = conv_out[-1]
        # 输出特征大小
        input_size = conv5.size()
        ppm_out = [conv5]
        roi = [] # fake rois, just used for pooling
        # 遍历每个样本
        for i in range(input_size[0]): # batch size
            roi.append(torch.Tensor([i, 0, 0, input_size[3], input_size[2]]).view(1, -1)) # b, x0, y0, x1, y1
        # 拼接所有输出，转成同样的数据类型
        roi = torch.cat(roi, dim=0).type_as(conv5)

        # ============================PPN Module=====================================
        ppm_out = [conv5]
        # 每个roi区域遍历roipooling
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(
                # 1*1 -> bn -> relu
                pool_conv(
                    F.interpolate(
                        # roipol，对backbone最后一次进行下采样
                        # pool_scale(conv5, roi.detach()), 
                        pool_scale(conv5),
                        # 原始图片的宽高
                        (input_size[2], input_size[3]), 
                        mode='bilinear', 
                        align_corners=False)
                        )
                    )
        # 拼接所有输出
        ppm_out = torch.cat(ppm_out, 1)
        # 融合所有特征
        f = self.ppm_last_conv(ppm_out)

        # ============================Head=====================================

        
        # ============================FPN Module=====================================
        # ============================FPN Module 计算=====================================
        fpn_feature_list = [f]
        # 遍历FPN每层特征
        for i in reversed(range(len(conv_out) - 1)):
            # 获取特定层特征
            conv_x = conv_out[i]
            # 1*1 -> bn -> relu
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            # 双线性插值向下采样
            f = F.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            # 残差链接
            f = conv_x + f
            # 追加特征
            fpn_feature_list.append(self.fpn_out[i](f))
        fpn_feature_list.reverse() # [P2 - P5]

           
        # ============================FPN Module 融合=====================================
        # 最底层特征大小
        output_size = fpn_feature_list[0].size()[2:]
        # 添加最底层特征
        fusion_list = [fpn_feature_list[0]]
        # 遍历所有fpn输出层
        for i in range(1, len(fpn_feature_list)):
            # 上采样到同一特征大小
            fusion_list.append(F.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        # 从第一维度拼接所有特征
        fusion_out = torch.cat(fusion_list, 1)
        # 融合所有层特征
        x = self.conv_fusion(fusion_out)
            
        return x
    
class resnet_with_upernet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet50()
        self.decoder = UPerNet(
            fc_dim=self.encoder.layer_out_channels[3],
            fpn_inplanes=self.encoder.layer_out_channels,
            fpn_dim=512
        )

    def forward(self, x):
        features = self.encoder.forward_layer(x)
        out = self.decoder(features)
        return out

if __name__ == "__main__":
    input_tensor = torch.randn(10, 3, 256, 256, dtype=torch.float)

#     net = resnet50()
#     # print(net)
#     # for name, param in net.named_parameters():
#     #     if 'layers_fuse' not in name:
#     #         print(name)
#     #         param.requires_grad = False
        
#     pretrained_weights = "./resnet50.pth"
#     state_dict = torch.load(pretrained_weights, map_location="cpu")
#     # for name, weight in state_dict.items():
#     #     print(name)

#     msg = net.load_state_dict(state_dict, strict=False)
#     print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

#     feat, attns, qkvs = net.get_intermediate_feat(input_tensor)

#     print(feat[0].shape)

    net_with_upernet = resnet_with_upernet()

    # print(net_with_upernet)

    for name, param in net_with_upernet.named_parameters():
        if 'decoder' not in name:
            param.requires_grad = False
        
    pretrained_weights = "./resnet50.pth"
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    for name, weight in state_dict.items():
        print(name)

    msg = net_with_upernet.encoder.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

    out_with_upernet = net_with_upernet(input_tensor)

    print(out_with_upernet.shape)