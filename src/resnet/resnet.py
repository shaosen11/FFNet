import torch.nn as nn
import torch
import torch.nn.functional as F

class OPM(nn.Module):
    def __init__(self, inchannel: int):
        super().__init__()
        self.conv = nn.Conv2d(inchannel, 1, 1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))
    

class LayerFuse(nn.Module):
    def __init__(self, chnnels, window_size):
        super().__init__()
        self.out_channels = chnnels
        self.window_size = window_size

        self.opms = nn.ModuleList([
            OPM(self.out_channels[i])
        for i in range(len(self.out_channels))])

        self.up_convs = nn.ModuleList([
            nn.Conv2d(self.out_channels[i - 1], self.out_channels[i], kernel_size=1)
        for i in range(len(self.out_channels))])

        self.down_convs = nn.ModuleList([
            nn.Conv2d(self.out_channels[i], self.out_channels[i - 1], kernel_size=1)
        for i in range(len(self.out_channels))])

        self.bns1 = nn.ModuleList([
            nn.BatchNorm2d(self.out_channels[i])
        for i in range(len(self.out_channels))])

        self.act1 = nn.ReLU()

        self.bns2 = nn.ModuleList([
            nn.BatchNorm2d(self.out_channels[i])
        for i in range(len(self.out_channels))])

        self.act2 = nn.ReLU()

    def forward(self, features, n=1):
        feat = []
        last_feature = None
        for i in range(len(features) - 1, -1, -1):
            # 获取当前特征
            feature = features[i]
            
            # 最后一层无操作
            if i + 1 == len(features):
                last_feature = feature
            else:
                # 当前层升维
                feature = self.up_convs[i + 1](feature)
                # 上一层的宽高
                next_window_size = self.window_size[i]
                last_feature = F.interpolate(last_feature, (next_window_size, next_window_size), mode='bilinear')
                # 物体推荐模块
                mask = self.opms[i + 1](last_feature)
                last_feature = mask * last_feature + (1 - mask) * feature
                # # relu
                # last_feature = self.act1(last_feature)
                # # 归一化
                # last_feature = self.bns1[i + 1](last_feature)
                # 降维
                last_feature = self.down_convs[i + 1](last_feature)
                # # relu
                # last_feature = self.act2(last_feature)
                # # 归一化
                # last_feature = self.bns2[i](last_feature)

            if i < n:
                feat.append(last_feature)
        return feat


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

        self.features = []

        self.layer_fuse_out_channels = [256, 512, 1024, 2048]
        # self.window_size = [56, 28, 14, 7]
        self.layer_fuse_window_size = [64, 32, 16, 8]

        self.layer_fuse_input_resolution = [64, 32, 16, 8]

        # 特征融合
        self.layers_fuse = LayerFuse(self.layer_fuse_out_channels, 
                                     self.layer_fuse_window_size)

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
    

    def get_intermediate_feat(self, x, n=1):
        feat = []
        attns = []
        qkvs = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        self.features = []
        x = self.layer1(x)
        self.features.append(x)
        x = self.layer2(x)
        self.features.append(x)
        x = self.layer3(x)
        self.features.append(x)
        x = self.layer4(x)
        self.features.append(x)

        # 特征融合
        feat = self.layers_fuse(self.features, n = 1)

        return feat, attns, qkvs




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


# if __name__ == "__main__":
#     input_tensor = torch.randn(10, 3, 256, 256, dtype=torch.float)

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