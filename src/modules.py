import torch

from utils import *
import torch.nn.functional as F
import dino.vision_transformer as vits
import SwinV2.swin_transformer_v2 as swinv2
import resnet.resnet as resnet


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class DinoFeaturizer(nn.Module):

    def __init__(self, dim, cfg):
        super().__init__()
        self.cfg = cfg
        # 嵌入层维度
        self.dim = dim
        # patch size，划分多少个patch
        patch_size = self.cfg.dino_patch_size
        self.patch_size = patch_size
        # 特征类型
        self.feat_type = self.cfg.dino_feat_type
        # vit small
        arch = self.cfg.model_type
        # 获取模型
        self.model = vits.__dict__[arch](
            patch_size=patch_size,
            num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval().cuda()
        self.dropout = torch.nn.Dropout2d(p=.1)

        # 预训练权重
        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError("Unknown arch and patch size")

        if cfg.pretrained_weights is not None:
            # 读取权重
            state_dict = torch.load(cfg.pretrained_weights, map_location="cpu")
            state_dict = state_dict["teacher"]
            # 删除前缀
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

            # state_dict = {k.replace("projection_head", "mlp"): v for k, v in state_dict.items()}
            # state_dict = {k.replace("prototypes", "last_layer"): v for k, v in state_dict.items()}

            # 加载权重
            msg = self.model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(cfg.pretrained_weights, msg))
        else:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            self.model.load_state_dict(state_dict, strict=True)

        # 嵌入层特征维度
        if arch == "vit_small":
            self.n_feats = 384
        else:
            self.n_feats = 768

        # KNN聚类头
        self.cluster1 = self.make_clusterer(self.n_feats)
        # 聚类头类型
        self.proj_type = cfg.projection_type
        # 非线性聚类头
        if self.proj_type == "nonlinear":
            self.cluster2 = self.make_nonlinear_clusterer(self.n_feats)

    # KNN聚类头
    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))  # ,

    # 非线性聚类头
    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    def forward(self, img, n=1, return_class_feat=False):
        self.model.eval()
        with torch.no_grad():
            assert (img.shape[2] % self.patch_size == 0)
            assert (img.shape[3] % self.patch_size == 0)

            # get selected layer activations
            # 获取每层交互特征，注意力，qkv
            feat, attn, qkv = self.model.get_intermediate_feat(img, n=n)
            feat, attn, qkv = feat[0], attn[0], qkv[0]

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            # 返回特征类型
            if self.feat_type == "feat":
                # image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
                image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            elif self.feat_type == "KK":
                image_k = qkv[1, :, :, 1:, :].reshape(feat.shape[0], 6, feat_h, feat_w, -1)
                B, H, I, J, D = image_k.shape
                image_feat = image_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
            else:
                raise ValueError("Unknown feat type:{}".format(self.feat_type))

            if return_class_feat:
                return feat[:, :1, :].reshape(feat.shape[0], 1, 1, -1).permute(0, 3, 1, 2)

        # 聚类
        if self.proj_type is not None:
            code = self.cluster1(self.dropout(image_feat))
            if self.proj_type == "nonlinear":
                code += self.cluster2(self.dropout(image_feat))
        else:
            code = image_feat

        # dropout
        if self.cfg.dropout:
            return self.dropout(image_feat), code
        else:
            return image_feat, code

class SwinFeaturizer(nn.Module):
    def __init__(self, dim, cfg):
        super().__init__()
        self.cfg = cfg
        # 嵌入层维度
        self.dim = dim
        # patch size，划分多少个patch
        self.patch_size = self.cfg.swin_patch_size
        # 特征类型
        self.feat_type = self.cfg.dino_feat_type
        
        # swin small
        arch = "swinv2_" + cfg.swin_model_type + "_" + "window" + str(cfg.swin_window_size)

        if cfg.swin_is_classify:
            arch += "_class"
        print("arch:", arch)

        # 获取模型
        self.model = swinv2.__dict__[arch](img_size=cfg.swin_img_size, 
                                           window_size=cfg.swin_window_size,
                                           patch_size=cfg.swin_patch_size,
                                           new_num_classes=100,
                                           is_classify=False)
        self.model.train().cuda()

        for name, param in self.model.named_parameters():
            if 'layers_fuse' not in name:
                param.requires_grad = False
        
        if cfg.swin_is_classify:
            if cfg.swin_model_type == "base" and cfg.swin_window_size == 16:
                pretrained_weights = "./SwinV2/swinv2_base_patch4_window16_256-pre.pth"
            else:
                raise ValueError("Unknown model type and window size")
        else:    
            if cfg.swin_model_type == "tiny" and cfg.swin_window_size == 8:
                pretrained_weights = "./SwinV2/swinv2_tiny_patch4_window8_256.pth"
            elif cfg.swin_model_type == "tiny" and cfg.swin_window_size == 16:
                pretrained_weights = "./SwinV2/swinv2_tiny_patch4_window16_256.pth"
            elif cfg.swin_model_type == "small" and cfg.swin_window_size == 8:
                pretrained_weights = "./SwinV2/swinv2_small_patch4_window8_256.pth"
            elif cfg.swin_model_type == "small" and cfg.swin_window_size == 16:
                pretrained_weights = "./SwinV2/swinv2_small_patch4_window16_256.pth"
            elif cfg.swin_model_type == "base" and cfg.swin_window_size == 8:
                pretrained_weights = "./SwinV2/swinv2_base_patch4_window8_256.pth"
            elif cfg.swin_model_type == "base" and cfg.swin_window_size == 16:
                pretrained_weights = "./SwinV2/swinv2_base_patch4_window16_256.pth"
            else:
                raise ValueError("Unknown model type and window size")

        print("pretrained_weights:", pretrained_weights)

        if pretrained_weights is not None:
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            msg = self.model.load_state_dict(state_dict['model'], strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

        if cfg.swin_model_type == "base":
            self.n_feats = 256
        else:
            self.n_feats = 192

        # self.n_feats = 1024

        # KNN聚类头
        self.cluster1 = self.make_clusterer(self.n_feats)
        # 聚类头类型
        self.proj_type = cfg.projection_type
        # 非线性聚类头
        if self.proj_type == "nonlinear":
            self.cluster2 = self.make_nonlinear_clusterer(self.n_feats)
        self.dropout = torch.nn.Dropout2d(p=.1)

    # KNN聚类头
    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))  # ,

    # 非线性聚类头
    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))
    
    def forward(self, img, n=1, return_class_feat=False):
        assert (img.shape[2] % self.patch_size == 0)
        assert (img.shape[3] % self.patch_size == 0)

        # get selected layer activations
        # 获取每层交互特征，注意力，qkv
        feat, attn, qkv = self.model.get_intermediate_feat(img, n=n)
        feat = feat[-1]

        feat_h = img.shape[2] // self.patch_size
        feat_w = img.shape[3] // self.patch_size

        # feat_h = img.shape[2] // (self.patch_size * 8)
        # feat_w = img.shape[3] // (self.patch_size * 8)


        # 返回特征类型
        if self.feat_type == "feat":
            image_feat = feat[:, :, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
        elif self.feat_type == "KK":
            image_k = qkv[1, :, :, 1:, :].reshape(feat.shape[0], 6, feat_h, feat_w, -1)
            B, H, I, J, D = image_k.shape
            image_feat = image_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
        else:
            raise ValueError("Unknown feat type:{}".format(self.feat_type))

        if return_class_feat:
            return feat[:, :1, :].reshape(feat.shape[0], 1, 1, -1).permute(0, 3, 1, 2)

        # print("image_feat:", image_feat.shape)

        # 聚类
        if self.proj_type is not None:
            code = self.cluster1(self.dropout(image_feat))
            if self.proj_type == "nonlinear":
                code += self.cluster2(self.dropout(image_feat))
        else:
            code = image_feat

        # dropout
        if self.cfg.dropout:
            return self.dropout(image_feat), code
        else:
            return image_feat, code


class ResnetFeaturizer(nn.Module):
    def __init__(self, dim, cfg):
        super().__init__()
        self.cfg = cfg
        # 嵌入层维度
        self.dim = dim
        # 特征类型
        self.feat_type = self.cfg.dino_feat_type
        
        # swin small
        arch = self.cfg.resnet_model_type

        # 获取模型
        self.model = resnet.__dict__[arch]()
        self.model.train().cuda()

        for name, param in self.model.named_parameters():
            if 'layers_fuse' not in name:
                param.requires_grad = False

        if cfg.resnet_pretrained_weights is not None:
            state_dict = torch.load(cfg.resnet_pretrained_weights, map_location="cpu")
            msg = self.model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(cfg.resnet_pretrained_weights, msg))


        self.n_feats = 256

        # KNN聚类头
        self.cluster1 = self.make_clusterer(self.n_feats)
        # 聚类头类型
        self.proj_type = cfg.projection_type
        # 非线性聚类头
        if self.proj_type == "nonlinear":
            self.cluster2 = self.make_nonlinear_clusterer(self.n_feats)
        self.dropout = torch.nn.Dropout2d(p=.1)

    # KNN聚类头
    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))  # ,

    # 非线性聚类头
    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))
    
    def forward(self, img, n=1, return_class_feat=False):
        # get selected layer activations
        # 获取每层交互特征，注意力，qkv
        feat, attn, qkv = self.model.get_intermediate_feat(img, n=n)
        image_feat = feat[0]

        # 聚类
        if self.proj_type is not None:
            code = self.cluster1(self.dropout(image_feat))
            if self.proj_type == "nonlinear":
                code += self.cluster2(self.dropout(image_feat))
        else:
            code = image_feat

        # dropout
        if self.cfg.dropout:
            return self.dropout(image_feat), code
        else:
            return image_feat, code

class ResizeAndClassify(nn.Module):

    def __init__(self, dim: int, size: int, n_classes: int):
        super(ResizeAndClassify, self).__init__()
        self.size = size
        self.predictor = torch.nn.Sequential(
            torch.nn.Conv2d(dim, n_classes, (1, 1)),
            torch.nn.LogSoftmax(1))

    def forward(self, x):
        return F.interpolate(self.predictor.forward(x), self.size, mode="bilinear", align_corners=False)
    

class ClusterLookup(nn.Module):

    def __init__(self, dim: int, n_classes: int):
        super(ClusterLookup, self).__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.clusters = torch.nn.Parameter(torch.randn(n_classes, dim))

    def reset_parameters(self):
        with torch.no_grad():
            self.clusters.copy_(torch.randn(self.n_classes, self.dim))

    def forward(self, x, alpha, log_probs=False):
        # 参数归一化
        normed_clusters = F.normalize(self.clusters, dim=1)
        # 输入特征归一化
        normed_features = F.normalize(x, dim=1)
        # 点积
        inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)

        # 聚类概率
        if alpha is None:
            cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), self.clusters.shape[0]) \
                .permute(0, 3, 1, 2).to(torch.float32)
        else:
            cluster_probs = nn.functional.softmax(inner_products * alpha, dim=1)

        # 聚类损失
        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()
        if log_probs:
            return nn.functional.log_softmax(inner_products * alpha, dim=1)
        else:
            return cluster_loss, cluster_probs


class FeaturePyramidNet(nn.Module):

    @staticmethod
    def _helper(x):
        # TODO remove this hard coded 56
        return F.interpolate(x, 56, mode="bilinear", align_corners=False).unsqueeze(-1)

    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)),
            LambdaLayer(FeaturePyramidNet._helper))

    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)),
            LambdaLayer(FeaturePyramidNet._helper))

    def __init__(self, granularity, cut_model, dim, continuous):
        super(FeaturePyramidNet, self).__init__()
        self.layer_nums = [5, 6, 7]
        self.spatial_resolutions = [7, 14, 28, 56]
        self.feat_channels = [2048, 1024, 512, 3]
        self.extra_channels = [128, 64, 32, 32]
        self.granularity = granularity
        self.encoder = NetWithActivations(cut_model, self.layer_nums)
        self.dim = dim
        self.continuous = continuous
        self.n_feats = self.dim

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        assert granularity in {1, 2, 3, 4}
        self.cluster1 = self.make_clusterer(self.feat_channels[0])
        self.cluster1_nl = self.make_nonlinear_clusterer(self.feat_channels[0])

        if granularity >= 2:
            # self.conv1 = DoubleConv(self.feat_channels[0], self.extra_channels[0])
            # self.conv2 = DoubleConv(self.extra_channels[0] + self.feat_channels[1], self.extra_channels[1])
            self.conv2 = DoubleConv(self.feat_channels[0] + self.feat_channels[1], self.extra_channels[1])
            self.cluster2 = self.make_clusterer(self.extra_channels[1])
        if granularity >= 3:
            self.conv3 = DoubleConv(self.extra_channels[1] + self.feat_channels[2], self.extra_channels[2])
            self.cluster3 = self.make_clusterer(self.extra_channels[2])
        if granularity >= 4:
            self.conv4 = DoubleConv(self.extra_channels[2] + self.feat_channels[3], self.extra_channels[3])
            self.cluster4 = self.make_clusterer(self.extra_channels[3])

    def c(self, x, y):
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        with torch.no_grad():
            feats = self.encoder(x)
        low_res_feats = feats[self.layer_nums[-1]]

        all_clusters = []

        # all_clusters.append(self.cluster1(low_res_feats) + self.cluster1_nl(low_res_feats))
        all_clusters.append(self.cluster1(low_res_feats))

        if self.granularity >= 2:
            # f1 = self.conv1(low_res_feats)
            # f1_up = self.up(f1)
            f1_up = self.up(low_res_feats)
            f2 = self.conv2(self.c(f1_up, feats[self.layer_nums[-2]]))
            all_clusters.append(self.cluster2(f2))
        if self.granularity >= 3:
            f2_up = self.up(f2)
            f3 = self.conv3(self.c(f2_up, feats[self.layer_nums[-3]]))
            all_clusters.append(self.cluster3(f3))
        if self.granularity >= 4:
            f3_up = self.up(f3)
            final_size = self.spatial_resolutions[-1]
            f4 = self.conv4(self.c(f3_up, F.interpolate(
                x, (final_size, final_size), mode="bilinear", align_corners=False)))
            all_clusters.append(self.cluster4(f4))

        avg_code = torch.cat(all_clusters, 4).mean(4)

        if self.continuous:
            clusters = avg_code
        else:
            clusters = torch.log_softmax(avg_code, 1)

        return low_res_feats, clusters


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def average_norm(t):
    return t / t.square().sum(1, keepdim=True).sqrt().mean()


def tensor_correlation(a, b):
    return torch.einsum("nchw,ncij->nhwij", a, b)


# 用于对输入的张量t进行采样，采样的坐标值由参数coords给出``
def sample(t: torch.Tensor, coords: torch.Tensor):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)


@torch.jit.script
def super_perm(size: int, device: torch.device):
    perm = torch.randperm(size, device=device, dtype=torch.long)
    perm[perm == torch.arange(size, device=device)] += 1
    return perm % size


# 用于从稀疏张量中提取非零元素的位置，并将这些位置转化为坐标值。
def sample_nonzero_locations(t, target_size):
    nonzeros = torch.nonzero(t)
    coords = torch.zeros(target_size, dtype=nonzeros.dtype, device=nonzeros.device)
    n = target_size[1] * target_size[2]
    for i in range(t.shape[0]):
        selected_nonzeros = nonzeros[nonzeros[:, 0] == i]
        if selected_nonzeros.shape[0] == 0:
            selected_coords = torch.randint(t.shape[1], size=(n, 2), device=nonzeros.device)
        else:
            selected_coords = selected_nonzeros[torch.randint(len(selected_nonzeros), size=(n,)), 1:]
        coords[i, :, :, :] = selected_coords.reshape(target_size[1], target_size[2], 2)
    coords = coords.to(torch.float32) / t.shape[1]
    coords = coords * 2 - 1
    return torch.flip(coords, dims=[-1])


class ContrastiveCorrelationLoss(nn.Module):

    def __init__(self, cfg, ):
        super(ContrastiveCorrelationLoss, self).__init__()
        self.cfg = cfg

    # 标准化张量
    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2

    # 张量平展(新增)
    def flatten_feature(self, feats):
        batch_size, C, H, W = feats.size()
        feats_flattened = feats.view(batch_size * H * W, C)

        return feats_flattened
    
    # 相似度衡量(新增)
    def similar(self, f1, f2):
        """
            f1, f2: 二维输入;
            return: 一维张量
            衡量方法: 余弦相似度
        """
        # sim = torch.sum(f1*f2)/((torch.sqrt(torch.sum(f1*f1))*(torch.sqrt(torch.sum(f2*f2)))))
        sim = F.cosine_similarity(f1, f2, dim=1)
        return sim
    
    # SegNCE损失函数
    def helper_SegNCE(self, f, f_pos, f_negs):
        """
            input:
                f: 原样本, [batch_size, C, 11, 11]
                f_pos: 正样本,
                f_negs: 负样本list
            return:
                SegNCE loss
        """
        # 张量平展
        f_flatten = self.flatten_feature(self.standard_scale(f))
        f_pos_flatten = self.flatten_feature(self.standard_scale(f_pos))
        # 温度参数读取
        t = self.cfg.temperature

        denominator = 0
        for f_neg in f_negs:
            f_neg_flatten = self.flatten_feature(self.standard_scale(f_neg))
            denominator += torch.exp(self.similar(f_flatten, f_neg_flatten)/t)
        denominator = torch.sum(denominator).item()
        molecular = torch.exp(self.similar(f_flatten, f_pos_flatten)/t)
        loss = -torch.sum(torch.log((molecular/denominator)))
        return loss


    def helper(self, f1, f2, c1, c2, shift):
        with torch.no_grad():
            # Comes straight from backbone which is currently frozen. this saves mem.
            # 数首先对输入的特征图f1和f2进行标准化处理，并使用tensor_correlation方法计算它们之间的相关性矩阵fd
            fd = tensor_correlation(norm(f1), norm(f2))

            # 归一化
            if self.cfg.pointwise:
                old_mean = fd.mean()
                fd -= fd.mean([3, 4], keepdim=True)
                fd = fd - fd.mean() + old_mean

        # 函数使用tensor_correlation方法计算输入的两个卷积结果c1和c2之间的相关性矩阵cd
        cd = tensor_correlation(norm(c1), norm(c2))

        if self.cfg.zero_clamp:
            min_val = 0.0
        else:
            min_val = -9999.0

        # 如果stabilize为True，则将cd的值截断到[-9999.0, 0.8]之间，并将fd与参数shift之差乘以截断后的cd
        if self.cfg.stabalize:
            loss = - cd.clamp(min_val, .8) * (fd - shift)
        # 否则，将fd与参数shift之差乘以cd。
        else:
            loss = - cd.clamp(min_val) * (fd - shift)

        return loss, cd

    # def forward(self,
    #             orig_feats: torch.Tensor, orig_feats_pos: torch.Tensor,
    #             orig_salience: torch.Tensor, orig_salience_pos: torch.Tensor,
    #             orig_code: torch.Tensor, orig_code_pos: torch.Tensor,
    #             ):

    #     # 
    #     coord_shape = [orig_feats.shape[0], self.cfg.feature_samples, self.cfg.feature_samples, 2]

    #     # 默认False
    #     if self.cfg.use_salience:
    #         # 从稀疏张量中提取非零元素的位置，并将这些位置转化为坐标值。
    #         coords1_nonzero = sample_nonzero_locations(orig_salience, coord_shape)
    #         coords2_nonzero = sample_nonzero_locations(orig_salience_pos, coord_shape)
    #         # 对coord_shape均匀采样，乘2保证正数，减1进行归一化到[0,2]
    #         coords1_reg = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
    #         coords2_reg = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
    #         # 将大于0.1的进行保存
    #         mask = (torch.rand(coord_shape[:-1], device=orig_feats.device) > .1).unsqueeze(-1).to(torch.float32)
    #         # 
    #         coords1 = coords1_nonzero * mask + coords1_reg * (1 - mask)
    #         coords2 = coords2_nonzero * mask + coords2_reg * (1 - mask)
    #     else:
    #         # 对coord_shape均匀采样，乘2保证正数，减1进行归一化到[0,2]
    #         coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
    #         coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1

    #     # 原始输入
    #     # 用于对输入的张量orig_feats进行采样，采样的坐标值由参数coords给出
    #     feats = sample(orig_feats, coords1)
    #     # 采样
    #     code = sample(orig_code, coords1)

    #     # KNN输入
    #     # 采样
    #     feats_pos = sample(orig_feats_pos, coords2)
    #     # 采样
    #     code_pos = sample(orig_code_pos, coords2)

    #     # 自身的损失，自身的相关性
    #     pos_intra_loss, pos_intra_cd = self.helper(
    #         feats, feats, code, code, self.cfg.pos_intra_shift)
    #     # knn的损失，knn的相关性
    #     pos_inter_loss, pos_inter_cd = self.helper(
    #         feats, feats_pos, code, code_pos, self.cfg.pos_inter_shift)

    #     neg_losses = []
    #     neg_cds = []
    #     # 遍历负样本，5个
    #     for i in range(self.cfg.neg_samples):
    #         # 将orig_feats都置换成不同的元素
    #         perm_neg = super_perm(orig_feats.shape[0], orig_feats.device)
    #         # 采样负样本特征
    #         feats_neg = sample(orig_feats[perm_neg], coords2)
    #         # 采样负样本编码
    #         code_neg = sample(orig_code[perm_neg], coords2)
    #         # 负样本损失，负样本相关性
    #         neg_inter_loss, neg_inter_cd = self.helper(
    #             feats, feats_neg, code, code_neg, self.cfg.neg_inter_shift)
    #         neg_losses.append(neg_inter_loss)
    #         neg_cds.append(neg_inter_cd)
    #     neg_inter_loss = torch.cat(neg_losses, axis=0)
    #     neg_inter_cd = torch.cat(neg_cds, axis=0)

    #     return (pos_intra_loss.mean(),
    #             pos_intra_cd,
    #             pos_inter_loss.mean(),
    #             pos_inter_cd,
    #             neg_inter_loss,
    #             neg_inter_cd)

    # def forward(self,
    #             orig_feats: torch.Tensor, orig_feats_pos: torch.Tensor,
    #             aug_feats: torch.Tensor, aug_feats_pos:torch.Tensor):
    #     """
    #         feats:[batch_size, C, H, W]
    #     """
    #     coord_shape = [orig_feats.shape[0], self.cfg.feature_samples, self.cfg.feature_samples, 2]  # [16, 11, 11, 2]

    #     # 最终生成的 coords1 和 coords2 张量是随机坐标，其中每个坐标点都由两个浮点数值表示，位于 [-1, 1) 区间内。
    #     # 这些坐标将用于在原始特征张量中进行采样，从而获取对应的样本特征用于计算对比损失。
    #     coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1 # [16, 11, 11, 2]
    #     coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
		
	# 	# 采用视角coords1采样knn分支
    #     # 采样img
    #     feats = sample(orig_feats, coords1) # [batch_size, C, 11, 11]
    #     # 采样img_pos
    #     feats_pos = sample(orig_feats_pos, coords1)
		
	# 	# 使用coords2采样aug分支
    #     # 采样aug_feats
    #     feats_aug = sample(aug_feats, coords2)

    #     # 采样aug_feats_pos
    #     feats_aug_pos = sample(aug_feats_pos, coords2)

    #     # 生成负样本列表
    #     neg_list_knn = []
    #     neg_list_aug = []
    #     for i in range(self.cfg.neg_samples):   # neg_samples=5
    #         # 生成一个具有原始特征张量 orig_feats 批次大小的随机排列张量 perm_neg
    #         perm_neg = super_perm(orig_feats.shape[0], orig_feats.device)
    #         # 根据分支不同采用不同的coords采样策略
    #         feats_neg_1 = sample(orig_feats[perm_neg], coords2)
    #         feats_neg_2 = sample(orig_feats[perm_neg], coords1)
    #         # 新增，创建负样本列表neg_list
    #         neg_list_knn.append(feats_neg_1)
    #         neg_list_aug.append(feats_neg_2)

    #     loss_knn = self.helper_SegNCE(feats, feats_pos, neg_list_knn)
    #     loss_aug = self.helper_SegNCE(feats_aug, feats_aug_pos, neg_list_aug)
    #     losses = loss_knn + loss_aug

    #     return (losses, loss_knn, loss_aug)
    
    def forward(self,
            feats: torch.Tensor, feats_pos: torch.Tensor):
        """
            feats:[batch_size, C, H, W]
        """
        coord_shape = [feats.shape[0], self.cfg.feature_samples, self.cfg.feature_samples, 2]  # [16, 11, 11, 2]

        # 最终生成的 coords1 和 coords2 张量是随机坐标，其中每个坐标点都由两个浮点数值表示，位于 [-1, 1) 区间内。
        # 这些坐标将用于在原始特征张量中进行采样，从而获取对应的样本特征用于计算对比损失。
        coords = torch.rand(coord_shape, device=feats.device) * 2 - 1 # [16, 11, 11, 2]
		
		# 采用视角coords1采样knn分支
        # 采样img
        feats = sample(feats, coords) # [batch_size, C, 11, 11]
        # 采样img_pos
        feats_pos = sample(feats_pos, coords)

        # 生成负样本列表
        neg_list_knn = []
        for i in range(self.cfg.neg_samples):   # neg_samples=5
            # 生成一个具有原始特征张量 orig_feats 批次大小的随机排列张量 perm_neg
            perm_neg = super_perm(feats.shape[0], feats.device)
            # 根据分支不同采用不同的coords采样策略
            feats_neg_1 = sample(feats[perm_neg], coords)
            # 新增，创建负样本列表neg_list
            neg_list_knn.append(feats_neg_1)

        return self.helper_SegNCE(feats, feats_pos, neg_list_knn)

class Decoder(nn.Module):
    def __init__(self, code_channels, feat_channels):
        super().__init__()
        self.linear = torch.nn.Conv2d(code_channels, feat_channels, (1, 1))
        self.nonlinear = torch.nn.Sequential(
            torch.nn.Conv2d(code_channels, code_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(code_channels, code_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(code_channels, feat_channels, (1, 1)))

    def forward(self, x):
        return self.linear(x) + self.nonlinear(x)


class NetWithActivations(torch.nn.Module):
    def __init__(self, model, layer_nums):
        super(NetWithActivations, self).__init__()
        self.layers = nn.ModuleList(model.children())
        self.layer_nums = []
        for l in layer_nums:
            if l < 0:
                self.layer_nums.append(len(self.layers) + l)
            else:
                self.layer_nums.append(l)
        self.layer_nums = set(sorted(self.layer_nums))

    def forward(self, x):
        activations = {}
        for ln, l in enumerate(self.layers):
            x = l(x)
            if ln in self.layer_nums:
                activations[ln] = x
        return activations


class ContrastiveCRFLoss(nn.Module):

    def __init__(self, n_samples, alpha, beta, gamma, w1, w2, shift):
        super(ContrastiveCRFLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.w1 = w1
        self.w2 = w2
        self.n_samples = n_samples
        self.shift = shift

    def forward(self, guidance, clusters):
        device = clusters.device
        assert (guidance.shape[0] == clusters.shape[0])
        assert (guidance.shape[2:] == clusters.shape[2:])
        h = guidance.shape[2]
        w = guidance.shape[3]

        coords = torch.cat([
            torch.randint(0, h, size=[1, self.n_samples], device=device),
            torch.randint(0, w, size=[1, self.n_samples], device=device)], 0)

        selected_guidance = guidance[:, :, coords[0, :], coords[1, :]]
        coord_diff = (coords.unsqueeze(-1) - coords.unsqueeze(1)).square().sum(0).unsqueeze(0)
        guidance_diff = (selected_guidance.unsqueeze(-1) - selected_guidance.unsqueeze(2)).square().sum(1)

        sim_kernel = self.w1 * torch.exp(- coord_diff / (2 * self.alpha) - guidance_diff / (2 * self.beta)) + \
                     self.w2 * torch.exp(- coord_diff / (2 * self.gamma)) - self.shift

        selected_clusters = clusters[:, :, coords[0, :], coords[1, :]]
        cluster_sims = torch.einsum("nka,nkb->nab", selected_clusters, selected_clusters)
        return -(cluster_sims * sim_kernel)
