from utils import *
from modules import *
from data import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
import torch.multiprocessing
import seaborn as sns
from pytorch_lightning.callbacks import ModelCheckpoint
import sys

torch.multiprocessing.set_sharing_strategy('file_system')

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_class_labels(dataset_name):
    if dataset_name.startswith("cityscapes"):
        return [
            'road', 'sidewalk', 'parking', 'rail track', 'building',
            'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
            'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'caravan', 'trailer', 'train',
            'motorcycle', 'bicycle']
    elif dataset_name == "cocostuff27":
        return [
            "electronic", "appliance", "food", "furniture", "indoor",
            "kitchen", "accessory", "animal", "outdoor", "person",
            "sports", "vehicle", "ceiling", "floor", "food",
            "furniture", "rawmaterial", "textile", "wall", "window",
            "building", "ground", "plant", "sky", "solid",
            "structural", "water"]
    elif dataset_name == "voc":
        return [
            'background',
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    elif dataset_name == "potsdam":
        return [
            'roads and cars',
            'buildings and clutter',
            'trees and vegetation']
    else:
        raise ValueError("Unknown Dataset {}".format(dataset_name))


class LitUnsupervisedSegmenter(pl.LightningModule):
    def __init__(self, n_classes, cfg):
        super().__init__()
        self.cfg = cfg
        # 多少类
        self.n_classes = n_classes

        # 
        if not cfg.continuous:
            dim = n_classes
        else:
            dim = cfg.dim

        # 数据集路径
        data_dir = join(cfg.output_root, "data")
        
        # 特征提取融合层
        if cfg.arch == "feature-pyramid":
            cut_model = load_model(cfg.model_type, data_dir).cuda()
            self.net = FeaturePyramidNet(cfg.granularity, cut_model, dim, cfg.continuous)
        elif cfg.arch == "dino":
            self.net = DinoFeaturizer(dim, cfg)
        elif cfg.arch == "swin":
            self.net = SwinFeaturizer(dim, cfg)
        elif cfg.arch == "resnet":
            self.net = ResnetFeaturizer(dim, cfg)
        else:
            raise ValueError("Unknown arch {}".format(cfg.arch))

        # 聚类概率
        self.train_cluster_probe = ClusterLookup(dim, n_classes)

        # 聚类概率
        self.cluster_probe = ClusterLookup(dim, n_classes + cfg.extra_clusters)
        # 线性概率
        self.linear_probe = nn.Conv2d(dim, n_classes, (1, 1))

        # decoder
        self.decoder = nn.Conv2d(dim, self.net.n_feats, (1, 1))

        # 
        self.cluster_metrics = UnsupervisedMetrics(
            "test/cluster/", n_classes, cfg.extra_clusters, True)
        self.linear_metrics = UnsupervisedMetrics(
            "test/linear/", n_classes, 0, False)

        # 
        self.test_cluster_metrics = UnsupervisedMetrics(
            "final/cluster/", n_classes, cfg.extra_clusters, True)
        self.test_linear_metrics = UnsupervisedMetrics(
            "final/linear/", n_classes, 0, False)

        # 损失函数
        self.linear_probe_loss_fn = torch.nn.CrossEntropyLoss()
        # CRF损失函数
        self.crf_loss_fn = ContrastiveCRFLoss(
            cfg.crf_samples, cfg.alpha, cfg.beta, cfg.gamma, cfg.w1, cfg.w2, cfg.shift)

        # 对比损失
        self.contrastive_corr_loss_fn = ContrastiveCorrelationLoss(cfg)
        for p in self.contrastive_corr_loss_fn.parameters():
            p.requires_grad = False

        self.automatic_optimization = False

        if self.cfg.dataset_name.startswith("cityscapes"):
            self.label_cmap = create_cityscapes_colormap()
        else:
            self.label_cmap = create_pascal_label_colormap()

        self.val_steps = 0
        self.save_hyperparameters()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net(x)[1]

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        # 网络优化器，线性优化器，聚类优化器
        net_optim, linear_probe_optim, cluster_probe_optim = self.optimizers()

        # 梯度归零
        net_optim.zero_grad()
        linear_probe_optim.zero_grad()
        cluster_probe_optim.zero_grad()

        with torch.no_grad():
            ind = batch["ind"]
            # 输入
            img = batch["img"]
            # 输入增强
            img_aug = batch["img_aug"]
            # 裁剪增强
            coord_aug = batch["coord_aug"]
            # 图片位置
            img_pos = batch["img_pos"]
            img_aug_pos = batch["img_aug_pos"]
            # 标签
            label = batch["label"]
            # 标签位置
            label_pos = batch["label_pos"]

        # 特征，编码
        feats, code = self.net(img)
        # feats_aug, code_aug = self.net(img_aug)
        if self.cfg.correspondence_weight > 0:
            feats_pos, code_pos = self.net(img_pos)
            # feats_aug_pos, code_aug_pos = self.net(img_aug_pos)
            
        log_args = dict(sync_dist=False, rank_zero_only=True)

        # # 使用标签
        # if self.cfg.use_true_labels:
        #     signal = one_hot_feats(label + 1, self.n_classes + 1)
        #     signal_pos = one_hot_feats(label_pos + 1, self.n_classes + 1)
        # # 不使用标签
        # else:
        #     signal = feats
        #     signal_pos = feats_pos

        loss = 0

        # # 打印日志
        # should_log_hist = (self.cfg.hist_freq is not None) and \
        #                   (self.global_step % self.cfg.hist_freq == 0) and \
        #                   (self.global_step > 0)
        
        # # 真实标签，KNN标签
        # if self.cfg.use_salience:
        #     salience = batch["mask"].to(torch.float32).squeeze(1)
        #     salience_pos = batch["mask_pos"].to(torch.float32).squeeze(1)
        # else:
        #     salience = None
        #     salience_pos = None

        # 自身损失，自身相关性
        # knn损失，knn相关性
        # 负样本损失，负样本相关性
        if self.cfg.correspondence_weight > 0:
            # (
            #     pos_intra_loss, pos_intra_cd,
            #     pos_inter_loss, pos_inter_cd,
            #     neg_inter_loss, neg_inter_cd,
            # ) = self.contrastive_corr_loss_fn(
            #     signal, signal_pos,
            #     salience, salience_pos,
            #     code, code_pos,
            # )
            # (loss, loss_knn, loss_aug) = self.contrastive_corr_loss_fn(
            #     feats, feats_pos,
            #     feats_aug, feats_aug_pos
            # )

            loss_knn = self.contrastive_corr_loss_fn(
                feats, feats_pos
            )

            loss = loss_knn

            # loss_aug = self.contrastive_corr_loss_fn(
            #     feats_aug, feats_aug_pos
            # )

            # loss = loss_aug

            # loss = loss_knn + loss_aug


            # # 记录日志
            # if should_log_hist:
            #     self.logger.experiment.add_histogram("intra_cd", pos_intra_cd, self.global_step)
            #     self.logger.experiment.add_histogram("inter_cd", pos_inter_cd, self.global_step)
            #     self.logger.experiment.add_histogram("neg_cd", neg_inter_cd, self.global_step)
            # neg_inter_loss = neg_inter_loss.mean()
            # pos_intra_loss = pos_intra_loss.mean()
            # pos_inter_loss = pos_inter_loss.mean()
            # self.log('loss/pos_intra', pos_intra_loss, **log_args)
            # self.log('loss/pos_inter', pos_inter_loss, **log_args)
            # self.log('loss/neg_inter', neg_inter_loss, **log_args)
            # self.log('cd/pos_intra', pos_intra_cd.mean(), **log_args)
            # self.log('cd/pos_inter', pos_inter_cd.mean(), **log_args)
            # self.log('cd/neg_inter', neg_inter_cd.mean(), **log_args)

            # # 累计损失
            # loss += (self.cfg.pos_inter_weight * pos_inter_loss +
            #          self.cfg.pos_intra_weight * pos_intra_loss +
            #          self.cfg.neg_inter_weight * neg_inter_loss) * self.cfg.correspondence_weight

        # # 还原分支，默认为0
        # if self.cfg.rec_weight > 0:
        #     # 还原特征
        #     rec_feats = self.decoder(code)
        #     # 还原损失
        #     rec_loss = -(norm(rec_feats) * norm(feats)).sum(1).mean()
        #     self.log('loss/rec', rec_loss, **log_args)
        #     # 累计损失
        #     loss += self.cfg.rec_weight * rec_loss

        # # 数据增强分支，默认为0
        # if self.cfg.aug_alignment_weight > 0:
        #     # 数据增强编码
        #     orig_feats_aug, orig_code_aug = self.net(img_aug)
        #     # 放缩到原始输入编码大小
        #     downsampled_coord_aug = resize(
        #         coord_aug.permute(0, 3, 1, 2),
        #         orig_code_aug.shape[2]).permute(0, 2, 3, 1)
        #     # 这段代码计算了对齐增强后的特征图code和增强前的特征图orig_code_aug之间的相似度，
        #     # 该损失项的值越小，说明对齐增强的效果越好，即增强前后的特征图之间的相似度越高。
        #     aug_alignment = -torch.einsum(
        #         "bkhw,bkhw->bhw",
        #         norm(sample(code, downsampled_coord_aug)),
        #         norm(orig_code_aug)
        #     ).mean()
        #     self.log('loss/aug_alignment', aug_alignment, **log_args)
        #     # 记录损失
        #     loss += self.cfg.aug_alignment_weight * aug_alignment

        # # CRF，默认为0
        # if self.cfg.crf_weight > 0:
        #     # CRF损失
        #     crf = self.crf_loss_fn(
        #         resize(img, 56),
        #         norm(resize(code, 56))
        #     ).mean()
        #     self.log('loss/crf', crf, **log_args)
        #     # 记录损失
        #     loss += self.cfg.crf_weight * crf

        # 展平标签
        flat_label = label.reshape(-1)
        # 标签大于0，且小于类属
        mask = (flat_label >= 0) & (flat_label < self.n_classes)

        # detached_code = torch.clone(code.detach())
        detached_code = torch.clone(code.detach())

        # 计算线性输出
        linear_logits = self.linear_probe(detached_code)
        # 上采样
        linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
        # 转化维度
        linear_logits = linear_logits.permute(0, 2, 3, 1).reshape(-1, self.n_classes)
        # 计算线性损失
        linear_loss = self.linear_probe_loss_fn(linear_logits[mask], flat_label[mask]).mean()
        # 记录损失
        loss += linear_loss
        self.log('loss/linear', linear_loss, **log_args)

        # 聚类损失
        cluster_loss, cluster_probs = self.cluster_probe(detached_code, None)
        # 记录损失
        loss += cluster_loss
        self.log('loss/cluster', cluster_loss, **log_args)
        self.log('loss/total', loss, **log_args)

        # 反向传播
        self.manual_backward(loss)
        # 梯度下降
        net_optim.step()
        cluster_probe_optim.step()
        linear_probe_optim.step()

        # 重设探测步数
        if self.cfg.reset_probe_steps is not None and self.global_step == self.cfg.reset_probe_steps:
            print("RESETTING PROBES")
            self.linear_probe.reset_parameters()
            self.cluster_probe.reset_parameters()
            self.trainer.optimizers[1] = torch.optim.Adam(list(self.linear_probe.parameters()), lr=5e-3)
            self.trainer.optimizers[2] = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=5e-3)

        # 创建一个新的文件
        if self.global_step % 2000 == 0 and self.global_step > 0:
            print("RESETTING TFEVENT FILE")
            # Make a new tfevent file
            self.logger.experiment.close()
            self.logger.experiment._get_file_writer()

        return loss

    def on_train_start(self):
        tb_metrics = {
            **self.linear_metrics.compute(),
            **self.cluster_metrics.compute()
        }
        self.logger.log_hyperparams(self.cfg, tb_metrics)

    def validation_step(self, batch, batch_idx):
        # 获取输入
        img = batch["img"]
        # 获取标签
        label = batch["label"]
        self.net.eval().cuda()

        with torch.no_grad():
            # 获取编码
            feats, code = self.net(img)
            # 上采样
            code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)

            # 线性输出
            linear_preds = self.linear_probe(code)
            # 获取最大概率坐标
            linear_preds = linear_preds.argmax(1)
            self.linear_metrics.update(linear_preds, label)

            # 获取聚类输出
            cluster_loss, cluster_preds = self.cluster_probe(code, None)
            # 获取最大概率坐标
            cluster_preds = cluster_preds.argmax(1)
            self.cluster_metrics.update(cluster_preds, label)

            return {
                'img': img[:self.cfg.n_images].detach().cpu(),
                'linear_preds': linear_preds[:self.cfg.n_images].detach().cpu(),
                "cluster_preds": cluster_preds[:self.cfg.n_images].detach().cpu(),
                "label": label[:self.cfg.n_images].detach().cpu()}

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        with torch.no_grad():
            tb_metrics = {
                **self.linear_metrics.compute(),
                **self.cluster_metrics.compute(),
            }

            if self.trainer.is_global_zero and not self.cfg.submitting_to_aml:
                #output_num = 0
                output_num = random.randint(0, len(outputs) -1)
                output = {k: v.detach().cpu() for k, v in outputs[output_num].items()}

                fig, ax = plt.subplots(4, self.cfg.n_images, figsize=(self.cfg.n_images * 3, 4 * 3))
                for i in range(self.cfg.n_images):
                    ax[0, i].imshow(prep_for_plot(output["img"][i]))
                    ax[1, i].imshow(self.label_cmap[output["label"][i]])
                    ax[2, i].imshow(self.label_cmap[output["linear_preds"][i]])
                    ax[3, i].imshow(self.label_cmap[self.cluster_metrics.map_clusters(output["cluster_preds"][i])])
                ax[0, 0].set_ylabel("Image", fontsize=16)
                ax[1, 0].set_ylabel("Label", fontsize=16)
                ax[2, 0].set_ylabel("Linear Probe", fontsize=16)
                ax[3, 0].set_ylabel("Cluster Probe", fontsize=16)
                remove_axes(ax)
                plt.tight_layout()
                add_plot(self.logger.experiment, "plot_labels", self.global_step)

                if self.cfg.has_labels:
                    fig = plt.figure(figsize=(13, 10))
                    ax = fig.gca()
                    hist = self.cluster_metrics.histogram.detach().cpu().to(torch.float32)
                    hist /= torch.clamp_min(hist.sum(dim=0, keepdim=True), 1)
                    sns.heatmap(hist.t(), annot=False, fmt='g', ax=ax, cmap="Blues")
                    ax.set_xlabel('Predicted labels')
                    ax.set_ylabel('True labels')
                    names = get_class_labels(self.cfg.dataset_name)
                    if self.cfg.extra_clusters:
                        names = names + ["Extra"]
                    ax.set_xticks(np.arange(0, len(names)) + .5)
                    ax.set_yticks(np.arange(0, len(names)) + .5)
                    ax.xaxis.tick_top()
                    ax.xaxis.set_ticklabels(names, fontsize=14)
                    ax.yaxis.set_ticklabels(names, fontsize=14)
                    colors = [self.label_cmap[i] / 255.0 for i in range(len(names))]
                    [t.set_color(colors[i]) for i, t in enumerate(ax.xaxis.get_ticklabels())]
                    [t.set_color(colors[i]) for i, t in enumerate(ax.yaxis.get_ticklabels())]
                    # ax.yaxis.get_ticklabels()[-1].set_color(self.label_cmap[0] / 255.0)
                    # ax.xaxis.get_ticklabels()[-1].set_color(self.label_cmap[0] / 255.0)
                    plt.xticks(rotation=90)
                    plt.yticks(rotation=0)
                    ax.vlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_xlim())
                    ax.hlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_ylim())
                    plt.tight_layout()
                    add_plot(self.logger.experiment, "conf_matrix", self.global_step)

                    all_bars = torch.cat([
                        self.cluster_metrics.histogram.sum(0).cpu(),
                        self.cluster_metrics.histogram.sum(1).cpu()
                    ], axis=0)
                    ymin = max(all_bars.min() * .8, 1)
                    ymax = all_bars.max() * 1.2

                    fig, ax = plt.subplots(1, 2, figsize=(2 * 5, 1 * 4))
                    ax[0].bar(range(self.n_classes + self.cfg.extra_clusters),
                              self.cluster_metrics.histogram.sum(0).cpu(),
                              tick_label=names,
                              color=colors)
                    ax[0].set_ylim(ymin, ymax)
                    ax[0].set_title("Label Frequency")
                    ax[0].set_yscale('log')
                    ax[0].tick_params(axis='x', labelrotation=90)

                    ax[1].bar(range(self.n_classes + self.cfg.extra_clusters),
                              self.cluster_metrics.histogram.sum(1).cpu(),
                              tick_label=names,
                              color=colors)
                    ax[1].set_ylim(ymin, ymax)
                    ax[1].set_title("Cluster Frequency")
                    ax[1].set_yscale('log')
                    ax[1].tick_params(axis='x', labelrotation=90)

                    plt.tight_layout()
                    add_plot(self.logger.experiment, "label frequency", self.global_step)

            if self.global_step > 2:
                self.log_dict(tb_metrics)

                if self.trainer.is_global_zero and self.cfg.azureml_logging:
                    from azureml.core.run import Run
                    run_logger = Run.get_context()
                    for metric, value in tb_metrics.items():
                        run_logger.log(metric, value)

            self.linear_metrics.reset()
            self.cluster_metrics.reset()

    def configure_optimizers(self):
        main_params = list(self.net.parameters())

        if self.cfg.rec_weight > 0:
            main_params.extend(self.decoder.parameters())

        net_optim = torch.optim.Adam(main_params, lr=self.cfg.lr)
        linear_probe_optim = torch.optim.Adam(list(self.linear_probe.parameters()), lr=5e-3)
        cluster_probe_optim = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=5e-3)

        return net_optim, linear_probe_optim, cluster_probe_optim


@hydra.main(config_path="configs", config_name="train_config.yml")
def my_app(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    # 数据集文件
    pytorch_data_dir = cfg.pytorch_data_dir
    # 输出路径
    data_dir = join(cfg.output_root, "data")
    # 输出日志
    log_dir = join(cfg.output_root, "logs")
    # 输出权重文件                                                                                                                                                                                                                                                      
    checkpoint_dir = join(cfg.output_root, "checkpoints")

    # 前缀
    prefix = "{}/{}_{}".format(cfg.log_dir, cfg.dataset_name, cfg.experiment_name)
    name = '{}_date_{}'.format(prefix, datetime.now().strftime('%b%d_%H-%M-%S'))
    cfg.full_name = prefix

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 随机种子
    seed_everything(seed=0)

    print(data_dir)
    print(cfg.output_root)

    # 几何增强
    geometric_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(size=cfg.res, scale=(0.8, 1.0))
    ])
    # 颜色增强
    photometric_transforms = T.Compose([
        T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
        T.RandomGrayscale(.2),
        T.RandomApply([T.GaussianBlur((5, 5))])
    ])

    sys.stdout.flush()

    # 训练集
    train_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=cfg.crop_type,
        image_set="train",
        transform=get_transform(cfg.input_res, False, cfg.loader_crop_type),
        target_transform=get_transform(cfg.input_res, True, cfg.loader_crop_type),
        cfg=cfg,
        aug_geometric_transform=geometric_transforms,
        aug_photometric_transform=photometric_transforms,
        num_neighbors=cfg.num_neighbors,
        mask=True,
        pos_images=True,
        pos_labels=True
    )

    if cfg.dataset_name == "voc":
        val_loader_crop = None
    else:
        val_loader_crop = "center"

    # 验证集
    val_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=None,
        image_set="val",
        transform=get_transform(cfg.input_res, False, val_loader_crop),
        target_transform=get_transform(cfg.input_res, True, val_loader_crop),
        mask=True,
        cfg=cfg,
    )

    #val_dataset = MaterializedDataset(val_dataset)
    # 加载训练集
    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    # batch size
    if cfg.submitting_to_aml:
        val_batch_size = 16
    else:
        val_batch_size = cfg.batch_size

    # 加载验证集
    val_loader = DataLoader(val_dataset, val_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = LitUnsupervisedSegmenter(train_dataset.n_classes, cfg)

    # 日志
    tb_logger = TensorBoardLogger(
        join(log_dir, name),
        default_hp_metric=False
    )

    if cfg.submitting_to_aml:
        gpu_args = dict(gpus=1, val_check_interval=250)

        if gpu_args["val_check_interval"] > len(train_loader):
            gpu_args.pop("val_check_interval")

    else:
        gpu_args = dict(gpus=-1, accelerator='ddp', val_check_interval=cfg.val_freq)
        # gpu_args = dict(gpus=1, accelerator='ddp', val_check_interval=cfg.val_freq)

        if gpu_args["val_check_interval"] > len(train_loader) // 4:
            gpu_args.pop("val_check_interval")

    trainer = Trainer(
        log_every_n_steps=cfg.scalar_log_freq,
        logger=tb_logger,
        # max_steps=cfg.max_steps,
        max_epochs=cfg.max_epochs,
        callbacks=[
            ModelCheckpoint(
                dirpath=join(checkpoint_dir, name),
                every_n_train_steps=400,
                save_top_k=2,
                monitor="test/cluster/mIoU",
                mode="max",
            )
        ],
        **gpu_args
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    prep_args()
    my_app()
