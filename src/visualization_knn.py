from utils import *
from modules import *
from data import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
import torch.multiprocessing
import sys
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

torch.multiprocessing.set_sharing_strategy('file_system')


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
    prefix = "{}/{}_{}".format("visualization_knn", cfg.dataset_name, cfg.experiment_name)
    name = '{}_date_{}'.format(prefix, datetime.now().strftime('%b%d_%H-%M-%S'))
    cfg.full_name = prefix

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 随机种子
    seed_everything(seed=0)

    print(data_dir)
    print(cfg.output_root)

    # 日志
    # tb_logger = TensorBoardLogger(
    #     join(log_dir, name),
    #     default_hp_metric=False
    # )

    # 指定 TensorBoard 日志路径
    logdir = join(log_dir, name)
    print("=====================" + log_dir + "=====================")
    # 日志
    writer = SummaryWriter(logdir)

    

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

    # 验证集
    val_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=cfg.crop_type,
        image_set="val",
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

    train_batch_size = 4

    # 加载训练集
    train_loader = DataLoader(train_dataset, train_batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    step = 0
    for pack in tqdm(train_loader):
        # 获取图片
        img = pack["img"]
        img_nns = pack["img_pos"]
        fig, ax = plt.subplots(img.shape[0], 2, figsize=(8, 8))
        for i in range(img.shape[0]):
            ax[i, 0].imshow(prep_for_plot(img[i]))
            ax[i, 1].imshow(prep_for_plot(img_nns[i]))
        remove_axes(ax)
        plt.tight_layout()
        # 将图形对象写入 TensorBoard
        writer.add_figure('train_plot_labels', fig, global_step=step)
        plt.close(fig)
        writer.flush()
        step += 1

    val_batch_size = 4

    # 加载验证集
    val_loader = DataLoader(val_dataset, val_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    step = 0
    for pack in tqdm(val_loader):
        # 获取图片
        img = pack["img"]
        img_nns = pack["img_pos"]
        fig, ax = plt.subplots(img.shape[0], 2, figsize=(8, 8))
        for i in range(img.shape[0]):
            ax[i, 0].imshow(prep_for_plot(img[i]))
            ax[i, 1].imshow(prep_for_plot(img_nns[i]))
        remove_axes(ax)
        plt.tight_layout()
        # 将图形对象写入 TensorBoard
        writer.add_figure('val_plot_labels', fig, global_step=step)
        plt.close(fig)
        writer.flush()
        step += 1

    writer.close()

    # batch size
    if cfg.submitting_to_aml:
        val_batch_size = 16
    else:
        val_batch_size = cfg.batch_size

    # 加载验证集
    val_loader = DataLoader(val_dataset, val_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    




if __name__ == "__main__":
    my_app()