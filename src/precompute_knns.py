from data import ContrastiveSegDataset
from modules import *
import os
from os.path import join
import hydra
import numpy as np
import torch.multiprocessing
import torch.multiprocessing
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.seed import seed_everything
from tqdm import tqdm

import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


# 获取图片特征
def get_feats(model, loader):
    all_feats = []
    for pack in tqdm(loader):
        # 获取图片
        img = pack["img"]
        # 输入模型，获取特征
        feats = F.normalize(model.forward(img.cuda()).mean([2, 3]), dim=1)
        # 记录特征
        all_feats.append(feats.to("cpu", non_blocking=True))
    return torch.cat(all_feats, dim=0).contiguous()


def my_prep_for_plot(img, rescale=True, resize=None):
    if resize is not None:
        img = F.interpolate(img.unsqueeze(0), resize, mode="bilinear")

    plot_img = unnorm(img).cpu().permute(1, 2, 0)
    if rescale:
        plot_img = (plot_img - plot_img.min()) / (plot_img.max() - plot_img.min())
    return plot_img

def write_img(dataset, ind, writer, writer_step, global_step):
    fig, ax = plt.subplots(writer_step, 6, figsize=(12, 12))
    for i in range(ind.shape[0]):
        img = dataset[ind[i][0]]['img']
        img = torch.clone(img)
        ax[i, 0].imshow(my_prep_for_plot(img))
        for j in range(5):
            img_knn = dataset[ind[i][j + 1]]['img']
            img_knn = torch.clone(img_knn)
            ax[i, j + 1].imshow(my_prep_for_plot(img_knn))
    remove_axes(ax)
    plt.tight_layout()
    writer.add_figure("plot_labels", fig, global_step)
    plt.close(fig)
    writer.flush()


@hydra.main(config_path="configs", config_name="train_config.yml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # 数据集路径
    pytorch_data_dir = cfg.pytorch_data_dir
    # 数据集输出路径
    data_dir = join(cfg.output_root, "data")
    # 数据集输出日志
    log_dir = join(cfg.output_root, "logs")

    # 前缀
    prefix = "{}/{}_{}".format("visualization_knn", cfg.dataset_name, cfg.experiment_name)
    name = '{}_date_{}'.format(prefix, datetime.now().strftime('%b%d_%H-%M-%S'))
    # cfg.full_name = prefix

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(join(pytorch_data_dir, "nns"), exist_ok=True)

    logdir = join(log_dir, name)
    print("=====================" + log_dir + "=====================")
    # 日志
    writer = SummaryWriter(logdir)

    # 随机种子
    seed_everything(seed=0)

    # 数据集模式
    image_sets = ["val", "train"]
    # 数据集名称
    # dataset_names = ["cocostuff27", "cityscapes", "potsdam"]
    dataset_names = ["cityscapes"]
    # 裁剪类型
    crop_types = ["five", None]

    # Uncomment these lines to run on custom datasets
    #dataset_names = ["directory"]
    #crop_types = [None]

    res = 256
    n_batches = 128

    if cfg.arch == "dino":
        # 加载vit模型
        from modules import DinoFeaturizer, LambdaLayer
        no_ap_model = torch.nn.Sequential(
            DinoFeaturizer(20, cfg),  # dim doesent matter
            LambdaLayer(lambda p: p[0]),
        ).cuda()
    elif cfg.arch == "swin":
        # 加载vit模型
        from modules import SwinFeaturizer, LambdaLayer
        no_ap_model = torch.nn.Sequential(
            SwinFeaturizer(20, cfg),  # dim doesent matter
            LambdaLayer(lambda p: p[0]),
        ).cuda()
    else:
        cut_model = load_model(cfg.model_type, join(cfg.output_root, "data")).cuda()
        no_ap_model = nn.Sequential(*list(cut_model.children())[:-1]).cuda()
    par_model = torch.nn.DataParallel(no_ap_model)


    global_step = 0
    # 遍历裁剪类型
    for crop_type in crop_types:
        # 遍历train, val, test
        for image_set in image_sets:
            # 遍历数据集
            for dataset_name in dataset_names:
                # 数据集名称
                nice_dataset_name = cfg.dir_dataset_name if dataset_name == "directory" else dataset_name

                # 特征缓存文件
                feature_cache_file = join(pytorch_data_dir, "nns", "nns_{}_{}_{}_{}_{}.npz".format(
                    cfg.model_type, nice_dataset_name, image_set, crop_type, res))

                if not os.path.exists(feature_cache_file):
                    print("{} not found, computing".format(feature_cache_file))
                    # 数据集，并设置数据增强
                    dataset = ContrastiveSegDataset(
                        pytorch_data_dir=pytorch_data_dir,
                        dataset_name=dataset_name,
                        crop_type=crop_type,
                        image_set=image_set,
                        transform=get_transform(res, False, "center"),
                        target_transform=get_transform(res, True, "center"),
                        cfg=cfg,
                    )

                    # 加载数据集
                    loader = DataLoader(dataset, 256, shuffle=False, num_workers=cfg.num_workers, pin_memory=False)

                    with torch.no_grad():
                        # 获取图片特征
                        normed_feats = get_feats(par_model, loader)
                        
                        all_nns = []
                        # 每步只处理step张特征
                        step = normed_feats.shape[0] // n_batches
                        print("number of dataset :", normed_feats.shape[0])
                        print("batch size:", n_batches)
                        print("step:", step)

                        for i in tqdm(range(0, normed_feats.shape[0], step)):
                            # 清空缓存
                            torch.cuda.empty_cache()
                            # 获取step张特征
                            batch_feats = normed_feats[i:i + step, :]
                            # 将step张特征和batch内的特征计算相似度
                            pairwise_sims = torch.einsum("nf,mf->nm", batch_feats, normed_feats)
                            # 记录特征矩阵相似度
                            top_index = torch.topk(pairwise_sims, 30)[1]
                            write_step = 5
                            for j in range(0, top_index.shape[0], write_step):
                                write_top_index = top_index[j:j + write_step]
                                write_img(dataset, write_top_index, writer, write_step, global_step)
                                global_step += 1
                            all_nns.append(top_index)
                            del pairwise_sims
                        # 拼接一个batch内的特征
                        nearest_neighbors = torch.cat(all_nns, dim=0)

                        # 保存文件
                        np.savez_compressed(feature_cache_file, nns=nearest_neighbors.numpy())
                        print("Saved NNs", cfg.model_type, nice_dataset_name, image_set)


if __name__ == "__main__":
    prep_args()
    my_app()
