import sys

from tqdm import tqdm
import torch

from .distributed_utils import reduce_value, is_main_process, warmup_lr_scheduler


def train_one_epoch(model, optimizer, data_loader, device, epoch, use_amp=False, warmup=True):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    enable_amp = use_amp and "cuda" in device.type
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    sample_num = 0
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        with torch.cuda.amp.autocast(enabled=enable_amp):
            pred = model(images.to(device))
            loss = 0
            loss_list = []
            accu_num_list = []
            for i in range(len(pred)):
                # loss += loss_function(pred[i], labels.to(device))
                # pred_classes = torch.max(pred[i], dim=1)[1]
                # accu_num += torch.eq(pred_classes, labels.to(device)).sum()

                layer_loss = loss_function(pred[i], labels.to(device))
                loss += layer_loss
                loss_list.append(layer_loss)
                pred_classes = torch.max(pred[i], dim=1)[1]
                layer_acc_num = torch.eq(pred_classes, labels.to(device)).sum()
                accu_num += layer_acc_num
                accu_num_list.append(layer_acc_num)
            
            # loss = loss_function(pred, labels.to(device))

            # pred_classes = torch.max(pred, dim=1)[1]
            # accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss = reduce_value(loss, average=True)
        accu_loss += loss.detach()


        # 在进程0中打印平均loss
        if is_main_process():
            info = "[epoch {}] loss: {:.3f}, train_acc: {:.3f}, lr: {:.5f}".format(
                epoch,
                accu_loss.item() / (step + 1),
                accu_num.item() / (sample_num * 4),
                optimizer.param_groups[0]["lr"])
            loss_info = " loss_0: {:.3f}, loss_1: {:.3f}, loss_2: {:.3f}, loss_3: {:.3f},".format(
                loss_list[0], loss_list[1], loss_list[2], loss_list[3])
            acc_info = " acc_0: {:.3f}, acc_1: {:.3f}, acc_2: {:.3f}, acc_3: {:.3f},".format(
                accu_num_list[0] / (sample_num + 1), 
                accu_num_list[1] / (sample_num + 1), 
                accu_num_list[2] / (sample_num + 1), 
                accu_num_list[3] / (sample_num + 1))
            data_loader.desc = info + "\t" + loss_info + "\t" + acc_info

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if lr_scheduler is not None:  # 如果使用warmup训练，逐渐调整学习率
            lr_scheduler.step()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    for i in range(len(accu_num_list)):
            accu_num_list[i] = accu_num_list[i] / (sample_num + 1)

    return accu_loss.item() / (step + 1), loss_list, accu_num_list


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 验证集样本个数
    num_samples = len(data_loader.dataset)

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        for i in range(len(pred)):
            pred_max = torch.max(pred[i], dim=1)[1]
            sum_num += torch.eq(pred_max, labels.to(device)).sum()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    sum_num = reduce_value(sum_num, average=False)
    acc = sum_num.item() / (num_samples * 4)

    return acc






