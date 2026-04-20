import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import save_image
from torchmetrics.functional import structural_similarity_index_measure

# 导入自定义模块
from model import RGB
from losses import CombinedLoss
from dataloader import create_dataloaders

# 全局配置项
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_PIXEL = 1.0  # 图像像素最大值


# 计算峰值信噪比(PSNR)
def compute_psnr(pred_img: torch.Tensor, gt_img: torch.Tensor, align_mean: bool = True) -> float:
    if align_mean:
        # 转换为灰度图并计算均值
        pred_gray = pred_img.mean(dim=1)
        gt_gray = gt_img.mean(dim=1)

        pred_mean = pred_gray.mean()
        gt_mean = gt_gray.mean()

        # 均值对齐并裁剪像素范围
        pred_img = torch.clamp(pred_img * (gt_mean / pred_mean), 0, MAX_PIXEL)

    # 计算均方误差
    mse_loss = F.mse_loss(pred_img, gt_img, reduction='mean')
    if mse_loss == 0:
        return float('inf')

    # 计算PSNR
    psnr_score = 20 * torch.log10(MAX_PIXEL / torch.sqrt(mse_loss))
    return psnr_score.item()


# 计算结构相似性指数(SSIM)
def compute_ssim(pred_img: torch.Tensor, gt_img: torch.Tensor, align_mean: bool = True) -> float:
    if align_mean:
        # 转换为灰度图并保留维度
        pred_gray = pred_img.mean(dim=1, keepdim=True)
        gt_gray = gt_img.mean(dim=1, keepdim=True)

        pred_mean = pred_gray.mean()
        gt_mean = gt_gray.mean()

        # 均值对齐并裁剪像素范围
        pred_img = torch.clamp(pred_img * (gt_mean / pred_mean), 0, MAX_PIXEL)

    # 计算SSIM
    ssim_score = structural_similarity_index_measure(
        pred_img, gt_img, data_range=MAX_PIXEL
    )
    return ssim_score.item()


@torch.no_grad()
def evaluate_model(model: nn.Module, val_loader: torch.utils.data.DataLoader) -> tuple[float, float]:
    model.eval()
    psnr_sum = 0.0
    ssim_sum = 0.0
    data_loader_len = len(val_loader)

    for low_light_img, normal_light_img in val_loader:
        # 数据移至指定设备
        low_img = low_light_img.to(DEVICE)
        gt_img = normal_light_img.to(DEVICE)

        # 模型推理
        pred_img = model(low_img)

        # 累加评估指标
        psnr_sum += compute_psnr(pred_img, gt_img)
        ssim_sum += compute_ssim(pred_img, gt_img)

    # 计算平均值
    avg_psnr = psnr_sum / data_loader_len
    avg_ssim = ssim_sum / data_loader_len
    return avg_psnr, avg_ssim


# 配置数据加载器
def setup_dataloaders() -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    # 统一路径：通过环境变量适配不同机器
    base_dir = os.environ.get('DATA_BASE_DIR', 'data')

    # 数据路径配置
    train_low_path = f'{base_dir}/LOLv1/Train/input'
    train_high_path = f'{base_dir}/LOLv1/Train/target'
    test_low_path = f'{base_dir}/LOLv1/Test/input'
    test_high_path = f'{base_dir}/LOLv1/Test/target'

    # 备选数据集路径
    # train_low_path = f'{base_dir}/five/train/input'
    # train_high_path = f'{base_dir}/five/train/gt'
    # test_low_path = f'{base_dir}/five/test/input'
    # test_high_path = f'{base_dir}/five/test/gt'

    # train_low_path = f'{base_dir}/SICE/train/low'
    # train_high_path = f'{base_dir}/SICE/train/gt'
    # test_low_path = f'{base_dir}/SICE/test/low'
    # test_high_path = f'{base_dir}/SICE/test/gt'

    # train_low_path = f'{base_dir}/SICE/train/over'
    # train_high_path = f'{base_dir}/SICE/train/gt'
    # test_low_path = f'{base_dir}/SICE/test/over'
    # test_high_path = f'{base_dir}/SICE/test/gt'

    # 校验路径是否存在
    for path in [train_low_path, train_high_path, test_low_path, test_high_path]:
        if not os.path.exists(path):
            print(f"警告：路径不存在 {path}，请检查DATA_BASE_DIR环境变量")

    # 创建数据加载器
    train_loader, test_loader = create_dataloaders(
        train_low=train_low_path,
        train_high=train_high_path,
        test_low=test_low_path,
        test_high=test_high_path,
        crop_size=256,
        batch_size=1
    )

    print(f'训练集批次数量: {len(train_loader)}; 测试集批次数量: {len(test_loader)}')
    return train_loader, test_loader


# 训练
def train_model():
    # 超参数配置
    lr = 2e-4
    epochs = 1000
    print(f'学习率: {lr}; 训练轮数: {epochs}')

    # 初始化数据加载器
    train_loader, test_loader = setup_dataloaders()

    # 初始化模型
    model = RGB().to(DEVICE)
    # 多卡训练
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f'使用 {torch.cuda.device_count()} 张GPU训练')

    # 损失函数、优化器、学习率调度器
    loss_fn = CombinedLoss(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()

    # 最佳模型跟踪
    best_psnr_score = 0.0
    print('开始模型训练...')

    # 训练循环
    for epoch_idx in range(epochs):
        model.train()
        epoch_loss = 0.0

        # 批次训练
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 数据移至设备
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播 + 混合精度训练
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            # 反向传播 + 优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 累加损失
            epoch_loss += loss.item()

        # 验证模型
        val_psnr, val_ssim = evaluate_model(model, test_loader)
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(
            f'第 {epoch_idx + 1}/{epochs} 轮 | 损失: {avg_epoch_loss:.4f} | PSNR: {val_psnr:.4f} | SSIM: {val_ssim:.4f}')

        # 更新学习率
        scheduler.step()

        # 保存最佳模型
        if val_psnr > best_psnr_score:
            best_psnr_score = val_psnr
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'保存最佳模型 - PSNR: {best_psnr_score:.4f}')


if __name__ == '__main__':
    # 启动训练
    train_model()