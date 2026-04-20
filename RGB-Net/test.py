import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure
from model import RGB
from dataloader import create_dataloaders
import os
import numpy as np
from torchvision.utils import save_image
from lpips import LPIPS  # 导入LPIPS相似度计算模块


# 计算峰值信噪比(PSNR)
def compute_psnr(pred_img, gt_img, max_pixel=1.0, align_mean=True):
    if align_mean:
        # 转换为灰度图并计算均值
        pred_gray = pred_img.mean(dim=1)
        gt_gray = gt_img.mean(dim=1)

        mean_pred = pred_gray.mean()
        mean_gt = gt_gray.mean()
        # 均值对齐并限制像素范围
        pred_img = torch.clamp(pred_img * (mean_gt / mean_pred), 0, 1)

    # 计算均方误差
    mse_error = F.mse_loss(pred_img, gt_img, reduction='mean')
    if mse_error == 0:
        return float('inf')
    # 计算PSNR
    psnr_score = 20 * torch.log10(max_pixel / torch.sqrt(mse_error))
    return psnr_score.item()


# 计算结构相似性指数(SSIM)
def compute_ssim(pred_img, gt_img, max_pixel=1.0, align_mean=True):
    if align_mean:
        # 转换为灰度图（保留维度）并计算均值
        pred_gray = pred_img.mean(dim=1, keepdim=True)
        gt_gray = gt_img.mean(dim=1, keepdim=True)

        mean_pred = pred_gray.mean()
        mean_gt = gt_gray.mean()
        # 均值对齐并限制像素范围
        pred_img = torch.clamp(pred_img * (mean_gt / mean_pred), 0, 1)

    # 计算SSIM
    ssim_score = structural_similarity_index_measure(
        pred_img, gt_img, data_range=max_pixel
    )
    return ssim_score.item()


# 计算LPIPS（感知相似度）
def compute_lpips_score(pred_img, gt_img, lpips_calculator):
    return lpips_calculator(pred_img, gt_img).item()


def save_inference_image(img_tensor, save_directory, image_index):
    save_path = os.path.join(save_directory, f'infer_{image_index}.png')
    save_image(img_tensor, save_path)


def evaluate_model_performance(model, test_loader, compute_device, save_dir):
    model.eval()
    psnr_total = 0.0
    ssim_total = 0.0
    lpips_total = 0.0

    # 初始化LPIPS计算器（VGG backbone）
    lpips_loss_fn = LPIPS(net='vgg').to(compute_device)

    with torch.no_grad():
        for batch_idx, (low_light_img, normal_light_img) in enumerate(test_loader):
            # 数据移至计算设备
            low_img = low_light_img.to(compute_device)
            gt_img = normal_light_img.to(compute_device)

            # 模型推理
            pred_img = model(low_img)
            pred_img = torch.clamp(pred_img, 0, 1)

            # 保存推理图像
            save_inference_image(pred_img, save_dir, batch_idx)

            # 累加PSNR
            batch_psnr = compute_psnr(pred_img, gt_img)
            psnr_total += batch_psnr

            # 累加SSIM
            batch_ssim = compute_ssim(pred_img, gt_img)
            ssim_total += batch_ssim

            # 累加LPIPS
            batch_lpips = compute_lpips_score(pred_img, gt_img, lpips_loss_fn)
            lpips_total += batch_lpips

    # 计算平均值
    avg_psnr = psnr_total / len(test_loader)
    avg_ssim = ssim_total / len(test_loader)
    avg_lpips = lpips_total / len(test_loader)

    return avg_psnr, avg_ssim, avg_lpips


def init_experiment_config():
    # 统一路径风格（与train.py一致，可通过环境变量/配置文件修改）
    base_dir = os.environ.get('DATA_BASE_DIR', '/root/autodl-tmp')

    # 数据集路径配置（按需取消注释切换）
    test_low_dir = f'{base_dir}/LOLv1/Test/input'
    test_high_dir = f'{base_dir}/LOLv1/Test/target'

    # 备选数据集路径
    # test_low_dir = f'{base_dir}/LOLv2/Synthetic/Test/Low'
    # test_high_dir = f'{base_dir}/LOLv2/Synthetic/Test/Normal'
    # test_low_dir = f'{base_dir}/LOLv2/Real_captured/Test/Low'
    # test_high_dir = f'{base_dir}/LOLv2/Real_captured/Test/Normal'
    # test_low_dir = f'{base_dir}/five/test/input'
    # test_high_dir = f'{base_dir}/five/test/gt'
    # test_low_dir = f'{base_dir}/SICE/test/low'
    # test_high_dir = f'{base_dir}/SICE/test/gt'
    # test_low_dir = f'{base_dir}/SICE/test/over'
    # test_high_dir = f'{base_dir}/SICE/test/gt'

    # 模型权重路径（与train.py保存路径统一）
    model_weights_path = 'best_model.pth'

    # 设备配置
    compute_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_parts = test_low_dir.strip('/').split('/')
    dataset_label = [p for p in dataset_parts if any(key in p for key in ['LOL', 'five', 'SICE'])][0]
    result_save_dir = os.path.join('results', dataset_label)
    os.makedirs(result_save_dir, exist_ok=True)

    # 创建数据加载器
    _, test_data_loader = create_dataloaders(
        train_low=None, train_high=None,
        test_low=test_low_dir, test_high=test_high_dir,
        crop_size=None, batch_size=1
    )
    print(f'测试集加载器长度: {len(test_data_loader)}')

    return test_data_loader, compute_device, model_weights_path, result_save_dir


def main():
    """主函数：加载模型、执行评估、输出结果"""
    # 初始化实验配置
    test_loader, device, weights_path, result_dir = init_experiment_config()

    # 加载模型
    inference_model = RGB().to(device)

    state_dict = torch.load(weights_path, map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    inference_model.load_state_dict(state_dict)
    print(f'模型权重加载完成: {weights_path}')

    # 执行模型评估
    avg_psnr, avg_ssim, avg_lpips = evaluate_model_performance(
        inference_model, test_loader, device, result_dir
    )

    # 输出评估结果
    print(f'评估结果 - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}')


if __name__ == '__main__':
    main()