import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ms_ssim

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=True).features[:16]  # block3_conv3
        self.loss_model = vgg.to(device).eval()
        for param in self.loss_model.parameters():
            param.requires_grad = False

    def forward(self, y_true, y_pred):
        device = next(self.loss_model.parameters()).device
        y_true, y_pred = y_true.to(device), y_pred.to(device)
        return F.mse_loss(self.loss_model(y_true), self.loss_model(y_pred))

# ==================== 各项损失函数 ====================
def smooth_l1_loss(y_true, y_pred):
    return F.smooth_l1_loss(y_true, y_pred)

def perceptual_loss(y_true, y_pred, loss_model, align_mean=True):
    # 新增：与测试逻辑对齐的均值校准
    if align_mean:
        pred_gray = y_pred.mean(dim=1, keepdim=True)
        gt_gray = y_true.mean(dim=1, keepdim=True)
        mean_pred = pred_gray.mean()
        mean_gt = gt_gray.mean()
        y_pred = torch.clamp(y_pred * (mean_gt / mean_pred), 0, 1)
    return loss_model(y_true, y_pred)

def histogram_loss(y_true, y_pred, bins=256):
    y_true_hist = torch.histc(y_true, bins=bins, min=0.0, max=1.0)
    y_pred_hist = torch.histc(y_pred, bins=bins, min=0.0, max=1.0)
    y_true_hist = y_true_hist / y_true_hist.sum()
    y_pred_hist = y_pred_hist / y_pred_hist.sum()
    return torch.mean(torch.abs(y_true_hist - y_pred_hist))

def psnr_loss(y_true, y_pred):
    mse = F.mse_loss(y_true, y_pred)
    if mse == 0:
        return torch.tensor(0.0, device=mse.device)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return 40.0 - torch.mean(psnr)

def color_loss(y_true, y_pred):
    mean_true = torch.mean(y_true, dim=[2, 3], keepdim=True)  # [B,3,1,1]
    mean_pred = torch.mean(y_pred, dim=[2, 3], keepdim=True)
    return torch.mean(torch.abs(mean_true - mean_pred))

def multiscale_ssim_loss(y_true, y_pred, max_val=1.0):
    return 1.0 - ms_ssim(y_true, y_pred, data_range=max_val, size_average=True)

class CombinedLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.perceptual_model = VGGPerceptualLoss(device)

        self.lambda1 = 1.00    # L_δ (smooth_l1)
        self.lambda2 = 0.03    # L_Perc
        self.lambda3 = 0.1     # L_Hist
        self.lambda4 = 0.03    # L_PSNR
        self.lambda5 = 0.35    # L_Color^MAE
        self.lambda6 = 0.6     # L_MS-SSIM

    def forward(self, y_true, y_pred):
        L_delta = smooth_l1_loss(y_true, y_pred)
        L_perc = perceptual_loss(y_true, y_pred, self.perceptual_model)
        L_hist = histogram_loss(y_true, y_pred)
        L_psnr = psnr_loss(y_true, y_pred)
        L_color = color_loss(y_true, y_pred)
        L_ms_ssim = multiscale_ssim_loss(y_true, y_pred)

        total_loss = (
            self.lambda1 * L_delta
            + self.lambda2 * L_perc
            + self.lambda3 * L_hist
            + self.lambda4 * L_psnr
            + self.lambda5 * L_color
            + self.lambda6 * L_ms_ssim
        )
        # 移除重复的torch.mean（各损失已做均值）
        return total_loss