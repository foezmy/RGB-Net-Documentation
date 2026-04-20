import torch
from torchprofile import profile_macs
from model import RGB

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RGB().to(device)

input_tensor = torch.randn(1, 3, 256, 256).to(device)

macs = profile_macs(model, input_tensor)
flops = macs * 2
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

gflops = flops / (1024*1024*1024)
mflops = gflops * 1024  # 转换为MFLOPs

print(f"Model FLOPs (G): {gflops:.4f} G")
print(f"Model FLOPs (M): {mflops:.4f} M")

print(f"Model MACs (G): {macs / (1024*1024*1024):.4f} G")

print(f"Model params (M): {num_params / 1e6:.4f}")
print(f"Model params: {num_params}")