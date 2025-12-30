import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class TxtLogger:
    def __init__(self, save_dir, model,filename="train_log.txt"):
        os.makedirs(os.path.join(save_dir, model), exist_ok=True)
        self.log_path = os.path.join(save_dir, model, filename)
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                f.write("=== Training Log Started ===\n")
        else:
            with open(self.log_path, 'a') as f:
                f.write("=== Training Log Resume ===\n")

    def log(self, message):
        """写入日志并同步打印"""
        print(message)
        with open(self.log_path, 'a') as f:
            f.write(message + "\n")


class VAELoss(nn.Module):
    def __init__(self, kld_weight=0.00025):
        super().__init__()
        self.kld_weight = kld_weight

    def forward(self, recons, input, mu, log_var):
        # 1. 重构损失 (MSE)
        recons_loss = F.mse_loss(recons, input, reduction='sum') / input.shape[0] 
        # 或者使用 BCE (取决于你的输入归一化方式，如果是 [-1, 1] 推荐 MSE)
        
        # 2. KL 散度损失
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + self.kld_weight * kld_loss
        return loss, recons_loss, kld_loss