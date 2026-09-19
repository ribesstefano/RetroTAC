import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class HurdleLoss(nn.Module):
    def __init__(self, lambda_reg: float = 1.0):
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds expected shape: [batch_size, 2] (index 0: logit, index 1: regression magnitude)
        logits = preds[..., 0]
        reg_preds = preds[..., 1]
        
        # Mask for non-zero targets
        is_positive = (targets > 0).float()
        
        # Classification Loss (Hurdle)
        bce_loss = F.binary_cross_entropy_with_logits(logits, is_positive, reduction='none')
        
        # Regression Loss (Masked MSE)
        mse_loss = F.mse_loss(reg_preds, targets, reduction='none')
        masked_mse = mse_loss * is_positive
        
        return (bce_loss + self.lambda_reg * masked_mse).mean()


class ZeroInflatedNLLLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds expected shape: [batch_size, 3] (index 0: logit, 1: mu, 2: unconstrained sigma)
        pi_logits = preds[..., 0]
        mu = preds[..., 1]
        # Force sigma to be strictly positive
        sigma = F.softplus(preds[..., 2]) + self.eps 
        
        # Log probabilities computed stably
        log_pi = F.logsigmoid(pi_logits)
        log_one_minus_pi = F.logsigmoid(-pi_logits)
        
        is_zero = (targets == 0).float()
        is_positive = (targets > 0).float()
        
        # NLL for exactly zero targets
        nll_zero = -log_pi
        
        # NLL for positive targets: -log N(y | mu, sigma)
        nll_positive = -log_one_minus_pi + 0.5 * math.log(2 * math.pi) + torch.log(sigma) + 0.5 * ((targets - mu) / sigma)**2
        
        return (is_zero * nll_zero + is_positive * nll_positive).mean()