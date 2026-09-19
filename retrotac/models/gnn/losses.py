import math
import torch
import torch.nn.functional as F
from chemprop.nn.metrics import ChempropMetric

class ChempropHurdleLoss(ChempropMetric):
    def __init__(self, lambda_reg: float = 1.0, task_weights=None):
        super().__init__(task_weights=task_weights)
        self.lambda_reg = lambda_reg

    def _calc_unreduced_loss(self, preds: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        # Assuming preds shape is (batch_size, num_tasks * 2)
        # Reshape to (batch_size, num_tasks, 2) to separate logit and magnitude per task
        preds = preds.view(preds.shape[0], -1, 2)
        
        logits = preds[..., 0]
        reg_preds = preds[..., 1]
        
        is_positive = (targets > 0).float()
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, is_positive, reduction='none')
        mse_loss = F.mse_loss(reg_preds, targets, reduction='none')
        masked_mse = mse_loss * is_positive
        
        # Return tensor of shape (batch_size, num_tasks). 
        # ChemPropMetric handles masking and reduction.
        return bce_loss + self.lambda_reg * masked_mse


class ChempropZINLLLoss(ChempropMetric):
    def __init__(self, eps: float = 1e-6, task_weights=None):
        super().__init__(task_weights=task_weights)
        self.eps = eps

    def _calc_unreduced_loss(self, preds: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        # Assuming preds shape is (batch_size, num_tasks * 3)
        # Reshape to (batch_size, num_tasks, 3) 
        preds = preds.view(preds.shape[0], -1, 3)
        
        pi_logits = preds[..., 0]
        mu = preds[..., 1]
        sigma = F.softplus(preds[..., 2]) + self.eps
        
        log_pi = F.logsigmoid(pi_logits)
        log_one_minus_pi = F.logsigmoid(-pi_logits)
        
        is_zero = (targets == 0).float()
        is_positive = (targets > 0).float()
        
        nll_zero = -log_pi
        nll_positive = -log_one_minus_pi + 0.5 * math.log(2 * math.pi) + torch.log(sigma) + 0.5 * ((targets - mu) / sigma)**2
        
        return is_zero * nll_zero + is_positive * nll_positive