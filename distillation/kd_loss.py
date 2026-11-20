from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureProjection(nn.Module):
    """1x1 convolutional regressors aligning student stage channels to teacher channels."""

    def __init__(self, student_channels: List[int], teacher_channels: List[int]):
        super().__init__()
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(t_ch),
            )
            for s_ch, t_ch in zip(student_channels, teacher_channels)
        ])

    def forward(
        self,
        student_feats: List[torch.Tensor],
        teacher_feats: List[torch.Tensor],
    ) -> torch.Tensor:
        feat_loss = 0.0
        for adapter, s_f, t_f in zip(self.adapters, student_feats, teacher_feats):
            proj = adapter(s_f)
            if proj.shape[-2:] != t_f.shape[-2:]:
                proj = F.interpolate(proj, size=t_f.shape[-2:], mode="bilinear", align_corners=False)
            proj_norm = F.normalize(proj.float(), p=2, dim=1)
            t_norm = F.normalize(t_f.detach().float(), p=2, dim=1)
            feat_loss = feat_loss + F.mse_loss(proj_norm, t_norm)
        return feat_loss / len(self.adapters)


class DistillationLoss(nn.Module):
    """Combined Soft-Target + Temperature Softmax + Hard-Label + Feature Distillation."""

    def __init__(
        self,
        student_channels: List[int],
        teacher_channels: List[int],
        temperature: float = 4.0,
        alpha: float = 0.7,
        beta: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
        self.feature_proj = FeatureProjection(student_channels, teacher_channels)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        student_feats: List[torch.Tensor],
        teacher_feats: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, dict]:
        T = self.temperature
        soft_student = F.log_softmax(student_logits.float() / T, dim=1)
        soft_teacher = F.softmax(teacher_logits.detach().float() / T, dim=1)
        loss_soft = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)

        loss_hard = self.ce_loss(student_logits.float(), targets)
        loss_feat = (
            self.feature_proj(student_feats, teacher_feats)
            if self.beta > 0
            else torch.tensor(0.0, device=student_logits.device)
        )

        total_loss = (1.0 - self.alpha) * loss_hard + self.alpha * loss_soft + self.beta * loss_feat
        return total_loss, {
            "total": total_loss.item(),
            "hard": loss_hard.item(),
            "soft": loss_soft.item(),
            "feat": loss_feat.item(),
        }
