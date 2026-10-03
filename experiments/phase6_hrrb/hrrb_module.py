#!/usr/bin/env python3
"""
experiments/phase6_hrrb/hrrb_module.py

Phase 6: High-Resolution Residual Bypass (HRRB) Module & Wrapper
================================================================

Scientific Hypothesis:
    Does Candidate B's main pathway suffer from false bridges and topology errors
    because it lacks an independent high-resolution feature pathway to the final prediction?
    
Architecture Contract:
    Input: RGB [B, 3, 448, 448]
        ↓
    Conv 3x3, stride=1, padding=1, 3->8
        ↓
    GELU
        ↓
    Conv 3x3, stride=2, padding=1, 8->16
        ↓
    [B, 16, 224, 224]
        ↓
    Conv 1x1, 16->1
        ↓
    [B, 1, 224, 224]
        ↓
    Bilinear x2 (align_corners=False)
        ↓
    detail_residual [B, 1, 448, 448]

Prediction Composition:
    final_logits = main_logits_448 + detail_residual

Initialization Contract (Zero-Init Identity at t=0):
    Conv2d(16, 1, kernel_size=1) weights and bias initialized to 0.
    detail_residual(t=0) == 0.
    final_logits(t=0) == main_logits_448 bit-exactly.
"""

from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class HighResolutionResidualBypass(nn.Module):
    """
    Independent High-Resolution Residual Bypass (HRRB) branch.
    Takes normalized image [B, 3, 448, 448] and outputs detail_residual [B, 1, 448, 448].
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        # Stage 1: 448x448, 3 -> 8
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.act = nn.GELU()

        # Stage 2: 224x224, 8 -> 16
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
        )

        # Stage 3: Projection 16 -> 1 @ 224x224
        self.proj = nn.Conv2d(
            in_channels=16,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        Kaiming normal for conv1 and conv2, EXACT ZERO for final projection.
        Guarantees detail_residual == 0 at initialization.
        """
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)

        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

        # Zero initialization for final 1x1 projection
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [B, 3, 448, 448]
        Returns:
            detail_residual (torch.Tensor): [B, 1, 448, 448]
        """
        target_size = (x.shape[2], x.shape[3])  # (448, 448)

        feat_448 = self.act(self.conv1(x))  # [B, 8, 448, 448]
        feat_224 = self.act(self.conv2(feat_448))  # [B, 16, 224, 224]
        res_224 = self.proj(feat_224)  # [B, 1, 224, 224]

        # Bilinear x2 upsampling to 448x448
        detail_residual = F.interpolate(
            res_224,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        return detail_residual


class CandidateBWithHRRB(nn.Module):
    """
    Composite Model combining 100% frozen Candidate B with trainable HRRB branch.
    """

    def __init__(self, main_model: nn.Module, hrrb_branch: Optional[HighResolutionResidualBypass] = None):
        super().__init__()
        self.main_model = main_model
        self.hrrb = hrrb_branch if hrrb_branch is not None else HighResolutionResidualBypass()

        # Strict Freeze of main model
        for param in self.main_model.parameters():
            param.requires_grad = False
        self.main_model.eval()

        # Ensure HRRB parameters are trainable
        for param in self.hrrb.parameters():
            param.requires_grad = True

    def train(self, mode: bool = True):
        """
        Override train mode: HRRB toggles train/eval, but main_model stays strictly in eval().
        """
        super().train(mode)
        self.main_model.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): [B, 3, 448, 448]
            return_components (bool): If True, returns (final_logits, main_logits, detail_residual).
        Returns:
            final_logits (torch.Tensor): [B, 1, 448, 448]
            or tuple if return_components is True.
        """
        # 1. Main Candidate B path (evaluated in no_grad mode or detached if needed)
        with torch.no_grad():
            main_logits = self.main_model(x)

        # 2. HRRB residual path
        detail_residual = self.hrrb(x)

        # 3. Additive composition
        final_logits = main_logits + detail_residual

        if return_components:
            return final_logits, main_logits, detail_residual
        return final_logits
