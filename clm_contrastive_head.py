"""
CLM-Style Disaggregated Contrastive Decision Head for Qwen3-VL Forensic Auditing (MAD-Guard).

Bridges the gap between single-task scalar direct heads (`noul_scorer`, 93.10%) and
pretrained linguistic priors (`AR-SFT`, 94.90%) by replacing fixed linear classifiers
with Stanford x NVIDIA CLM's dual-tower SwiGLU residual projection heads (`4096 -> 1536 -> 512`)
and an HBM-pinned action embedding cache (`VectorArena`).

Key Capabilities:
1. Direct compatibility with `CLM_v0.1-8B.pt` (since Qwen3-VL-8B shares the 4096-dim Qwen3-8B text hidden space).
2. Disaggregated State vs. Action Encoding: Forensic authenticity & manipulation category descriptions
   are encoded once and cached in NPU HBM; online inference requires only a single multimodal forward pass
   plus a sub-millisecond `zq @ za.T` cosine similarity matmul.
3. Open-Vocabulary Category Generalization: Unlike a fixed `Linear(512, 7)` classifier, new manipulation
   categories can be added at test time simply by appending their textual criteria embeddings.
"""

import os
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_DIM = 4096
PROJ_DIM = 512

# Default semantic forensic criteria for Binary Authenticity & 7-Class Attribution
DEFAULT_BINARY_CRITERIA: Dict[str, str] = {
    "real": (
        "Authentic, unaltered photograph or document captured by a physical sensor, "
        "exhibiting natural optical noise, consistent lighting geometry, and intact structural boundaries."
    ),
    "fake": (
        "Synthetic AI-generated or digitally manipulated image exhibiting diffusion texture smoothing, "
        "blending boundary artifacts, structural layout tampering, or deepfake facial splicing."
    ),
}

DEFAULT_CATEGORY_CRITERIA: Dict[str, str] = {
    "deepfake": "Facial identity swap, expression reenactment, or localized facial blending artifacts.",
    "object": "Synthesized or inpainted foreground objects with inconsistent shadows or edge halos.",
    "satellite": "Manipulated overhead remote-sensing or satellite imagery with tiled synthesis textures.",
    "animal": "AI-generated biological subjects with unnatural fur texture or anatomical anomalies.",
    "doc": "Tampered document text, forged signatures, font kerning inconsistencies, or pixel cloning.",
    "human": "Full-body synthetic human generation with hand, limb, or clothing fold distortions.",
    "scene": "Whole-scene diffusion synthesis with global perspective, reflection, or frequency anomalies.",
}


def make_clm_head(
    width: int = 1536,
    depth: int = 3,
    proj: int = PROJ_DIM,
    activation: str = "gelu",
    layernorm: bool = True,
    residual: bool = True,
    hidden: int = HIDDEN_DIM,
) -> nn.Module:
    """Constructs a CLM-compatible projection tower (`4096 -> width -> ... -> proj`) matching `CLM_v0.1-8B.pt`."""
    act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}.get(activation, nn.GELU)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(hidden, width)
            self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 2))
            self.norms = nn.ModuleList(
                (nn.LayerNorm(width) if layernorm else nn.Identity()) for _ in range(depth - 2)
            )
            self.out = nn.Linear(width, proj)
            self.act = act()
            self.residual = residual

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.act(self.inp(x))
            for lin, nrm in zip(self.hidden, self.norms):
                h = self.act(nrm(lin(x)))
                x = x + h if self.residual else h
            return self.out(x)

    return Head()


class CLMForensicDecisionHead(nn.Module):
    """
    Hybrid CLM Dual-Tower Contrastive + Uncertainty-Gated Decision Head for MAD-Guard.

    Replaces randomly initialized scalar/linear classifiers (`noul_scorer`, `choice_classifier`)
    with CLM's pretrained dual-tower contrastive heads (`state_head` & `action_head`), while
    retaining MAD-Guard's uncertainty-conditioned `act_head`.
    """

    def __init__(
        self,
        hidden_size: int = HIDDEN_DIM,
        num_categories: int = 7,
        clm_ckpt_path: Optional[str] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_categories = num_categories

        if clm_ckpt_path and os.path.exists(clm_ckpt_path):
            ck = torch.load(clm_ckpt_path, map_location="cpu", weights_only=False)
            cfg = dict(ck["cfg"])
            kw = dict(
                width=cfg["width"],
                depth=cfg["depth"],
                proj=ck.get("projection_dim", cfg.get("projection_dim", PROJ_DIM)),
                activation=cfg.get("activation", "swiglu"),
                layernorm=cfg.get("layernorm", True),
                residual=cfg.get("residual", True),
                hidden=cfg.get("hidden_size", hidden_size),
            )
            self.state_head = make_clm_head(**kw)
            self.action_head = make_clm_head(**kw)
            self.state_head.load_state_dict(ck["state_head"])
            self.action_head.load_state_dict(ck["action_head"])
            init_scale = float(torch.as_tensor(ck["logit_scale"]).item())
        else:
            self.state_head = make_clm_head(hidden=hidden_size)
            self.action_head = make_clm_head(hidden=hidden_size)
            init_scale = 2.6592  # log(14.28) default CLM inverse temperature

        self.logit_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

        # Learnable or pre-cached HBM action embeddings for binary [real, fake] and K categories.
        # When precomputed from Qwen3-8B / Qwen3-VL text encoder, call `register_cached_criteria(...)`.
        self.register_buffer("binary_criteria_emb", torch.randn(2, hidden_size) * 0.02)
        self.register_buffer("category_criteria_emb", torch.randn(num_categories, hidden_size) * 0.02)

        # Uncertainty-conditioned audit gate [pass: 0, intercept: 1]
        self.act_head = nn.Sequential(
            nn.Linear(PROJ_DIM + 3, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    @torch.no_grad()
    def register_cached_criteria(
        self,
        binary_emb: torch.Tensor,
        category_emb: torch.Tensor,
    ) -> None:
        """Pins pre-extracted text embeddings of binary and category criteria into HBM buffers."""
        assert binary_emb.shape == (2, self.hidden_size)
        assert category_emb.shape[1] == self.hidden_size
        self.binary_criteria_emb.copy_(binary_emb.float())
        if category_emb.shape[0] == self.category_criteria_emb.shape[0]:
            self.category_criteria_emb.copy_(category_emb.float())
        else:
            self.category_criteria_emb = category_emb.float().to(self.binary_criteria_emb.device)
            self.num_categories = category_emb.shape[0]

    def forward(self, pooled_hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled_f = pooled_hidden.float()
        scale = self.logit_scale.exp().clamp(min=1.0, max=100.0)

        # 1. Project multimodal state into L2-normalized 512-d CLM decision space
        z_state = F.normalize(self.state_head(pooled_f), dim=-1)  # [B, 512]

        # 2. Project binary [real, fake] criteria & compute contrastive authenticity logits
        z_bin = F.normalize(self.action_head(self.binary_criteria_emb), dim=-1)  # [2, 512]
        bin_logits = scale * (z_state @ z_bin.t())  # [B, 2]
        noul_logit = bin_logits[:, 1] - bin_logits[:, 0]  # log-odds P(fake)/P(real)
        fake_prob = F.softmax(bin_logits, dim=-1)[:, 1]

        # 3. Project manipulation category criteria & compute contrastive attribution logits
        z_cat = F.normalize(self.action_head(self.category_criteria_emb), dim=-1)  # [K, 512]
        choice_logits = scale * (z_state @ z_cat.t())  # [B, K]

        # 4. Uncertainty-conditioned action gate
        p_safe = fake_prob.clamp(1e-6, 1.0 - 1e-6)
        binary_entropy = -(p_safe * torch.log(p_safe) + (1.0 - p_safe) * torch.log(1.0 - p_safe))
        margin = (fake_prob - 0.5).abs()
        uncertainty_feats = torch.stack([binary_entropy, margin, fake_prob], dim=-1)
        act_logits = self.act_head(torch.cat([z_state, uncertainty_feats], dim=-1))

        return {
            "noul_logit": noul_logit,
            "fake_prob": fake_prob,
            "choice_logits": choice_logits,
            "act_logits": act_logits,
            "z_state": z_state,
        }


def compute_clm_forensic_loss(
    outputs: Dict[str, torch.Tensor],
    y_bin: torch.Tensor,
    y_cat: torch.Tensor,
    lambda_choice: float = 0.5,
    lambda_act: float = 0.3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Joint Contrastive InfoNCE + Uncertainty-Gated Action Loss for CLMForensicDecisionHead.
    """
    loss_bin = F.binary_cross_entropy_with_logits(outputs["noul_logit"], y_bin.float())
    loss_cat = F.cross_entropy(outputs["choice_logits"], y_cat.long())
    loss_act = F.cross_entropy(outputs["act_logits"], y_bin.long())
    total = loss_bin + lambda_choice * loss_cat + lambda_act * loss_act
    return total, {
        "loss_total": float(total.item()),
        "loss_bin": float(loss_bin.item()),
        "loss_cat": float(loss_cat.item()),
        "loss_act": float(loss_act.item()),
    }

