import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from peft import LoraConfig, get_peft_model, PeftModel  # type: ignore
except ImportError:
    LoraConfig = None
    get_peft_model = None
    PeftModel = None

try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor  # type: ignore
except ImportError:
    Qwen3VLForConditionalGeneration = None
    AutoProcessor = None

CATE_LIST = ["deepfake", "object", "satellite", "animal", "doc", "human", "scene"]
CATE2ID = {c: i for i, c in enumerate(CATE_LIST)}
ID2CATE = {i: c for i, c in enumerate(CATE_LIST)}


class LayaDecisionHead(nn.Module):
    """
    Laya Multi-Task Calibrated Decision Head.
    
    Mounted on top of Qwen3-VL multimodal pooled representation.
    1. noul_scorer: Binary Deepfake Risk Scorer (output logit for fake vs real)
    2. choice_classifier: Manipulation Technique / Category Classifier (7 categories)
    3. act_head: Safety Audit Gate Decision ([pass, intercept]) conditioned on
                 pooled representation + uncertainty features (entropy, margin)
    """

    def __init__(self, hidden_size: int = 4096, num_categories: int = 7, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_categories = num_categories

        # 1. noul_scorer: Calibrated risk probability P(fake)
        self.noul_scorer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )

        # 2. choice_classifier: Category classification
        self.choice_classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_categories)
        )

        # 3. act_head: Audit Gate Action [pass: 0, intercept: 1]
        # Combines pooled representation + uncertainty features (entropy, margin, raw_prob)
        self.act_head = nn.Sequential(
            nn.Linear(hidden_size + 3, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )

        # Learnable temperature scaling for probability calibration
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, pooled_hidden: torch.Tensor):
        # Cast to float32 for stable scoring and calibration
        pooled_f = pooled_hidden.float()

        noul_logit = self.noul_scorer(pooled_f).squeeze(-1)  # [B]
        choice_logits = self.choice_classifier(pooled_f)     # [B, num_categories]

        # Calibrated probability
        temp = self.temperature.clamp(min=0.1, max=5.0)
        fake_prob = torch.sigmoid(noul_logit / temp)

        # Compute uncertainty features for audit gate
        p_safe = fake_prob.clamp(1e-6, 1.0 - 1e-6)
        binary_entropy = -(p_safe * torch.log(p_safe) + (1.0 - p_safe) * torch.log(1.0 - p_safe))
        margin = (fake_prob - 0.5).abs()
        uncertainty_feats = torch.stack([binary_entropy, margin, fake_prob], dim=-1)

        act_logits = self.act_head(torch.cat([pooled_f, uncertainty_feats], dim=-1))  # [B, 2]

        return {
            "noul_logit": noul_logit,
            "fake_prob": fake_prob,
            "choice_logits": choice_logits,
            "act_logits": act_logits
        }


class Qwen3VLLayaModel(nn.Module):
    """
    Qwen3-VL Multimodal Backbone + Laya Decision Head.
    
    Single forward pass performs:
    - Multimodal visual-text understanding (Qwen3-VL-8B BF16 with LoRA)
    - Deepfake authenticity calibration (noul)
    - Forensic manipulation categorization (choice)
    - Audit gate decision action (act)
    """

    def __init__(
        self,
        base_model_path: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        num_categories: int = 7,
        torch_dtype=torch.bfloat16,
        device_map=None,
        use_gradient_checkpointing: bool = True
    ):
        super().__init__()
        print(f"Loading Qwen3-VL base model from {base_model_path}...")
        self.base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        if use_gradient_checkpointing:
            print("Enabling gradient checkpointing on base model...")
            self.base_model.gradient_checkpointing_enable()
            if hasattr(self.base_model, "enable_input_require_grads"):
                self.base_model.enable_input_require_grads()

        hidden_size = self.base_model.config.text_config.hidden_size

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.backbone = get_peft_model(self.base_model, lora_config)
        self.decision_head = LayaDecisionHead(hidden_size=hidden_size, num_categories=num_categories)
        self.num_categories = num_categories

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None, **kwargs):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        last_hidden = outputs.hidden_states[-1]  # [B, Seq_len, hidden_size]

        # Extract representation from the last non-padded token position
        if attention_mask is not None:
            last_token_idx = attention_mask.sum(dim=1) - 1
            last_token_idx = last_token_idx.clamp(min=0, max=last_hidden.size(1) - 1)
            batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
            pooled = last_hidden[batch_indices, last_token_idx]
        else:
            pooled = last_hidden[:, -1, :]

        decision_out = self.decision_head(pooled)
        return decision_out

    def save_laya_model(self, output_dir: str, processor=None):
        os.makedirs(output_dir, exist_ok=True)
        # 1. Save LoRA adapter weights
        lora_dir = os.path.join(output_dir, "lora_adapter")
        self.backbone.save_pretrained(lora_dir)

        # 2. Save Laya Decision Head weights
        head_path = os.path.join(output_dir, "laya_decision_head.pt")
        torch.save(self.decision_head.state_dict(), head_path)

        # 3. Save Config
        meta = {
            "num_categories": self.num_categories,
            "categories": CATE_LIST,
            "hidden_size": self.decision_head.hidden_size,
            "temperature": float(self.decision_head.temperature.data.item())
        }
        with open(os.path.join(output_dir, "laya_config.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # 4. Save processor
        if processor is not None:
            processor.save_pretrained(output_dir)

        print(f"[✓] Qwen3-VL Laya model saved successfully to: {output_dir}")

    @classmethod
    def load_laya_model(
        cls,
        base_model_path: str,
        checkpoint_dir: str,
        torch_dtype=torch.bfloat16,
        device="npu:0"
    ):
        lora_dir = os.path.join(checkpoint_dir, "lora_adapter")
        head_path = os.path.join(checkpoint_dir, "laya_decision_head.pt")
        cfg_path = os.path.join(checkpoint_dir, "laya_config.json")

        with open(cfg_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        print(f"Loading base model {base_model_path} onto {device}...")
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=device
        )
        print(f"Loading LoRA adapter from {lora_dir}...")
        backbone = PeftModel.from_pretrained(base_model, lora_dir)

        model = cls.__new__(cls)
        super(Qwen3VLLayaModel, model).__init__()
        model.base_model = base_model
        model.backbone = backbone
        model.num_categories = meta["num_categories"]
        hidden_size = base_model.config.text_config.hidden_size

        model.decision_head = LayaDecisionHead(hidden_size=hidden_size, num_categories=model.num_categories)
        model.decision_head.load_state_dict(torch.load(head_path, map_location="cpu"))
        model.decision_head.to(device)
        model.decision_head.temperature.data.fill_(meta.get("temperature", 1.0))
        return model

