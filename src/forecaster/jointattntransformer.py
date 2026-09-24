import math
from typing import Callable, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights



class ResNet18Encoder(nn.Module):
    def __init__(self, in_ch=2, pretrained_imagenet=False, out_dim=512):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained_imagenet else None
        m = resnet18(weights=weights)

        # 2-channel conv1
        old = m.conv1
        m.conv1 = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=False)

        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.l1, self.l2, self.l3, self.l4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = out_dim

    def forward(self, x):
        # x: (B,2,128,128)
        x = self.stem(x)
        x = self.l1(x); x = self.l2(x); x = self.l3(x); x = self.l4(x)   # (B,512,4,4)
        x = self.pool(x).flatten(1)                                      # (B,512)
        return x

def load_encoder_from_autoencoder(encoder: ResNet18Encoder, ae_ckpt_path: str):
    sd = torch.load(ae_ckpt_path, map_location="cpu")

    # If you saved full model.state_dict(), sd is already a state_dict.
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    # Map AE keys -> encoder keys
    mapped = {}
    for k, v in sd.items():
        if k.startswith("encoder_stem."):
            mapped["stem." + k[len("encoder_stem."):]] = v
        elif k.startswith("encoder_l1."):
            mapped["l1." + k[len("encoder_l1."):]] = v
        elif k.startswith("encoder_l2."):
            mapped["l2." + k[len("encoder_l2."):]] = v
        elif k.startswith("encoder_l3."):
            mapped["l3." + k[len("encoder_l3."):]] = v
        elif k.startswith("encoder_l4."):
            mapped["l4." + k[len("encoder_l4."):]] = v

    missing, unexpected = encoder.load_state_dict(mapped, strict=False)
    print("Missing:", missing)
    print("Unexpected:", unexpected)
    return encoder


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for (L, B, D)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (L, B, D)
        x = x + self.pe[: x.size(0)]
        # return self.dropout(x)
        return x

#============= BEST MODEL ======================

class RecordingTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    Same as nn.TransformerEncoderLayer, but stores self-attention weights.
    Stores:
      self.last_attn_weights: (B, num_heads, L, L) if average_attn_weights=False
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn_weights = None  # set on forward

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        # IMPORTANT: ask for weights and keep per-head weights
        attn_out, attn_w = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,  # => (B, heads, L, L)
            is_causal=is_causal
        )
        self.last_attn_weights = attn_w.detach()
        return self.dropout1(attn_out)


################ FINAL MODEL ##################
class TsEncoderThenJointAttn(nn.Module):
    """
    Stage 1: Temporal encoder (TransformerEncoder) -> ts_tokens (B,T,D)
    Stage 2: Joint self-attention over [ts_tokens ; img_tokens] -> pooled -> horizon head

    Inputs:
      - x: (B, T, F)              raw time series features
      - meta: (B, R)              one-hot region metadata (optional, recommended)
      - img_tokens: (B, S, D) or (B, D)  precomputed image/map tokens (already encoded elsewhere)

    Output:
      - (B, H) horizons
    """

    def __init__(
        self,
        x_dim: int,
        meta_dim: int,
        d_model: int = 128,
        weeks_ahead: int = 4,
        # Stage 1 temporal encoder
        ts_layers: int = 1,
        ts_heads: int = 4,
        # Stage 2 joint attention encoder
        joint_layers: int = 1,
        joint_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 5000,
        pool: str = "mean",           # pool over temporal tokens only: "last" or "mean"
        use_meta: bool = True,
    ):
        super().__init__()
        if pool not in {"last", "mean"}:
            raise ValueError(f"pool must be 'last' or 'mean', got {pool}")

        self.d_model = d_model
        self.weeks_ahead = weeks_ahead
        self.pool = pool
        self.use_meta = use_meta

        # --- Project raw time series to model dim ---
        self.x_proj = nn.Linear(x_dim, d_model, bias=False)

        # --- Region meta conditioning (one-hot -> d_model) ---
        self.meta_proj = nn.Linear(meta_dim, d_model, bias=False) if use_meta else None

        # ----- Load encoded imgs -------
        # load image encoder and make sure to freeze weights
        img_encoder = ResNet18Encoder(in_ch=2, pretrained_imagenet=False)
        # self.img_encoder = img_encoder

        #### End to End Training - Comment below #####
        chkpt_path = "../../stage0/"
        self.img_encoder = load_encoder_from_autoencoder(img_encoder, chkpt_path + "resnet18_ae_cch_temp_best.pt")
        # make sure the encoder is frozen OR comment for finetuning on the downstrean task
        self.img_encoder.eval()
        for p in self.img_encoder.parameters():
            p.requires_grad = False
        
        self.img_proj = nn.Linear(512, d_model, bias=False)
    
        # --- Positional encodings ---
        self.ts_pos = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        self.joint_pos = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)

        # --- Stage 1: Temporal encoder ---
        ts_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=ts_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=False,  # (L,B,D)
        )
        self.ts_encoder = nn.TransformerEncoder(ts_layer, num_layers=ts_layers)

        # --- Stage 2: Joint self-attention encoder ---
        joint_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=joint_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=False,
        )
        
        self.joint_encoder = nn.TransformerEncoder(joint_layer, num_layers=joint_layers)

        # joint_layer = RecordingTransformerEncoderLayer(
        #     d_model=d_model,
        #     nhead=joint_heads,
        #     dim_feedforward=4 * d_model,
        #     dropout=dropout,
        #     batch_first=False,
        # )
        # self.joint_encoder = nn.TransformerEncoder(joint_layer, num_layers=joint_layers)

        # ----- For week gating ---
        self.head_temp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, weeks_ahead),
        )

        self.head_fused = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, weeks_ahead),
        )

        # gate per horizon
        self.g_logit = nn.Parameter(torch.zeros(weeks_ahead))  # (H,)

        #ablation:
        # self.g_logit = nn.Parameter(torch.zeros(1))  # single scalar

        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, weeks_ahead),
        )
        # ----------------------------

        # --- Forecast head ---
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, weeks_ahead),
        )


    @torch.no_grad()
    def modality_mass_per_head(self,attn):
        """
        Quantify “temporal ↔ image” attention mass
        """
        # attn: (B, heads, L, L)
        B, H, L, _ = attn.shape
        idx = torch.arange(L, device=attn.device)
        ts = (idx % 2 == 0)
        img = ~ts

        out = []
        for h in range(H):
            a = attn[:, h]  # (B, L, L)
            out.append({
                "ts->ts":  a[:, ts][:, :, ts].mean().item(),
                "ts->img": a[:, ts][:, :, img].mean().item(),
                "img->ts": a[:, img][:, :, ts].mean().item(),
                "img->img":a[:, img][:, :, img].mean().item(),
            })
        return out

    @torch.no_grad()
    def ts_query_to_img_by_time(self, attn: torch.Tensor):
        """
        “How much does each week consult images?”
        Returns a length-T vector: for each temporal query token t,
        average attention mass going to all image key tokens.
        """
        a = attn.mean(dim=1)          # (B, L, L)
        B, L, _ = a.shape
        idx = torch.arange(L, device=a.device)
        ts_q = (idx % 2 == 0)
        img_k = ~ts_q

        # a[:, ts_q, img_k] -> (B, T, T)
        per_t = a[:, ts_q][:, :, img_k].mean(dim=-1)  # (B, T)
        return per_t.mean(dim=0)  # (T,)

    def get_last_joint_attn(self):
        return self.joint_encoder.layers[0].last_attn_weights  # (B, heads, L, L)


    def forward(self, x, x_mask, meta, img):
        """
        x: (B, T, F)
        x_mask: ignored (kept for compatibility)
        meta: (B, R) one-hot (optional if use_meta=False)
        img_tokens: (B, S, D) or (B, D)

        returns: (B, H)
        """
        # ---- Stage 0: get img from img encoder ------
        B, Tin, C, H, W = img.shape
        img_flat = img.view(B * Tin, C, H, W)
        with torch.no_grad():
            img_emb_flat = self.img_encoder(img_flat) 
        img_emb = img_emb_flat.view(B, Tin, -1)        # [B, Tin, img_emb_dim]
        img_tokens = self.img_proj(img_emb)              # [B, Tin, D]
        
        if img_tokens.dim() == 2:
            img_tokens = img_tokens.unsqueeze(1)  # (B,1,D)

        B, T, _ = x.shape

        # --- Stage 1: encode time-series into ts_tokens ---
        ts = self.x_proj(x)  # (B,T,D)

        if self.use_meta:
            m = self.meta_proj(meta.float()).unsqueeze(1)  # (B,1,D)
            ts = ts + m
            img_tokens = img_tokens + m 

        # pos enc for temporal sequence
        ts = ts.permute(1, 0, 2)     # (T,B,D)
        ts = self.ts_pos(ts) 
        ts = self.ts_encoder(ts)     # (T,B,D)

        # ts = self.ts_pos(ts) #checking
        ts = self.joint_pos(ts)
        ts_tokens = ts.permute(1, 0, 2)  # (B,T,D)

        img_tokens = self.joint_pos(img_tokens.permute(1, 0, 2))# (T,B,D)
        # img_tokens = self.ts_pos(img_tokens.permute(1, 0, 2))# (T,B,D)

        img_tokens = img_tokens.permute(1, 0, 2) # (B,T,D)

        # # --- Stage 2: joint self-attention over [ts_tokens ; img_tokens] ---
        B, T, D = ts_tokens.shape
        joint = torch.stack([ts_tokens, img_tokens], dim=2).reshape(B, 2*T, D)  # ts1,img1,ts2,img2,...
                        # (B, 2T, D)

        joint = joint.permute(1, 0, 2)                     # (T+S,B,D)
        # joint = self.joint_pos(joint)
        joint = self.joint_encoder(joint)                  # (T+S,B,D)
        joint_bt = joint.permute(1, 0, 2)                  # (B, T+S, D)

        # -------- Gating Learn Parameter per Horizon -----
        temporal_out = joint_bt[:, 0::2, :]  # positions 0,2,4,... are ts tokens
        pooled_ts = temporal_out[:, -1, :]  
        pooled_fused = joint_bt[:, -1, :]

        y_temp = self.head_temp(pooled_ts)       # (B,H)
        y_fused = self.head_fused(pooled_fused)  # (B,H)


        g = torch.sigmoid(self.g_logit).view(1, -1)  # (1,H), broadcast to (B,H)
        
        # gating ablations 
        # single learned weight
        # g = torch.sigmoid(self.g_logit).view(1, 1)  # (1,1), broadcast to (B,H)
        # uniform
        # g = 0.5

        y = (1 - g) * y_temp + g * y_fused       # (B,H)
        return y.unsqueeze(-1)                   # (B,H,1)
        
        # ----------------------------------------------------

        ##########################################################################################
