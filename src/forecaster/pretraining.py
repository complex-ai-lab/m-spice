import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torch.cuda.amp import autocast, GradScaler
import os
from PIL import Image
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from pathlib import Path

def extract_rec_ids(temp_dir):
    """
    Returns list of rec_ids like:
      ['01_202043', 'US_202043', ...]
    """
    rec_ids = []

    for p in Path(temp_dir).glob("temperature_*.png"):
        # temperature_01_202043.png
        name = p.stem                      # temperature_01_202043
        rec_id = name.replace("temperature_", "")
        rec_ids.append(rec_id)

    return sorted(rec_ids)

def split_rec_ids_by_epiweek(
    rec_ids,
    train_start=202043,
    train_end=202339,
    val_start=202340,
    val_end=202416,
):
    train, val = [], []

    for rid in rec_ids:
        epiweek = int(rid.split("_")[-1])

        if train_start <= epiweek <= train_end:
            train.append(rid)
        elif val_start <= epiweek <= val_end:
            val.append(rid)

    return train, val


# ========== Dataloader

class  MapsPretrainPNG(torch.utils.data.Dataset):
    def __init__(
        self,
        recs,
        temp_dir,
        covid_dir,
        img_in_channels: int = 2,
        img_height: int = 128,
        img_width: int = 128,
    ):
        """
        PNG-based multimodal dataset.

        Args:
            region: [N]
            metas: [N, meta_dim]
            X: (N, T, F+1) if with_week_id else (N, T, F)
            mask_X: (N, T)
            y: (N, W, out_dim)
            mask_y: (N, W)
            recs: per-window record IDs; can be scalar or per-time-step IDs
            temp_dir: directory with temperature PNGs
            covid_dir: directory with covid PNGs (reusing hum_dir slot)
            with_week_id: if True, last feature in X is week_id
            img_in_channels: 1 (covid only) or 2 (temp+covid)
        """
    
        self.recs = recs
        self.temp_dir = temp_dir
        self.covid_dir = covid_dir
        self.img_in_channels = img_in_channels
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self):
        return len(self.recs)

    def _load_single_map(self, path: str) -> np.ndarray:
        """
        Load a single PNG as grayscale and resize to (H, W).
        Returns array of shape [H, W] in float32, scaled to [0, 1].
        """
        if not os.path.exists(path):
            print('path does not exist')
            # print(path)
            return np.zeros((self.img_height, self.img_width), dtype=np.float32)

        img = Image.open(path).convert("L")  # grayscale
        if img.size != (self.img_width, self.img_height):
            img = img.resize((self.img_width, self.img_height), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        return arr

    def _canonicalize_epi_for_covid(self, rec_str: str) -> str:
        """
        Map any epiweek (region_YYYYWW) to the canonical epi-year 2023-2024
        for COVID heatmaps.

        Example:
        - '12_202020' -> '12_202320'
        - '12_202004' -> '12_202404'
        """
        # expect something like '12_202043' or '202043'
        region, epi_part = rec_str.split("_", 1)

        # robust guard
        if len(epi_part) < 6:
            # if format is weird, just return original
            return rec_str

        year = int(epi_part[:4])
        week = int(epi_part[4:6])

        # Map week to canonical epi-year:
        # Weeks 20–53 -> 2023, Weeks 1–19 -> 2024
        if 19 <= week <= 53:
            new_year = 2023
            if week == 53:
                return f"{region}_202352"
        else:
            new_year = 2024

        new_epi = f"{new_year}{week:02d}"

        return f"{region}_{new_epi}"

    def get_region(self, rec_str: str) -> str:
        region, epi_part = rec_str.split("_", 1)
        return region

    def __getitem__(self, idx):
        # ----- rec IDs -----
        rec_str = self.recs[idx]

        # build filepaths; adjust suffixes to your naming scheme
        #changed rec_str to fips_epiweek
        temp_path = os.path.join(self.temp_dir, f"temperature_{rec_str}.png")

        # covid: remap epi-year to canonical 2023–2024 season
        covid_rec_str = self._canonicalize_epi_for_covid(rec_str)
        covid_path = os.path.join(self.covid_dir, f"cch_{covid_rec_str}.png")

        # load maps
        temp_map = self._load_single_map(temp_path)  # [H, W]
        covid_map = self._load_single_map(covid_path)  # [H, W]

        if self.img_in_channels == 1:
            # e.g. covid-only
            img = covid_map[None, ...]               # [1, H, W]
        else:
            # temp + covid → 2 channels
            img = np.stack([temp_map, covid_map], axis=0)  # [2, H, W

        # ----- to tensors -----
        maps = torch.as_tensor(img, dtype=torch.float32)

        if idx == 0:
            print("DEBUG Dataset shapes:")
            print("rec_str: ", rec_str)
            print("maps : ", maps.shape)            # [C, H, W] 

        return (rec_str, maps)


class ResNet18Autoencoder(nn.Module):
    """
    Autoencoder for 128x128 inputs using ResNet-18 as encoder.

    - Input:  (B, 2, 128, 128)  # temperature + covid
    - Output: (B, out_ch, 128, 128)  # reconstruct both or just covid
    """
    def __init__(self, in_ch=2, out_ch=1, pretrained=True):
        super().__init__()

        # --- Encoder (ResNet-18) ---
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        enc = resnet18(weights=weights)

        # Replace conv1 to accept 2 channels
        old_conv1 = enc.conv1
        enc.conv1 = nn.Conv2d(in_ch, old_conv1.out_channels,
                              kernel_size=old_conv1.kernel_size,
                              stride=old_conv1.stride,
                              padding=old_conv1.padding,
                              bias=False)

        # Initialize 2-channel conv1 from pretrained 3-channel weights (if used)
        if pretrained:
            with torch.no_grad():
                # old weights shape: (64, 3, 7, 7)
                w = old_conv1.weight
                # Make 2ch weights by taking first 2 channels + scaled average of all 3
                # (simple + stable)
                enc.conv1.weight[:, :2].copy_(w[:, :2])
                # If you prefer: enc.conv1.weight.copy_(w.mean(dim=1, keepdim=True).repeat(1,2,1,1))
        self.encoder_stem = nn.Sequential(
            enc.conv1, enc.bn1, enc.relu, enc.maxpool
        )
        self.encoder_l1 = enc.layer1  # 64
        self.encoder_l2 = enc.layer2  # 128
        self.encoder_l3 = enc.layer3  # 256
        self.encoder_l4 = enc.layer4  # 512

        self.encoder = nn.Sequential(
                        self.encoder_stem,
                        self.encoder_l1,
                        self.encoder_l2,
                        self.encoder_l3,
                        self.encoder_l4,
                    )

        # For 128x128 input:
        # after stem (maxpool): 32x32
        # layer1: 32x32 (64)
        # layer2: 16x16 (128)
        # layer3:  8x8 (256)
        # layer4:  4x4 (512)

        # --- Decoder (lightweight upsampling) ---
        # (B, 512, 4,4) -> (B, out_ch, 128,128)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # 128x128
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        z = self.encoder(x) # (B, 512, 4, 4)
        y = self.dec(z)
        return y

def reconstruction_loss(y_hat, x, mode="covid"):
    # y_hat: (B,1,H,W), target covid is x[:,1:2]
    if mode == "covid":
        target = x[:, 1:2, :, :]
    elif mode == 'temp':
        target = x[:, 0:1, :, :]
    elif mode == "both":
        target = x[:, 0:2, :, :]  # temp + covid
        temp_loss = torch.mean((y_hat[:, 0:1] - x[:, 0:1]) ** 2)
        covid_loss = torch.mean((y_hat[:, 1:2] - x[:, 1:2]) ** 2)
        print("Covid Loss: ", covid_loss.item(), "Temp Loss: ", temp_loss.item())
    return torch.mean((y_hat - target) ** 2)


def _to_uint8(img2d: torch.Tensor) -> np.ndarray:
    """img2d: (H,W) float tensor -> uint8 for saving"""
    x = img2d.detach().float().cpu().numpy()
    x = np.nan_to_num(x)
    # robust min/max for visualization
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi <= lo:
        lo, hi = x.min(), x.max() if x.max() > x.min() else (0.0, 1.0)
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    return (255 * x).astype(np.uint8)

@torch.no_grad()
def save_recon_examples(model, loader, out_dir, epoch, device="cuda", mode="covid", max_batches=1):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    saved = 0
    for b, (rec_id, x) in enumerate(loader):
        if b >= max_batches:
            break

        x = x.to(device).float()
        y_hat = model(x)

        if mode == "covid":
            tgts = {"covid": x[:, 1, :, :]}
            preds = {"covid": y_hat[:, 0, :, :]}
        elif mode == "temp":
            tgts = {"temp": x[:, 0, :, :]}
            preds = {"temp": y_hat[:, 0, :, :]}
        elif mode == "both":
            tgts = {
                "temp": x[:, 0, :, :],
                "covid": x[:, 1, :, :]
            }
            preds = {
                "temp": y_hat[:, 0, :, :],
                "covid": y_hat[:, 1, :, :]
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")

        B = x.shape[0]
        num_samples = min(4, B)
        indices = torch.randperm(B)[:num_samples]
        # for i in indices:
            # i = i.item()
        for i in range(0, B):
            rid = rec_id[i]
            if isinstance(rid, (bytes, bytearray)):
                rid = rid.decode("utf-8")
            rid = str(rid).replace("/", "_")

            for name in tgts:
                Image.fromarray(_to_uint8(tgts[name][i])).save(
                    os.path.join(out_dir, f"ep{epoch:03d}_{saved}_{rid}_{name}_tgt.png")
                )
                Image.fromarray(_to_uint8(preds[name][i])).save(
                    os.path.join(out_dir, f"ep{epoch:03d}_{saved}_{rid}_{name}_pred.png")
                )

            saved += 1
# @torch.no_grad()
# def save_recon_examples(model, loader, out_dir, epoch, device="cuda", mode='covid', max_batches=1):
#     os.makedirs(out_dir, exist_ok=True)
#     model.eval()

#     saved = 0
#     for b, (rec_id, x) in enumerate(loader):
#         print("How many batches do we have? : ", b)
#         if b >= max_batches:
#             break
#         x = x.to(device).float()
#         y_hat = model(x)

#         if mode == "covid":
#             # covid target is channel 2 in input; prediction is channel 0 if out_ch=1
#             tgt = x[:, 1, :, :]              # (B,H,W)
#         elif mode == 'temp':
#             # temp target is channel 1 in input; prediction is channel 0 if out_ch=1
#             tgt = x[:, 0, :, :]              # (B,H,W)
#         pred = y_hat[:, 0, :, :]         # (B,H,W)

#         # save first 4 items in batch
#         # B = min(4, x.shape[0])
#         B = x.shape[0]
#         # B = 32
#         for i in range(0, B, 2):
#             rid = rec_id[i]
#             if isinstance(rid, (bytes, bytearray)):
#                 rid = rid.decode("utf-8")
#             rid = str(rid).replace("/", "_")

#             Image.fromarray(_to_uint8(tgt[i])).save(
#                 os.path.join(out_dir, f"ep{epoch:03d}_{saved}_{rid}_tgt.png")
#             )
#             Image.fromarray(_to_uint8(pred[i])).save(
#                 os.path.join(out_dir, f"ep{epoch:03d}_{saved}_{rid}_pred.png")
#             )
#             saved += 1


def train_autoencoder(
    model,
    train_loader,
    val_loader=None,
    epochs=20,
    lr=1e-3,
    mode="covid",  # "covid" or "temp"
    device="cuda",
    save_path = "resnet18_ae_temp.pt"
):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=(device.startswith("cuda")))
    
    best = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for rec_id, x in train_loader:
            x = x.to(device, non_blocking=True).float()

            opt.zero_grad(set_to_none=True)

            with autocast(enabled=(device.startswith("cuda"))):
                y_hat = model(x)
                loss = reconstruction_loss(y_hat, x, mode=mode)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()

        train_loss = running / max(1, len(train_loader))

        if val_loader is not None:
            model.eval()
            vrun = 0.0
            with torch.no_grad():
                for rec_id, x in val_loader:
                    x = x.to(device, non_blocking=True).float()
                    y_hat = model(x)
                    vloss = reconstruction_loss(y_hat, x, mode=mode)
                    vrun += vloss.item()
            val_loss = vrun / max(1, len(val_loader))
            print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")
        else:
            print(f"Epoch {epoch:03d} | train {train_loss:.6f}")
        
        
        if epoch % 5 == 0 and epoch > 50:
            save_recon_examples(
                model,
                val_loader if val_loader is not None else train_loader,
                out_dir="../../rebuttal/results/recon_debug_both",
                epoch=epoch,
                device=device,
                mode=mode
            )

        # ---- SAVE ----
        ckpt = {
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),  # <-- encoder only
            "optimizer_state_dict": opt.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epochs": epochs,
            "lr": lr,
            "mode": mode,
        }
        if val_loss < best:
            best = val_loss
            torch.save(ckpt, save_path.replace(".pt", "_best.pt"))

    # ---- SAVE ----
    ckpt = {
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": model.encoder.state_dict(),  # <-- encoder only
        "optimizer_state_dict": opt.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epochs": epochs,
        "lr": lr,
        "mode": mode,
    }
    torch.save(ckpt, save_path)
    print(f"Saved checkpoint to: {save_path}")


    return model

if __name__ == "__main__":
    temp_dir = "data/temperature_new"
    covid_dir = "data/cch_new"

    rec_ids = extract_rec_ids(temp_dir)
    train_rec_ids, val_rec_ids = split_rec_ids_by_epiweek(rec_ids)
    
    print("train_rec_ids: ", train_rec_ids)
    print("val_rec_ids:", val_rec_ids)

    train_ds = MapsPretrainPNG(train_rec_ids, 
                                temp_dir=temp_dir, 
                                covid_dir=covid_dir, 
                                img_in_channels=2)
    val_ds   = MapsPretrainPNG(val_rec_ids,   
                                temp_dir=temp_dir, 
                                covid_dir=covid_dir, 
                                img_in_channels=2)

    train_loader = DataLoader(train_ds, 
                            batch_size=64, 
                            shuffle=True, 
                            num_workers=4, 
                            pin_memory=True)
    val_loader   = DataLoader(val_ds,   
                            batch_size=64, 
                            shuffle=False, 
                            num_workers=4, 
                            pin_memory=True)

    model = ResNet18Autoencoder(in_ch=2, out_ch=2, pretrained=True)
    train_autoencoder(
    model,
    train_loader,
    val_loader=val_loader,
    epochs=70,
    save_path = "../../rebuttal/results/resnet18_ae_temp_cch_both.pt",
    mode='covid'
)

epochs = range(1, len(train_total) + 1)

plt.figure(figsize=(7, 5))
plt.plot(epochs, train_total, label="Train total")
plt.plot(epochs, val_total, label="Val total")
plt.plot(epochs, train_covid, label="Train COVID")
plt.plot(epochs, train_temp, label="Train temp")
plt.plot(epochs, val_covid, label="Val COVID")
plt.plot(epochs, val_temp, label="Val temp")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.savefig("joint_reconstruction_losses.png", dpi=300)

# Example to load 
# ckpt = torch.load("resnet18_ae_cch_temp.pt", map_location="cpu")
# autoenc = ResNet18Autoencoder(in_ch=2, out_ch=2, pretrained=False)
# autoenc.encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)

# for p in autoenc.encoder.parameters():
#     p.requires_grad = False
# autoenc.encoder.eval()

