def plot_training_curves(train_losses, train_dices, val_losses, val_dices, save_dir="visualizations"):
    """Trace les courbes de loss et dice."""
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = list(range(1, len(train_losses) + 1))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(epochs, train_losses, 'o-', label='Train', linewidth=2, color='#2196F3')
    ax1.plot(epochs, val_losses, 's--', label='Val', linewidth=2, color='#FF5722')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Evolution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Dice
    ax2.plot(epochs, train_dices, 'o-', label='Train', linewidth=2, color='#4CAF50')
    ax2.plot(epochs, val_dices, 's--', label='Val', linewidth=2, color='#FF9800')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Lesion Dice', fontsize=12)
    ax2.set_title('Dice Score Evolution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Courbes d'entraînement mises à jour")


def plot_confusion_matrix(targets, preds, epoch, save_dir="visualizations", optimal_threshold=None, optimal_preds=None):
    """Trace la matrice de confusion (lésion vs pas lésion) par epoch."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Déterminer si on affiche 1 ou 2 matrices
    show_optimal = optimal_threshold is not None and optimal_preds is not None
    n_cols = 2 if show_optimal else 1
    
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]
    
    def draw_cm(ax, targets, preds, title_suffix=""):
        cm = confusion_matrix(targets, preds, labels=[0, 1])
        
        # Calcul des métriques
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        cm_percent = cm.astype('float') / cm.sum() * 100
        
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Sain (Prédit)', 'Lésion (Prédit)'],
                    yticklabels=['Sain (Réel)', 'Lésion (Réel)'],
                    cbar=True)
        
        for i in range(2):
            for j in range(2):
                text = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
                ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', 
                       fontsize=14, fontweight='bold',
                       color='white' if cm[i, j] > cm.max()/2 else 'black')
        
        ax.set_xlabel('Prédiction', fontsize=12)
        ax.set_ylabel('Réalité', fontsize=12)
        
        title = f'{title_suffix}\n'
        title += f'Acc: {accuracy:.2%} | Sens: {sensitivity:.2%} | Spec: {specificity:.2%} | Prec: {precision:.2%}'
        ax.set_title(title, fontsize=11, fontweight='bold')
        
        return accuracy
    
    # Matrice avec seuil par défaut (100 pixels)
    acc_default = draw_cm(axes[0], targets, preds, "Seuil par défaut (100 px)")
    
    # Matrice avec seuil optimal
    if show_optimal:
        acc_optimal = draw_cm(axes[1], targets, optimal_preds, f"Seuil optimal ({optimal_threshold} px)")
    
    plt.suptitle(f'Matrices de Confusion - Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_epoch_{epoch:02d}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  🔢 Matrice de confusion sauvegardée (Epoch {epoch})")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_checkpoint(path, model, metadata):
    torch.save({"model_state": model.state_dict(), **metadata}, path)
    print(f"  ✓ Checkpoint sauvegardé : {path}")

def load_checkpoint(path, device, dropout_rate=0.3):
    model = SimpleResUNet_ConvNeXtTiny(dropout_rate=dropout_rate)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt
"""
Par Enis Kaygun - 2025
SEGMENTEUR MÉDICAL - ResUNet++ (ConvNeXt Tiny)

RÔLE: Segmenteur spécialisé dans la pipeline cascade
- Backend: ConvNeXt Tiny
- Loss: Batch Dice (Évite le score parfait artificiel sur images vides)
"""

import numpy as np
import csv
import os
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import types
import albumentations as A


# ===== RLE ENCODING/DECODING (INTACT) =====

def rle_decode(mask_rle, shape):
    if pd.isna(mask_rle) or str(mask_rle).strip() == "-1":
        return np.zeros(shape, dtype=np.float32)
    
    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)
    array = np.asarray([int(x) for x in mask_rle.split()])
    starts = array[0::2]
    lengths = array[1::2]
    
    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 1
        current_position += lengths[index]
    
    return mask.reshape(w, h).T.astype(np.float32)


def rle_encode(mask):
    height, width = mask.shape
    mask_t = mask.T
    
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0
    
    for x in range(width):
        for y in range(height):
            currentColor = 1 if mask_t[x][y] > 0.5 else 0
            
            if currentColor != lastColor:
                if currentColor == 1:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            
            lastColor = currentColor
            currentPixel += 1
    
    if runStart > -1:
        rle.append(str(runStart))
        rle.append(str(runLength))
    
    return " ".join(rle) if len(rle) > 0 else "-1"


def aggregate_rle(df_rle, image_shape=(1024, 1024)):
    def agg(group):
        rles = [s for s in group.astype(str) if s.strip() != "-1"]
        if len(rles) == 0:
            return "-1"
        if len(rles) == 1:
            return rles[0]
        masks = [rle_decode(rle, image_shape) for rle in rles]
        combined_mask = np.zeros(image_shape, dtype=np.float32)
        for mask in masks:
            combined_mask = np.maximum(combined_mask, mask)
        return rle_encode(combined_mask)
    return df_rle.groupby("ImageId")["EncodedPixels"].apply(agg).reset_index()


# ===== AUGMENTATION =====

def get_training_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.3, border_mode=0),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=0,
            border_mode=0,
            p=0.3
        )
    ])


# ===== DATASET =====

class AugmentedSegmentationDataset(Dataset):
    def __init__(self, df, dicom_dir, mode="train", apply_augmentation=True):
        self.df = df.reset_index(drop=True)
        self.dicom_dir = dicom_dir
        self.mode = mode
        self.apply_augmentation = apply_augmentation and (mode == "train")
        
        if self.apply_augmentation:
            self.transform = get_training_augmentation()
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["ImageId"]
        
        path = os.path.join(self.dicom_dir, image_id + ".dcm")
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        
        # Normalisation robuste
        p01 = np.percentile(img, 1)
        p99 = np.percentile(img, 99)
        img = np.clip(img, p01, p99)
        img = (img - p01) / (p99 - p01 + 1e-6)
        
        h, w = img.shape
        
        if self.mode == "train" and "EncodedPixels" in row:
            mask = rle_decode(row["EncodedPixels"], (h, w))
        else:
            mask = np.zeros((h, w), dtype=np.float32)
        
        img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        
        img_512 = F.interpolate(img_t, size=(512, 512), mode="bilinear", align_corners=False)
        mask_512 = F.interpolate(mask_t, size=(512, 512), mode="nearest")
        
        img_np = img_512.squeeze().numpy()
        mask_np = mask_512.squeeze().numpy()
        
        if self.transform is not None:
            augmented = self.transform(image=img_np, mask=mask_np)
            img_np = augmented['image']
            mask_np = augmented['mask']
        
        # Output [1, 512, 512]
        img_final = torch.from_numpy(img_np).unsqueeze(0).float()
        mask_final = torch.from_numpy(mask_np).unsqueeze(0).float()
        
        return img_final, mask_final, image_id


# ===== MODÈLE : RESUNET CONVNEXT TINY =====

class ConvBlock(nn.Module):
    """Block de convolution pour le décodeur."""
    def __init__(self, in_ch, out_ch, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.activation = nn.GELU()
        
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    
    def forward(self, x):
        res = self.shortcut(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        x += res
        return self.activation(x)


class SimpleResUNet_ConvNeXtTiny(nn.Module):
    """
    U-Net avec Backbone ConvNeXt Tiny.
    Structure adaptée car ConvNeXt downsample par 4 dès le stem.
    """
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        print(f"🔧 Init ResUNet (ConvNeXt Tiny) - 1 Channel Input")
        
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 
        backbone = convnext_tiny(weights=weights)
        
        # Adaptation Input (3ch -> 1ch)
        original_stem = backbone.features[0][0]
        self.stem = nn.Conv2d(1, 96, kernel_size=4, stride=4, bias=True)
        with torch.no_grad():
            self.stem.weight.data = original_stem.weight.data.mean(dim=1, keepdim=True)
            self.stem.bias.data = original_stem.bias.data
            
        self.stem_norm = backbone.features[0][1]
        
        self.stage0 = backbone.features[1]
        self.down1 = backbone.features[2]
        self.stage1 = backbone.features[3]
        self.down2 = backbone.features[4]
        self.stage2 = backbone.features[5]
        self.down3 = backbone.features[6]
        self.stage3 = backbone.features[7]
        
        dec_channels = [384, 192, 96, 64]
        
        self.up3 = nn.ConvTranspose2d(768, dec_channels[0], 2, stride=2)
        self.dec3 = ConvBlock(384 + dec_channels[0], dec_channels[0], dropout_rate)
        
        self.up2 = nn.ConvTranspose2d(dec_channels[0], dec_channels[1], 2, stride=2)
        self.dec2 = ConvBlock(192 + dec_channels[1], dec_channels[1], dropout_rate)
        
        self.up1 = nn.ConvTranspose2d(dec_channels[1], dec_channels[2], 2, stride=2)
        self.dec1 = ConvBlock(96 + dec_channels[2], dec_channels[2], dropout_rate)
        
        self.up0 = nn.ConvTranspose2d(dec_channels[2], dec_channels[3], 2, stride=2)
        self.dec0 = ConvBlock(dec_channels[3], dec_channels[3], dropout_rate)
        
        self.final_up = nn.ConvTranspose2d(dec_channels[3], 32, 2, stride=2)
        self.final_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.stem_norm(x)
        s0 = self.stage0(x)
        
        x = self.down1(s0)
        s1 = self.stage1(x)
        
        x = self.down2(s1)
        s2 = self.stage2(x)
        
        x = self.down3(s2)
        b = self.stage3(x)
        
        d3 = self.up3(b)
        d3 = torch.cat([d3, s2], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, s1], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, s0], dim=1)
        d1 = self.dec1(d1)
        
        d0 = self.up0(d1)
        d0 = self.dec0(d0)
        
        out = self.final_up(d0)
        out = self.final_conv(out)
        
        return out

    def get_encoder_params(self):
        return [
            *self.stem.parameters(), *self.stem_norm.parameters(),
            *self.stage0.parameters(), *self.down1.parameters(),
            *self.stage1.parameters(), *self.down2.parameters(),
            *self.stage2.parameters(), *self.down3.parameters(),
            *self.stage3.parameters()
        ]
    
    def get_decoder_params(self):
        return [
            *self.up3.parameters(), *self.dec3.parameters(),
            *self.up2.parameters(), *self.dec2.parameters(),
            *self.up1.parameters(), *self.dec1.parameters(),
            *self.up0.parameters(), *self.dec0.parameters(),
            *self.final_up.parameters(), *self.final_conv.parameters()
        ]


# ===== BATCH DICE LOSS =====

def batch_dice_loss(logits, targets, smooth=1.0):
    """
    Calcule le Dice sur l'ensemble du Batch comme un seul volume.
    Évite le Dice=1.0 sur les images vides si le batch contient au moins une lésion.
    """
    probs = torch.sigmoid(logits)
    probs_flat = probs.view(-1)
    targets_flat = targets.view(-1)
    
    intersection = (probs_flat * targets_flat).sum()
    union = probs_flat.sum() + targets_flat.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def combo_loss_segmenter(logits, targets, bce_weight=0.5, dice_weight=0.5, lesion_weight=3.0):
    """
    Combinaison BCE + Batch Dice.
    """
    loss_dice = batch_dice_loss(logits, targets)
    
    bce_raw = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    
    has_lesion = (targets.sum(dim=(1, 2, 3)) > 0).float().view(-1, 1, 1, 1)
    weights = torch.ones_like(bce_raw) + (has_lesion * (lesion_weight - 1))
    
    loss_bce = (bce_raw * weights).mean()
    
    return bce_weight * loss_bce + dice_weight * loss_dice

# ===== TRAINING =====

def train_epoch(model, loader, optimizer, device, scaler=None):
    """
    Entraîne une epoch avec la Combo Loss (Batch Dice + Weighted BCE).
    """
    model.train()
    running_loss = 0.0
    running_lesion_dice = 0.0
    n_batches = 0
    
    for batch_idx, (imgs, masks, _) in enumerate(loader):
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss = combo_loss_segmenter(logits, masks, bce_weight=0.5, dice_weight=0.5, lesion_weight=4.0)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = combo_loss_segmenter(logits, masks, bce_weight=0.5, dice_weight=0.5, lesion_weight=4.0)
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            mask_sums = masks.sum(dim=(1, 2, 3))
            has_lesion = mask_sums > 0
            
            if has_lesion.any():
                probs = torch.sigmoid(logits)
                inter = (probs * masks).sum(dim=(1, 2, 3))
                union = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
                dices = (2. * inter + 1e-6) / (union + 1e-6)
                batch_lesion_dice = dices[has_lesion].mean().item()
            else:
                batch_lesion_dice = 0.0
        
        running_loss += loss.item()
        running_lesion_dice += batch_lesion_dice
        n_batches += 1
        
        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f} | Lesion Dice: {batch_lesion_dice:.4f}")
    
    return {
        "loss": running_loss / n_batches,
        "lesion_dice": running_lesion_dice / n_batches
    }


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    """
    Évaluation : Sépare le score sur les lésions et sur les sains.
    Retourne aussi les données brutes pour l'optimisation du seuil.
    """
    model.eval()
    running_loss = 0.0
    
    all_dices_lesion = []
    all_false_positives = []
    all_targets = []
    all_preds = []
    all_pred_pixels = []  # Nombre de pixels prédits par image
    
    for imgs, masks, _ in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        logits = model(imgs)
        loss = combo_loss_segmenter(logits, masks)
        running_loss += loss.item()
        
        probs = torch.sigmoid(logits)
        preds_bin = (probs > threshold).float()
        
        batch_size = imgs.size(0)
        for i in range(batch_size):
            true_mask = masks[i, 0]
            pred_mask = preds_bin[i, 0]
            
            is_lesion = true_mask.sum() > 0
            pred_pixels = pred_mask.sum().item()
            all_pred_pixels.append(pred_pixels)
            
            if is_lesion:
                inter = (pred_mask * true_mask).sum()
                union = pred_mask.sum() + true_mask.sum()
                dice = (2. * inter + 1e-6) / (union + 1e-6)
                all_dices_lesion.append(dice.item())
                
                all_targets.append(1)
                all_preds.append(1 if pred_pixels > 100 else 0)
            else:
                all_false_positives.append(pred_pixels)
                
                all_targets.append(0)
                all_preds.append(1 if pred_pixels > 100 else 0)
    
    avg_loss = running_loss / len(loader)
    avg_lesion_dice = np.mean(all_dices_lesion) if all_dices_lesion else 0.0
    avg_fp_pixels = np.mean(all_false_positives) if all_false_positives else 0.0
    
    return {
        "loss": avg_loss,
        "lesion_dice": avg_lesion_dice,
        "avg_fp_pixels": avg_fp_pixels,
        "targets": all_targets,
        "preds": all_preds,
        "pred_pixels": all_pred_pixels
    }


def find_optimal_threshold(targets, pred_pixels, thresholds=None):
    """
    Trouve le seuil de pixels optimal pour maximiser l'accuracy de classification.
    """
    if thresholds is None:
        thresholds = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000]
    
    targets = np.array(targets)
    pred_pixels = np.array(pred_pixels)
    
    best_acc = 0.0
    best_threshold = 100
    results = []
    
    for thresh in thresholds:
        preds = (pred_pixels > thresh).astype(int)
        acc = accuracy_score(targets, preds)
        results.append((thresh, acc))
        
        if acc > best_acc:
            best_acc = acc
            best_threshold = thresh
    
    return best_threshold, best_acc, results


# ===== VISUALIZATIONS =====

DICE_SMOOTH = 1.0

def compute_dice_score(pred, target, smooth=DICE_SMOOTH):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


@torch.no_grad()
def visualize_batch(model, loader, device, epoch, threshold=0.5, save_dir="visualizations"):
    """Visualise un batch à chaque epoch."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Collecter un seul batch (pas plusieurs) pour économiser la VRAM
    all_imgs, all_masks, all_ids = [], [], []
    for batch_idx, (imgs, masks, image_ids) in enumerate(loader):
        all_imgs.append(imgs)
        all_masks.append(masks)
        all_ids.extend(image_ids)
        if batch_idx >= 0:  # Un seul batch
            break
    
    imgs = torch.cat(all_imgs, dim=0).to(device)
    masks = torch.cat(all_masks, dim=0).to(device)
    
    mask_sums = masks.sum(dim=(1, 2, 3)).cpu().numpy()
    
    indices_with_lesions = np.where(mask_sums > 100)[0]
    indices_without_lesions = np.where(mask_sums == 0)[0]
    
    selected_indices = []
    
    if len(indices_with_lesions) >= 2:
        sorted_lesion_indices = indices_with_lesions[np.argsort(-mask_sums[indices_with_lesions])]
        selected_indices.extend(sorted_lesion_indices[:2].tolist())
    elif len(indices_with_lesions) > 0:
        selected_indices.extend(indices_with_lesions.tolist())
    
    remaining = 4 - len(selected_indices)
    if len(indices_without_lesions) >= remaining:
        selected_indices.extend(np.random.choice(indices_without_lesions, remaining, replace=False).tolist())
    else:
        all_indices = list(range(len(imgs)))
        remaining_indices = [i for i in all_indices if i not in selected_indices]
        if remaining_indices:
            selected_indices.extend(np.random.choice(remaining_indices, 
                                                    min(remaining, len(remaining_indices)), 
                                                    replace=False).tolist())
    
    n_show = min(4, len(selected_indices))
    selected_indices = selected_indices[:n_show]
    
    logits = model(imgs)
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    fig, axes = plt.subplots(n_show, 4, figsize=(16, 4*n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    
    for plot_idx, img_idx in enumerate(selected_indices):
        img_np = imgs[img_idx, 0].cpu().numpy()
        mask_np = masks[img_idx, 0].cpu().numpy()
        prob_np = probs[img_idx, 0].cpu().numpy()
        pred_np = preds[img_idx, 0].cpu().numpy()
        
        dice = compute_dice_score(pred_np, mask_np, smooth=DICE_SMOOTH)
        has_lesion = "✓ Lésion" if mask_np.sum() > 0 else "✗ Sain"
        
        axes[plot_idx, 0].imshow(img_np, cmap='gray')
        axes[plot_idx, 0].set_title(f'Image {has_lesion}\n{all_ids[img_idx]}', fontsize=10)
        axes[plot_idx, 0].axis('off')
        
        axes[plot_idx, 1].imshow(img_np, cmap='gray')
        if mask_np.sum() > 0:
            axes[plot_idx, 1].imshow(mask_np, alpha=0.5, cmap='Reds')
        axes[plot_idx, 1].set_title('Ground Truth', fontsize=10)
        axes[plot_idx, 1].axis('off')
        
        axes[plot_idx, 2].imshow(img_np, cmap='gray')
        im = axes[plot_idx, 2].imshow(prob_np, alpha=0.6, cmap='jet', vmin=0, vmax=1)
        axes[plot_idx, 2].set_title(f'Probabilités', fontsize=10)
        axes[plot_idx, 2].axis('off')
        plt.colorbar(im, ax=axes[plot_idx, 2], fraction=0.046)
        
        axes[plot_idx, 3].imshow(img_np, cmap='gray')
        if pred_np.sum() > 0:
            axes[plot_idx, 3].imshow(pred_np, alpha=0.5, cmap='Blues')
        axes[plot_idx, 3].set_title(f'Prédiction\nDice={dice:.3f}', fontsize=10)
        axes[plot_idx, 3].axis('off')
    
    plt.suptitle(f'Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'training_epoch_{epoch:02d}.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 Visualisation sauvegardée : {save_path}")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_checkpoint(path, model, metadata):
    torch.save({"model_state": model.state_dict(), **metadata}, path)
    print(f"  ✓ Checkpoint sauvegardé : {path}")

def load_checkpoint(path, device, dropout_rate=0.3):
    model = SimpleResUNet_ConvNeXtTiny(dropout_rate=dropout_rate)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt

# ===== PREDICTION =====

@torch.no_grad()
def predict(model, loader, device, threshold=0.5, min_pixels=50):
    model.eval()
    predictions = []
    print("\n🔮 Génération des prédictions (ConvNeXt)...")
    
    for imgs, _, image_ids in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.sigmoid(logits)
        pred_masks = (probs >= threshold).float()
        
        for i, image_id in enumerate(image_ids):
            msk = pred_masks[i, 0].cpu().numpy()
            if msk.sum() < min_pixels: msk[:] = 0
            predictions.append({"ImageId": image_id, "EncodedPixels": rle_encode(msk)})
            
    return predictions

# ===== CONFIGURATION =====

class Config:
    MODE = "train" 
    DICOM_DIR = "dicom-images-train"
    RLE_CSV = "trainSet-rle.csv"
    PREDICT_DICOM_DIR = "dicom-images-valid"
    OUTPUT_DIR = "outputs"
    CHECKPOINT = "best_convnext_tiny_segmenter.pth"
    PREDICTIONS_CSV = "submission.csv"
    
    EPOCHS = 25 # Nombre d'epochs
    LR = 1e-4 # Learning rate pour le décodeur
    LR_ENCODER_RATIO = 0.8 # Ratio du LR pour l'encodeur (backbone)
    BATCH_SIZE = 8 # Taille de batch
    DROPOUT_RATE = 0.5 # Dropout dans le décodeur
    
    EVAL_THRESHOLD = 0.5 # Seuil de probabilité pour l'évaluation
    MIN_PIXELS = 300 # Nombre minimum de pixels pour considérer une prédiction comme une lésion
    
    LESION_RATIO = 0.6  # Ratio de lésions dans le dataset (0.6 = 60% lésions, 40% sains)
    LESION_WEIGHT = 3.0 # Poids pour la loss des images avec lésion
    
    SEED = 42 # Graine pour la reproductibilité
    NUM_WORKERS = 0 # DataLoader workers (0 pour Windows)

def get_config():
    args = types.SimpleNamespace()
    for k, v in Config.__dict__.items():
        if not k.startswith("__"): setattr(args, k.lower(), v)
    return args

def prepare_segmenter_dataset(df, lesion_ratio=0.85, random_state=42):
    df_pos = df[df["HasLesion"] == 1]
    df_neg = df[df["HasLesion"] == 0]
    n_pos = len(df_pos)
    
    n_neg_target = int(n_pos * (1 - lesion_ratio) / lesion_ratio)
    df_neg_sampled = df_neg.sample(n=min(len(df_neg), n_neg_target), random_state=random_state)
    
    return pd.concat([df_pos, df_neg_sampled]).sample(frac=1, random_state=random_state).reset_index(drop=True)

# ===== MAIN =====

def main():
    args = get_config()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    
    print("="*60)
    print("🚀 SEGMENTEUR V3 - CONVNEXT TINY + BATCH DICE")
    print("="*60)
    
    if args.mode == "train":
        # 1. Data Loading
        df_rle = pd.read_csv(args.rle_csv)
        df_rle_agg = aggregate_rle(df_rle)
        image_ids = [f[:-4] for f in os.listdir(args.dicom_dir) if f.endswith('.dcm')]
        df = pd.DataFrame({"ImageId": image_ids}).merge(df_rle_agg, on="ImageId", how="left")
        df["EncodedPixels"] = df["EncodedPixels"].fillna("-1")
        df["HasLesion"] = (df["EncodedPixels"] != "-1").astype(int)
        
        df = prepare_segmenter_dataset(df, lesion_ratio=args.lesion_ratio)
        
        n_lesion = (df["HasLesion"] == 1).sum()
        n_sain = (df["HasLesion"] == 0).sum()
        print(f"📊 Dataset: {len(df)} images (Lésions: {n_lesion} [{100*n_lesion/len(df):.1f}%], Sains: {n_sain} [{100*n_sain/len(df):.1f}%])")
        
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["HasLesion"], random_state=args.seed)
        
        train_loader = DataLoader(
            AugmentedSegmentationDataset(train_df, args.dicom_dir, mode="train"),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
        )
        val_loader = DataLoader(
            AugmentedSegmentationDataset(val_df, args.dicom_dir, mode="train", apply_augmentation=False),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        
        # 2. Model Setup
        model = SimpleResUNet_ConvNeXtTiny(dropout_rate=args.dropout_rate).to(DEVICE)
        
        optimizer = optim.AdamW([
            {'params': model.get_encoder_params(), 'lr': args.lr * args.lr_encoder_ratio},
            {'params': model.get_decoder_params(), 'lr': args.lr}
        ], weight_decay=1e-2)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=2, verbose=True)
        scaler = torch.amp.GradScaler('cuda') if DEVICE == "cuda" else None
        
        os.makedirs("visualizations", exist_ok=True)
        
        # 3. Training Loop
        best_dice = 0.0
        train_losses, train_dices = [], []
        val_losses, val_dices = [], []
        
        for epoch in range(1, args.epochs + 1):
            print(f"\nEPOCH {epoch}/{args.epochs}")
            
            # Train
            t_metrics = train_epoch(model, train_loader, optimizer, DEVICE, scaler)
            train_losses.append(t_metrics['loss'])
            train_dices.append(t_metrics['lesion_dice'])
            
            # Val
            v_metrics = evaluate(model, val_loader, DEVICE, threshold=args.eval_threshold)
            val_losses.append(v_metrics['loss'])
            val_dices.append(v_metrics['lesion_dice'])
            
            print(f"✅ Train Loss: {t_metrics['loss']:.4f} | Lesion Dice: {t_metrics['lesion_dice']:.4f}")
            print(f"🔍 Val Loss:   {v_metrics['loss']:.4f} | Lesion Dice: {v_metrics['lesion_dice']:.4f} | FP Pixels: {v_metrics['avg_fp_pixels']:.1f}")
            
            # Visualize À CHAQUE EPOCH
            visualize_batch(model, val_loader, DEVICE, epoch, threshold=args.eval_threshold)
            
            # Courbes d'entraînement
            plot_training_curves(train_losses, train_dices, val_losses, val_dices)
            
            # Matrice de confusion À CHAQUE EPOCH (avec seuil optimal)
            optimal_thresh, optimal_acc, _ = find_optimal_threshold(v_metrics['targets'], v_metrics['pred_pixels'])
            optimal_preds = (np.array(v_metrics['pred_pixels']) > optimal_thresh).astype(int).tolist()
            plot_confusion_matrix(v_metrics['targets'], v_metrics['preds'], epoch, 
                                  optimal_threshold=optimal_thresh, optimal_preds=optimal_preds)
            print(f"  🎯 Seuil optimal: {optimal_thresh} px (Acc: {optimal_acc:.2%})")
            
            # Sauvegarde
            if v_metrics['lesion_dice'] > best_dice:
                best_dice = v_metrics['lesion_dice']
                print(f"💾 NEW BEST MODEL! ({best_dice:.4f})")
                save_checkpoint(args.checkpoint, model, {"best_dice": best_dice})
            
            scheduler.step(v_metrics['lesion_dice'])
            
            # Libérer la mémoire
            torch.cuda.empty_cache()

    elif args.mode == "predict":
        # Prediction Logic
        model, _ = load_checkpoint(args.checkpoint, DEVICE, args.dropout_rate)
        files = [f[:-4] for f in os.listdir(args.predict_dicom_dir) if f.endswith('.dcm')]
        test_loader = DataLoader(
            AugmentedSegmentationDataset(pd.DataFrame({"ImageId": files}), args.predict_dicom_dir, mode="predict"),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        preds = predict(model, test_loader, DEVICE, threshold=args.eval_threshold, min_pixels=args.min_pixels)
        os.makedirs(args.output_dir, exist_ok=True)
        pd.DataFrame(preds).to_csv(args.output_dir + "/" + args.predictions_csv, index=False)
        print("Done.")

if __name__ == "__main__":
    main()