"""
Par Enis Kaygun - 2025
CLASSIFIEUR MÉDICAL - EfficientNet-B3 (Haute Sensibilité)

RÔLE: "Le Filet" - Ne jamais rater une lésion
- Objectif: Recall ~95-99% (haute sensibilité)
- Accepte quelques faux positifs (le segmenteur corrigera)
- Filtre rapide pour les 77% d'images saines

STRATÉGIE:
- WeightedRandomSampler: batchs équilibrés 50/50
- Focal Loss: focus sur les cas difficiles
- Validation: proportions réelles pour calibrer le seuil
- Seuil calibré via courbe ROC pour maximiser Recall

ARCHITECTURE:
- EfficientNet-B3 (~12M params, features=1536)
- Classification binaire pure (pas de segmentation)
- Dropout avant la couche finale
"""

import numpy as np
import csv
import os
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score, 
                             precision_score, roc_curve, auc, f1_score, fbeta_score)
import matplotlib.pyplot as plt
import seaborn as sns
import timm
import types
import albumentations as A
import os

# ===== AUGMENTATION GRAYSCALE =====

def get_training_augmentation():
    """Augmentations pour images médicales GRAYSCALE."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3, border_mode=0),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=0,
            border_mode=0,
            p=0.3
        )
    ])


# ===== DATASET CLASSIFIEUR =====

class ClassificationDataset(Dataset):
    """Dataset pour classification binaire (lésion/pas lésion)."""
    
    def __init__(self, df, dicom_dir, mode="train", apply_augmentation=True):
        self.df = df.reset_index(drop=True)
        self.dicom_dir = dicom_dir
        self.mode = mode
        self.apply_augmentation = apply_augmentation and (mode == "train")
        
        if self.apply_augmentation:
            self.transform = get_training_augmentation()
            print(f"  ✓ Augmentation activée")
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["ImageId"]
        label = row["HasLesion"]
        
        path = os.path.join(self.dicom_dir, image_id + ".dcm")
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        
        # Normalisation robuste
        p01 = np.percentile(img, 1)
        p99 = np.percentile(img, 99)
        img = np.clip(img, p01, p99)
        img = (img - p01) / (p99 - p01 + 1e-6)
        
        # Resize vers 512x512
        img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        img_512 = F.interpolate(img_t, size=(512, 512), mode="bilinear", align_corners=False)
        img_np = img_512.squeeze().numpy()
        
        # Augmentation
        if self.transform is not None:
            augmented = self.transform(image=img_np)
            img_np = augmented['image']
        
        # Conversion finale
        img_final = torch.from_numpy(img_np).unsqueeze(0).float()  # [1, 512, 512]
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return img_final, label_tensor, image_id
    
    def get_labels(self):
        """Retourne les labels pour le WeightedRandomSampler."""
        return self.df["HasLesion"].values


# ===== MODÈLE : EFFICIENTNET-B3 CLASSIFIEUR =====

class EfficientNetB3Classifier(nn.Module):
    """
    Classifieur binaire basé sur EfficientNet-B3.
    EfficientNet-B3 : ~12M params, features=1536
    
    Entrée : [B, 1, 512, 512] (GRAYSCALE)
    Sortie : [B, 1] (logit pour probabilité de lésion)
    """
    
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        
        print(f"🔧 Initialisation EfficientNet-B3 Classifieur")
        print(f"   Dropout rate : {dropout_rate}")
        
        # EfficientNet-B3 pré-entraîné, adapté pour 1 canal
        self.backbone = timm.create_model(
            'efficientnet_b3',
            pretrained=True,
            in_chans=1,
            num_classes=0,  # Retirer la tête de classification
            global_pool='avg'
        )
        
        # Nombre de features en sortie du backbone
        n_features = self.backbone.num_features  # 1536 pour EfficientNet-B3
        
        # Tête de classification avec dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(n_features, 1)
        )
        
        print(f"   Features backbone : {n_features}")
        print(f"   Paramètres : {sum(p.numel() for p in self.parameters())/1e6:.1f}M")
    
    def forward(self, x):
        features = self.backbone(x)  # [B, 1536]
        logits = self.classifier(features)  # [B, 1]
        return logits.squeeze(-1)  # [B]
    
    def get_backbone_params(self):
        """Retourne les paramètres du backbone."""
        return list(self.backbone.parameters())
    
    def get_classifier_params(self):
        """Retourne les paramètres de la tête de classification."""
        return list(self.classifier.parameters())


# ===== LOSS FUNCTIONS =====

class FocalLoss(nn.Module):
    """
    Focal Loss pour classification déséquilibrée.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    - alpha: poids pour la classe positive (lésion)
    - gamma: facteur de focus (gamma=0 → BCE standard)
      - gamma=2: cas difficiles ont 100x plus d'importance que cas faciles
    
    Pour notre cas (22% positifs):
    - alpha=0.75: compense le déséquilibre
    - gamma=2: focus sur les lésions difficiles à détecter
    """
    
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Probabilité de la vraie classe
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Poids alpha pour chaque exemple
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal Loss
        focal_loss = alpha_t * focal_weight * bce
        
        return focal_loss.mean()


class WeightedBCELoss(nn.Module):
    """
    BCE avec poids pour la classe positive.
    
    pos_weight=5 signifie: "rater une lésion coûte 5x plus cher"
    """
    
    def __init__(self, pos_weight=5.0):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor([self.pos_weight], device=logits.device)
        )


# ===== TRAINING =====

def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """Entraîne une epoch."""
    model.train()
    running_loss = 0.0
    all_targets = []
    all_probs = []
    n_samples = 0
    
    for batch_idx, (imgs, labels, _) in enumerate(loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        batch_size = imgs.size(0)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
        
        running_loss += loss.item() * batch_size
        n_samples += batch_size
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")
    
    # Métriques avec seuil 0.5
    all_preds = (np.array(all_probs) > 0.5).astype(int)
    all_targets = np.array(all_targets).astype(int)
    
    return {
        "loss": running_loss / n_samples,
        "accuracy": accuracy_score(all_targets, all_preds),
        "recall": recall_score(all_targets, all_preds, zero_division=0),
        "precision": precision_score(all_targets, all_preds, zero_division=0),
        "probs": all_probs,
        "targets": all_targets
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Évalue le modèle et retourne les probabilités pour calibration."""
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []
    all_image_ids = []
    n_samples = 0
    
    for imgs, labels, image_ids in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        batch_size = imgs.size(0)
        
        logits = model(imgs)
        loss = criterion(logits, labels)
        
        probs = torch.sigmoid(logits)
        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        all_image_ids.extend(image_ids)
        
        running_loss += loss.item() * batch_size
        n_samples += batch_size
    
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets).astype(int)
    
    # Métriques avec seuil 0.5 (par défaut)
    all_preds = (all_probs > 0.5).astype(int)
    
    return {
        "loss": running_loss / n_samples,
        "accuracy": accuracy_score(all_targets, all_preds),
        "recall": recall_score(all_targets, all_preds, zero_division=0),
        "precision": precision_score(all_targets, all_preds, zero_division=0),
        "f1": f1_score(all_targets, all_preds, zero_division=0),
        "probs": all_probs,
        "targets": all_targets,
        "preds": all_preds,
        "image_ids": all_image_ids
    }


def find_optimal_threshold(targets, probs, target_recall=0.95):
    """
    Trouve le seuil optimal pour atteindre un recall cible
    avec la meilleur précision possible.
    
    Args:
        targets: labels réels (0/1)
        probs: probabilités prédites
        target_recall: recall minimum souhaité (default 95%)
    
    Returns:
        threshold, metrics_at_threshold
    """
    # Tester plusieurs seuils
    thresholds_to_test = np.arange(0.05, 0.96, 0.01)
    
    best_threshold = 0.5
    best_precision = 0.0
    best_f1 = 0.0
    
    results = []
    
    for thresh in thresholds_to_test:
        preds = (probs >= thresh).astype(int)
        recall = recall_score(targets, preds, zero_division=0)
        precision = precision_score(targets, preds, zero_division=0)
        f1 = f1_score(targets, preds, zero_division=0)
        
        results.append({
            'threshold': thresh,
            'recall': recall,
            'precision': precision,
            'f1': f1
        })
        
        # Si le recall est suffisant, on cherche la meilleure precision
        if recall >= target_recall:
            if precision > best_precision:
                best_precision = precision
                best_threshold = thresh
                best_f1 = f1
    
    # Calculer aussi la courbe ROC pour l'AUC
    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)
    
    # Métriques finales au seuil optimal
    preds_optimal = (probs >= best_threshold).astype(int)
    
    # Calculer spécificité
    tn = np.sum((targets == 0) & (preds_optimal == 0))
    fp = np.sum((targets == 0) & (preds_optimal == 1))
    specificity = tn / (tn + fp + 1e-6)
    
    return {
        "threshold": best_threshold,
        "recall": recall_score(targets, preds_optimal, zero_division=0),
        "precision": precision_score(targets, preds_optimal, zero_division=0),
        "f1": f1_score(targets, preds_optimal, zero_division=0),
        "accuracy": accuracy_score(targets, preds_optimal),
        "specificity": specificity,
        "roc_auc": roc_auc,
        "all_results": results,
        "fpr": fpr,
        "tpr": tpr
    }


# ===== VISUALIZATIONS =====

def plot_roc_curve(targets, probs, optimal_threshold, epoch, save_dir="visualizations"):
    """Courbe ROC avec seuil optimal marqué."""
    os.makedirs(save_dir, exist_ok=True)
    
    fpr, tpr, thresholds = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)
    
    # Trouver le point correspondant au seuil optimal
    idx = np.argmin(np.abs(thresholds - optimal_threshold))
    
    plt.figure(figsize=(10, 8))
    
    # Courbe ROC
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    # Point optimal
    plt.scatter([fpr[idx]], [tpr[idx]], c='red', s=200, zorder=5, 
                label=f'Seuil optimal = {optimal_threshold:.3f}')
    plt.annotate(f'Recall={tpr[idx]:.2%}\nSpec={1-fpr[idx]:.2%}',
                xy=(fpr[idx], tpr[idx]), xytext=(fpr[idx]+0.1, tpr[idx]-0.1),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.xlabel('Taux de Faux Positifs (1 - Spécificité)', fontsize=12)
    plt.ylabel('Taux de Vrais Positifs (Sensibilité/Recall)', fontsize=12)
    plt.title(f'Courbe ROC - Classifieur EfficientNet-B3 (Epoch {epoch})\nObjectif: Maximiser Recall + Precision', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'classifier_roc_epoch_{epoch:02d}.png'), dpi=150)
    plt.close()


def plot_confusion_matrix_classifier(y_true, y_pred, epoch, threshold, save_dir="visualizations"):
    """Matrice de confusion pour le classifieur."""
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp + 1e-6)
    else:
        spec = 0.0
    
    plt.figure(figsize=(10, 8))
    
    if cm.size == 4:
        labels = ['TN\n(Vrai Sain)', 'FP\n(Fausse Alerte)', 
                  'FN\n(Lésion RATÉE!)', 'TP\n(Vrai Malade)']
        counts = [f"{v:0.0f}" for v in cm.flatten()]
        percs = [f"{v:.1%}" for v in cm.flatten()/np.sum(cm)]
        annot = [f"{l}\n{c}\n{p}" for l, c, p in zip(labels, counts, percs)]
        annot = np.asarray(annot).reshape(2, 2)
        
        # Couleurs personnalisées: FN en rouge (c'est le pire cas!)
        cmap = sns.color_palette("Blues", as_cmap=True)
        
        ax = sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False,
                    xticklabels=['Prédit Sain', 'Prédit Suspect'],
                    yticklabels=['Réel Sain', 'Réel Malade'])
        
        # Mettre en évidence les FN
        if cm[1, 0] > 0:  # Si FN > 0
            ax.add_patch(plt.Rectangle((0, 1), 1, 1, fill=False, edgecolor='red', lw=3))
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    plt.title(f'Classifieur EfficientNet-B3 - Epoch {epoch} (seuil={threshold:.3f})\n'
              f'Recall: {recall:.1%} | Precision: {precision:.1%} | Spec: {spec:.1%}', 
              fontsize=12)
    plt.ylabel('Vérité Terrain')
    plt.xlabel('Prédiction')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'classifier_cm_epoch_{epoch:02d}.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def plot_threshold_analysis(targets, probs, epoch, save_dir="visualizations"):
    """Analyse des métriques en fonction du seuil."""
    os.makedirs(save_dir, exist_ok=True)
    
    thresholds = np.arange(0.05, 0.96, 0.05)
    recalls = []
    precisions = []
    specificities = []
    f1s = []
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        recalls.append(recall_score(targets, preds, zero_division=0))
        precisions.append(precision_score(targets, preds, zero_division=0))
        
        tn = np.sum((targets == 0) & (preds == 0))
        fp = np.sum((targets == 0) & (preds == 1))
        specificities.append(tn / (tn + fp + 1e-6))
        
        f1s.append(f1_score(targets, preds, zero_division=0))
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(thresholds, recalls, 'b-', linewidth=2, label='Recall (Sensibilité)', marker='o')
    plt.plot(thresholds, precisions, 'g-', linewidth=2, label='Precision', marker='s')
    plt.plot(thresholds, specificities, 'r-', linewidth=2, label='Spécificité', marker='^')
    plt.plot(thresholds, f1s, 'm--', linewidth=2, label='F1-Score', marker='d')
    
    # Ligne de recall cible
    plt.axhline(y=0.95, color='blue', linestyle=':', alpha=0.7, label='Recall cible (95%)')
    
    plt.xlabel('Seuil de décision', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Analyse des métriques vs Seuil - Epoch {epoch}', fontsize=14)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'classifier_threshold_analysis_{epoch:02d}.png'), dpi=150)
    plt.close()


def plot_training_curves_classifier(log_path, save_dir="visualizations"):
    """Courbes d'entraînement du classifieur."""
    if not os.path.exists(log_path):
        return
    
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(log_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(df['epoch'], df['train_loss'], 'o-', label='Train', linewidth=2, color='#2196F3')
    axes[0, 0].plot(df['epoch'], df['val_loss'], 's--', label='Validation', linewidth=2, color='#FF5722')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (Focal)')
    axes[0, 0].set_title('Loss Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Recall (le plus important!)
    axes[0, 1].plot(df['epoch'], df['train_recall'], 'o-', label='Train', linewidth=2, color='#4CAF50')
    axes[0, 1].plot(df['epoch'], df['val_recall'], 's--', label='Val (seuil 0.5)', linewidth=2, color='#FF9800')
    if 'val_recall_calibrated' in df.columns:
        axes[0, 1].plot(df['epoch'], df['val_recall_calibrated'], '^-', label='Val (seuil calibré)', 
                       linewidth=2, color='#E91E63')
    axes[0, 1].axhline(y=0.95, color='red', linestyle=':', alpha=0.7, label='Cible 95%')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_title('Recall Evolution (⭐ Métrique clé)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])
    
    # Precision
    axes[1, 0].plot(df['epoch'], df['train_precision'], 'o-', label='Train', linewidth=2, color='#9C27B0')
    axes[1, 0].plot(df['epoch'], df['val_precision'], 's--', label='Validation', linewidth=2, color='#00BCD4')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.05])
    
    # F2-Score
    if 'val_f2_calibrated' in df.columns:
        axes[1, 1].plot(df['epoch'], df['val_f2_calibrated'], 'o-', linewidth=2, color='#FF5722')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F2-Score')
        axes[1, 1].set_title('Évolution du F2-Score (métrique optimisée)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1.05])
    else:
        axes[1, 1].text(0.5, 0.5, 'F2-Score\nnon disponible', 
                       ha='center', va='center', fontsize=14)
        axes[1, 1].axis('off')
    
    plt.suptitle('Entraînement Classifieur EfficientNet-B3 - Métriques', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'classifier_training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ===== UTILS =====

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(path, model, metadata):
    ckpt = {"model_state": model.state_dict(), **metadata}
    torch.save(ckpt, path)
    print(f"  ✓ Checkpoint sauvegardé : {path}")


def load_checkpoint(path, device, dropout_rate=0.5):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = EfficientNetB3Classifier(dropout_rate=dropout_rate)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt


def create_weighted_sampler(dataset):
    """
    Crée un WeightedRandomSampler pour équilibrer les batchs 50/50.
    
    Même si le dataset est déséquilibré (77/23), chaque batch
    aura ~50% positifs et ~50% négatifs.
    """
    labels = dataset.get_labels()
    
    # Compter les classes
    class_counts = np.bincount(labels)
    n_neg, n_pos = class_counts[0], class_counts[1]
    
    print(f"\n📊 WeightedRandomSampler :")
    print(f"   Négatifs (sains)   : {n_neg}")
    print(f"   Positifs (lésions) : {n_pos}")
    
    # Poids inversement proportionnels à la fréquence
    weight_neg = 1.0 / n_neg
    weight_pos = 1.0 / n_pos
    
    # Poids pour chaque échantillon
    weights = np.array([weight_neg if l == 0 else weight_pos for l in labels])
    weights = torch.from_numpy(weights).double()
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    print(f"   → Batchs équilibrés ~50/50 ✓")
    
    return sampler


# ===== CONFIGURATION =====

class Config:
    # Mode
    MODE = "train"  # "train" ou "predict"
    
    # Données
    DICOM_DIR = "dicom-images-train"
    RLE_CSV = "trainSet-rle.csv"
    PREDICT_DICOM_DIR = "dicom-images-valid"
    
    # Sorties
    OUTPUT_DIR = "outputs"
    CHECKPOINT = "best_classifier_efficientnet_b3.pth"
    PREDICTIONS_CSV = "classifier_predictions.csv"
    
    # Architecture
    MODEL = "efficientnet_b3_classifier"
    
    # Hyperparamètres
    EPOCHS = 20 # Nombre d'epochs
    LR = 7e-5 # Learning rate pour la tête de classification
    LR_BACKBONE_RATIO = 0.5  # Backbone 2x plus lent
    BATCH_SIZE = 8 # Taille de batch
    
    # Régularisation
    DROPOUT_RATE = 0.5 # Dropout avant la couche finale
    
    # Loss
    LOSS_TYPE = "focal"  # "focal" ou "weighted_bce"
    FOCAL_ALPHA = 0.75   # Poids classe positive
    FOCAL_GAMMA = 1.0    # Focus sur cas difficiles
    BCE_POS_WEIGHT = 2.0 # Si weighted_bce
    
    # Calibration
    TARGET_RECALL = 0.95  # Recall cible pour calibration du seuil
    
    # Augmentation
    USE_AUGMENTATION = True # Augmentations pendant l'entraînement
    
    # Sampler
    USE_WEIGHTED_SAMPLER = True # Équilibrer les batchs 50/50
    
    # Scheduler
    SCHEDULER_PATIENCE = 1 # Epochs sans amélioration avant réduction LR
    
    # Seed
    SEED = 42 # Pour reproductibilité
    NUM_WORKERS = 0 # DataLoader workers (0 pour Windows)


def get_config():
    args = types.SimpleNamespace()
    for key, value in vars(Config).items():
        if not key.startswith('_'):
            setattr(args, key.lower(), value)
    return args


# ===== MAIN =====

def main():
    args = get_config()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Ajout pour confirmer le GPU
    if DEVICE == "cuda":
        print(f"🎮 GPU utilisé : {torch.cuda.get_device_name(0)}")
        print(f"   VRAM totale : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("="*80)
    print("🎯 CLASSIFIEUR MÉDICAL V2 - EFFICIENTNET-B3 (Pipeline Cascade)")
    print("="*80)
    print(f"Device        : {DEVICE}")
    print(f"Mode          : {args.mode}")
    print(f"Architecture  : {args.model}")
    print(f"Rôle          : Filtre haute sensibilité (ne jamais rater une lésion)")
    print(f"Batch Size    : {args.batch_size}")
    print(f"Learning Rate : {args.lr} (classifier) / {args.lr * args.lr_backbone_ratio:.2e} (backbone)")
    print(f"Dropout       : {args.dropout_rate}")
    print(f"Loss          : {args.loss_type.upper()} (α={args.focal_alpha}, γ={args.focal_gamma})")
    print(f"Recall cible  : {args.target_recall:.0%}")
    print(f"Sampler       : {'WeightedRandom (50/50)' if args.use_weighted_sampler else 'Standard'}")
    print(f"Optimisation  : F2-Score (Recall=2x, Precision=1x)")
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # ===== MODE TRAINING =====
    if args.mode == "train":
        print("\n" + "="*80)
        print("CHARGEMENT DES DONNÉES")
        print("="*80)
        
        # Charger les labels depuis le CSV RLE
        df_rle = pd.read_csv(args.rle_csv, dtype={"EncodedPixels": str})
        
        # Créer le DataFrame avec HasLesion
        image_ids = [f[:-4] for f in os.listdir(args.dicom_dir) if f.endswith('.dcm')]
        df = pd.DataFrame({"ImageId": image_ids})
        
        # Marquer les images avec lésions
        images_with_lesions = df_rle[df_rle["EncodedPixels"].str.strip() != "-1"]["ImageId"].unique()
        df["HasLesion"] = df["ImageId"].isin(images_with_lesions).astype(int)
        
        n_pos = df["HasLesion"].sum()
        n_neg = len(df) - n_pos
        
        print(f"✓ Total images : {len(df)}")
        print(f"✓ Avec lésions : {n_pos} ({n_pos/len(df):.1%})")
        print(f"✓ Sans lésions : {n_neg} ({n_neg/len(df):.1%})")
        
        # Split STRATIFIÉ (garder les proportions réelles dans val!)
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=args.seed, stratify=df["HasLesion"]
        )
        
        print(f"\n✓ Train : {len(train_df)} images ({train_df['HasLesion'].sum()} pos)")
        print(f"✓ Val   : {len(val_df)} images ({val_df['HasLesion'].sum()} pos)")
        print(f"  → Val garde les proportions réelles pour calibration du seuil")
        
        # Datasets
        print(f"\n📦 Création des datasets...")
        train_ds = ClassificationDataset(
            train_df, args.dicom_dir, 
            mode="train", 
            apply_augmentation=args.use_augmentation
        )
        val_ds = ClassificationDataset(
            val_df, args.dicom_dir, 
            mode="val",
            apply_augmentation=False
        )
        
        # Sampler pour équilibrer les batchs
        if args.use_weighted_sampler:
            train_sampler = create_weighted_sampler(train_ds)
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, sampler=train_sampler,
                num_workers=args.num_workers, pin_memory=True
            )
        else:
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True
            )
        
        # Validation sans sampler (proportions réelles)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        
        # ===== TRAINING =====
        print("\n" + "="*80)
        print("ENTRAÎNEMENT CLASSIFIEUR")
        print("="*80)
        
        model = EfficientNetB3Classifier(dropout_rate=args.dropout_rate)
        model.to(DEVICE)
        
        # Optimizer avec LR différentiel
        backbone_params = model.get_backbone_params()
        classifier_params = model.get_classifier_params()
        
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * args.lr_backbone_ratio, 'name': 'backbone'},
            {'params': classifier_params, 'lr': args.lr, 'name': 'classifier'}
        ], weight_decay=1e-4)
        
        print(f"\n📊 Optimiseur configuré :")
        print(f"   Backbone LR   : {args.lr * args.lr_backbone_ratio:.2e}")
        print(f"   Classifier LR : {args.lr:.2e}")
        
        # Loss
        if args.loss_type == "focal":
            criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
            print(f"   Loss: Focal (α={args.focal_alpha}, γ={args.focal_gamma})")
        else:
            criterion = WeightedBCELoss(pos_weight=args.bce_pos_weight)
            print(f"   Loss: Weighted BCE (pos_weight={args.bce_pos_weight})")
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=args.scheduler_patience, verbose=True
        )
        
        scaler = torch.amp.GradScaler('cuda') if DEVICE == "cuda" else None
        
        # Log
        log_path = "training_log_classifier.csv"
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "lr_backbone", "lr_classifier", 
                "train_loss", "train_recall", "train_precision",
                "val_loss", "val_recall", "val_precision", "val_f1",
                "optimal_threshold", "val_recall_calibrated", "val_precision_calibrated", "val_f2_calibrated"
            ])
        
        best_f2 = 0.0
        best_recall = 0.0
        best_precision = 0.0
        best_threshold = 0.5
        
        for epoch in range(1, args.epochs + 1):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch}/{args.epochs}")
            print(f"{'='*60}")
            
            lr_backbone = optimizer.param_groups[0]['lr']
            lr_classifier = optimizer.param_groups[1]['lr']
            print(f"Learning Rate: backbone={lr_backbone:.2e}, classifier={lr_classifier:.2e}")
            
            # Train
            train_metrics = train_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
            
            # Validation
            val_metrics = evaluate(model, val_loader, criterion, DEVICE)
            
            # Calibration du seuil
            threshold_info = find_optimal_threshold(
                val_metrics['targets'], val_metrics['probs'], 
                target_recall=args.target_recall
            )
            optimal_threshold = threshold_info['threshold']
            
            # Métriques avec seuil calibré
            preds_calibrated = (val_metrics['probs'] >= optimal_threshold).astype(int)
            recall_calibrated = recall_score(val_metrics['targets'], preds_calibrated, zero_division=0)
            precision_calibrated = precision_score(val_metrics['targets'], preds_calibrated, zero_division=0)
            f2_calibrated = fbeta_score(val_metrics['targets'], preds_calibrated, beta=2, zero_division=0)
            
            print(f"\nRésultats :")
            print(f"  Train - Loss: {train_metrics['loss']:.4f} | Recall: {train_metrics['recall']:.2%}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f} | Recall@0.5: {val_metrics['recall']:.2%}")
            print(f"  🎯 Seuil calibré: {optimal_threshold:.3f}")
            print(f"     Recall: {recall_calibrated:.2%} | Precision: {precision_calibrated:.2%} | F2: {f2_calibrated:.3f}")
            print(f"     ROC AUC: {threshold_info['roc_auc']:.3f}")
            
            # Log
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch, lr_backbone, lr_classifier,
                    train_metrics['loss'], train_metrics['recall'], train_metrics['precision'],
                    val_metrics['loss'], val_metrics['recall'], val_metrics['precision'], val_metrics['f1'],
                    optimal_threshold, recall_calibrated, precision_calibrated, f2_calibrated
                ])
            
            # Visualisations
            if True:
                plot_roc_curve(val_metrics['targets'], val_metrics['probs'], 
                              optimal_threshold, epoch)
                plot_confusion_matrix_classifier(
                    val_metrics['targets'], preds_calibrated, 
                    epoch, optimal_threshold
                )
                plot_threshold_analysis(val_metrics['targets'], val_metrics['probs'], epoch)
                plot_training_curves_classifier(log_path)
            
            # Sauvegarder le meilleur modèle (basé sur F2-score)
            if f2_calibrated > best_f2:
                best_f2 = f2_calibrated
                best_recall = recall_calibrated
                best_precision = precision_calibrated
                best_threshold = optimal_threshold
                
                print(f"\n✅ Nouveau meilleur modèle !")
                print(f"   F2-Score  : {f2_calibrated:.3f}")
                print(f"   Recall    : {best_recall:.2%}")
                print(f"   Precision : {best_precision:.2%}")
                print(f"   Seuil     : {best_threshold:.3f}")
                
                save_checkpoint(args.checkpoint, model, {
                    "best_f2": best_f2,
                    "best_recall": best_recall,
                    "best_precision": best_precision,
                    "best_threshold": best_threshold,
                    "roc_auc": threshold_info['roc_auc']
                })
            
            # Scheduler basé sur F2-score
            scheduler.step(f2_calibrated)
        
        print("\n" + "="*80)
        print("✅ ENTRAÎNEMENT CLASSIFIEUR TERMINÉ !")
        print("="*80)
        print(f"Meilleur F2-Score : {best_f2:.3f}")
        print(f"Meilleur Recall   : {best_recall:.2%}")
        print(f"Meilleure Precision : {best_precision:.2%}")
        print(f"Seuil optimal     : {best_threshold:.3f}")
        print(f"\n💡 Utiliser ce seuil en production pour filtrer les images")
    
    # ===== MODE PREDICTION =====
    elif args.mode == "predict":
        print("\n" + "="*80)
        print("PRÉDICTION (CLASSIFIEUR)")
        print("="*80)
        
        model, ckpt = load_checkpoint(args.checkpoint, DEVICE, dropout_rate=args.dropout_rate)
        best_threshold = ckpt.get('best_threshold', 0.5)
        print(f"✓ Modèle chargé")
        print(f"✓ Seuil calibré : {best_threshold:.3f}")
        print(f"✓ F2-Score attendu : {ckpt.get('best_f2', 'N/A')}")
        print(f"✓ Recall attendu : {ckpt.get('best_recall', 'N/A')}")
        
        image_ids = [f[:-4] for f in os.listdir(args.predict_dicom_dir) if f.endswith('.dcm')]
        df_test = pd.DataFrame({"ImageId": image_ids, "HasLesion": 0})
        
        test_ds = ClassificationDataset(
            df_test, args.predict_dicom_dir, 
            mode="predict",
            apply_augmentation=False
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        
        # Prédiction
        model.eval()
        predictions = []
        
        print("\n🔮 Classification des images...")
        with torch.no_grad():
            for batch_idx, (imgs, _, image_ids_batch) in enumerate(test_loader):
                imgs = imgs.to(DEVICE)
                logits = model(imgs)
                probs = torch.sigmoid(logits).cpu().numpy()
                
                for image_id, prob in zip(image_ids_batch, probs):
                    is_suspect = prob >= best_threshold
                    predictions.append({
                        "ImageId": image_id,
                        "Probability": prob,
                        "IsSuspect": int(is_suspect)
                    })
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"  Traité : {(batch_idx+1)*args.batch_size}/{len(test_ds)} images")
        
        df_pred = pd.DataFrame(predictions)
        output_path = os.path.join(args.output_dir, args.predictions_csv)
        df_pred.to_csv(output_path, index=False)
        
        n_suspect = df_pred["IsSuspect"].sum()
        print(f"\n✓ Prédictions sauvegardées : {output_path}")
        print(f"  Total images  : {len(df_pred)}")
        print(f"  Suspectes     : {n_suspect} ({n_suspect/len(df_pred):.1%}) → à envoyer au segmenteur")
        print(f"  Saines        : {len(df_pred) - n_suspect} ({1-n_suspect/len(df_pred):.1%}) → masque vide")


if __name__ == "__main__":
    main()