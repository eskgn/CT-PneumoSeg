# Détection de pneumothorax par pipeline en cascade

Le projet vise à détecter les pneumothorax sur imagerie médicale et à générer un masque précis de la région pathologique.

## 🎯 Stratégies d'entraînement

Pour éviter de faire tourner un modèle de segmentation lourd sur chaque examen (la majorité étant sains), l'architecture repose sur une cascade à **deux étapes** :
1. **Le classifieur** : analyse l'image et écarte immédiatement les scans normaux.
2. **Le segmenter** : traite uniquement les cas suspects pour délimiter la lésion. Si le classifieur a produit un faux positif, le segmenter peut encore renvoyer un masque vide pour corriger l'erreur.


```
┌─────────────────────────────────────────────────────────────────────────┐
│ PIPELINE EN CASCADE                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ Image ──► [CLASSIFIEUR] ──► Suspect? ──OUI──► [SEGMENTER] ──► Masque    │
│                                   │                      │              │
│                                   └───────► NON ──► Vide └──► Vide      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Architecture

### Étape de classification

**Modèle :** EfficientNet-B3  
**Rôle :** Classification binaire (présence ou absence de lésion)  
**Priorité :** Sensibilité maximale (rappel élevé) pour limiter strictement les faux négatifs  

**Spécifications :**
- **Entrée :** Niveaux de gris 512×512
- **Sortie :** Score de probabilité entre 0 et 1
- **Fonction de perte :** Focal Loss (alpha = 0,75, gamma = 2,0)
- **Optimisation :** Score F2 (rappel pondéré deux fois plus que la précision)
- **Entraînement :** Échantillonnage pondéré (*Weighted Random Sampler*) pour des lots équilibrés

**Calibration du seuil :**  
Le seuil du classifieur est ajusté pour atteindre environ 95 % de rappel. Quelques faux positifs sont acceptés car :

- Ils sont éliminés par le segmentateur à l'étape 2.
- La majorité des images saines est tout de même correctement filtrée dès l'étape 1.


### Étape de segmentation

**Modèle :** U-Net avec encodeur ConvNeXt-Tiny  
**Rôle :** Segmentation au pixel (localisation précise de la lésion)  
**Priorité :** Localisation exacte avec un minimum de faux positifs  

**Composants du segmentateur :**
- **Encodeur :** ConvNeXt-Tiny (pré-entraîné sur ImageNet)
- **Décodeur :** Blocs de convolution résiduels avec connexions directes
- **Entrée :** Niveaux de gris 512×512
- **Sortie :** Masque de segmentation 512×512
- **Fonction de perte :** Perte combinée (BCE + Batch Dice)
- **Activation :** GELU

### Batch Dice

Au lieu de calculer le coefficient Dice image par image (ce qui attribue un score parfait de 1,0 lorsque le masque réel et la prédiction sont tous deux vides), le calcul est effectué globalement sur l'ensemble du lot. Cela empêche le modèle d'apprendre à prédire systématiquement des masques vides pour les images saines.

## 🔧 Configuration

Tous les seuils sont configurables dans `PipelineConfig`:

```python
class PipelineConfig:
    # Seuil du classifieur (calibré pour un rappel élevé)
    CLASSIFIER_THRESHOLD = 0.28  # plus bas = plus sensible

    # Seuil de probabilité du segmentateur
    SEGMENTER_THRESHOLD = 0.94

    # Taille minimale de la lésion (pixels)
    MIN_PIXELS = 100  # filtre le bruit et les artefacts
```

## 🗂️ Project Structure

```
├── classifier_efficientnet_b3.py   # Étape 1 : Classifieur binaire
├── segmenter_convnext_tiny.py      # Étape 2 : Segmenter U-Net
├── pipeline_inference.py           # Pipeline d'inférence combiné
├── visualizations/                 # Courbes d'entraînement, matrices de confusion
└── outputs/                        # Fichiers CSV de prédictions
```

- [EfficientNet: Rethinking Model Scaling](https://arxiv.org/abs/1905.11946)
- [ConvNeXt: A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
