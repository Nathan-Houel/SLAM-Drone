# SLAM 3D — Reconstruction et fusion de nuages de points (drone DJI Tello)

## Description

Pipeline de reconstruction 3D par **Structure-from-Motion (SfM)** à partir d'images capturées par un drone DJI Tello, avec **fusion multi-parcours par ICP**. Utilise **pycolmap** (bindings Python de COLMAP) pour le SfM et **PyVista** pour la visualisation interactive.

Pour une description détaillée de chaque étape et des choix techniques, voir [description.md](description.md).

## Prérequis

- **Python** 3.9+
- **Windows** 10/11
- (Optionnel) **GPU NVIDIA avec CUDA** pour accélérer l'extraction SIFT et le matching

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Scripts

### 1. `slam_reconstruction.py` — Reconstruction SfM

Exécute le pipeline complet : crop des images, extraction SIFT, matching séquentiel, reconstruction sparse, export PLY et visualisation.

```powershell
# Reconstruction complète
python slam_reconstruction.py --clean

# Choisir un parcours différent
python slam_reconstruction.py --images Images/parcours_2 --output output_p2 --clean

# Sans visualisation
python slam_reconstruction.py --clean --no-viz

# Visualiser un résultat existant
python slam_reconstruction.py --viz-only output/sparse/nuage_sparse_filtre.ply

# Ajuster le filtrage de distance
python slam_reconstruction.py --clean --max-dist 5.0

# Désactiver le crop
python slam_reconstruction.py --clean --crop-top 0
```

#### Options

| Option           | Défaut            | Description                                          |
|------------------|-------------------|------------------------------------------------------|
| `--images`, `-i` | `Images/parcours_2` | Chemin vers le dossier d'images                   |
| `--output`, `-o` | `output2`         | Dossier de sortie                                    |
| `--clean`        | —                 | Supprimer le dossier de sortie avant de relancer     |
| `--no-viz`       | —                 | Ne pas lancer la visualisation 3D                    |
| `--viz-only`     | —                 | Visualiser un fichier PLY existant                   |
| `--overlap`      | `15`              | Nb d'images voisines pour le matching séquentiel     |
| `--max-features` | `8192`            | Nb max de features SIFT par image                    |
| `--no-gpu`       | —                 | Forcer le CPU                                        |
| `--gpu-index`    | `0`               | Index du GPU à utiliser                              |
| `--crop-top`     | `25`              | % du haut de l'image à couper (0 = désactivé)       |
| `--max-dist`     | `0`               | Rayon max de filtrage (0 = auto via box-plot Q75+1.5×IQR) |

---

### 2. `merge_icp.py` — Fusion de nuages par ICP

Fusionne deux nuages de points `.ply` issus de parcours différents via un recalage **ICP multi-échelle** (Iterative Closest Point). Implémenté sans Open3D, utilise uniquement scipy, numpy et pyvista.

```powershell
# Fusion de base avec visualisation
python merge_icp.py output/sparse/nuage_sparse_filtre.ply output2/sparse/nuage_sparse_filtre.ply --viz

# Spécifier un fichier de sortie
python merge_icp.py output/sparse/nuage_sparse_filtre.ply output2/sparse/nuage_sparse_filtre.ply -o resultat.ply

# Avec sous-échantillonnage pour accélérer l'ICP
python merge_icp.py output/sparse/nuage_sparse_filtre.ply output2/sparse/nuage_sparse_filtre.ply --voxel-size 0.05 --viz

# Sans alignement initial (ICP brut)
python merge_icp.py output/sparse/nuage_sparse_filtre.ply output2/sparse/nuage_sparse_filtre.ply --no-initial --viz
```

#### Options

| Option           | Défaut            | Description                                          |
|------------------|-------------------|------------------------------------------------------|
| `ply1`           | —                 | Nuage cible (référence)                              |
| `ply2`           | —                 | Nuage source (sera aligné sur le premier)            |
| `--output`, `-o` | `merged_icp.ply`  | Fichier de sortie                                    |
| `--voxel-size`   | `0.0`             | Taille du voxel pour sous-échantillonner avant ICP   |
| `--max-iter`     | `200`             | Itérations max pour ICP                              |
| `--tolerance`    | `1e-7`            | Seuil de convergence                                 |
| `--max-dist`     | `0`               | Distance max pour rejeter les outliers (0 = auto)    |
| `--traj1`        | auto              | Trajectoire du nuage cible (`.ply`, auto-détectée)   |
| `--traj2`        | auto              | Trajectoire du nuage source (`.ply`, auto-détectée)  |
| `--traj-points`  | `10`              | Nb de points de trajectoire pour l'alignement initial|
| `--no-initial`   | —                 | Pas d'alignement initial                             |
| `--dedup`        | `0.0`             | Taille voxel pour déduplications après fusion        |
| `--viz`          | —                 | Visualiser avant/après + résultat fusionné           |

#### Fonctionnalités

- **Pré-rotations automatiques** des nuages pour corriger l'orientation (flip vertical/horizontal)
- **Alignement initial par trajectoires** (fin traj1 ↔ début traj2), fallback PCA ou centroïdes
- **ICP multi-échelle** (4 passes : voxel×4, ×2, ×1, résolution complète) avec rejet progressif d'outliers
- **Mise sur le même plan Z** avant ICP pour garantir la cohérence spatiale
- **Visualisation 3 fenêtres** : avant ICP, après ICP, nuage fusionné (avec trajectoires)

## Fichiers générés

```
output/                         # Parcours 1
├── database.db                 # Base COLMAP (features + matches)
├── images_cropped/             # Images après crop du haut
├── sparse/
│   ├── 0/                      # Modèle sparse (format COLMAP binaire)
│   │   ├── cameras.bin
│   │   ├── images.bin
│   │   └── points3D.bin
│   ├── nuage_sparse.ply        # Nuage de points 3D complet
│   ├── nuage_sparse_filtre.ply # Nuage filtré (points proches uniquement)
│   └── trajectoire_drone.ply   # Trajectoire du drone

output2/                        # Parcours 2 (même structure)
├── ...

merged_icp.ply                  # Nuage fusionné (sortie de merge_icp.py)
```

## Contrôles de la visualisation 3D

| Contrôle              | Action      |
|-----------------------|-------------|
| Clic gauche + glisser | Rotation    |
| Molette               | Zoom        |
| Clic droit + glisser  | Translation |
| `R`                   | Reset vue   |
| `Q`                   | Quitter     |

## Résolution de problèmes

| Problème | Solution |
|----------|----------|
| Aucun match trouvé | Augmenter `--overlap` (20-25) ou vérifier les images (flou, trop peu) |
| Reconstruction vide | Minimum ~20 images nécessaires, le drone doit se déplacer |
| pycolmap ne s'installe pas | `pip install pycolmap --only-binary :all:` |
| Trajectoire non fermée (drift) | Normal en matching séquentiel — voir [description.md](description.md) pour les détails |
| ICP ne converge pas bien | Essayer `--voxel-size 0.05` ou ajuster `--traj-points` |
