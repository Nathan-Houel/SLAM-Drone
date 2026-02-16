"""
SLAM 3D — Reconstruction 3D a partir d'images de drone DJI Tello

Auteurs : Youri Larose - Maxence Robineau - Nathan Houel
"""

import sys
import time
import struct
import shutil
import argparse
import threading
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

CHEMIN_IMAGES_DEFAUT    = r"Images\parcours_1"
DOSSIER_SORTIE_DEFAUT   = r"output"

MODELE_CAMERA           = "OPENCV"
UTILISER_GPU            = True
GPU_INDEX               = "0"

#  Paramètres SIFT
SIFT_MAX_FEATURES       = 8192   # Nombre max de features par image
SIFT_FIRST_OCTAVE       = -1     # -1 → upscale ×2 (crucial pour basse résolution)
SIFT_NUM_OCTAVES        = 4      # Nombre d'octaves
SIFT_OCTAVE_RESOLUTION  = 3      # Images par octave
SIFT_PEAK_THRESHOLD     = 0.004  # Seuil de détection (bas = plus de features)
SIFT_EDGE_THRESHOLD     = 10     # Seuil d'arête

#  Matching séquentiel
MATCHING_OVERLAP        = 15     # Nb d'images voisines à comparer
MATCHING_LOOP_DETECTION = False  # Détection de boucle (nécessite vocab tree)

#  Mapper (reconstruction)
MAPPER_MIN_NUM_MATCHES  = 10     # Matches min pour une paire

CROP_TOP_PERCENT        = 25     # Pourcentage du haut de l'image a couper
FILTRE_DISTANCE_MAX     = 0      # Rayon max autour de la trajectoire


# =============================================================================
# UTILITAIRES
# =============================================================================

def lister_images(dossier: Path, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tiff")) -> list:
    return sorted(f for f in dossier.iterdir() if f.is_file() and f.suffix.lower() in extensions)


def ecrire_ply(chemin: Path, points: np.ndarray, couleurs: np.ndarray = None):
    n = len(points)
    with open(chemin, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
        )
        if couleurs is not None and len(couleurs) == n:
            header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        header += "end_header\n"
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(struct.pack("<fff", *points[i].astype(np.float32)))
            if couleurs is not None and len(couleurs) == n:
                f.write(struct.pack("<BBB", *couleurs[i].astype(np.uint8)))


def extraire_nuage_sparse(reconstruction):
    pts, cols = [], []
    for _, p in reconstruction.points3D.items():
        pts.append(p.xyz)
        cols.append(p.color)
    if not pts:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    return np.array(pts, dtype=np.float64), np.array(cols, dtype=np.uint8)


def extraire_poses_cameras(reconstruction):
    return [np.array(img.projection_center()) for _, img in reconstruction.images.items()]


def spinner(label, t_start, stop_event):
    symboles = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']
    i = 0
    while not stop_event.is_set():
        elapsed = time.time() - t_start
        print(f"\r  {symboles[i % len(symboles)]} {label} ... {elapsed:.0f}s", end="", flush=True)
        i += 1
        stop_event.wait(0.15)
    elapsed = time.time() - t_start
    print(f"\r  ✓ {label} termine en {elapsed:.1f}s" + " " * 20)


# =============================================================================
# ETAPES DU PIPELINE
# =============================================================================

def etape_crop_images(image_path, output_path, crop_top_pct):
    print(f"\n{'='*60}\nETAPE 0 : Crop des images ({crop_top_pct}% du haut)\n{'='*60}")
    cropped_path = output_path / "images_cropped"
    cropped_path.mkdir(parents=True, exist_ok=True)
    images = lister_images(image_path)

    for img_file in tqdm(images, desc="  Crop", unit="img", ncols=80):
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        h, w = img.shape[:2]
        cropped = img[int(h * crop_top_pct / 100.0):, :, :]
        cv2.imwrite(str(cropped_path / img_file.name), cropped)

    return cropped_path


def etape_extraction_features(database_path, image_path):
    import pycolmap
    print(f"\n{'='*60}\nETAPE 1 : Extraction SIFT\n{'='*60}")

    opts = pycolmap.FeatureExtractionOptions()
    opts.sift.max_num_features  = SIFT_MAX_FEATURES
    opts.sift.first_octave      = SIFT_FIRST_OCTAVE
    opts.sift.num_octaves       = SIFT_NUM_OCTAVES
    opts.sift.octave_resolution = SIFT_OCTAVE_RESOLUTION
    opts.sift.peak_threshold    = SIFT_PEAK_THRESHOLD
    opts.sift.edge_threshold    = SIFT_EDGE_THRESHOLD
    opts.use_gpu   = UTILISER_GPU
    opts.gpu_index = GPU_INDEX

    t0 = time.time()
    stop = threading.Event()
    t = threading.Thread(target=spinner, args=("Extraction SIFT", t0, stop), daemon=True)
    t.start()

    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(image_path),
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model=MODELE_CAMERA,
        extraction_options=opts,
    )

    stop.set(); t.join()


def etape_matching(database_path):
    import pycolmap
    print(f"\n{'='*60}\nETAPE 2 : Matching sequentiel (overlap={MATCHING_OVERLAP})\n{'='*60}")

    match_opts = pycolmap.FeatureMatchingOptions()
    match_opts.max_num_matches = 32768
    match_opts.use_gpu   = UTILISER_GPU
    match_opts.gpu_index = GPU_INDEX

    pair_opts = pycolmap.SequentialPairingOptions()
    pair_opts.overlap = MATCHING_OVERLAP
    pair_opts.quadratic_overlap = True
    pair_opts.loop_detection = MATCHING_LOOP_DETECTION

    t0 = time.time()
    stop = threading.Event()
    t = threading.Thread(target=spinner, args=("Matching", t0, stop), daemon=True)
    t.start()

    try:
        pycolmap.match_sequential(
            database_path=str(database_path),
            matching_options=match_opts,
            pairing_options=pair_opts,
        )
    except Exception:
        stop.set(); t.join()
        print("  Fallback -> matching exhaustif")
        t0 = time.time()
        stop = threading.Event()
        t = threading.Thread(target=spinner, args=("Matching exhaustif", t0, stop), daemon=True)
        t.start()
        pycolmap.match_exhaustive(database_path=str(database_path), matching_options=match_opts)

    stop.set(); t.join()


def etape_reconstruction(database_path, image_path, sparse_path):
    import pycolmap
    print(f"\n{'='*60}\nETAPE 3 : Reconstruction sparse\n{'='*60}")

    sparse_path.mkdir(parents=True, exist_ok=True)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.min_num_matches = MAPPER_MIN_NUM_MATCHES
    opts.multiple_models = True

    t0 = time.time()
    stop = threading.Event()
    t = threading.Thread(target=spinner, args=("Reconstruction 3D", t0, stop), daemon=True)
    t.start()

    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(image_path),
        output_path=str(sparse_path),
        options=opts,
    )

    stop.set(); t.join()

    if not reconstructions:
        raise RuntimeError("Reconstruction echouee — aucun modele cree.")

    best_id = max(reconstructions, key=lambda i: reconstructions[i].num_reg_images())
    best = reconstructions[best_id]
    print(f"  Meilleur modele : #{best_id} — {best.num_reg_images()} images, {best.num_points3D()} points")

    for idx, rec in reconstructions.items():
        d = sparse_path / str(idx)
        d.mkdir(parents=True, exist_ok=True)
        rec.write(d)

    return best


def etape_export_ply(reconstruction, sparse_path):
    print(f"\n{'='*60}\nETAPE 4 : Export PLY\n{'='*60}")
    ply_path = sparse_path / "nuage_sparse.ply"

    try:
        reconstruction.export_PLY(str(ply_path))
    except Exception:
        points, couleurs = extraire_nuage_sparse(reconstruction)
        if len(points) == 0:
            raise RuntimeError("Aucun point 3D a exporter.")
        ecrire_ply(ply_path, points, couleurs)

    poses = extraire_poses_cameras(reconstruction)
    if poses:
        traj_path = sparse_path / "trajectoire_drone.ply"
        ecrire_ply(traj_path, np.array(poses), np.full((len(poses), 3), [255, 0, 0], dtype=np.uint8))

    print(f"  {reconstruction.num_points3D()} points exportes")
    return ply_path


def etape_filtrage_distance(reconstruction, sparse_path, max_dist):
    print(f"\n{'='*60}\nETAPE 4b : Filtrage par distance\n{'='*60}")

    poses = extraire_poses_cameras(reconstruction)
    if not poses:
        return sparse_path / "nuage_sparse.ply"

    points, couleurs = extraire_nuage_sparse(reconstruction)
    if len(points) == 0:
        return sparse_path / "nuage_sparse.ply"

    from scipy.spatial import cKDTree
    distances, _ = cKDTree(np.array(poses)).query(points, k=1)

    if max_dist <= 0:
        q25, q50, q75 = np.percentile(distances, [25, 50, 75])
        max_dist = q75 + 1.5 * (q75 - q25)
        print(f"  Seuil auto : {max_dist:.2f}")

    mask = distances <= max_dist
    pts_f, col_f = points[mask], couleurs[mask]
    nb_suppr = len(points) - len(pts_f)
    print(f"  {len(points)} -> {len(pts_f)} points ({nb_suppr} supprimes, {nb_suppr*100/len(points):.1f}%)")

    ply_out = sparse_path / "nuage_sparse_filtre.ply"
    ecrire_ply(ply_out, pts_f, col_f)
    return ply_out


# =============================================================================
# VISUALISATION
# =============================================================================

def visualiser_nuage(ply_path, trajectoire_path=None):
    import pyvista as pv
    from scipy.interpolate import CubicSpline

    if not ply_path.exists():
        print(f"  Fichier introuvable : {ply_path}")
        return

    nuage = pv.read(str(ply_path))
    if nuage.n_points == 0:
        print("  Nuage vide.")
        return

    plotter = pv.Plotter(window_size=[1280, 720])
    plotter.set_background("black")

    has_rgb = any("rgb" in n.lower() or "color" in n.lower() or "red" in n.lower() for n in nuage.array_names)
    if has_rgb:
        plotter.add_mesh(nuage, scalars=nuage.array_names[0], rgb=True, point_size=3.0, render_points_as_spheres=False, label="Nuage de points")
    else:
        plotter.add_mesh(nuage, color="white", point_size=3.0, render_points_as_spheres=False, label="Nuage de points")

    if trajectoire_path and trajectoire_path.exists():
        traj = pv.read(str(trajectoire_path))
        if traj.n_points > 0:
            plotter.add_mesh(traj, color="red", point_size=8.0, render_points_as_spheres=True, label="Trajectoire drone")

    plotter.add_legend(bcolor=(30, 30, 30), face="circle", size=(0.15, 0.15))

    plotter.add_title("Reconstruction 3D Parcours 2", font_size=10, color="white")
    plotter.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    global MATCHING_OVERLAP, SIFT_MAX_FEATURES, UTILISER_GPU, GPU_INDEX
    global CROP_TOP_PERCENT, FILTRE_DISTANCE_MAX

    parser = argparse.ArgumentParser(description="SLAM 3D — Drone Tello")
    parser.add_argument("--images", "-i", default=CHEMIN_IMAGES_DEFAUT)
    parser.add_argument("--output", "-o", default=DOSSIER_SORTIE_DEFAUT)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--viz-only", type=str, default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overlap", type=int, default=MATCHING_OVERLAP)
    parser.add_argument("--max-features", type=int, default=SIFT_MAX_FEATURES)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--gpu-index", type=str, default=GPU_INDEX)
    parser.add_argument("--crop-top", type=float, default=CROP_TOP_PERCENT)
    parser.add_argument("--max-dist", type=float, default=FILTRE_DISTANCE_MAX)
    args = parser.parse_args()

    MATCHING_OVERLAP    = args.overlap
    SIFT_MAX_FEATURES   = args.max_features
    UTILISER_GPU        = not args.no_gpu
    GPU_INDEX           = args.gpu_index
    CROP_TOP_PERCENT    = args.crop_top
    FILTRE_DISTANCE_MAX = args.max_dist

    base_dir      = Path(__file__).parent
    image_path    = base_dir / args.images
    output_path   = base_dir / args.output
    sparse_path   = output_path / "sparse"
    database_path = output_path / "database.db"

    # Mode visualisation seule
    if args.viz_only:
        ply = Path(args.viz_only)
        if not ply.exists():
            ply = base_dir / args.viz_only
        traj = ply.parent / "trajectoire_drone.ply"
        visualiser_nuage(ply, traj if traj.exists() else None)
        return

    # Nettoyage
    if args.clean and output_path.exists():
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    sparse_path.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()

    # Verifications
    if not image_path.exists():
        print(f"Dossier d'images introuvable : {image_path}")
        sys.exit(1)

    images = lister_images(image_path)
    if len(images) < 3:
        print(f"Seulement {len(images)} image(s). Minimum 3 requises.")
        sys.exit(1)

    print(f"\n  Images : {len(images)} | GPU : {'oui' if UTILISER_GPU else 'non'}")

    t_debut = time.time()

    try:
        # Etape 0 : Crop
        img_sfm = image_path
        if CROP_TOP_PERCENT > 0:
            img_sfm = etape_crop_images(image_path, output_path, CROP_TOP_PERCENT)

        # Etape 1 : Extraction SIFT
        etape_extraction_features(database_path, img_sfm)

        # Etape 2 : Matching
        etape_matching(database_path)

        # Etape 3 : Reconstruction
        reconstruction = etape_reconstruction(database_path, img_sfm, sparse_path)

        # Etape 4 : Export PLY
        etape_export_ply(reconstruction, sparse_path)

        # Etape 4b : Filtrage distance
        ply_filtre = etape_filtrage_distance(reconstruction, sparse_path, FILTRE_DISTANCE_MAX)

        duree = time.time() - t_debut
        print(f"\n{'='*60}\n  Pipeline termine en {duree:.1f}s ({duree/60:.1f} min)\n{'='*60}")

        # Etape 5 : Visualisation
        if not args.no_viz:
            traj_path = sparse_path / "trajectoire_drone.ply"
            visualiser_nuage(ply_filtre, traj_path if traj_path.exists() else None)

    except RuntimeError as e:
        print(f"\nERREUR : {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrompu.")
        sys.exit(130)


if __name__ == "__main__":
    main()
