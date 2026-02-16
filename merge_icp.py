"""
Fusion de deux nuages de points .ply par recalage ICP (Iterative Closest Point).

Implementation sans Open3D — utilise uniquement scipy, numpy et pyvista.

Usage :
    python merge_icp.py <fichier1.ply> <fichier2.ply> [options]

Exemple :
    python merge_icp.py output/sparse/nuage_sparse_filtre.ply output2/sparse/nuage_sparse_filtre.ply --viz
    python merge_icp.py output/sparse/nuage_sparse_filtre.ply output2/sparse/nuage_sparse_filtre.ply -o merged_icp.ply --max-iter 100
"""

import argparse
import sys
import struct
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import pyvista as pv


# =============================================================================
# CHARGEMENT / SAUVEGARDE
# =============================================================================

def load_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Charge un .ply et retourne (points Nx3, couleurs Nx3 ou None)."""
    path = Path(path)
    if not path.exists():
        print(f"Erreur : le fichier '{path}' n'existe pas.")
        sys.exit(1)
    cloud = pv.read(str(path))
    points = np.array(cloud.points, dtype=np.float64)

    # Chercher les couleurs RGB
    colors = None
    for key in cloud.point_data.keys():
        if "rgb" in key.lower() or "color" in key.lower() or "red" in key.lower():
            colors = np.array(cloud.point_data[key], dtype=np.uint8)
            break

    print(f"  {path.name} : {len(points)} points")
    return points, colors


def save_ply(path: Path, points: np.ndarray, colors: np.ndarray = None):
    """Sauvegarde un nuage au format PLY binaire."""
    n = len(points)
    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
        )
        if colors is not None and len(colors) == n:
            header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        header += "end_header\n"
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(struct.pack("<fff", *points[i].astype(np.float32)))
            if colors is not None and len(colors) == n:
                f.write(struct.pack("<BBB", *colors[i].astype(np.uint8)))


# =============================================================================
# SOUS-ECHANTILLONNAGE
# =============================================================================

def voxel_downsample(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Sous-echantillonne par voxel. Retourne (points_down, indices)."""
    if voxel_size <= 0:
        return points, np.arange(len(points))

    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)

    return points[unique_idx], unique_idx


# =============================================================================
# MISE A PLAT (PCA)
# =============================================================================

def flatten_to_plane(points: np.ndarray):
    """Projette le nuage sur le plan XY en alignant la normale du plan
    dominant (direction de plus faible variance) sur l'axe Z.

    Retourne (points_aplatis, R, centroid) pour pouvoir appliquer
    la meme transformation a d'autres donnees (ex: trajectoire).
    """
    centroid = points.mean(axis=0)
    centered = points - centroid

    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)   # tri croissant
    # eigenvalues[0] -> plus petite variance = normale du plan dominant
    normal = eigenvectors[:, 0]
    v1     = eigenvectors[:, 1]
    v2     = eigenvectors[:, 2]   # plus grande variance

    # Matrice de rotation : v2 -> X, v1 -> Y, normal -> Z
    R = np.array([v2, v1, normal], dtype=np.float64)

    # Assurer un repere direct (det = +1)
    if np.linalg.det(R) < 0:
        R[2, :] *= -1

    aligned = (R @ centered.T).T + centroid
    return aligned, R, centroid


def apply_plane_transform(pts, R, centroid):
    """Applique la meme transformation de mise a plat a un autre jeu de points."""
    return (R @ (pts - centroid).T).T + centroid


# =============================================================================
# ICP
# =============================================================================

def best_fit_transform(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule la meilleure transformation rigide (R, t) qui aligne A sur B.
    A, B : Nx3
    Retourne R (3x3), t (3,)
    """
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Correction de reflexion
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A
    return R, t


def icp(source: np.ndarray, target: np.ndarray,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        max_dist: float = None) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Iterative Closest Point.

    Parametres :
        source : Nx3 — nuage a aligner
        target : Mx3 — nuage de reference
        max_iterations : nombre max d'iterations
        tolerance : seuil de convergence sur la variation d'erreur
        max_dist : distance max pour rejeter les outliers (None = pas de rejet)

    Retourne :
        R : 3x3 — matrice de rotation finale
        t : 3   — vecteur de translation final
        errors : liste des erreurs RMS a chaque iteration
    """
    src = source.copy()
    tree = cKDTree(target)

    R_total = np.eye(3)
    t_total = np.zeros(3)
    errors = []

    for i in range(max_iterations):
        # 1. Trouver les correspondances les plus proches
        distances, indices = tree.query(src, k=1)

        # 2. Rejeter les outliers si max_dist est specifie
        if max_dist is not None:
            mask = distances <= max_dist
            if mask.sum() < 10:
                print(f"    Iteration {i+1}: trop peu d'inliers ({mask.sum()}), arret.")
                break
            src_matched = src[mask]
            tgt_matched = target[indices[mask]]
            rms = np.sqrt(np.mean(distances[mask] ** 2))
        else:
            src_matched = src
            tgt_matched = target[indices]
            rms = np.sqrt(np.mean(distances ** 2))

        errors.append(rms)

        # 3. Calculer la transformation optimale
        R_step, t_step = best_fit_transform(src_matched, tgt_matched)

        # 4. Appliquer la transformation
        src = (R_step @ src.T).T + t_step

        # Accumuler
        R_total = R_step @ R_total
        t_total = R_step @ t_total + t_step

        # 5. Verifier la convergence
        if i > 0 and abs(errors[-1] - errors[-2]) < tolerance:
            print(f"    Convergence atteinte a l'iteration {i+1} (delta={abs(errors[-1] - errors[-2]):.2e})")
            break

    return R_total, t_total, errors


def compute_initial_alignment(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Alignement initial par PCA (rotation + translation)."""
    centroid_s = source.mean(axis=0)
    centroid_t = target.mean(axis=0)
    _, _, Vt_s = np.linalg.svd(source - centroid_s, full_matrices=False)
    _, _, Vt_t = np.linalg.svd(target - centroid_t, full_matrices=False)
    R = Vt_t.T @ Vt_s
    if np.linalg.det(R) < 0:
        Vt_s[-1, :] *= -1
        R = Vt_t.T @ Vt_s
    t = centroid_t - R @ centroid_s
    return R, t


def compute_initial_alignment_simple(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Alignement initial par translation des centroides uniquement."""
    t = target.mean(axis=0) - source.mean(axis=0)
    return np.eye(3), t


def compute_trajectory_alignment(traj1_pts: np.ndarray, traj2_pts: np.ndarray,
                                  n_points: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Alignement initial par trajectoires.

    Calcule la transformation rigide (R, t) qui aligne le debut de la
    trajectoire 2 sur la fin de la trajectoire 1.

    Utilise les N derniers points de traj1 et les N premiers de traj2
    pour estimer la rotation, puis translate pour faire coincider
    le premier point de traj2 avec le dernier point de traj1.
    """
    # Nombre de points a utiliser pour l'estimation
    n = min(n_points, len(traj1_pts), len(traj2_pts))

    if n >= 3:
        # Utiliser les N derniers/premiers points pour estimer R et t
        end_traj1 = traj1_pts[-n:]    # fin de la trajectoire 1
        start_traj2 = traj2_pts[:n]   # debut de la trajectoire 2
        R, t = best_fit_transform(start_traj2, end_traj1)
    else:
        # Pas assez de points : translation pure
        R = np.eye(3)
        t = traj1_pts[-1] - traj2_pts[0]

    return R, t


# =============================================================================
# VISUALISATION
# =============================================================================

def visualiser(target_pts, target_col, source_pts, source_col,
               aligned_pts, merged_pts, merged_col,
               traj1=None, traj2=None, traj2_aligned=None):
    """Affiche avant/apres alignement + resultat fusionne."""

    # Fenetre 1 : Avant alignement
    print("\n  [Fenetre 1] Avant alignement (rouge = source, bleu = cible)")
    p1 = pv.Plotter(window_size=[1280, 720])
    p1.set_background("black")
    p1.add_points(pv.PolyData(target_pts), color="dodgerblue", point_size=2, label="Nuage 1")
    p1.add_points(pv.PolyData(source_pts), color="red", point_size=2, label="Nuage 2")
    if traj1 is not None:
        p1.add_points(pv.PolyData(traj1), color="cyan", point_size=6,
                      render_points_as_spheres=True, label="Traj 1")
    if traj2 is not None:
        p1.add_points(pv.PolyData(traj2), color="orange", point_size=6,
                      render_points_as_spheres=True, label="Traj 2")
    p1.add_legend(bcolor=(30, 30, 30), face="circle", size=(0.15, 0.18))
    p1.add_title("Avant ICP", font_size=10, color="white")
    p1.show()

    # Fenetre 2 : Apres alignement
    print("  [Fenetre 2] Apres alignement (vert = source aligne, bleu = cible)")
    p2 = pv.Plotter(window_size=[1280, 720])
    p2.set_background("black")
    p2.add_points(pv.PolyData(target_pts), color="dodgerblue", point_size=2, label="Nuage 1")
    p2.add_points(pv.PolyData(aligned_pts), color="lime", point_size=2, label="Nuage 2 aligne")
    if traj1 is not None:
        p2.add_points(pv.PolyData(traj1), color="cyan", point_size=6,
                      render_points_as_spheres=True, label="Traj 1")
    if traj2_aligned is not None:
        p2.add_points(pv.PolyData(traj2_aligned), color="yellow", point_size=6,
                      render_points_as_spheres=True, label="Traj 2 alignee")
    p2.add_legend(bcolor=(30, 30, 30), face="circle", size=(0.15, 0.18))
    p2.add_title("Apres ICP", font_size=10, color="white")
    p2.show()

    # Fenetre 3 : Nuage fusionne
    print("  [Fenetre 3] Nuage fusionne")
    p3 = pv.Plotter(window_size=[1280, 720])
    p3.set_background("black")
    cloud = pv.PolyData(merged_pts)
    if merged_col is not None:
        cloud.point_data["RGB"] = merged_col
        p3.add_mesh(cloud, scalars="RGB", rgb=True, point_size=2,
                    render_points_as_spheres=False, label="Nuage fusionne")
    else:
        p3.add_points(cloud, color="white", point_size=2, label="Nuage fusionne")
    # Trajectoire fusionnee
    if traj1 is not None and traj2_aligned is not None:
        traj_merged = np.vstack([traj1, traj2_aligned])
        p3.add_points(pv.PolyData(traj_merged), color="red", point_size=6,
                      render_points_as_spheres=True, label="Trajectoire")
    p3.add_legend(bcolor=(30, 30, 30), face="circle", size=(0.15, 0.12))
    p3.add_title("Nuage fusionne (ICP)", font_size=10, color="white")
    p3.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fusion de nuages de points par ICP (sans Open3D)"
    )
    parser.add_argument("ply1", help="Nuage cible (reference)")
    parser.add_argument("ply2", help="Nuage source (sera aligne sur le premier)")
    parser.add_argument("--output", "-o", default="merged_icp.ply",
                        help="Fichier de sortie (defaut: merged_icp.ply)")
    parser.add_argument("--voxel-size", type=float, default=0.0,
                        help="Taille du voxel pour sous-echantillonner avant ICP (0 = desactive)")
    parser.add_argument("--max-iter", type=int, default=200,
                        help="Iterations max pour ICP (defaut: 200)")
    parser.add_argument("--tolerance", type=float, default=1e-7,
                        help="Seuil de convergence (defaut: 1e-7)")
    parser.add_argument("--max-dist", type=float, default=0,
                        help="Distance max pour rejeter les outliers (0 = auto)")
    parser.add_argument("--traj1", type=str, default=None,
                        help="Trajectoire du nuage cible (.ply). Auto-detecte si absent.")
    parser.add_argument("--traj2", type=str, default=None,
                        help="Trajectoire du nuage source (.ply). Auto-detecte si absent.")
    parser.add_argument("--traj-points", type=int, default=10,
                        help="Nb de points de trajectoire a utiliser pour l'alignement (defaut: 10)")
    parser.add_argument("--no-initial", action="store_true",
                        help="Pas d'alignement initial")
    parser.add_argument("--dedup", type=float, default=0.0,
                        help="Taille voxel pour deduplication apres fusion (0 = desactive)")
    parser.add_argument("--viz", action="store_true",
                        help="Visualiser avant/apres + resultat fusionne")
    args = parser.parse_args()

    # ---- Chargement ----
    print("\n" + "=" * 60)
    print("ETAPE 1 : Chargement des nuages")
    print("=" * 60)
    target_pts, target_col = load_ply(args.ply1)
    source_pts, source_col = load_ply(args.ply2)

    # Rotation 180° de la cible autour de X (flip vertical : tete en bas -> tete en haut)
    centroid_tgt = target_pts.mean(axis=0)
    R_flip_tgt = np.array([[1, 0, 0],
                           [0,-1, 0],
                           [0, 0,-1]], dtype=np.float64)
    target_pts = (R_flip_tgt @ (target_pts - centroid_tgt).T).T + centroid_tgt
    print(f"  Rotation 180° (axe X) appliquee au nuage cible (flip vertical)")

    # Rotation de 180° du nuage source autour de son centroide (axes Y + X)
    centroid_src = source_pts.mean(axis=0)
    R180 = np.array([[-1, 0, 0],
                     [ 0,-1, 0],
                     [ 0, 0, 1]], dtype=np.float64)
    source_pts = (R180 @ (source_pts - centroid_src).T).T + centroid_src
    print(f"  Rotation 180° appliquee au nuage source (axes Y+X, flip vertical+horizontal)")

    # ---- Mise sur le meme plan Z ----
    z_offset = target_pts[:, 2].mean() - source_pts[:, 2].mean()
    source_pts[:, 2] += z_offset
    print(f"  Nuages alignes sur le meme plan Z (offset source = {z_offset:.4f})")

    # ---- Sous-echantillonnage pour ICP ----
    print("\n" + "=" * 60)
    print("ETAPE 2 : Preparation")
    print("=" * 60)

    if args.voxel_size > 0:
        target_down, _ = voxel_downsample(target_pts, args.voxel_size)
        source_down, _ = voxel_downsample(source_pts, args.voxel_size)
        print(f"  Sous-echantillonnage : cible {len(target_pts)} -> {len(target_down)}, "
              f"source {len(source_pts)} -> {len(source_down)}")
    else:
        target_down = target_pts
        source_down = source_pts
        print(f"  Pas de sous-echantillonnage (cible: {len(target_down)}, source: {len(source_down)})")

    # ---- Chargement des trajectoires ----
    traj1_path = Path(args.traj1) if args.traj1 else Path(args.ply1).parent / "trajectoire_drone.ply"
    traj2_path = Path(args.traj2) if args.traj2 else Path(args.ply2).parent / "trajectoire_drone.ply"

    traj1_pts, traj2_pts = None, None
    if traj1_path.exists() and traj2_path.exists():
        traj1_pts, _ = load_ply(traj1_path)
        traj2_pts, _ = load_ply(traj2_path)
        # Appliquer la meme rotation flip vertical a la trajectoire 1
        traj1_pts = (R_flip_tgt @ (traj1_pts - centroid_tgt).T).T + centroid_tgt
        # Appliquer la meme rotation 180° a la trajectoire 2
        traj2_pts = (R180 @ (traj2_pts - centroid_src).T).T + centroid_src
        # Meme offset Z que le nuage source
        traj2_pts[:, 2] += z_offset
        print(f"  Trajectoires detectees : traj1={len(traj1_pts)} pts, traj2={len(traj2_pts)} pts")
    else:
        missing = []
        if not traj1_path.exists(): missing.append(str(traj1_path))
        if not traj2_path.exists(): missing.append(str(traj2_path))
        print(f"  Trajectoires non trouvees : {', '.join(missing)}")
        print(f"  -> Fallback sur alignement PCA/centroides")

    # ---- Alignement initial ----
    src_icp = source_down.copy()
    T_init = np.eye(4)
    if not args.no_initial:
        if traj1_pts is not None and traj2_pts is not None:
            # Alignement par trajectoires : debut traj2 -> fin traj1
            R_init, t_init = compute_trajectory_alignment(
                traj1_pts, traj2_pts, n_points=args.traj_points
            )
            src_icp = (R_init @ source_down.T).T + t_init
            T_init[:3, :3] = R_init
            T_init[:3, 3] = t_init
            print(f"  Alignement initial par TRAJECTOIRES (n={min(args.traj_points, len(traj1_pts), len(traj2_pts))})")
            print(f"    Fin traj1   : [{traj1_pts[-1][0]:.4f}, {traj1_pts[-1][1]:.4f}, {traj1_pts[-1][2]:.4f}]")
            print(f"    Debut traj2 : [{traj2_pts[0][0]:.4f}, {traj2_pts[0][1]:.4f}, {traj2_pts[0][2]:.4f}]")
            # Verifier ou le debut de traj2 atterrit apres transformation
            debut_traj2_aligne = R_init @ traj2_pts[0] + t_init
            print(f"    Debut traj2 aligne -> [{debut_traj2_aligne[0]:.4f}, {debut_traj2_aligne[1]:.4f}, {debut_traj2_aligne[2]:.4f}]")
        else:
            # Fallback PCA / centroides
            R_init, t_init = compute_initial_alignment(source_down, target_down)
            src_pca = (R_init @ source_down.T).T + t_init
            R_simple, t_simple = compute_initial_alignment_simple(source_down, target_down)
            src_simple = source_down + t_simple
            tree_check = cKDTree(target_down)
            d_pca, _ = tree_check.query(src_pca, k=1)
            d_simple, _ = tree_check.query(src_simple, k=1)
            if np.median(d_pca) <= np.median(d_simple):
                src_icp = src_pca
                T_init[:3, :3] = R_init
                T_init[:3, 3] = t_init
                print(f"  Alignement initial PCA")
            else:
                src_icp = src_simple
                T_init[:3, 3] = t_simple
                print(f"  Alignement initial (centroides)")

    # ---- ICP multi-echelle ----
    print("\n" + "=" * 60)
    print(f"ETAPE 3 : ICP multi-echelle (max {args.max_iter} iter/passe)")
    print("=" * 60)

    # Definir les echelles (du plus grossier au plus fin)
    if args.voxel_size > 0:
        base_voxel = args.voxel_size
    else:
        # Estimer une taille de voxel raisonnable a partir de l'etendue du nuage
        extent = target_down.max(axis=0) - target_down.min(axis=0)
        base_voxel = np.max(extent) / 50.0

    scales = [
        {"voxel": base_voxel * 4.0, "max_iter": min(args.max_iter, 50),  "label": "grossiere"},
        {"voxel": base_voxel * 2.0, "max_iter": min(args.max_iter, 80),  "label": "moyenne"},
        {"voxel": base_voxel * 1.0, "max_iter": min(args.max_iter, 120), "label": "fine"},
        {"voxel": 0,                "max_iter": args.max_iter,           "label": "pleine res."},
    ]

    t0 = time.time()
    R_accum = np.eye(3)
    t_accum = np.zeros(3)
    all_errors = []

    for si, scale in enumerate(scales):
        vx = scale["voxel"]
        label = scale["label"]
        mi = scale["max_iter"]

        # Sous-echantillonner pour cette echelle
        if vx > 0:
            tgt_s, _ = voxel_downsample(target_down, vx)
            # Appliquer la transformation accumulee au source puis sous-echantillonner
            src_cur = (R_accum @ source_down.T).T + t_accum
            if not args.no_initial:
                src_cur = (T_init[:3, :3] @ source_down.T).T + T_init[:3, 3]
                src_cur = (R_accum @ src_cur.T).T + t_accum
            src_s, _ = voxel_downsample(src_cur, vx)
        else:
            tgt_s = target_down
            src_cur = (R_accum @ source_down.T).T + t_accum
            if not args.no_initial:
                src_cur = (T_init[:3, :3] @ source_down.T).T + T_init[:3, 3]
                src_cur = (R_accum @ src_cur.T).T + t_accum
            src_s = src_cur

        # Calcul du seuil outlier pour cette echelle
        tree_s = cKDTree(tgt_s)
        dists_s, _ = tree_s.query(src_s, k=1)
        q75 = np.percentile(dists_s, 75)
        iqr = q75 - np.percentile(dists_s, 25)
        # Seuil plus permissif aux grandes echelles, plus strict aux petites
        outlier_mult = 3.0 - si * 0.5  # 3.0, 2.5, 2.0, 1.5
        max_dist_s = q75 + outlier_mult * iqr
        if args.max_dist > 0:
            max_dist_s = args.max_dist

        print(f"\n  Passe {si+1}/4 ({label}) : {len(src_s)} vs {len(tgt_s)} pts, "
              f"seuil={max_dist_s:.4f}, max_iter={mi}")

        R_step, t_step, errs = icp(src_s, tgt_s,
                                   max_iterations=mi,
                                   tolerance=args.tolerance,
                                   max_dist=max_dist_s)
        all_errors.extend(errs)

        if errs:
            print(f"    RMS : {errs[0]:.6f} -> {errs[-1]:.6f} ({len(errs)} iter)")

        # Accumuler la transformation
        R_accum = R_step @ R_accum
        t_accum = R_step @ t_accum + t_step

    elapsed = time.time() - t0
    print(f"\n    Erreur RMS finale : {all_errors[-1]:.6f}")
    print(f"    Total : {len(all_errors)} iterations en {elapsed:.2f}s")

    # Construire la matrice 4x4 de l'ICP
    T_icp = np.eye(4)
    T_icp[:3, :3] = R_accum
    T_icp[:3, 3] = t_accum

    # Combiner avec l'alignement initial
    T = T_icp @ T_init

    R_final = T[:3, :3]
    t_final = T[:3, 3]

    print(f"\n  Matrice de transformation 4x4 :")
    print(np.array2string(T, precision=6, suppress_small=True))

    # Appliquer sur le nuage source COMPLET
    aligned_pts = (R_final @ source_pts.T).T + t_final

    # Appliquer sur la trajectoire 2 si disponible
    traj2_aligned = None
    if traj2_pts is not None:
        traj2_aligned = (R_final @ traj2_pts.T).T + t_final

    # ---- Fusion ----
    print("\n" + "=" * 60)
    print("ETAPE 4 : Fusion")
    print("=" * 60)

    merged_pts = np.vstack([target_pts, aligned_pts])
    if target_col is not None and source_col is not None:
        merged_col = np.vstack([target_col, source_col])
    else:
        merged_col = None

    if args.dedup > 0:
        n_before = len(merged_pts)
        merged_pts, dedup_idx = voxel_downsample(merged_pts, args.dedup)
        if merged_col is not None:
            merged_col = merged_col[dedup_idx]
        print(f"  Deduplication : {n_before} -> {len(merged_pts)} points")

    print(f"  Nuage fusionne : {len(merged_pts)} points")

    # ---- Sauvegarde ----
    out = Path(args.output)
    save_ply(out, merged_pts, merged_col)
    print(f"\n  Sauvegarde dans : {out}")

    # ---- Visualisation ----
    if args.viz:
        print("\n" + "=" * 60)
        print("VISUALISATION")
        print("=" * 60)
        visualiser(target_pts, target_col, source_pts, source_col,
                   aligned_pts, merged_pts, merged_col,
                   traj1_pts, traj2_pts, traj2_aligned)

    print("\nTermine.")


if __name__ == "__main__":
    main()
