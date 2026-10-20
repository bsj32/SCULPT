#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#SCULPT python script using ROIs/MultiROis derived from Dragonfly 3D

#imports
import os, glob
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from skimage import exposure
from skimage.filters import frangi
from skimage.morphology import (
    remove_small_objects,
    skeletonize_3d,
    ball,
    binary_closing,
    binary_erosion,
)
from skimage.measure import label

from scipy.ndimage import binary_dilation, convolve, distance_transform_edt
from scipy.spatial import cKDTree
from scipy.stats import norm

from skan import Skeleton, summarize

#file paths
BASE_DIR = r"C:\Users\Estroff-lab\Desktop\USER_DATA\bsj32\python (canaliculi)" #base folder
DATA_DIR = os.path.join(BASE_DIR, "data") 
RAW_DIR  = os.path.join(DATA_DIR, "raw") #raw TIFF stack
MASK_DIR = os.path.join(DATA_DIR, "masks") 
LAC_DIR  = os.path.join(DATA_DIR, "lacunae") #where cartilage, haversian canals and lacunae npz files are [derived from Dragonfly 3D]

#setting voxel size in microns (z,y,x)
VOXEL_SIZE = (0.310, 0.2965235, 0.2965235)

#file names of cartilage, haversian canals and lacunae 
LACUNAE_NPZ   = os.path.join(LAC_DIR, "AM3_2_lacunae.npz")
CARTILAGE_NPZ = os.path.join(LAC_DIR, "AM3_2_cartilage.npz")
VESSELS_NPZ   = os.path.join(LAC_DIR, "AM3_2_vessels.npz")

#connectivity & cleanup
MIN_OBJ_SIZE_VOX = 80
CLOSE_RADIUS_VOX = 1
MIN_BRANCH_UM    = 5.0

REMOVE_LACUNA_INTERIOR = True
RIM_SHELL_ITERS = 1
DO_RIM_ARTIFACT_REMOVAL = True
DO_ENDPOINT_TO_LACUNA_BRIDGE = True
LACUNA_ENDPOINT_GAP_VOX = 2

ENDPOINT_GAP_VOX = 2
DO_ENDPOINT_BRIDGE = True

#tunning for masks
TUNE_SKIP_MASK_PIXELS_BELOW = 1000

#threshold for full stack
FORCE_PCTL = 96.0

#outputs
OUT_BIN_STACK    = os.path.join(BASE_DIR, "canaliculi_binary_stack.tif")
OUT_SKEL_STACK   = os.path.join(BASE_DIR, "canaliculi_skeleton_stack.tif")
OUT_BRANCH_CSV   = os.path.join(BASE_DIR, "canaliculi_branch_metrics_skan.csv")
OUT_LAC_DIST_CSV = os.path.join(BASE_DIR, "lacunae_to_canaliculi_metrics.csv")
OUT_LAC_PROC_CSV = os.path.join(BASE_DIR, "lacunae_extrusions_per_lacuna.csv")

#histograms & summary outputs
OUT_HIST_LEN    = os.path.join(BASE_DIR, "hist_length_um.png")
OUT_HIST_PROC   = os.path.join(BASE_DIR, "hist_extrusions_per_lacuna.png")
OUT_GAUSS_CSV   = os.path.join(BASE_DIR, "canaliculi_gaussian_summary.csv")
OUT_MASTER_CSV  = os.path.join(BASE_DIR, "LCN_master_summary.csv")

#helpers
def load_mask(mask_path: str) -> np.ndarray:
    m = tiff.imread(mask_path)

    if m.ndim == 3 and m.shape[-1] == 4:
        return (m[..., 3] > 0)

    raise ValueError(f"Unexpected mask shape for {mask_path}: {m.shape}")

def mask_to_raw_name(mask_filename: str) -> str:
    base = mask_filename

    if base.endswith(".tif~"):
        base = base[:-1]

    if base.endswith("-2.tif"):
        base = base.replace("-2.tif", "*.tif")

    return base

def robust01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)

    lo = np.percentile(img, 1)
    hi = np.percentile(img, 99)

    out = (img - lo) / (hi - lo + 1e-8)

    return np.clip(out, 0, 1).astype(np.float32)

def segment_slice(
    raw2d: np.ndarray,
    sigmas: np.ndarray,
    pctl: float,
    tissue_pctl: float = 99.5,
    min_size_2d: int = 10
) -> np.ndarray: #raw2d -> normalize -> CLAHE -> Frangi ->percentile threshold -> remove small objects

    r01 = robust01(raw2d)

    r_eq = exposure.equalize_adapthist(
        r01,
        clip_limit=0.01
    )

    raw_u8 = (r01 * 255).astype(np.uint8)

    tissue = raw_u8 < np.percentile(
        raw_u8,
        tissue_pctl
    )

    v = frangi(
        r_eq,
        sigmas=sigmas,
        black_ridges=False
    ) * tissue

    thr = np.percentile(
        v[tissue],
        pctl
    )

    binary = v > thr

    binary = remove_small_objects(
        binary,
        min_size=min_size_2d
    )

    return binary


def coverage_score(
    binary: np.ndarray,
    manual: np.ndarray,
    r_px: int = 3
) -> float:

    sk2d = skeletonize_3d(
        binary[np.newaxis, ...]
    )[0].astype(bool)

    dist_to_skel = distance_transform_edt(~sk2d)

    if manual.any():
        return float(
            (dist_to_skel[manual] <= r_px).mean()
        )

    return 0.0


def load_binary_mask_from_npz(
    npz_path: str
) -> np.ndarray:

    d = np.load(npz_path)

    combined = np.zeros_like(
        next(iter(d.values())),
        dtype=bool
    )

    for k in d.files:
        combined |= (d[k] > 0)

    return combined


def load_lacuna_labels_from_npz(
    npz_path: str
) -> np.ndarray:

    lac = np.load(npz_path)

    labels = np.zeros_like(
        next(iter(lac.values())),
        dtype=np.int32
    )

    for idx, key in enumerate(
        lac.files,
        start=1
    ):
        labels[lac[key] > 0] = idx

    return labels


def z_bridge(
    binary_bool: np.ndarray
) -> np.ndarray: #bridge in z axis

    b = binary_bool.copy()

    if b.shape[0] < 3:
        return b

    b[1:-1] |= (
        b[:-2] &
        b[2:]
    )

    return b


def find_skeleton_endpoints(
    skel: np.ndarray
) -> np.ndarray:

    k = np.ones(
        (3, 3, 3),
        dtype=np.uint8
    )

    k[1, 1, 1] = 0

    nbrs = convolve(
        skel.astype(np.uint8),
        k,
        mode="constant",
        cval=0
    )

    return skel & (nbrs == 1)


def connect_endpoints(
    binary: np.ndarray,
    max_gap_vox: int = 2
) -> np.ndarray:

    sk = skeletonize_3d(
        binary.astype(bool)
    ).astype(bool)

    ep = find_skeleton_endpoints(sk)

    if ep.sum() < 2:
        return binary

    ep_dil = binary_dilation(
        ep,
        structure=ball(max_gap_vox)
    )

    bridge = ep_dil & (~binary)

    win = (2 * max_gap_vox + 1)

    ep_count = convolve(
        ep.astype(np.uint8),
        np.ones(
            (win, win, win),
            dtype=np.uint8
        ),
        mode="constant",
        cval=0
    )

    bridge &= (ep_count >= 2)

    return binary | bridge


def connect_endpoints_to_lacuna(
    binary: np.ndarray,
    lac_mask: np.ndarray,
    max_gap_vox: int = 2
) -> np.ndarray:

    if lac_mask is None:
        return binary

    sk = skeletonize_3d(
        binary.astype(bool)
    ).astype(bool)

    ep = find_skeleton_endpoints(sk)

    ep_coords = np.argwhere(ep)

    if ep_coords.shape[0] == 0:
        return binary

    lac_shell = (
        binary_dilation(
            lac_mask,
            structure=ball(1)
        )
        & (~lac_mask)
    )

    shell_coords = np.argwhere(lac_shell)

    if shell_coords.shape[0] == 0:
        return binary

    tree = cKDTree(shell_coords)

    dists, idxs = tree.query(
        ep_coords,
        k=1
    )

    out = binary.copy()

    for ep_pt, d, idx in zip(
        ep_coords,
        dists,
        idxs
    ):

        if d <= max_gap_vox:

            target = shell_coords[idx]

            n_steps = (
                int(
                    np.max(
                        np.abs(
                            target - ep_pt
                        )
                    )
                )
                + 1
            )

            zz = np.round(
                np.linspace(
                    ep_pt[0],
                    target[0],
                    n_steps
                )
            ).astype(int)

            yy = np.round(
                np.linspace(
                    ep_pt[1],
                    target[1],
                    n_steps
                )
            ).astype(int)

            xx = np.round(
                np.linspace(
                    ep_pt[2],
                    target[2],
                    n_steps
                )
            ).astype(int)

            out[
                zz,
                yy,
                xx
            ] = True

    return out


def remove_lacuna_rim_artifact(
    bin_stack: np.ndarray,
    lac_mask: np.ndarray,
    shell_iters: int = 1
) -> np.ndarray:

    if lac_mask is None:
        return bin_stack

    shell = (
        binary_dilation(
            lac_mask,
            iterations=shell_iters
        )
        & (~lac_mask)
    )

    out = bin_stack.copy()

    cc = label(
        bin_stack,
        connectivity=1
    )

    comp_ids = np.unique(cc)
    comp_ids = comp_ids[comp_ids > 0]

    removed = 0

    for cid in comp_ids:

        comp = (cc == cid)

        touches_shell = np.any(
            comp & shell
        )

        extends_outward = np.any(
            comp &
            (~shell) &
            (~lac_mask)
        )

        if (
            touches_shell
            and
            not extends_outward
        ):
            out[comp] = False
            removed += 1

    print(
        f"Removed {removed} rim-only components"
    )

    return out


#metrics
def build_branch_table(
    skel_bool: np.ndarray,
    spacing_zyx,
    min_branch_um: float = 8.0
):

    sk = Skeleton(
        skel_bool,
        spacing=spacing_zyx
    )

    df = summarize(
        sk,
        separator='-'
    ).copy()

    if "branch-distance" in df.columns:
        df = df[
            df["branch-distance"]
            >= float(min_branch_um)
        ].copy()

    return df, sk


#lacunae metrics
def lacuna_metrics_kdtree(
    lac_labels: np.ndarray,
    skel: np.ndarray,
    spacing_zyx
) -> pd.DataFrame:

    skel_bool = skel.astype(bool)

    skel_coords = np.argwhere(
        skel_bool
    )

    if skel_coords.shape[0] == 0:
        raise RuntimeError(
        )

    vz, vy, vx = spacing_zyx

    skel_um = (
        skel_coords.astype(np.float32)
        * np.array(
            [vz, vy, vx],
            dtype=np.float32
        )
    )

    tree = cKDTree(skel_um)

    lac_ids = np.unique(lac_labels)
    lac_ids = lac_ids[lac_ids > 0]

    erode_selem = ball(1)

    rows = []

    for lac_id in lac_ids:

        lac = (
            lac_labels == lac_id
        )

        er = binary_erosion(
            lac,
            footprint=erode_selem
        )

        boundary = (
            lac &
            (~er)
        )

        coords = np.argwhere(
            boundary
        )

        if coords.size == 0:
            coords = np.argwhere(lac)

        coords_um = (
            coords.astype(np.float32)
            * np.array(
                [vz, vy, vx],
                dtype=np.float32
            )
        )

        d_um, _ = tree.query(
            coords_um,
            k=1,
            workers=-1
        )

        rows.append({
            "lacuna_id": int(lac_id),

            "min_dist_to_canaliculus_um":
                float(np.min(d_um))
                if len(d_um)
                else np.nan,

            "mean_dist_to_canaliculus_um":
                float(np.mean(d_um))
                if len(d_um)
                else np.nan,
        })

    out = pd.DataFrame(rows)

    out.to_csv(
        OUT_LAC_DIST_CSV,
        index=False
    )

    print(
        f"Saved lacunae distance metrics (KD-tree): "
        f"{OUT_LAC_DIST_CSV}"
    )

    return out

#number of canaliculi per lacunae
def lacuna_extrusion_count(
    lac_labels: np.ndarray,
    skel_bool: np.ndarray,
    shell_iters: int = 1,
    contact_connectivity: int = 3,
    min_contact_cluster_vox: int = 1
) -> pd.DataFrame:

    lac_ids = np.unique(lac_labels)
    lac_ids = lac_ids[lac_ids > 0]

    rows = []

    for lac_id in lac_ids:

        lac = (
            lac_labels == lac_id
        )

        shell = (
            binary_dilation(
                lac,
                iterations=shell_iters
            )
            & (~lac)
        )

        contact_mask = (
            skel_bool &
            shell
        )

        contact_cc = label(
            contact_mask,
            connectivity=contact_connectivity
        )

        cluster_ids = np.unique(
            contact_cc
        )

        cluster_ids = cluster_ids[
            cluster_ids > 0
        ]

        valid_cluster_sizes = []

        for cid in cluster_ids:

            nvox = int(
                np.count_nonzero(
                    contact_cc == cid
                )
            )

            if nvox >= min_contact_cluster_vox:
                valid_cluster_sizes.append(
                    nvox
                )

        extrusion_count = len(
            valid_cluster_sizes
        )

        rows.append({
            "lacuna_id": int(lac_id),

            "extrusions_per_lacuna":
                int(extrusion_count),

            "contact_cluster_mean_size_vox":
                float(
                    np.mean(
                        valid_cluster_sizes
                    )
                )
                if len(valid_cluster_sizes)
                else np.nan,

            "contact_cluster_max_size_vox":
                float(
                    np.max(
                        valid_cluster_sizes
                    )
                )
                if len(valid_cluster_sizes)
                else np.nan,
        })

    out = pd.DataFrame(rows)

    out.to_csv(
        OUT_LAC_PROC_CSV,
        index=False
    )

    print(
        f"Saved lacuna extrusion counts: "
        f"{OUT_LAC_PROC_CSV}"
    )

    return out


#tuning
def tune_from_masks():

    mask_files = sorted(
        glob.glob(
            os.path.join(
                MASK_DIR,
                "*.tif*"
            )
        )
    )

    if not mask_files:

        print(
            f"No masks found in {MASK_DIR}. "
            f"Falling back to defaults."
        )

        return (
            np.array(
                [0.6, 1.0, 1.4, 1.8]
            ),
            96.0
        )

    sigma_grids = [
        np.linspace(
            0.4,
            1.4,
            5
        ),

        np.linspace(
            0.6,
            1.8,
            5
        ),

        np.linspace(
            0.8,
            2.2,
            5
        ),
    ]

    pctls = [
        94.0,
        94.5,
        95.0,
        95.5,
        96.0,
        96.5,
        97.0,
        97.5
    ]

    pairs = []

    for mp in mask_files:

        mname = os.path.basename(mp)

        raw_name = mask_to_raw_name(
            mname
        )

        rp = os.path.join(
            RAW_DIR,
            raw_name
        )

        if not os.path.exists(rp):

            print(
                f"Skipping mask (no matching raw): "
                f"{mname} -> expected {raw_name}"
            )

            continue

        manual = load_mask(mp)

        raw = tiff.imread(rp)

        if manual.sum() < TUNE_SKIP_MASK_PIXELS_BELOW:

            print(
                f"Skipping small mask for tuning: "
                f"{mname} "
                f"(alpha pixels={manual.sum()})"
            )

            continue

        pairs.append(
            (
                raw,
                manual,
                raw_name
            )
        )

    if not pairs:

        print(
            "All masks were too small for tuning. "
            "Falling back to defaults."
        )

        return (
            np.array(
                [0.6, 1.0, 1.4, 1.8]
            ),
            96.0
        )

    best_cov = -1.0
    best_detail = None

    for sigmas in sigma_grids:

        for p in pctls:

            covs = []

            for raw, manual, _nm in pairs:

                pred = segment_slice(
                    raw,
                    sigmas=sigmas,
                    pctl=p
                )

                covs.append(
                    coverage_score(
                        pred,
                        manual,
                        r_px=3
                    )
                )

            mean_cov = float(
                np.mean(covs)
            )

            if mean_cov > best_cov:

                best_cov = mean_cov

                best_detail = (
                    mean_cov,
                    p,
                    sigmas
                )

    mean_cov, p_best, sigmas_best = best_detail

    print("tuning")

    print(
        f"sigmas = {sigmas_best}"
    )

    print(
        f"percentile = {p_best}"
    )

    print(
        f"mean manual coverage (@3px) = "
        f"{mean_cov:.3f}"
    )

    return (
        sigmas_best,
        p_best
    )

#find canaliculi
def load_raw_stack():

    raw_files = sorted(
        glob.glob(
            os.path.join(
                RAW_DIR,
                "*.tif*"
            )
        )
    )

    if not raw_files:
        raise RuntimeError(
            f"No raw .tif slices found in {RAW_DIR}"
        )

    stack = np.stack(
        [
            tiff.imread(f)
            for f in raw_files
        ],
        axis=0
    )

    names = [
        os.path.basename(f)
        for f in raw_files
    ]

    return names, stack


def segment_full_stack(
    raw_stack: np.ndarray,
    sigmas: np.ndarray,
    pctl: float
) -> np.ndarray:

    Z = raw_stack.shape[0]

    bin_stack = np.zeros(
        raw_stack.shape,
        dtype=np.uint8
    )

    for z in range(Z):

        bin_stack[z] = segment_slice(
            raw_stack[z],
            sigmas=sigmas,
            pctl=pctl
        ).astype(np.uint8)

        if (z % 25) == 0:

            print(
                f"Segmented {z+1}/{Z}"
            )

    return bin_stack.astype(bool)


#canaliculi density
def compute_length_density(
    skel_stack: np.ndarray,
    exclude_mask: np.ndarray,
    lac_mask: np.ndarray,
    spacing_zyx
):

    voxel_volume_um3 = float(
        np.prod(spacing_zyx)
    )

    mean_voxel_step_um = float(
        np.mean(spacing_zyx)
    )

    valid_mask = ~exclude_mask

    if lac_mask is not None:
        valid_mask &= ~lac_mask

    analyzed_volume_um3 = float(
        np.count_nonzero(valid_mask)
        * voxel_volume_um3
    )

    total_length_um = float(
        np.count_nonzero(
            skel_stack & valid_mask
        )
        * mean_voxel_step_um
    )

    length_density = (
        total_length_um
        / analyzed_volume_um3
        if analyzed_volume_um3 > 0
        else np.nan
    )

    return {
        "total_length_um":
            total_length_um,

        "analyzed_volume_um3":
            analyzed_volume_um3,

        "canalicular_length_density_um_per_um3":
            length_density,
    }


#histograms and gaussian fits
def save_hist_with_best_fit(
    data,
    title,
    xlabel,
    out_png,
    bins=20,
    fit_mode="linear"
):

    d = pd.to_numeric(
        pd.Series(data),
        errors="coerce"
    ).dropna()

    if fit_mode == "log":
        d = d[d > 0]

    if len(d) < 5:

        plt.figure(
            figsize=(6, 4)
        )

        plt.text(
            0.5,
            0.5,
            "No valid data",
            ha='center',
            va='center'
        )

        plt.title(title)
        plt.axis('off')
        plt.tight_layout()

        plt.savefig(
            out_png,
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        return {
            "n": len(d),
            "mean": np.nan,
            "std": np.nan,
            "fit_mu": np.nan,
            "fit_sigma": np.nan,
            "fit_mode": fit_mode
        }

    mean_raw = float(
        d.mean()
    )

    std_raw = float(
        d.std()
    )

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    if fit_mode == "log":

        counts, edges, _ = ax.hist(
            d,
            bins=np.logspace(
                np.log10(d.min()),
                np.log10(d.max()),
                bins + 1
            ),
            color="#4C72B0",
            edgecolor="black",
            alpha=0.7
        )

        logd = np.log10(d)

        mu_fit, sigma_fit = norm.fit(
            logd
        )

        x = np.logspace(
            np.log10(d.min()),
            np.log10(d.max()),
            500
        )

        logx = np.log10(x)

        pdf_log = norm.pdf(
            logx,
            mu_fit,
            sigma_fit
        )

        bin_width_log = np.mean(
            np.diff(
                np.log10(edges)
            )
        )

        pdf_scaled = (
            pdf_log
            * len(d)
            * bin_width_log
        )

        ax.plot(
            x,
            pdf_scaled,
            'r-',
            linewidth=2,
            label='Gaussian fit'
        )

        ax.set_xscale('log')

        ax.xaxis.set_major_formatter(
            ticker.ScalarFormatter()
        )

        ax.xaxis.set_minor_formatter(
            ticker.NullFormatter()
        )

    else:

        counts, edges, _ = ax.hist(
            d,
            bins=bins,
            color="#4C72B0",
            edgecolor="black",
            alpha=0.7
        )

        mu_fit, sigma_fit = norm.fit(
            d
        )

        x = np.linspace(
            d.min(),
            d.max(),
            500
        )

        pdf = norm.pdf(
            x,
            mu_fit,
            sigma_fit
        )

        bin_width = (
            edges[1]
            - edges[0]
        )

        pdf_scaled = (
            pdf
            * len(d)
            * bin_width
        )

        ax.plot(
            x,
            pdf_scaled,
            'r-',
            linewidth=2,
            label='Gaussian fit'
        )

    ax.axvline(
        mean_raw,
        color='black',
        linestyle='--',
        linewidth=1.5,
        label='Mean'
    )

    stats_text = (
        f"mean = {mean_raw:.3f}\n"
        f"std = {std_raw:.3f}\n"
        f"N = {len(d)}"
    )

    ax.text(
        0.98,
        0.88,
        stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        fontsize=9,
        bbox=dict(
            facecolor='white',
            alpha=0.7,
            edgecolor='none',
            boxstyle='round,pad=0.3'
        )
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(
        title,
        fontsize=11
    )

    ax.legend(
        fontsize=8
    )

    plt.tight_layout()

    plt.savefig(
        out_png,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    return {
        "n": int(len(d)),
        "mean": mean_raw,
        "std": std_raw,
        "fit_mu": float(mu_fit),
        "fit_sigma": float(sigma_fit),
        "fit_mode": fit_mode
    }


#create summaries
def write_master_summary(
    df_branch,
    df_lac_proc,
    length_density_metrics,
    gaussian_results
):

    lengths = (
        df_branch["branch-distance"]
        .dropna()
    )

    extr = (
        df_lac_proc["extrusions_per_lacuna"]
        .dropna()
    )

    summary = {

        "total_branches":
            int(len(lengths)),

        "mean_length_um":
            float(lengths.mean()),

        "std_length_um":
            float(lengths.std()),

        "median_length_um":
            float(lengths.median()),


        "mean_extrusions_per_lacuna":
            float(extr.mean())
            if len(extr)
            else np.nan,

        "std_extrusions_per_lacuna":
            float(extr.std())
            if len(extr)
            else np.nan,

        "median_extrusions_per_lacuna":
            float(extr.median())
            if len(extr)
            else np.nan,


        "total_canalicular_length_um":
            float(
                length_density_metrics[
                    "total_length_um"
                ]
            ),

        "analyzed_tissue_volume_um3":
            float(
                length_density_metrics[
                    "analyzed_volume_um3"
                ]
            ),

        "canalicular_length_density_um_per_um3":
            float(
                length_density_metrics[
                    "canalicular_length_density_um_per_um3"
                ]
            ),


        "hist_length_mean":
            float(
                gaussian_results[
                    "branch-distance"
                ]["mean"]
            )
            if "branch-distance"
            in gaussian_results
            else np.nan,

        "hist_length_std":
            float(
                gaussian_results[
                    "branch-distance"
                ]["std"]
            )
            if "branch-distance"
            in gaussian_results
            else np.nan,

        "hist_length_fit_mu":
            float(
                gaussian_results[
                    "branch-distance"
                ]["fit_mu"]
            )
            if "branch-distance"
            in gaussian_results
            else np.nan,

        "hist_length_fit_sigma":
            float(
                gaussian_results[
                    "branch-distance"
                ]["fit_sigma"]
            )
            if "branch-distance"
            in gaussian_results
            else np.nan,


        "hist_extrusions_mean":
            float(
                gaussian_results[
                    "extrusions_per_lacuna"
                ]["mean"]
            )
            if "extrusions_per_lacuna"
            in gaussian_results
            else np.nan,

        "hist_extrusions_std":
            float(
                gaussian_results[
                    "extrusions_per_lacuna"
                ]["std"]
            )
            if "extrusions_per_lacuna"
            in gaussian_results
            else np.nan,

        "hist_extrusions_fit_mu":
            float(
                gaussian_results[
                    "extrusions_per_lacuna"
                ]["fit_mu"]
            )
            if "extrusions_per_lacuna"
            in gaussian_results
            else np.nan,

        "hist_extrusions_fit_sigma":
            float(
                gaussian_results[
                    "extrusions_per_lacuna"
                ]["fit_sigma"]
            )
            if "extrusions_per_lacuna"
            in gaussian_results
            else np.nan,
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        OUT_MASTER_CSV,
        index=False
    )

    print(
        f"Saved master summary CSV: "
        f"{OUT_MASTER_CSV}"
    )


#run
sigmas, pctl_tuned = tune_from_masks()

pctl = max(
    float(pctl_tuned),
    float(FORCE_PCTL)
)

print(
    f"\nUsing FINAL percentile = "
    f"{pctl} "
    f"(tuned was {pctl_tuned})"
)


#load raw TIFF stack
names, raw_stack = load_raw_stack()

print(
    f"Loaded raw stack: "
    f"{raw_stack.shape} "
    f"({len(names)} slices)"
)

#labels
if os.path.exists(LACUNAE_NPZ):

    lac_labels = load_lacuna_labels_from_npz(
        LACUNAE_NPZ
    )

    lac_mask = (
        lac_labels > 0
    )

    lac_exclude = (
        lac_mask.copy()
        if REMOVE_LACUNA_INTERIOR
        else None
    )

else:

    lac_labels = None
    lac_mask = None
    lac_exclude = None

    print(
        "Lacunae NPZ not found — "
        "skipping lacuna metrics and "
        "lacuna interior exclusion."
    )


#layers that get excluded [cartilage & haversian canals]
exclude_mask = np.zeros(
    raw_stack.shape,
    dtype=bool
)

cartilage_mask = None
vessels_mask = None


if os.path.exists(CARTILAGE_NPZ):

    cartilage_mask = (
        load_binary_mask_from_npz(
            CARTILAGE_NPZ
        )
    )

    exclude_mask |= (
        cartilage_mask
    )

else:

    print(
        "Cartilage NPZ not found — "
        "skipping cartilage exclusion."
    )


if os.path.exists(VESSELS_NPZ):

    vessels_mask = (
        load_binary_mask_from_npz(
            VESSELS_NPZ
        )
    )

    exclude_mask |= (
        vessels_mask
    )

else:

    print(
        "Vessels NPZ not found — "
        "skipping vessel exclusion."
    )


print(
    f"Non-lacuna exclusion mask coverage: "
    f"{exclude_mask.mean()*100:.2f}% "
    f"of volume"
)


#canaliculi extraction

bin_stack = segment_full_stack(
    raw_stack,
    sigmas=sigmas,
    pctl=pctl
)

#cartilage & haversian canal exclusions
bin_stack[
    exclude_mask
] = False


#remove lacunae interior
if lac_exclude is not None:

    bin_stack[
        lac_exclude
    ] = False


#connection recovery
bin_stack = remove_small_objects(
    bin_stack,
    min_size=MIN_OBJ_SIZE_VOX
)

bin_stack = z_bridge(
    bin_stack
)


if DO_ENDPOINT_BRIDGE:

    bin_stack = connect_endpoints(
        bin_stack,
        max_gap_vox=ENDPOINT_GAP_VOX
    )


if (
    DO_ENDPOINT_TO_LACUNA_BRIDGE
    and
    (lac_mask is not None)
):

    bin_stack = connect_endpoints_to_lacuna(
        bin_stack,
        lac_mask=lac_mask,
        max_gap_vox=LACUNA_ENDPOINT_GAP_VOX
    )


if (
    DO_RIM_ARTIFACT_REMOVAL
    and
    (lac_mask is not None)
):

    bin_stack = remove_lacuna_rim_artifact(
        bin_stack,
        lac_mask=lac_mask,
        shell_iters=RIM_SHELL_ITERS
    )


if CLOSE_RADIUS_VOX > 0:

    bin_stack = binary_closing(
        bin_stack,
        ball(CLOSE_RADIUS_VOX)
    )


bin_stack = remove_small_objects(
    bin_stack,
    min_size=MIN_OBJ_SIZE_VOX
)


#save the binary stack
tiff.imwrite(
    OUT_BIN_STACK,
    bin_stack.astype(np.uint8)
)

print(
    f"Saved binary stack: "
    f"{OUT_BIN_STACK}"
)


#skeletonize
skel_stack = skeletonize_3d(
    bin_stack
).astype(bool)


#keep cartilage/vessel exclusions off skeleton
skel_stack[
    exclude_mask
] = False


#make sure lacuna interior is not skeletonized
if lac_mask is not None:

    skel_stack[
        lac_mask
    ] = False


#canaliculi density
length_density_metrics = (
    compute_length_density(
        skel_stack=skel_stack,
        exclude_mask=exclude_mask,
        lac_mask=lac_mask,
        spacing_zyx=VOXEL_SIZE
    )
)


print(
    f"Total canalicular length (um): "
    f"{length_density_metrics['total_length_um']:.3f}"
)

print(
    f"Analyzed tissue volume (um^3): "
    f"{length_density_metrics['analyzed_volume_um3']:.3f}"
)

print(
    f"Canalicular length density (um/um^3): "
    f"{length_density_metrics['canalicular_length_density_um_per_um3']:.6e}"
)


#canaliculi metric - length

df_branch, sk_obj = build_branch_table(
    skel_bool=skel_stack,
    spacing_zyx=VOXEL_SIZE,
    min_branch_um=MIN_BRANCH_UM
)

df_branch.to_csv(
    OUT_BRANCH_CSV,
    index=False
)

print(
    f"Saved branch metrics (length): "
    f"{OUT_BRANCH_CSV}"
)


#saving skeleton image

tiff.imwrite(
    OUT_SKEL_STACK,
    skel_stack.astype(np.uint8)
)

print(
    f"Saved skeleton stack: "
    f"{OUT_SKEL_STACK}"
)


#number of canaliculi per lacunae
df_lac_dist = None
df_lac_proc = None


if lac_labels is not None:

    df_lac_dist = lacuna_metrics_kdtree(
        lac_labels,
        skel_stack,
        spacing_zyx=VOXEL_SIZE
    )

    df_lac_proc = lacuna_extrusion_count(
        lac_labels=lac_labels,
        skel_bool=skel_stack,
        shell_iters=1,
        contact_connectivity=3,
        min_contact_cluster_vox=1
    )


#histogram making
gaussian_results = {}


#canaliculi length
gaussian_results[
    "branch-distance"
] = save_hist_with_best_fit(

    df_branch[
        "branch-distance"
    ],

    "Canaliculi branch length",

    "Length (µm)",

    OUT_HIST_LEN,

    bins=20,

    fit_mode="log"
)


#canaliculi per lacunae
if df_lac_proc is not None:

    gaussian_results[
        "extrusions_per_lacuna"
    ] = save_hist_with_best_fit(

        df_lac_proc[
            "extrusions_per_lacuna"
        ],

        "Canalicular extrusions per lacuna",

        "# canaliculi extruding from lacuna",

        OUT_HIST_PROC,

        bins=15,

        fit_mode="linear"
    )


#saving the summary
gauss_export_rows = []

for metric, vals in gaussian_results.items():

    row = {
        "Metric": metric
    }

    row.update(vals)

    gauss_export_rows.append(
        row
    )


pd.DataFrame(
    gauss_export_rows
).to_csv(
    OUT_GAUSS_CSV,
    index=False
)

print(
    f"Saved Gaussian summary CSV: "
    f"{OUT_GAUSS_CSV}"
)


#create a master summary
if df_lac_proc is not None:

    write_master_summary(
        df_branch,
        df_lac_proc,
        length_density_metrics,
        gaussian_results
    )


print("\nDONE")


#3D visualization in napari
import napari


binary_stack_v = tiff.imread(
    OUT_BIN_STACK
).astype(np.uint8)

skel_stack_v = tiff.imread(
    OUT_SKEL_STACK
).astype(np.uint8)


viewer = napari.Viewer(
    ndisplay=3
)


viewer.add_image(
    raw_stack,
    name="Raw",
    scale=VOXEL_SIZE,
    contrast_limits=(
        np.percentile(
            raw_stack,
            1
        ),
        np.percentile(
            raw_stack,
            99
        )
    ),
    opacity=0.5
)


viewer.add_labels(
    binary_stack_v,
    name="Canaliculi Binary",
    scale=VOXEL_SIZE,
    opacity=0.20
)


viewer.add_image(
    skel_stack_v,
    name="Canaliculi Skeleton",
    scale=VOXEL_SIZE,
    rendering="mip",
    opacity=1.0
)


if lac_labels is not None:

    viewer.add_labels(
        lac_labels.astype(np.uint16),
        name="Lacunae",
        scale=VOXEL_SIZE,
        opacity=1.0
    )


if cartilage_mask is not None:

    viewer.add_labels(
        cartilage_mask.astype(np.uint8),
        name="Cartilage (exclude)",
        scale=VOXEL_SIZE,
        opacity=0.10
    )


if vessels_mask is not None:

    viewer.add_labels(
        vessels_mask.astype(np.uint8),
        name="Vessels (exclude)",
        scale=VOXEL_SIZE,
        opacity=0.10
    )


if lac_mask is not None:

    viewer.add_labels(
        lac_mask.astype(np.uint8),
        name="Lacuna interior excluded",
        scale=VOXEL_SIZE,
        opacity=0.08
    )


print(
    "napari 3D viewer ready"
)

napari.run()

