#!/usr/bin/env python3
"""
scripts/diagnostics/phase6_t2_frozen_probe_image_grouped.py

Phase 6: Leakage-Safe Image-Level Regularized Frozen T2 Probe
=============================================================

Scientific Objective
--------------------
Determine whether the NECK-vs-CRACK separability (previously observed at AUC ~0.69
in unregularized / fixed-C settings) persists when:
1. Evaluation is 100% leakage-free via 5-Fold GroupKFold strictly partitioned by Image ID.
2. All preprocessing (StandardScaler, PCA, projection matrices) is fit strictly on training images of each fold.
3. Regularization strength C is selected strictly via inner 4-fold GroupKFold on training images.
4. Confidence intervals and paired contrasts are computed via image-level bootstrapping and fold-level paired testing.

Hard Locks
----------
* Diagnostic-only: No training, no backward, no optimizer, no model weight edits.
* Candidate B checkpoint and model parameters bitwise invariant before/after.
* Validation cohort only (N=348); Test N=1124 sealed.
* Setting A: 448x448, stride 448, reflect padding, tau=0.5.

Conditions Evaluated
--------------------
* PCA sweep: k in {32, 64, 128, 192, 256}
* Raw baseline: raw_288 (full 288-dim T2, StandardScaler only)
* Random projection controls: RP in {32, 64, 128} (data-agnostic Gaussian, normalized)
* Null control: shuffle_null (permuted target labels within each fold)
"""

import hashlib
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

sys.path.insert(0, 'SAGE_LITE')
from tools.run_phase6_c_topology_diagnostic import load_model_from_checkpoint
from scripts.diagnostics.phase6_stem_factorization_provenance import (
    isolate_bridged_pairs_and_rois,
    isolate_clean_pairs_and_rois,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_OUTER_FOLDS    = 5
N_INNER_FOLDS    = 4
MAX_PIX_PER_CLS  = 30
RNG_BASE         = 42
N_BOOTSTRAP_IMG  = 1000

LABEL_CRACK = 0
LABEL_NECK  = 1
LABEL_BG    = 2

# Regularization grid for inner CV
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

# Conditions
PCA_DIMS  = [32, 64, 128, 192, 256]
RAW_DIM   = 288
RAND_DIMS = [32, 64, 128]

CONDITIONS: Dict[str, Dict[str, Any]] = {}
for k in PCA_DIMS:
    CONDITIONS[f'pca_{k}']  = {'type': 'pca',  'k': k}
CONDITIONS['raw_288']       = {'type': 'raw',  'k': 288}
for k in RAND_DIMS:
    CONDITIONS[f'rand_{k}'] = {'type': 'rand', 'k': k}
CONDITIONS['shuffle_null']  = {'type': 'null', 'k': 32}


# ---------------------------------------------------------------------------
# Integrity Utilities
# ---------------------------------------------------------------------------

def compute_file_hash(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_model_param_hash(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode('utf-8'))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# T2 Feature Capture Engine
# ---------------------------------------------------------------------------

class T2FeatureCaptureEngine:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model     = model
        self.device    = device
        self.dec_b1    = model.decoder.decoder_blocks[1]
        self.conv1_seq = self.dec_b1.conv1
        self.current_patch_t2: Optional[torch.Tensor] = None
        self.pre_hook_handle = None
        self._register_hook()

    def _register_hook(self):
        def hook(module, args):
            self.current_patch_t2 = args[0].detach().clone().cpu()
            return args
        self.pre_hook_handle = self.conv1_seq.register_forward_pre_hook(hook)

    def cleanup(self):
        if self.pre_hook_handle:
            self.pre_hook_handle.remove()
            self.pre_hook_handle = None

    def run_image(self, image_rgb: np.ndarray, tile_size: int = 448, batch_size: int = 4):
        H, W = image_rgb.shape[:2]
        pad_h = (tile_size - H % tile_size) % tile_size
        pad_w = (tile_size - W % tile_size) % tile_size
        padded = np.pad(image_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        pH, pW = padded.shape[:2]

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        final_logits = np.zeros((pH, pW), dtype=np.float32)
        patches, coords = [], []
        for y in range(0, pH, tile_size):
            for x in range(0, pW, tile_size):
                pt = torch.from_numpy(padded[y:y+tile_size, x:x+tile_size]).permute(2,0,1).float().unsqueeze(0).to(self.device) / 255.0
                pt = (pt - mean) / std
                patches.append(pt)
                coords.append((y, x))

        patch_records = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch_pt = torch.cat(patches[i:i+batch_size], 0)
                self.current_patch_t2 = None
                logits = self.model(batch_pt)
                if logits.shape[2:] != (tile_size, tile_size):
                    logits = F.interpolate(logits, (tile_size, tile_size), mode='bilinear', align_corners=False)
                captured = self.current_patch_t2
                for b_idx, (py, px) in enumerate(coords[i:i+batch_size]):
                    final_logits[py:py+tile_size, px:px+tile_size] = logits[b_idx].squeeze().cpu().numpy()
                    patch_records.append({'coords': (py, px), 't2': captured[b_idx:b_idx+1]})

        logits_c = final_logits[:H, :W]
        pred_bin = (logits_c > 0.0).astype(np.uint8)
        return pred_bin, None, patch_records


def assemble_full_t2_maps(patch_records: List[Dict], pH: int, pW: int, tile_size: int = 448):
    fH, fW = pH // 8, pW // 8
    B_grid = np.zeros((192, fH, fW), dtype=np.float32)
    S_grid = np.zeros((96,  fH, fW), dtype=np.float32)
    for p in patch_records:
        py, px = p['coords']
        fy, fx = py // 8, px // 8
        t2 = p['t2'][0].numpy()
        B_grid[:, fy:fy+56, fx:fx+56] = t2[:192]
        S_grid[:, fy:fy+56, fx:fx+56] = t2[192:]
    return B_grid, S_grid


# ---------------------------------------------------------------------------
# Cohort Loading
# ---------------------------------------------------------------------------

def load_cohorts(consensus_csv: str, master_csv: str, ac_atten_csv: str) -> List[Dict]:
    df_meta   = pd.read_csv(consensus_csv)
    df_master = pd.read_csv(master_csv)
    df_prev   = pd.read_csv(ac_atten_csv)

    models = ['Base', 'A1', 'A2', 'B1', 'C1', 'D1', 'D2']
    bcols  = [f'{m}_bridge_events' for m in models]
    clean_mask  = (df_master[bcols] == 0).all(axis=1) & (df_master['gt_cc'] >= 2)
    clean_stems = set(df_master[clean_mask]['case_name'].tolist())

    c7_stems       = set(df_meta[df_meta['n_models_with_bridge'] == 7]['image_id'].tolist())
    d2_cured_stems = set(df_meta[df_meta['d2_transition_type'] == 'Cured_by_D2']['image_id'].tolist())

    c7a0            = df_prev[(df_prev['cohort'] == 'Consensus_7of7') & (df_prev['alpha'] == 0.0)]
    sensitive_stems = set(c7a0[c7a0['has_bridge'] == False]['image_id'].tolist())
    resistant_stems = set(c7a0[c7a0['has_bridge'] == True]['image_id'].tolist())

    samples = []
    for s in sorted(c7_stems):
        sub = 'Consensus_Resistant' if s in resistant_stems else 'Consensus_Sensitive'
        samples.append({'stem': s, 'cohort': 'Consensus_7of7', 'subgroup': sub})
    for s in sorted(d2_cured_stems):
        samples.append({'stem': s, 'cohort': 'Cured_by_D2', 'subgroup': 'Cured_by_D2'})
    for s in sorted(clean_stems):
        samples.append({'stem': s, 'cohort': 'Clean_Control', 'subgroup': 'Clean_Control'})
    return samples


# ---------------------------------------------------------------------------
# T2 Map Extraction & Caching
# ---------------------------------------------------------------------------

def extract_all_t2_maps(samples: List[Dict], engine: T2FeatureCaptureEngine, tile_size: int = 448) -> Dict[str, Dict]:
    cache = {}
    for item in tqdm(samples, desc='Extracting T2 maps'):
        stem     = item['stem']
        cohort   = item['cohort']
        subgroup = item['subgroup']

        img_bgr = cv2.imread(f'datasets/Crack500_ready/val/images/{stem}.jpg')
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(f'datasets/Crack500_ready/val/masks/{stem}.png', cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue
        target_bin = (gt_mask > 0).astype(np.uint8)

        H, W  = img_rgb.shape[:2]
        pad_h = (tile_size - H % tile_size) % tile_size
        pad_w = (tile_size - W % tile_size) % tile_size
        pH, pW = H + pad_h, W + pad_w

        pred_bin, _, patch_records = engine.run_image(img_rgb, tile_size=tile_size)
        if cohort in ('Consensus_7of7', 'Cured_by_D2'):
            neck_mask, crack_mask, bg_mask, _ = isolate_bridged_pairs_and_rois(pred_bin, target_bin)
        else:
            neck_mask, crack_mask, bg_mask, _ = isolate_clean_pairs_and_rois(target_bin)

        B_grid, S_grid = assemble_full_t2_maps(patch_records, pH, pW, tile_size)
        fH, fW = H // 8, W // 8
        T2_crop = np.concatenate([B_grid[:, :fH, :fW], S_grid[:, :fH, :fW]], axis=0)  # (288, fH, fW)

        m_neck  = (cv2.resize(neck_mask[:H, :W],  (fW, fH), interpolation=cv2.INTER_NEAREST) > 0)
        m_crack = (cv2.resize(crack_mask[:H, :W], (fW, fH), interpolation=cv2.INTER_NEAREST) > 0) & ~m_neck
        m_bg    = (cv2.resize(bg_mask[:H, :W],    (fW, fH), interpolation=cv2.INTER_NEAREST) > 0) & ~m_neck & ~m_crack

        cache[stem] = {
            'T2_crop': T2_crop,
            'm_neck':  m_neck,
            'm_crack': m_crack,
            'm_bg':    m_bg,
            'has_neck': bool(m_neck.any()),
            'n_neck':   int(m_neck.sum()),
            'subgroup': subgroup,
            'cohort':   cohort,
        }
    return cache


# ---------------------------------------------------------------------------
# Pixel Location Sampling & Dataset Construction
# ---------------------------------------------------------------------------

def sample_locations(mask: np.ndarray, max_n: int, rng: np.random.RandomState) -> np.ndarray:
    idxs = np.argwhere(mask)
    if len(idxs) == 0:
        return np.zeros((0, 2), dtype=int)
    if len(idxs) > max_n:
        idxs = idxs[rng.choice(len(idxs), max_n, replace=False)]
    return idxs


def extract_features_and_metadata(
    stems: List[str],
    cache: Dict[str, Dict],
    rng: np.random.RandomState,
    max_per_cls: int = MAX_PIX_PER_CLS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts raw 288-dim T2 vectors, labels, and image_id array.
    Returns:
        X: (N, 288) float32
        y: (N,) int (0=CRACK, 1=NECK, 2=BG)
        img_ids: (N,) str (image stem for grouping)
    """
    feats = []
    labels = []
    img_ids = []

    for stem in stems:
        d = cache[stem]
        feat_map = d['T2_crop']  # (288, fH, fW)

        # Neck
        n_locs = sample_locations(d['m_neck'], max_per_cls, rng)
        for (i, j) in n_locs:
            feats.append(feat_map[:, i, j])
            labels.append(LABEL_NECK)
            img_ids.append(stem)

        # Crack
        c_locs = sample_locations(d['m_crack'], max_per_cls, rng)
        for (i, j) in c_locs:
            feats.append(feat_map[:, i, j])
            labels.append(LABEL_CRACK)
            img_ids.append(stem)

        # Background
        b_locs = sample_locations(d['m_bg'], max_per_cls, rng)
        for (i, j) in b_locs:
            feats.append(feat_map[:, i, j])
            labels.append(LABEL_BG)
            img_ids.append(stem)

    if not feats:
        return np.zeros((0, 288), dtype=np.float32), np.zeros(0, dtype=int), np.zeros(0, dtype=object)

    return np.array(feats, dtype=np.float32), np.array(labels, dtype=int), np.array(img_ids, dtype=object)


# ---------------------------------------------------------------------------
# Stratified Snake Fold Allocation
# ---------------------------------------------------------------------------

def make_stratified_image_folds(bridge_stems: List[str], cache: Dict[str, Dict], n_folds: int = N_OUTER_FOLDS) -> Dict[str, int]:
    """
    Deterministically partitions bridge images into n_folds using snake allocation
    sorted by n_neck descending. Guarantees balanced neck pixels and exact image partition.
    """
    stems_sorted = sorted(bridge_stems, key=lambda s: cache[s]['n_neck'], reverse=True)
    assignments = {}
    for idx, s in enumerate(stems_sorted):
        cycle = (idx // n_folds) % 2
        if cycle == 0:
            fold = idx % n_folds
        else:
            fold = (n_folds - 1) - (idx % n_folds)
        assignments[s] = fold
    return assignments


# ---------------------------------------------------------------------------
# Projectors
# ---------------------------------------------------------------------------

def make_random_projection_matrix(n_features: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    P = rng.randn(n_features, k).astype(np.float32)
    norms = np.linalg.norm(P, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return P / norms


def fit_projector(X_train_sc: np.ndarray, cond_cfg: Dict[str, Any], seed: int):
    ctype = cond_cfg['type']
    k     = cond_cfg['k']
    if ctype in ('pca', 'null'):
        k_actual = min(k, X_train_sc.shape[1], X_train_sc.shape[0] - 1)
        pca = PCA(n_components=k_actual, random_state=seed)
        pca.fit(X_train_sc)
        return pca
    elif ctype == 'rand':
        return make_random_projection_matrix(288, k, seed)
    elif ctype == 'raw':
        return None
    else:
        raise ValueError(f'Unknown type {ctype}')


def transform_projector(X_sc: np.ndarray, cond_cfg: Dict[str, Any], proj_obj: Any) -> np.ndarray:
    ctype = cond_cfg['type']
    if ctype in ('pca', 'null'):
        return proj_obj.transform(X_sc).astype(np.float32)
    elif ctype == 'rand':
        return (X_sc @ proj_obj).astype(np.float32)
    elif ctype == 'raw':
        return X_sc.astype(np.float32)
    else:
        raise ValueError(f'Unknown type {ctype}')


# ---------------------------------------------------------------------------
# Inner Cross-Validation for Regularization C Selection
# ---------------------------------------------------------------------------

def tune_regularization_inner_cv(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    img_ids_train: np.ndarray,
    cond_cfg: Dict[str, Any],
    c_grid: List[float],
    n_inner_folds: int = N_INNER_FOLDS,
    seed: int = 42,
) -> float:
    """
    Selects best C strictly via inner GroupKFold by image on training images.
    Scaler and PCA are fit STRICTLY on inner-train images.
    """
    unique_imgs = sorted(list(set(img_ids_train)))
    # Deterministic inner fold allocation by image
    inner_assignments = {}
    for idx, s in enumerate(unique_imgs):
        inner_assignments[s] = idx % n_inner_folds

    inner_fold_scores: Dict[float, List[float]] = {C: [] for C in c_grid}

    for inf in range(n_inner_folds):
        in_val_imgs  = {s for s, f in inner_assignments.items() if f == inf}
        in_val_mask  = np.isin(img_ids_train, list(in_val_imgs))
        in_train_mask = ~in_val_mask

        X_itr, y_itr = X_train_raw[in_train_mask], y_train[in_train_mask]
        X_iva, y_iva = X_train_raw[in_val_mask],   y_train[in_val_mask]

        nc_iva_mask = (y_iva == LABEL_NECK) | (y_iva == LABEL_CRACK)
        if len(X_itr) < 10 or nc_iva_mask.sum() < 4 or len(np.unique(y_iva[nc_iva_mask])) < 2:
            continue

        # Fit scaler on inner-train
        in_scaler = StandardScaler()
        X_itr_sc = in_scaler.fit_transform(X_itr)
        X_iva_sc = in_scaler.transform(X_iva)

        # Fit projector on inner-train
        proj = fit_projector(X_itr_sc, cond_cfg, seed=seed + inf)
        X_itr_proj = transform_projector(X_itr_sc, cond_cfg, proj)
        X_iva_proj = transform_projector(X_iva_sc, cond_cfg, proj)

        if cond_cfg['type'] == 'null':
            # Shuffle labels in inner-train for null control
            rng_null = np.random.RandomState(seed + inf)
            y_itr = rng_null.permutation(y_itr)

        for C in c_grid:
            try:
                clf = LogisticRegression(C=C, max_iter=300, tol=1e-3, solver='lbfgs', random_state=seed)
                clf.fit(X_itr_proj, y_itr)
                proba = clf.predict_proba(X_iva_proj[nc_iva_mask])
                classes = list(clf.classes_)
                if LABEL_NECK in classes:
                    s = proba[:, classes.index(LABEL_NECK)]
                    auc = roc_auc_score((y_iva[nc_iva_mask] == LABEL_NECK).astype(int), s)
                    inner_fold_scores[C].append(float(auc))
            except Exception:
                pass

    # Select C with highest mean inner validation AUC
    best_c = c_grid[0]
    best_score = -1.0
    for C in c_grid:
        scores = inner_fold_scores[C]
        m = float(np.mean(scores)) if len(scores) > 0 else 0.5
        if m > best_score:
            best_score = m
            best_c = C

    return best_c


# ---------------------------------------------------------------------------
# Evaluation Routine
# ---------------------------------------------------------------------------

def compute_metrics(
    clf: LogisticRegression,
    X_proj: np.ndarray,
    y: np.ndarray,
    img_ids: np.ndarray,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes:
      - pooled_auc_nc: ROC-AUC pooled across all pixels
      - macro_auc_nc: mean of per-image ROC-AUCs (for images with both NECK & CRACK)
      - per-pixel scores, labels, and image_ids for bootstrapping
    """
    nc_mask = (y == LABEL_NECK) | (y == LABEL_CRACK)
    if nc_mask.sum() < 4 or len(np.unique(y[nc_mask])) < 2:
        return float('nan'), float('nan'), np.zeros(0), np.zeros(0), np.zeros(0)

    classes = list(clf.classes_)
    proba = clf.predict_proba(X_proj[nc_mask])
    if LABEL_NECK not in classes:
        return float('nan'), float('nan'), np.zeros(0), np.zeros(0), np.zeros(0)

    neck_col = classes.index(LABEL_NECK)
    scores = proba[:, neck_col]
    y_sub  = (y[nc_mask] == LABEL_NECK).astype(int)
    imgs_sub = img_ids[nc_mask]

    pooled_auc = float(roc_auc_score(y_sub, scores))

    # Per-image macro AUC
    img_aucs = []
    for uimg in np.unique(imgs_sub):
        im_mask = (imgs_sub == uimg)
        if len(np.unique(y_sub[im_mask])) == 2 and im_mask.sum() >= 4:
            try:
                im_auc = roc_auc_score(y_sub[im_mask], scores[im_mask])
                img_aucs.append(float(im_auc))
            except Exception:
                pass

    macro_auc = float(np.mean(img_aucs)) if len(img_aucs) > 0 else pooled_auc

    return pooled_auc, macro_auc, scores, y_sub, imgs_sub


# ---------------------------------------------------------------------------
# Image-Level Bootstrap CI Routine
# ---------------------------------------------------------------------------

def compute_image_bootstrap_ci(
    records_per_cond: Dict[str, Dict[str, Any]],
    n_boot: int = N_BOOTSTRAP_IMG,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Resamples test IMAGES with replacement (image-level cluster bootstrap),
    computes held-out AUC for each condition, and calculates 95% CIs.
    """
    rng = np.random.RandomState(seed)

    # Get universe of unique test images
    sample_cond = list(records_per_cond.keys())[0]
    all_imgs = np.unique(records_per_cond[sample_cond]['imgs'])
    n_imgs = len(all_imgs)

    boot_aucs: Dict[str, List[float]] = {c: [] for c in records_per_cond}

    for b in range(n_boot):
        boot_imgs = rng.choice(all_imgs, size=n_imgs, replace=True)
        # Count frequency of each image in this resample
        img_counts = pd.Series(boot_imgs).value_counts().to_dict()

        for c, rec in records_per_cond.items():
            scores = rec['scores']
            y_sub  = rec['y_sub']
            imgs   = rec['imgs']

            # Weight pixels by image sample count
            w_idx = []
            for i, img in enumerate(imgs):
                if img in img_counts:
                    w_idx.extend([i] * img_counts[img])

            if len(w_idx) == 0:
                continue
            w_idx = np.array(w_idx)
            y_b = y_sub[w_idx]
            s_b = scores[w_idx]

            if len(np.unique(y_b)) == 2:
                try:
                    auc_b = roc_auc_score(y_b, s_b)
                    boot_aucs[c].append(float(auc_b))
                except Exception:
                    pass

    ci_results = {}
    for c, vals in boot_aucs.items():
        if len(vals) > 0:
            ci_results[c] = {
                'boot_mean': float(np.mean(vals)),
                'boot_std':  float(np.std(vals)),
                'ci_lo':     float(np.percentile(vals, 2.5)),
                'ci_hi':     float(np.percentile(vals, 97.5)),
            }
        else:
            ci_results[c] = {'boot_mean': float('nan'), 'boot_std': float('nan'), 'ci_lo': float('nan'), 'ci_hi': float('nan')}

    return ci_results, boot_aucs


def compute_paired_contrasts_bootstrap(
    boot_aucs: Dict[str, List[float]],
    contrasts: List[Tuple[str, str, str]],
) -> List[Dict[str, Any]]:
    """
    Computes paired difference bootstrap distribution for each specified contrast.
    """
    rows = []
    for (name, c_a, c_b) in contrasts:
        vals_a = np.array(boot_aucs.get(c_a, []))
        vals_b = np.array(boot_aucs.get(c_b, []))
        min_len = min(len(vals_a), len(vals_b))
        if min_len == 0:
            continue
        diffs = vals_a[:min_len] - vals_b[:min_len]
        rows.append({
            'contrast': name,
            'cond_A':   c_a,
            'cond_B':   c_b,
            'diff_mean': float(np.mean(diffs)),
            'diff_std':  float(np.std(diffs)),
            'ci_lo':     float(np.percentile(diffs, 2.5)),
            'ci_hi':     float(np.percentile(diffs, 97.5)),
            'p_val_zero': float(np.mean(diffs <= 0) if np.mean(diffs) > 0 else np.mean(diffs >= 0)) * 2,
        })
    return rows


# ---------------------------------------------------------------------------
# Decision Gate Logic
# ---------------------------------------------------------------------------

def evaluate_decision_gates(
    df_summary: pd.DataFrame,
    df_contrasts: pd.DataFrame,
) -> Dict[str, str]:
    """
    Evaluates H_A, H_B, H_C and generates claim verdict table.
    """
    get_stat = lambda c, col: float(df_summary[df_summary['condition'] == c][col].values[0])
    get_diff = lambda name: df_contrasts[df_contrasts['contrast'] == name].iloc[0]

    # Contrast lookups
    c_192_32 = get_diff('PCA192 - PCA32')
    c_128_32 = get_diff('PCA128 - PCA32')
    c_32_r32 = get_diff('PCA32 - RP32')
    c_64_r64 = get_diff('PCA64 - RP64')
    c_128_r128= get_diff('PCA128 - RP128')
    c_192_128= get_diff('PCA192 - PCA128')
    c_raw_192= get_diff('raw288 - PCA192')

    test_192 = get_stat('pca_192', 'mean_test_auc_pooled')
    gap_raw  = get_stat('raw_288', 'mean_train_test_gap')

    verdicts = {}

    # H_A: Higher dimensions really contain usable signal
    # SUPPORTED if PCA128/192 > PCA32 stably and CI excludes 0
    if c_192_32['ci_lo'] > 0 and c_128_32['ci_lo'] > 0:
        verdicts['H_A'] = 'SUPPORTED'
    elif c_192_32['diff_mean'] > 0.02:
        verdicts['H_A'] = 'PLAUSIBLE BUT NOT PROVEN'
    else:
        verdicts['H_A'] = 'NOT SUPPORTED'

    # H_B: Basis selection itself has a meaningful effect
    # SUPPORTED only if PCA_k > RP_k at same dimension with paired CI excluding 0
    if c_32_r32['ci_lo'] > 0 and (c_64_r64['ci_lo'] > 0 or c_128_r128['ci_lo'] > 0):
        verdicts['H_B'] = 'SUPPORTED'
    elif c_32_r32['ci_lo'] > 0 or c_64_r64['ci_lo'] > 0:
        verdicts['H_B'] = 'PLAUSIBLE BUT NOT PROVEN'
    else:
        verdicts['H_B'] = 'NOT SUPPORTED'

    # H_C: There is a real T2 representation ceiling
    # Plateau around ~0.68-0.70 from k>=128 and CI of differences include 0
    diff_plateau = abs(c_192_128['diff_mean']) < 0.02 and abs(c_raw_192['diff_mean']) < 0.02
    if 0.65 <= test_192 <= 0.72 and diff_plateau:
        verdicts['H_C'] = 'SUPPORTED'
    else:
        verdicts['H_C'] = 'PLAUSIBLE BUT NOT PROVEN'

    # Claims table
    claims = {}
    claims['T2 contains measurable separation information'] = (
        'SUPPORTED' if test_192 > 0.65 and get_stat('pca_32', 'mean_test_auc_pooled') > 0.60 else 'PLAUSIBLE'
    )
    claims['Higher dimensionality recovers real signal'] = verdicts['H_A']
    claims['PCA basis gives unique advantage'] = verdicts['H_B']
    claims['T2 has stable ~0.69 ceiling'] = verdicts['H_C']
    claims['Evidence justifies T2 intervention'] = (
        'PLAUSIBLE BUT NOT PROVEN' if verdicts['H_C'] == 'SUPPORTED' else 'NOT SUPPORTED'
    )
    claims['Evidence favors upstream representation change'] = (
        'PLAUSIBLE BUT NOT PROVEN' if verdicts['H_C'] == 'SUPPORTED' and test_192 < 0.72 else 'NOT SUPPORTED'
    )

    return verdicts, claims


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_all_figures(
    df_summary: pd.DataFrame,
    df_folds: pd.DataFrame,
    df_contrasts: pd.DataFrame,
    output_dir: str,
):
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Test AUC vs Dimension
    pca_conds = ['pca_32', 'pca_64', 'pca_128', 'pca_192', 'pca_256', 'raw_288']
    pca_k     = [32, 64, 128, 192, 256, 288]
    rp_conds  = ['rand_32', 'rand_64', 'rand_128']
    rp_k      = [32, 64, 128]

    get_stat = lambda c, col: float(df_summary[df_summary['condition'] == c][col].values[0])

    pca_test = [get_stat(c, 'mean_test_auc_pooled') for c in pca_conds]
    pca_err  = [get_stat(c, 'std_test_auc_pooled')  for c in pca_conds]
    pca_tr   = [get_stat(c, 'mean_train_auc')       for c in pca_conds]

    rp_test  = [get_stat(c, 'mean_test_auc_pooled') for c in rp_conds]
    rp_err   = [get_stat(c, 'std_test_auc_pooled')  for c in rp_conds]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(pca_k, pca_test, yerr=pca_err, fmt='o-', color='#1f77b4', linewidth=2, capsize=4, label='PCA Test AUC (5-fold mean±std)')
    ax.errorbar(rp_k,  rp_test,  yerr=rp_err,  fmt='^--', color='#d62728', linewidth=1.5, capsize=4, label='Random Projection Test AUC')
    ax.plot(pca_k, pca_tr, 's:', color='#ff7f0e', alpha=0.7, label='PCA Train AUC (inner-regularized)')
    ax.axhline(0.50, color='gray', linestyle=':', label='Chance')
    ax.axhline(get_stat('shuffle_null', 'mean_test_auc_pooled'), color='#7f7f7f', linestyle='--', label='Shuffle Null Control')

    for k, val in zip(pca_k, pca_test):
        ax.annotate(f'{val:.3f}', (k, val + 0.008), ha='center', fontsize=8, fontweight='bold', color='#1f77b4')
    for k, val in zip(rp_k, rp_test):
        ax.annotate(f'{val:.3f}', (k, val - 0.015), ha='center', fontsize=8, color='#d62728')

    ax.set_xlabel('Representation Dimension (k)', fontsize=11)
    ax.set_ylabel('ROC-AUC NECK vs CRACK', fontsize=11)
    ax.set_title('Leakage-Safe Image-Level 5-Fold GroupKFold Frozen T2 Probe\nRegularization C Selected via Inner 4-Fold CV', fontsize=11)
    ax.set_xticks(pca_k)
    ax.set_ylim(0.42, 1.0)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower right', fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 't2_image_grouped_auc_vs_dim.png'), dpi=160)
    plt.close()

    # 2. Train-Test Gap vs Dimension
    gaps     = [get_stat(c, 'mean_train_test_gap') for c in pca_conds]
    gap_errs = [get_stat(c, 'std_train_test_gap')  for c in pca_conds]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(pca_k, gaps, yerr=gap_errs, fmt='o-', color='#9467bd', linewidth=2, capsize=4)
    ax.axhline(0.0, color='black', linewidth=0.8)
    ax.axhline(0.10, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label='Gap=0.10 threshold')
    for k, g in zip(pca_k, gaps):
        ax.annotate(f'{g:+.3f}', (k, g + 0.005), ha='center', fontsize=8)
    ax.set_xlabel('Dimension (k)', fontsize=11)
    ax.set_ylabel('Train AUC - Test AUC Gap', fontsize=11)
    ax.set_title('Train-Test AUC Gap vs Dimension (Monitoring Capacity Ceiling)', fontsize=11)
    ax.set_xticks(pca_k)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 't2_image_grouped_gap_vs_dim.png'), dpi=160)
    plt.close()

    # 3. Paired Contrasts with 95% CIs
    fig, ax = plt.subplots(figsize=(10, 4.5))
    c_names = df_contrasts['contrast'].tolist()
    c_means = df_contrasts['diff_mean'].tolist()
    c_los   = df_contrasts['ci_lo'].tolist()
    c_his   = df_contrasts['ci_hi'].tolist()
    y_pos   = np.arange(len(c_names))

    err_lo = [m - lo for m, lo in zip(c_means, c_los)]
    err_hi = [hi - m for m, hi in zip(c_means, c_his)]

    colors = ['#2ca02c' if lo > 0 else ('#d62728' if hi < 0 else '#1f77b4') for lo, hi in zip(c_los, c_his)]
    for m, y, lo_val, hi_val, col in zip(c_means, y_pos, err_lo, err_hi, colors):
        ax.errorbar(m, y, xerr=[[lo_val], [hi_val]], fmt='o', color='black', ecolor=col, capsize=5, elinewidth=2, markersize=7)
    ax.axvline(0.0, color='black', linestyle='--', linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(c_names, fontsize=9.5)
    ax.set_xlabel('Paired Difference in AUC (Image-Level Bootstrap 95% CI)', fontsize=10.5)
    ax.set_title('Paired Statistical Contrasts (1,000 Image-Level Cluster Bootstraps)\nGreen: strictly positive CI | Blue: crosses zero', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5, axis='x')
    for y, m, lo, hi in zip(y_pos, c_means, c_los, c_his):
        ax.annotate(f'{m:+.3f} [{lo:+.3f}, {hi:+.3f}]', (m, y + 0.22), ha='center', fontsize=8)
    ax.set_ylim(-0.5, len(c_names) - 0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 't2_image_grouped_paired_contrasts.png'), dpi=160)
    plt.close()


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main():
    print('=' * 80)
    print('Phase 6: Leakage-Safe Image-Level Regularized Frozen T2 Probe')
    print('Evaluation with 5-Fold GroupKFold strictly by Image ID')
    print('=' * 80)

    config_path = 'results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml'
    ckpt_path   = 'results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth'
    output_dir  = 'results/diagnostics/phase6_t2_frozen_probe_image_grouped'
    os.makedirs(output_dir, exist_ok=True)

    # 1. Initial Integrity Verification
    print('\n[1/7] Computing initial model & checkpoint integrity hashes...')
    init_file_hash = compute_file_hash(ckpt_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    model = load_model_from_checkpoint(config_path, ckpt_path, device=device)
    init_param_hash = compute_model_param_hash(model)

    EXPECTED_CKPT_SHA  = '147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66'
    EXPECTED_PARAM_SHA = '4aeda58ce6d2fb32f5b772cdbf92dec7fb913f9510b392910efe6674b0adf5d6'
    assert init_file_hash  == EXPECTED_CKPT_SHA,  f'Checkpoint mismatch! {init_file_hash}'
    assert init_param_hash == EXPECTED_PARAM_SHA, f'Param mismatch! {init_param_hash}'
    print(f'Initial Checkpoint SHA256: {init_file_hash}')
    print(f'Initial Param      SHA256: {init_param_hash}')
    print('Bitwise anchor check: PASSED.')

    # 2. Cohort Loading & Feature Extraction
    print('\n[2/7] Loading cohort metadata & extracting T2 maps...')
    samples = load_cohorts(
        'results/diagnostics/phase6_bridge_localization/bridge_consensus_per_image.csv',
        'results/diagnostics/phase6_d2_topology/topology_7models_master_paired.csv',
        'results/diagnostics/phase6_stem_ac_attenuation/ac_attenuation_per_sample_alpha.csv',
    )
    engine = T2FeatureCaptureEngine(model, device)
    t0 = time.time()
    cache = extract_all_t2_maps(samples, engine, tile_size=448)
    engine.cleanup()
    print(f'T2 extraction completed in {time.time() - t0:.1f}s.')

    # Post-extraction bitwise invariance check
    assert compute_model_param_hash(model) == init_param_hash, 'HARD LOCK: model parameters changed!'
    assert compute_file_hash(ckpt_path)    == init_file_hash,  'HARD LOCK: checkpoint file changed!'
    print('Post-extraction bitwise invariance: PASSED.')

    # Filter to Consensus Resistant images with NECK pixels
    resistant_stems = sorted([s for s, d in cache.items() if d['subgroup'] == 'Consensus_Resistant'])
    bridge_stems    = [s for s in resistant_stems if cache[s]['has_neck']]
    print(f'\nConsensus Resistant images total: {len(resistant_stems)}')
    print(f'Images with valid NECK pixels:     {len(bridge_stems)}')

    total_neck = sum(cache[s]['n_neck'] for s in bridge_stems)
    print(f'Total NECK pixel instances:        {total_neck} (avg {total_neck/len(bridge_stems):.1f}/image)')

    # 3. Stratified Snake Image-Fold Partition
    print('\n[3/7] Partitioning images into 5 GroupKFold outer folds...')
    fold_assignments = make_stratified_image_folds(bridge_stems, cache, n_folds=N_OUTER_FOLDS)

    fold_stats = []
    for f in range(N_OUTER_FOLDS):
        f_stems = [s for s, fold in fold_assignments.items() if fold == f]
        f_neck  = sum(cache[s]['n_neck'] for s in f_stems)
        fold_stats.append({'fold': f, 'n_images': len(f_stems), 'n_neck_px': f_neck, 'stems': f_stems})
        print(f'  Fold {f}: {len(f_stems)} images, {f_neck} NECK pixels')

    # 4. Outer 5-Fold GroupKFold Execution
    print('\n[4/7] Running 5-Fold GroupKFold Outer Loop with Inner CV Regularization Tuning...')
    rng_data = np.random.RandomState(RNG_BASE)

    fold_results = []
    records_per_cond: Dict[str, Dict[str, List]] = {
        c: {'scores': [], 'y_sub': [], 'imgs': []} for c in CONDITIONS
    }

    for fold_idx in range(N_OUTER_FOLDS):
        test_stems  = [s for s, fold in fold_assignments.items() if fold == fold_idx]
        train_stems = [s for s, fold in fold_assignments.items() if fold != fold_idx]

        # Extract features strictly by image membership
        X_tr_raw, y_tr, img_ids_tr = extract_features_and_metadata(train_stems, cache, rng_data, max_per_cls=MAX_PIX_PER_CLS)
        X_te_raw, y_te, img_ids_te = extract_features_and_metadata(test_stems,  cache, rng_data, max_per_cls=MAX_PIX_PER_CLS)

        # Fit StandardScaler strictly on training images
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_raw)
        X_te_sc = scaler.transform(X_te_raw)

        print(f'\n--- Outer Fold {fold_idx + 1}/{N_OUTER_FOLDS} (Train: {len(train_stems)} imgs, Test: {len(test_stems)} imgs) ---')

        for cond_name, cond_cfg in CONDITIONS.items():
            ctype = cond_cfg['type']
            k     = cond_cfg['k']
            seed  = RNG_BASE * 100 + fold_idx * 10 + k

            # 1. Inner CV to select best regularization parameter C
            best_c = tune_regularization_inner_cv(
                X_tr_raw, y_tr, img_ids_tr, cond_cfg, C_GRID,
                n_inner_folds=N_INNER_FOLDS, seed=seed,
            )

            # 2. Fit projector strictly on outer training data
            proj = fit_projector(X_tr_sc, cond_cfg, seed=seed)
            X_tr_proj = transform_projector(X_tr_sc, cond_cfg, proj)
            X_te_proj = transform_projector(X_te_sc, cond_cfg, proj)

            # 3. Fit classifier with chosen C
            y_train_fit = y_tr.copy()
            if ctype == 'null':
                # Shuffle labels in outer-train as null baseline
                rng_null = np.random.RandomState(seed)
                y_train_fit = rng_null.permutation(y_train_fit)

            clf = LogisticRegression(C=best_c, max_iter=500, tol=1e-3, solver='lbfgs', random_state=seed)
            clf.fit(X_tr_proj, y_train_fit)

            # 4. Evaluate on outer training set
            tr_pool_auc, tr_macro_auc, _, _, _ = compute_metrics(clf, X_tr_proj, y_tr, img_ids_tr)

            # 5. Evaluate on outer held-out test set
            te_pool_auc, te_macro_auc, te_scores, te_y_sub, te_imgs_sub = compute_metrics(clf, X_te_proj, y_te, img_ids_te)

            gap = tr_pool_auc - te_pool_auc if not (np.isnan(tr_pool_auc) or np.isnan(te_pool_auc)) else float('nan')

            fold_results.append({
                'fold':               fold_idx,
                'condition':          cond_name,
                'k':                  k,
                'type':               ctype,
                'best_c':             best_c,
                'train_auc':          tr_pool_auc,
                'test_auc_pooled':    te_pool_auc,
                'test_auc_macro':     te_macro_auc,
                'train_test_gap':     gap,
                'n_train_imgs':       len(train_stems),
                'n_test_imgs':        len(test_stems),
                'n_train_neck_px':    int((y_tr == LABEL_NECK).sum()),
                'n_test_neck_px':     int((y_te == LABEL_NECK).sum()),
            })

            # Accumulate predictions across outer folds for image-level cluster bootstrapping
            records_per_cond[cond_name]['scores'].extend(te_scores.tolist())
            records_per_cond[cond_name]['y_sub'].extend(te_y_sub.tolist())
            records_per_cond[cond_name]['imgs'].extend(te_imgs_sub.tolist())

            print(f'  {cond_name:<14} k={k:>3} C*={best_c:<6} | Test AUC: {te_pool_auc:.4f} (macro: {te_macro_auc:.4f}) | Train: {tr_pool_auc:.4f} | Gap: {gap:+.4f}')

    # Convert accumulated arrays
    for c in records_per_cond:
        records_per_cond[c]['scores'] = np.array(records_per_cond[c]['scores'])
        records_per_cond[c]['y_sub']  = np.array(records_per_cond[c]['y_sub'])
        records_per_cond[c]['imgs']   = np.array(records_per_cond[c]['imgs'])

    df_folds = pd.DataFrame(fold_results)
    df_folds.to_csv(os.path.join(output_dir, 't2_image_grouped_folds.csv'), index=False)

    # 5. Image-Level Cluster Bootstrapping & Paired Contrasts
    print('\n[5/7] Computing Image-Level Cluster Bootstrap (1,000 resamples)...')
    ci_results, boot_aucs = compute_image_bootstrap_ci(records_per_cond, n_boot=N_BOOTSTRAP_IMG, seed=RNG_BASE)

    contrasts_def = [
        ('PCA192 - PCA32',   'pca_192',  'pca_32'),
        ('PCA128 - PCA32',   'pca_128',  'pca_32'),
        ('PCA32 - RP32',     'pca_32',   'rand_32'),
        ('PCA64 - RP64',     'pca_64',   'rand_64'),
        ('PCA128 - RP128',   'pca_128',  'rand_128'),
        ('PCA192 - PCA128',  'pca_192',  'pca_128'),
        ('raw288 - PCA192',  'raw_288',  'pca_192'),
    ]
    paired_contrasts = compute_paired_contrasts_bootstrap(boot_aucs, contrasts_def)
    df_contrasts = pd.DataFrame(paired_contrasts)
    df_contrasts.to_csv(os.path.join(output_dir, 't2_image_grouped_contrasts.csv'), index=False)

    # 6. Summary Aggregation
    print('\n[6/7] Aggregating Summary Metrics across 5 Outer Folds...')
    summary_rows = []
    for cond_name in CONDITIONS:
        sub = df_folds[df_folds['condition'] == cond_name]
        ci  = ci_results[cond_name]
        summary_rows.append({
            'condition':            cond_name,
            'k':                    CONDITIONS[cond_name]['k'],
            'type':                 CONDITIONS[cond_name]['type'],
            'mean_test_auc_pooled': float(sub['test_auc_pooled'].mean()),
            'std_test_auc_pooled':  float(sub['test_auc_pooled'].std()),
            'mean_test_auc_macro':  float(sub['test_auc_macro'].mean()),
            'std_test_auc_macro':   float(sub['test_auc_macro'].std()),
            'mean_train_auc':       float(sub['train_auc'].mean()),
            'std_train_auc':        float(sub['train_auc'].std()),
            'mean_train_test_gap':  float(sub['train_test_gap'].mean()),
            'std_train_test_gap':   float(sub['train_test_gap'].std()),
            'boot_ci_lo':           ci['ci_lo'],
            'boot_ci_hi':           ci['ci_hi'],
            'boot_mean':            ci['boot_mean'],
            'boot_std':             ci['boot_std'],
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(output_dir, 't2_image_grouped_summary.csv'), index=False)

    # Print summary table
    print('\n' + '=' * 88)
    print(f'{"Condition":<14} {"k":>4} | {"Test Pooled":>11} {"Macro":>8} | {"Train AUC":>9} {"Gap":>7} | {"Bootstrap 95% CI":>18}')
    print('-' * 88)
    for _, r in df_summary.iterrows():
        p_str = f"{r['mean_test_auc_pooled']:.4f}±{r['std_test_auc_pooled']:.3f}"
        m_str = f"{r['mean_test_auc_macro']:.4f}"
        tr_str= f"{r['mean_train_auc']:.4f}"
        g_str = f"{r['mean_train_test_gap']:+.3f}"
        ci_str= f"[{r['boot_ci_lo']:.3f}, {r['boot_ci_hi']:.3f}]"
        print(f"{r['condition']:<14} {int(r['k']):>4} | {p_str:>11} {m_str:>8} | {tr_str:>9} {g_str:>7} | {ci_str:>18}")
    print('=' * 88)

    # Print paired contrasts table
    print('\nPaired Statistical Contrasts (1,000 Image-Level Cluster Bootstraps):')
    print('-' * 70)
    for _, r in df_contrasts.iterrows():
        print(f"  {r['contrast']:<20}: diff={r['diff_mean']:+.4f} (std={r['diff_std']:.4f})  95% CI: [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]")
    print('-' * 70)

    # 7. Decision Gates & Visualizations
    print('\n[7/7] Evaluating Hypothesis Decision Gates and Plotting...')
    verdicts, claims = evaluate_decision_gates(df_summary, df_contrasts)

    gate_lines = [
        '=' * 80,
        'DECISION GATES — Leakage-Safe Image-Level 5-Fold GroupKFold Frozen T2 Probe',
        '=' * 80,
        '',
        f"H_A: Higher dimensions really contain usable signal -> {verdicts['H_A']}",
        f"     (PCA192 - PCA32 CI: [{df_contrasts[df_contrasts['contrast']=='PCA192 - PCA32']['ci_lo'].values[0]:+.4f}, {df_contrasts[df_contrasts['contrast']=='PCA192 - PCA32']['ci_hi'].values[0]:+.4f}])",
        '',
        f"H_B: Basis selection itself has a meaningful effect -> {verdicts['H_B']}",
        f"     (PCA32 - RP32 CI:   [{df_contrasts[df_contrasts['contrast']=='PCA32 - RP32']['ci_lo'].values[0]:+.4f}, {df_contrasts[df_contrasts['contrast']=='PCA32 - RP32']['ci_hi'].values[0]:+.4f}])",
        f"     (PCA128 - RP128 CI: [{df_contrasts[df_contrasts['contrast']=='PCA128 - RP128']['ci_lo'].values[0]:+.4f}, {df_contrasts[df_contrasts['contrast']=='PCA128 - RP128']['ci_hi'].values[0]:+.4f}])",
        '',
        f"H_C: There is a real T2 representation ceiling -> {verdicts['H_C']}",
        f"     (PCA192 test: {df_summary[df_summary['condition']=='pca_192']['mean_test_auc_pooled'].values[0]:.4f}, PCA128: {df_summary[df_summary['condition']=='pca_128']['mean_test_auc_pooled'].values[0]:.4f}, raw288: {df_summary[df_summary['condition']=='raw_288']['mean_test_auc_pooled'].values[0]:.4f})",
        '',
        'CLAIM VERDICT TABLE:',
        '-' * 80,
    ]
    for claim, verd in claims.items():
        gate_lines.append(f"{claim:<50} | {verd}")
    gate_lines.append('-' * 80)

    gate_text = '\n'.join(gate_lines)
    print(gate_text)
    with open(os.path.join(output_dir, 'decision_gate.txt'), 'w', encoding='utf-8') as f:
        f.write(gate_text)

    plot_all_figures(df_summary, df_folds, df_contrasts, output_dir)

    # Final bitwise invariance verification
    final_param_hash = compute_model_param_hash(model)
    final_file_hash  = compute_file_hash(ckpt_path)
    assert final_param_hash == init_param_hash, 'HARD LOCK: model parameters changed!'
    assert final_file_hash  == init_file_hash,  'HARD LOCK: checkpoint file changed!'
    print('\nFinal Checkpoint SHA256: ' + final_file_hash)
    print('Final Param      SHA256: ' + final_param_hash)
    print('Final bitwise invariance check: PASSED.')
    print('\nPhase 6 Leakage-Safe Image-Level Grouped Frozen T2 Probe Completed Successfully.')


if __name__ == '__main__':
    main()
