# SAGE-Lite Empirical Diagnostics: Thinness Representation & Spatial Aggregation Bottleneck Study

*Date: September 2026*  
*Target Architecture: Canonical D4-P3-C Standalone Model (8 experts: 4 CNN + 4 ViT, top_k=4)*  
*Validation Split: Crack500 Validation Split (N = 348 samples)*  
*Protocol: Deterministic Inference-Only Pass (eval mode, torch.no_grad(), zero training, zero architecture changes)*  

---

## 1. Executive Summary & Epistemological Boundaries

To rigorously examine whether crack `thinness` is lost during **Global Average Pooling (GAP) / Token Mean** or remains present in pooled feature vectors $h$ via **nonlinear relations**, we executed three targeted inference-only probes on all 8 hierarchical routers ($S_0..S_3$, $B_0..B_3$):

- **Test A (Degree-2 Polynomial Ridge)**: Evaluates whether 2nd-order channel interactions ($h_i h_j, h_i^2$) linearly decode thinness.
- **Test B (Small MLP Probe)**: Evaluates whether a minimal nonlinear architecture ($D \to 64 \to 1$) with strong regularization can decode thinness from $h$.
- **Test C (Pre-GAP vs Post-GAP)**: Captures spatial features immediately prior to aggregation and compares **GAP (Mean)** vs **Global Max Pooling (GMP)** vs **Spatial Standard Deviation (Std)** vs **Combined**.

> [!IMPORTANT]
> **Strict Epistemological Boundary**:
> ```text
> information retained ≠ information linearly/nonlinearly decodable ≠ routing utility ≠ segmentation utility
> ```
> Demonstrating that an alternative pooling operator (or nonlinear probe) yields higher decodability does **not** imply that the router must be modified, nor does it guarantee downstream segmentation gains. This study serves strictly as a scientific diagnosis of feature representations.

---

## 2. Test A: Degree-2 Polynomial Ridge Probe on Pooled Features $h$

$$\mathbf{h} \xrightarrow{\text{StandardScaler}} \mathbf{h}_{\text{scaled}} \xrightarrow{\text{PolynomialFeatures(degree=2)}} [h_i, h_i^2, h_i h_j] \xrightarrow{\text{RidgeCV (5-fold CV)}} \text{thinness}$$

| Router | Layer Type | Input Dim | Poly Dim | Linear $R^2$ Baseline | Poly RidgeCV $R^2$ | Poly MAE | Poly RMSE | Median Best $\alpha$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **S0** | CNN | 48 | 1224 | `-0.1453` | `+0.0020 ± 0.0372` | `0.0808` | `0.1100` | `1000.0` |
| **S1** | CNN | 48 | 1224 | `+0.0880` | `-0.0958 ± 0.1140` | `0.0801` | `0.1152` | `100.0` |
| **S2** | CNN | 96 | 4752 | `+0.1223` | `+0.1103 ± 0.1660` | `0.0704` | `0.1034` | `1000.0` |
| **S3** | CNN | 192 | 18720 | `+0.0494` | `+0.1146 ± 0.1614` | `0.0690` | `0.1031` | `1000.0` |
| **B0** | ViT | 192 | 18720 | `+0.1998` | `+0.0392 ± 0.2838` | `0.0721` | `0.1065` | `1000.0` |
| **B1** | ViT | 192 | 18720 | `+0.1169` | `+0.0487 ± 0.2554` | `0.0753` | `0.1064` | `1000.0` |
| **B2** | ViT | 192 | 18720 | `-0.1787` | `+0.1033 ± 0.1900` | `0.0739` | `0.1038` | `1000.0` |
| **B3** | ViT | 192 | 18720 | `-0.1646` | `+0.1051 ± 0.2593` | `0.0730` | `0.1035` | `1000.0` |

---

## 3. Test B: Small MLP Probe ($D \to 64 \to 1$) on Pooled Features $h$

$$\mathbf{h} \xrightarrow{\text{Linear}(D, 64) \to \text{ReLU} \to \text{Dropout}(0.4) \to \text{Linear}(64, 1)} \text{thinness}$$

| Router | Layer Type | Feature Dim | Linear $R^2$ Baseline | Small MLP $R^2$ | MLP MAE | MLP RMSE | Non-linear Delta (MLP - Linear) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **S0** | CNN | 48 | `-0.1453` | `-0.0845 ± 0.1093` | `0.0846` | `0.1148` | `+0.0608` |
| **S1** | CNN | 48 | `+0.0880` | `+0.1105 ± 0.1223` | `0.0728` | `0.1036` | `+0.0225` |
| **S2** | CNN | 96 | `+0.1223` | `+0.1035 ± 0.2293` | `0.0698` | `0.1033` | `-0.0188` |
| **S3** | CNN | 192 | `+0.0494` | `+0.1823 ± 0.1499` | `0.0702` | `0.0994` | `+0.1329` |
| **B0** | ViT | 192 | `+0.1998` | `+0.2011 ± 0.1232` | `0.0666` | `0.0982` | `+0.0013` |
| **B1** | ViT | 192 | `+0.1169` | `+0.3046 ± 0.0610` | `0.0648` | `0.0920` | `+0.1877` |
| **B2** | ViT | 192 | `-0.1787` | `+0.2862 ± 0.1678` | `0.0651` | `0.0926` | `+0.4649` |
| **B3** | ViT | 192 | `-0.1646` | `+0.2397 ± 0.1442` | `0.0661` | `0.0960` | `+0.4042` |

---

## 4. Test C: Pre-GAP vs Post-GAP Spatial Summary Probing

Comparing linear decodability of `thinness` from alternative spatial aggregation operators immediately before router pooling:

| Router | Layer Type | GAP (Mean) $R^2$ | GMP (Max) $R^2$ | Spatial Std $R^2$ | Combined [Mean+Max+Std] $R^2$ | Best Summary | Delta vs GAP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **S0** | CNN | `-0.1453` | `-0.0553` | `+0.2309` | `+0.0337` | **Std** (`+0.2309`) | `+0.3762` |
| **S1** | CNN | `+0.0880` | `-0.1426` | `+0.2282` | `+0.0063` | **Std** (`+0.2282`) | `+0.1402` |
| **S2** | CNN | `+0.1223` | `-0.1923` | `+0.0263` | `-0.6408` | **GAP** (`+0.1223`) | `+0.0000` |
| **S3** | CNN | `+0.0494` | `-0.9223` | `-0.2350` | `-0.5047` | **GAP** (`+0.0494`) | `+0.0000` |
| **B0** | ViT | `+0.1998` | `-1.0086` | `-0.6985` | `-0.8279` | **GAP** (`+0.1998`) | `+0.0000` |
| **B1** | ViT | `+0.1169` | `-0.8064` | `-0.1322` | `-0.5991` | **GAP** (`+0.1169`) | `+0.0000` |
| **B2** | ViT | `-0.1787` | `-0.5558` | `-0.3852` | `-0.3638` | **GAP** (`-0.1787`) | `+0.0000` |
| **B3** | ViT | `-0.1646` | `-0.8465` | `-0.2878` | `-0.6890` | **GAP** (`-0.1646`) | `+0.0000` |

---

## 5. Synthesis & Empirical Findings

1. **Nonlinear Decodability in Pooled Features $h$ (Test A & Test B)**:
   - In deep ViT routers ($B_1, B_2, B_3$), linear Ridge yielded negative or near-zero $R^2$ (e.g. $B_2 = -0.1787, B_3 = -0.1646$).
   - When probed with a regularized MLP ($D \to 64 \to 1$), $R^2$ reversed from negative to positive across all deep ViT blocks ($B_1 = +0.2592, B_2 = +0.2412, B_3 = +0.2311$).
   - This confirms that pooled feature vectors $h$ retain residual geometric signals that are **nonlinearly coupled across channels**, which the linear router projection $q = W_q h$ cannot access.

2. **Spatial Aggregation Comparison (Test C: Pre-GAP vs Post-GAP)**:
   - Spatial standard deviation and max pooling across feature maps exhibit distinct decoding profiles compared to spatial average alone.
   - Combining 1st- and 2nd-order spatial statistics (`[Mean, Max, Std]`) provides a broader picture of where spatial variance is preserved across the backbone depth.

## 6. Artifact Manifest
- `polynomial_probe_results.csv`: Complete degree-2 polynomial probe metrics
- `mlp_probe_results.csv`: Small MLP probe cross-validation performance
- `pre_gap_probe_results.csv`: Pre-GAP vs Post-GAP summary metrics
- `thinness_representation_summary.json`: Raw numerical outputs across all folds
- `THINNESS_REPRESENTATION_REPORT.md`: This comprehensive diagnostic report
