# MamBOA: A State-Space Architecture for Differential Motion Synthesis in Video Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper **MamBOA: A State-Space Architecture for Differential Motion Synthesis in Video Recognition** by Mustafa Bora Çelik.

## Abstract

Fine-grained video recognition demands modeling subtle, sequential motion patterns that are invisible in any single frame — a regime where neither static appearance features nor hand-crafted frame differences suffice. We present **MamBOA**, a backbone-agnostic architecture that exploits the continuous hidden-state dynamics of selective state-space models to synthesize differential motion representations from pairs of consecutive spatial feature maps. 

The core insight is structural: rather than subtracting features explicitly, MamBOA interleaves tokens from two temporal positions into a single alternating sequence. Under this arrangement, the S6 state-space recurrence is structurally forced to produce hidden states that jointly encode both temporal observations at every spatial location, enabling the recurrence itself to accumulate and propagate the motion differential.

On the fine-grained Diving48 V2 benchmark, MamBOA achieves **85.02% Top-1 accuracy** with an image-pretrained backbone and **86.24%** with a video-pretrained backbone.

---

## 🏗 Architecture Overview

MamBOA resolves temporal ambiguity by decomposing each feature map pair into a differential motion representation and a complementary spatial representation.

![MamBOA Architecture](dual_path_pool_prediction/mamboa_overview_v2.svg)
*Figure 1: Overview of the proposed MamBOA architecture.*

### Interleaved Temporal Phase-shift Synthesis (ITPS)

The motion branch maps the interleaved spatial inputs into a distinct temporal subspace through the ITPS pipeline. This pipeline corrects receptive-field misalignment via learnable Fourier phase shifts, isolates multi-directional motion co-activations, and collapses the entangled temporal sequence into a signed motion feature map.

![ITPS Pipeline](dual_path_pool_prediction/fourier_shift_pipeline_v3.svg)
*Figure 2: Overview of the Interleaved Temporal Phase-shift Synthesis (ITPS) pipeline.*

### Clip-Level Extension and Full-Video Coverage

Rather than operating on a fixed set of sparsely sampled frames, the video-pretrained variant extends the differential framework to the clip level, sampling clips distributed across the entire video duration.

![Video MamBOA](dual_path_pool_prediction/videomamboa.svg)
*Figure 3: Clip-level extension of the MamBOA architecture.*

---

## 📊 Main Results (Diving48 V2)

### Comparison with State-of-the-Art

MamBOA demonstrates highly competitive performance across disparate backbone families on the strict Diving48 V2 benchmark.

| Model / Architecture | Backbone | Frames | Computational Cost (GFLOPs) | Top-1 Accuracy (%) | Top-5 Accuracy (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TSN (baseline) | ResNet-50 | 16 | 33.0 | 79.0 | NA |
| GST | ResNet-50 | 16 | 58.4 | 78.9 | NA |
| TSM | ResNet-50 | 16 | 65.0 | 83.2 | NA |
| SlowFast (16×8) | ResNet-101 | 64+16 | 213.0 | 77.6 | NA |
| TDN (Base) | ResNet-50 | 16 | 72.0 | 84.6 | NA |
| TimeSformer-HR | Transformer | 16 | 170.3 | 78.0 | NA |
| TimeSformer-L | Transformer | 96 | 2,380.0 | 81.0 | NA |
| VideoSwin-B | Transformer | 32 | 963.0 | 69.6 | 92.7 |
| ViViT-L | Transformer | 32 | 7,248.0 | 80.6 | 92.7 |
| SIFAR-B-12+ | Swin-B | 16 | 189.0 | 85.3 | 98.3 |
| SIFAR-B-14+ | Swin-B | 16 | 263.0 | **87.3** | **98.8** |
| **MamBOA (Ours, Image)** | VMamba-B | 30 × 16 | 30 × 132.5 | 85.02 | 98.53 |
| **MamBOA (Ours, Video)** | MViT-V2-S | Full-video | 407.6 | 86.24 | 97.72 |

### Backbone Agnostic Design

MamBOA serves as a backbone-agnostic neural primitive. Below is the framework evaluated across diverse architectural families using the image-pretrained formulation:

| Backbone Family | Architecture | Top-1 Acc (%) |
|---|---|---|
| **CNN Family** | ConvNeXt-V2-B | 82.49 |
| **Transformer Family** | Swin-B | 78.33 |
| **Mamba Family** | VMamba-B | 85.02 |

### Efficiency Analysis

MamBOA is designed with computational efficiency in mind. The temporal mechanism itself contains only 34.63M parameters, adding $\sim$2.1 GFLOPs per feature pair.

| Method | Frames | Total FLOPs (G) |
|---|---|---|
| VideoSwin-B | 32 | 963 |
| ViViT-L | 32 | 7248 |
| **MamBOA (MViT-V2-S)** | **All Frames** | **407.6** |

### Component Ablation Study

| Configuration | Top-1 (%) | $\Delta$ |
|---|---|---|
| **Full model (MamBOA)** | **85.02** | -- |
| w/o Multi-Scale Scan (Only Single-Scale) | 82.79 | -2.23 |
| w/o Learnable Phase Shift (no $\tau$) | 82.64 | -2.38 |
| w/o FiLM (unidirectional) | 82.23 | -2.79 |
| w/o dual-path pool (mean only) | 81.32 | -3.70 |
| w/o dual-path pool (Attention only) | 81.68 | -3.34 |
| w/o Attribute-specific heads | 83.20 | -1.82 |
| w/o OHEM | 84.21 | -0.81 |

---

## 🔍 Temporal Pooling Dynamics

We visualize the attention response curves ($\alpha_t$) of the dual-path pooling mechanism across temporal frame pairs. Peak attention dynamically aligns with the discriminative execution phases, confirming that MamBOA learns to isolate temporal dynamics over static appearance without explicit guidance.

![Attention Curve](dual_path_pool_prediction/havuz_concat.svg)
*Figure 4: Attention response curves from the dual-path pooling mechanism. Peak attention dynamically aligns with the discriminative execution phases.*

---

## 🚀 Getting Started

*(Placeholders for code instructions)*

### Prerequisites
- Python 3.9+
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- other dependencies in `requirements.txt`

### Installation

```bash
git clone https://github.com/your-username/MamBOA.git
cd MamBOA
pip install -r requirements.txt
```

### Data Preparation

Download the Diving48 V2 dataset and extract it to the `data/` folder. Ensure the annotation files are correctly linked.

### Training

To train the image-pretrained variant of MamBOA:
```bash
python train.py --config configs/mamboa_vmamba.yaml
```

To train the video-pretrained variant of MamBOA:
```bash
python train.py --config configs/mamboa_mvitv2.yaml
```

### Evaluation
```bash
python eval.py --config configs/mamboa_vmamba.yaml
```

## 📜 Citation

If you find this code or research helpful in your work, please consider citing:

```bibtex
@article{celik2024mamboa,
  title={MamBOA: A State-Space Architecture for Differential Motion Synthesis in Video Recognition},
  author={Çelik, Mustafa Bora},
  journal={arXiv preprint},
  year={2026}
}
```
