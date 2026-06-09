# HPC Run Guide — DCAN on the RTX 5090

Step-by-step instructions to set up the environment, train the models, and produce **every
result table and graph** on the HPC workstation (RTX 5090, 32 GB; i9-14900K, 32 threads).

Companion docs: `CODE_STRUCTURE.md` (what the code is), `README.md` (quick reference),
`REPORT.md` (results & interpretation).

> Run everything from the `impl/` directory. Commands use `uv run` (the project's environment
> manager). All training/inference needs the GPU; tables and graphs are CPU-only.

---

## 1. Prerequisites

| Need | Detail |
|---|---|
| GPU | NVIDIA RTX 5090 (Blackwell, compute capability **sm_120**) |
| **PyTorch** | **≥ 2.7 with CUDA 12.8** (or newer). Older builds (e.g. cu121) **fail on Blackwell** with `no kernel image is available for execution on the device`. |
| CUDA driver | ≥ the one shipping CUDA 12.8 support (the captured box reports driver 595.x / CUDA 13.2 — fine). |
| Env manager | `uv` (https://docs.astral.sh/uv/) |
| Dataset | the `Dataset/` GlaS release one level up (`../Dataset`) |

---

## 2. Environment setup

```bash
cd impl

# 1) sync the project environment
uv sync

# 2) ensure a Blackwell-capable PyTorch (only if the default is too old)
uv pip install --upgrade "torch>=2.7" "torchvision>=0.22" --index-url https://download.pytorch.org/whl/cu128

# 3) verify the GPU is visible and usable
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); torch.randn(8,3,480,480,device='cuda').sum().item(); print('GPU OK')"
```
Expected last line: `... True NVIDIA GeForce RTX 5090` then `GPU OK`. If you see *no kernel
image*, the PyTorch build is too old → redo step 2.

---

## 3. Prepare the data (once)

```bash
uv run python scripts/prepare_data.py --dataset ../Dataset --out data
```
Produces:
- `data/{train,test_a,test_b}/images/` — RGB PNGs (train 85, test_a 60 = Part A, test_b 20 = Part B)
- `data/{...}/instances/` — uint16 instance masks (one id per gland)
- `data/splits/` — `fixed_train.txt` (71) / `fixed_val.txt` (14), `fold0/1_*` (stratified 2-fold),
  `all_train.txt` (all 85)

---

## 4. Sanity check (30 seconds, optional but recommended)

```bash
uv run python -c "
import torch; from dcan.models.dcan import DCAN
m=DCAN(pretrained=True, output_stride=8).cuda().eval()
o=m(torch.randn(1,3,480,480).cuda())
print('forward OK', tuple(o['obj_main'].shape), 'aux', len(o['obj_aux']))
"
```
Expect `forward OK (1, 2, 480, 480) aux 3`.

---

## 5. Run A — 2-fold cross-validation (the headline result)

Trains both folds, predicts the held-out fold, evaluates, and pools metrics over all 85 images
(each image is tested exactly once). Do this for DCAN **and** the FCN baseline.

```bash
uv run python scripts/run_cv.py --config configs/dcan.yaml
uv run python scripts/run_cv.py --config configs/fcn_baseline.yaml
```
Outputs:
- `runs/dcan_cv/fold0|fold1/` — per-fold checkpoints, predictions, `results.json`
- `runs/dcan_cv/results_cv.json` — **pooled mean ± std over 85 images** (the number to report)
- same under `runs/fcn_baseline_cv/`

> Tip: each `run_cv.py` runs two full trainings. To watch progress, training prints per-epoch
> `train_loss / val_loss / val_Dice`.

---

## 6. Run B — direct paper benchmark (train all 85 → Part A / Part B)

This matches the paper's exact protocol so your numbers line up with its Tables 1–3.

```bash
# DCAN
uv run python scripts/train.py    --config configs/dcan_all85.yaml
uv run python scripts/infer.py    --config configs/dcan_all85.yaml --ckpt runs/dcan_all85/best.pt --split test_a
uv run python scripts/evaluate.py --pred runs/dcan_all85/pred/test_a --gt data/test_a/instances --out runs/dcan_all85/results_test_a.json
uv run python scripts/infer.py    --config configs/dcan_all85.yaml --ckpt runs/dcan_all85/best.pt --split test_b
uv run python scripts/evaluate.py --pred runs/dcan_all85/pred/test_b --gt data/test_b/instances --out runs/dcan_all85/results_test_b.json

# FCN baseline (same four commands with configs/fcn_all85.yaml and runs/fcn_all85/)
uv run python scripts/train.py    --config configs/fcn_all85.yaml
uv run python scripts/infer.py    --config configs/fcn_all85.yaml --ckpt runs/fcn_all85/best.pt --split test_a
uv run python scripts/evaluate.py --pred runs/fcn_all85/pred/test_a --gt data/test_a/instances --out runs/fcn_all85/results_test_a.json
uv run python scripts/infer.py    --config configs/fcn_all85.yaml --ckpt runs/fcn_all85/best.pt --split test_b
uv run python scripts/evaluate.py --pred runs/fcn_all85/pred/test_b --gt data/test_b/instances --out runs/fcn_all85/results_test_b.json
```

---

## 7. Run C — single fixed-split (quick, optional)

A fast single model on the 71/14 split (useful for a quick smoke run before the full CV):
```bash
uv run python scripts/train.py    --config configs/dcan.yaml
uv run python scripts/infer.py    --config configs/dcan.yaml --ckpt runs/dcan/best.pt --split train
uv run python scripts/evaluate.py --pred runs/dcan/pred/train --gt data/train/instances --names data/splits/fixed_val.txt --out runs/dcan/results_val.json
```

---

## 8. Generate the result tables

```bash
# Cross-validated DCAN vs FCN + rank-sum (+ paper reference rows)
uv run python scripts/compare.py --paper \
    --runs DCAN=runs/dcan_cv/results_cv.json FCN=runs/fcn_baseline_cv/results_cv.json \
    --out runs/comparison_cv

# Paper benchmark: Part A
uv run python scripts/compare.py --paper \
    --runs DCAN=runs/dcan_all85/results_test_a.json FCN=runs/fcn_all85/results_test_a.json \
    --out runs/comparison_partA
# Part B: swap test_a → test_b
```
Each writes `<out>.md` and `<out>.csv` and prints the table + rank-sum.

---

## 9. Generate all graphs

```bash
# Training curves (loss / lr / w_a) + validation-loss comparison
uv run python scripts/export_curves.py --runs runs/dcan_cv/fold0 runs/fcn_baseline_cv/fold0

# Qualitative panels: Input | Ground truth | FCN (merged) | DCAN (separated)
uv run python scripts/visualize.py \
    --dcan runs/dcan_all85/pred/test_a --fcn runs/fcn_all85/pred/test_a \
    --images data/test_a/images --gt data/test_a/instances \
    --names test_a_1 test_a_2 test_a_3 --out runs/figures_partA
```
Figures produced:
- `runs/<run>/training_curves.png` — train/val loss, learning-rate schedule, `w_a` decay.
- `runs/val_loss_comparison.png` — DCAN vs FCN validation loss.
- `runs/figures_*/panel_<name>.png` and `montage_comparison.png` — the key visual showing DCAN
  separating touching glands where the FCN baseline merges them.

> `visualize.py` with no `--names` auto-picks the busiest validation images (most glands), which
> best showcase the contour-aware separation.

---

## 10. Artifact map — where everything lands

| Artifact | Path | Shows | Use for |
|---|---|---|---|
| CV metrics (per model) | `runs/{dcan,fcn_baseline}_cv/results_cv.json` | F1/Dice/Hausdorff mean±std over 85 | headline number |
| CV comparison table | `runs/comparison_cv.md` / `.csv` | DCAN vs FCN + rank-sum | report table |
| Part A/B metrics | `runs/{dcan,fcn}_all85/results_test_*.json` | scores on the paper's test sets | paper comparison |
| Part A/B table | `runs/comparison_partA.md`, `..._partB.md` | next to paper Tables 1–3 | report table |
| Training curves | `runs/*/training_curves.png` | convergence, LR/`w_a` schedule | slides |
| Val-loss comparison | `runs/val_loss_comparison.png` | DCAN vs FCN training | slides |
| Qualitative panels | `runs/figures*/panel_*.png`, `montage_comparison.png` | gland separation | **key demo visual** |
| Checkpoints | `runs/*/best.pt` (best val-Dice), `last.pt` | trained weights | reproduce / inference |
| Predictions | `runs/*/pred/<split>/*.png` | instance maps (uint16) | re-evaluate / visualise |
| TensorBoard | `runs/*/logs/` | live curves | `tensorboard --logdir runs` |

---

## 11. Expected results & how to read them

- **DCAN should beat the FCN baseline** on object-Dice and object-Hausdorff and win on
  **rank-sum** — that is the contour-awareness benefit (cleaner boundaries, separated glands).
- **Part A > Part B** for both (Part B is mostly malignant/deformed — every method drops there;
  the paper notes the same).
- Absolute numbers should rise well above the 2 GB-laptop run (Dice ~0.75) toward the paper's
  range, but **may not match the paper's exact decimals** (ImageNet vs PASCAL-VOC init + the
  original's heavy tuning). Method fidelity is reproduced; exact numbers are not guaranteed.
  See `REPORT.md §7` for the full gap explanation.

---

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `no kernel image is available for execution on the device` | PyTorch too old for Blackwell → install `torch>=2.7` cu128 (section 2). |
| CUDA out of memory | Lower `batch_size` in the config (8 → 4 → 2). The atrous /8 backbone is memory-heavier; 32 GB should hold batch 8, but reduce if needed. |
| Slow data loading | Ensure `num_workers: 16` in the config (the box has 32 threads). |
| Albumentations deprecation warnings | Harmless; ignore. |
| `best.pt` not created | It is saved on the first val-Dice evaluation (`eval_every`, default 5) — make sure `max_epochs ≥ eval_every`. |
| Want live curves | `uv run tensorboard --logdir runs` then open the printed URL. |

---

## 13. Rough time estimates (RTX 5090)

| Step | Approx time |
|---|---|
| `uv sync` + torch install | a few minutes (once) |
| `prepare_data.py` | < 1 minute |
| One training run (100 epochs, atrous /8, batch 8) | tens of minutes |
| `run_cv.py` (one model = 2 trainings + infer + eval) | roughly an hour |
| Inference on a test split | seconds–minutes |
| Tables + graphs | seconds |

(These are estimates; the 5090 is far faster than the laptop the project was prototyped on.)

---

## 14. One-block copy-paste (full pipeline)

```bash
cd impl
uv sync
uv run python -c "import torch;assert torch.cuda.is_available();print('GPU:',torch.cuda.get_device_name(0))"
uv run python scripts/prepare_data.py --dataset ../Dataset --out data

# Headline: 2-fold CV for both models
uv run python scripts/run_cv.py --config configs/dcan.yaml
uv run python scripts/run_cv.py --config configs/fcn_baseline.yaml
uv run python scripts/compare.py --paper --runs DCAN=runs/dcan_cv/results_cv.json FCN=runs/fcn_baseline_cv/results_cv.json --out runs/comparison_cv

# Paper benchmark: train all 85, evaluate Part A and Part B
for m in dcan fcn; do
  uv run python scripts/train.py --config configs/${m}_all85.yaml
  for s in test_a test_b; do
    uv run python scripts/infer.py    --config configs/${m}_all85.yaml --ckpt runs/${m}_all85/best.pt --split $s
    uv run python scripts/evaluate.py --pred runs/${m}_all85/pred/$s --gt data/$s/instances --out runs/${m}_all85/results_$s.json
  done
done
uv run python scripts/compare.py --paper --runs DCAN=runs/dcan_all85/results_test_a.json FCN=runs/fcn_all85/results_test_a.json --out runs/comparison_partA
uv run python scripts/compare.py --paper --runs DCAN=runs/dcan_all85/results_test_b.json FCN=runs/fcn_all85/results_test_b.json --out runs/comparison_partB

# Graphs
uv run python scripts/export_curves.py --runs runs/dcan_cv/fold0 runs/fcn_baseline_cv/fold0
uv run python scripts/visualize.py --dcan runs/dcan_all85/pred/test_a --fcn runs/fcn_all85/pred/test_a --images data/test_a/images --gt data/test_a/instances --out runs/figures_partA
```

> Note: in `configs/*_all85.yaml` the FCN model name is `fcn_baseline`; the loop above assumes
> config files `configs/fcn_all85.yaml` and run dir `runs/fcn_all85/` (both already provided).
