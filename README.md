# Cross-Room Few-Shot Adaptation

**Course:** CS60055 — Ubiquitous Computing  
**Project:** Hackathon Challenge 2  
**Topic:** Environment-aware, label-efficient activity recognition from WiFi Channel State Information (CSI)

A complete Python pipeline, from the raw challenge archives to predictions: Nexmon PCAP decoding, quality control, a common 256-subcarrier amplitude format, fixed 150 × 256 windows, room identification (Task 1), a self-supervised CSI encoder (Task 2), cross-room few-shot adaptation (Task 3), and `run_episode.py` for unseen episodes. Every model uses all **256 subcarriers**. No MATLAB code, `.mat` files or the source authors' notebook are used.

## Results at a glance

All numbers are **validation** results on the fixed challenge validation sets (Room A: supplied `validation/` folder; Room B: 25 captures per activity, seed 42). Design choices were compared on these sets, as the challenge allows (§8); they are not unseen-test results.

**Task 1: room identification** (`App/results/task1/`)

| Input (all 256 subcarriers, 150-observation windows) | Validation accuracy |
|---|---:|
| Per-packet gain removed (headline, used as the router) | **100%** |
| Per-window z-score (teammate's original setup) | 100% |
| Shortcut diagnostic: fraction of narrow 20 MHz packets only | 99.6% |
| Control: only full 80 MHz packets, 1 / 5 / 30 per window | 99.6% / 100% / 100% |

Window-length study (headline): 98.9% with 1 observation, 99.8% with 2, 100% from 5 observations upwards.

**Task 2: self-supervised encoder, frozen + linear probe** (`App/results/task2/`)

| Representation | Macro-F1 | Balanced accuracy |
|---|---:|---:|
| SSL encoder (contrastive, no labels) | **0.973** | 0.977 |
| Handcrafted features (baseline) | 0.982 | 0.986 |
| Same encoder, random weights (baseline) | 0.846 | 0.851 |

**Task 3: cross-room few-shot adaptation** (`App/results/task3/`; mean over 5 support draws, predicted-room routing; SSL features, per-room alignment, logistic regression, class balancing, and for target Room A the room-structure prior)

| Direction | K | Primary macro-F1 | Target-exclusive macro-F1 | Shared (SIT/STAND/WALK) macro-F1 |
|---|---:|---:|---:|---:|
| A → B | 1 | 0.241 | 0.314 | 0.193 |
| A → B | 5 | 0.388 | 0.647 | 0.216 |
| A → B | 10 | 0.406 | 0.692 | 0.215 |
| A → B | 20 | 0.461 | 0.824 | 0.220 |
| B → A | 1 | 0.405 | 0.366 | 0.445 |
| B → A | 5 | 0.572 | 0.639 | 0.506 |
| B → A | 10 | 0.619 | 0.718 | 0.520 |
| B → A | 20 | 0.617 | 0.730 | 0.504 |

Ablations (mean primary macro-F1 over both directions and all K): plain adaptation 0.364 → + class balancing 0.392 → + room-structure prior **0.464**. For target Room A the prior lifts the shared activities from ≈0.19 to ≈0.50 (`structure_prior_effect.csv`); it does not apply to target Room B. Routing contribution at K = 20: target A 0.330 without routing vs 0.617 with predicted-room routing; target B 0.229 vs 0.461 (predicted and oracle routing coincide: the router is 100% correct on validation).

**Main findings**
- About 70% of Room A observations are 20 MHz frames that fill only the lowest 20 MHz block (subcarriers 134–186); Room B is almost always full 80 MHz. This recording-configuration difference alone identifies the room (99.6%). With packet types matched, the rooms remain separable (100%), but room and recording session cannot be disentangled (§12).
- Target-exclusive activities are learned well from few labelled captures and improve steadily with K.
- Knowledge of the shared activities does not transfer between the rooms: Room B's consecutive observations carry no temporal information (block-averaged variation equals pure noise for every activity, including walking), so the motion cues that identify walking in Room A have no counterpart in Room B.
- For target Room A, the shared activities can instead be recovered from the target room's own structure, without any of its SIT/STAND/WALK labels: a label-free walking detector (walking is the only activity with strong fluctuation in Room A) and SIT/STAND examples taken from the start and end of the K-shot SITDOWN/STANDUP captures. No such structure is available when Room B is the target.
- SSL features are slightly ahead of handcrafted features in the final configuration (0.464 vs 0.443).

The required figures: `task3/fig_label_efficiency.png`, `task2/fig_tsne_by_room.png` and `fig_tsne_by_activity.png`, `task1/fig_room_accuracy_vs_window_length.png`, `task3/fig_routing_contribution.png`, `task3/fig_confusion_K5.png`.

## Quick start

### 1. Environment

Python 3.13. The versions in `requirements.txt` are the ones that produced every result.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

### 2. Data

Place the challenge archives (not tracked by Git) in `Source_Data/`:

```text
Source_Data/Experiment-2.zip     # Room A: 1,084 raw .pcap captures
Source_Data/Experiment-3.zip     # Room B: 1,201 amplitude .csv captures
```

### 3. Reproduce everything

Run from the project root, in this order. Times are approximate, on a laptop CPU.

```bash
python App/scripts/build_dataset.py               # ~10 min: decode, QC, intermediate CSVs, windows, manifests
python App/scripts/task1_room_identification.py   # ~5 min:  Task 1 + room router
python App/scripts/task2_self_supervised.py       # ~15 min: SSL encoder + linear probes + t-SNE
python App/scripts/task3_cross_room.py            # ~10 min: Task 3 grid, routing study, figures
python App/scripts/build_final_model.py           # seconds: the deployed system for run_episode.py
```

Every random choice is seeded; re-running reproduces the reported numbers exactly. `python App/scripts/task3_cross_room.py --replot` redraws the Task 3 figures from saved results.

### 4. Tests

```bash
python App/tests/test_preprocessing.py   # decoder, truncated captures, packet-width QC, windowing, 256-column input
python App/tests/test_few_shot.py        # support sets: development only, exactly K captures, nested; balancing
python App/tests/test_room_structure.py  # structure prior: applicability, transition examples, revision rules
python App/tests/test_episode.py         # run_episode reproduces the Task 3 validation predictions; raw-input path
```

## Running an episode

```bash
python run_episode.py EPISODE_DIR [--output FILE] [--rebuild-model]
```

`EPISODE_DIR` may contain:

| File | Meaning |
|---|---|
| `window_XXXX.npy` | one window: shape (150, 256), float32, CSI amplitude (the model-input contract, §10) |
| `*.pcap`, `*.csv` | raw captures (Room A Nexmon PCAP format, or 256 amplitude columns); they go through exactly the dataset-build preprocessing and become windows `<file>_w0`, `<file>_w1`, ... |
| `support.csv` (optional) | few-shot support for the target room: columns `window_file,label[,room]`. For each label it contains, it replaces the default support |
| `labels.csv` (optional) | ground truth `window_id,label[,room]`; only used to print scores |

Output: `predictions.csv` with columns `window_id, room, label, confidence`, plus `predictions_details.csv` with the router confidence and the input file of each window.

Each window is routed to a room by the Task 1 router, then classified by that room's Task 3 model (shared activities learned from the other room; target-exclusive activities from K = 20 support captures per activity, the official seed-42 draw 0). Windows routed to Room A are then revised by the room-structure prior. File names, folders and capture lengths are never used as features. On the full validation set, `run_episode.py` reproduces the reported Task 3 predictions exactly (tested).

Practice episodes built from validation data:

```bash
python App/scripts/make_demo_episode.py       # Episodes/demo_windows (window files) and Episodes/demo_raw (raw captures)
python run_episode.py Episodes/demo_windows
python run_episode.py Episodes/demo_raw
```

## Pipeline

```text
Source_Data/*.zip (read-only)
  │  Room A: PCAP → Ethernet/IPv4/UDP (port 5500) → Nexmon payload → BCM4366c0 packed CSI → 256 complex values → amplitude
  │  Room B: CSV (500 × 256) → seed-42 capture-level validation split
  ▼
Quality control (csi_quality.py): malformed, zero-information and repeated observations; packet width per 20 MHz block
  ▼
Intermediate_CSV/room_{A,B}/{development,validation}/   one CSV per capture, columns subcarrier_001 … subcarrier_256
  ▼
Windows: non-overlapping 150-observation windows centred in each capture → windows_X.npy + Windows/*/window_XXXXX.npy
  ▼
Normalization (per window, nothing fitted): log1p, then each observation's own mean removed (per-packet gain)
  ▼
Task 1 room router · Task 2 SSL encoder · Task 3 per-room alignment + logistic regression + class balancing
  │                                         + room-structure prior where the K-shot activities include transitions
  ▼
run_episode.py → predictions.csv (window_id, room, label, confidence)
```

Design notes:
- **Decoding.** The BCM4366c0 packed format (11-bit I and Q mantissas, shared 6-bit exponent, 10-bit autoscaling) follows the public Nexmon CSI project (`github.com/seemoo-lab/nexmon_csi`, `unpack_float.c`); it is implemented in our own Python, and a vectorized version is tested to be bit-identical. Truncated captures keep their complete packets.
- **Windowing.** One deterministic rule for both rooms: ⌊n/150⌋ windows, centred. Captures shorter than 150 valid observations give no window (12 in Room A, 3 of them validation). Each window's provenance (capture, rows, settings hash) is in `App/results/data/windows_manifest.csv`.
- **Leakage controls.** Splits are by capture; validation never fits anything. Per-room alignment statistics and class-balancing weights come from unlabelled development windows; target-room SIT/STAND/WALK labels are never read; support captures are drawn only from development data (tested).
- **SSL.** A three-layer temporal 1-D CNN (128 channels, mean + std pooling, 256-d embedding), trained with the NT-Xent contrastive loss on development windows of both rooms without any labels (augmentations: observation subsets, noise, smooth spectral tilt, subcarrier masking).
- **Room-structure prior** (`room_structure.py`). Used only for a target room whose K-shot activities include SITDOWN and STANDUP, i.e. Room A. (1) Walking detector: the within-block signal-to-noise of a window's full 80 MHz observations (all 256 subcarriers); the unlabelled target development windows are split by 2-means on its logarithm, and the high-fluctuation group is WALK. (2) Each SITDOWN/STANDUP support window also provides a STAND and a SIT example (its first and last 30%), and a small logistic regression learns SIT vs STAND in the target room. Windows that the adapted classifier assigns to a shared activity get SIT/STAND from this classifier, and detector windows become WALK unless the classifier says FALL. No target SIT/STAND/WALK label is used; the 30% segment and the 2-means split were fixed before scoring.

## Repository structure

```text
.
├── run_episode.py                     entry point for unseen episodes
├── requirements.txt                   pinned environment (all results)
├── requirements-notebook-freeze.txt   teammate's original environment snapshot (reference)
├── App/
│   ├── configs/experiment.yaml        every setting: paths, QC, windows, SSL, Task 3, final model
│   ├── scripts/
│   │   ├── build_dataset.py           Step 1: archives → intermediate CSVs → windows → manifests
│   │   ├── room_a_pcap.py             PCAP parsing and CSI decoding
│   │   ├── room_b_split.py            fixed Room B validation split
│   │   ├── csi_quality.py             observation QC and packet-width classes
│   │   ├── window_extraction.py       centred windowing rule
│   │   ├── room_fingerprint.py        normalization and handcrafted features
│   │   ├── task1_room_identification.py
│   │   ├── ssl_encoder.py             encoder, augmentations, contrastive loss
│   │   ├── task2_self_supervised.py
│   │   ├── few_shot_adaptation.py     support selection, alignment, heads, balancing, scoring
│   │   ├── room_structure.py          room-structure prior: walking detector, transition postures
│   │   ├── task3_cross_room.py
│   │   ├── episode.py                 deployed system: bundle, routing, prediction, raw inputs
│   │   ├── build_final_model.py
│   │   ├── make_demo_episode.py
│   │   └── archive_inspection.py, pcap_inspection.py, normalization.py, room_classifier.py,
│   │       dataset_builder.py         used by the exploration notebook
│   ├── tests/                         test_preprocessing.py, test_few_shot.py, test_room_structure.py, test_episode.py
│   ├── main_notebook.ipynb            exploration record: file inspection, decoder audits, first Task 1
│   └── results/
│       ├── data/                      manifests, QC report, preprocessing settings (tensors not tracked)
│       ├── task1/, task2/, task3/     metrics.csv, predictions, figures, trained router and encoder
│       └── final_model/               bundle.pkl used by run_episode.py
├── Source_Data/                       challenge archives (not tracked)
├── Intermediate_CSV/                  generated (not tracked)
└── Windows/                           generated window_XXXXX.npy files (not tracked)
```

Results follow the challenge formats: `metrics.csv` files use `experiment, target_room, K, metric, value`; prediction files use `window_id, room, label, confidence`.

## Data and activity labels

| Room | Activities | Development | Validation |
|---|---|---|---|
| A (Experiment 2, raw PCAP) | FALL, SIT, SITDOWN, STAND, STANDUP, WALK | 925 windows (1 per capture) | 147 windows |
| B (Experiment 3, amplitude CSV) | EMPTY, LYING, SIT, STAND, WALK | 3,228 windows (3 per capture) | 375 windows |

Shared activities: SIT, STAND, WALK. Room-specific activities are learned from K labelled target captures in Task 3.

## Limitations

- **Room vs session:** the rooms were recorded in separate sessions with different packet configurations; room identification may reflect session as well as environment.
- **Activity vs location:** Room A activities were recorded in partly different spatial zones.
- **Two rooms only;** no claim about unseen rooms. **Capture-disjoint, not subject-disjoint.**
- **Window-level metrics:** Room B validation has 3 windows per capture, so its 375 windows are not independent.
- **Room B data:** consecutive observations behave like independent noisy samples (no temporal coherence, values capped at 2,047, origin of the export unknown), which limits cross-room transfer of shared activities.
- **Room-structure prior:** it relies on two assumptions about the target room — walking is its only strongly fluctuating activity, and transition captures start and end in the stated postures. Both hold for Room A; the prior is not applicable to Room B. Its design was compared on the validation set, so its gain is a validation result.

## Notes

- `App/main_notebook.ipynb` is the teammate's exploration record. Its export cells write their own copies of the intermediate CSVs (names such as `sitting__Sitting_0.csv`) next to the pipeline's files; both contain identical values, and the pipeline only reads its own manifest paths.
- The technical report, contribution statement and AI-use disclosure required by §14 are submitted separately.

Prepared for the CS60055 Ubiquitous Computing Hackathon Challenge 2.
