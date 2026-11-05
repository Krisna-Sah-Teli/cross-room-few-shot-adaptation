# WiQ : Cross-Room Few-Shot Adaptation

**Course:** CS60055 - Ubiquitous Computing  
**Project:** Hackathon Challenge 2  
**Topic:** Environment-aware, label-efficient activity recognition from WiFi Channel State Information (CSI)

A complete Python pipeline, from the raw challenge archives to predictions: Nexmon PCAP decoding, quality control, a common 256-subcarrier amplitude format, fixed 150 × 256 windows, room identification (Task 1), a self-supervised CSI encoder (Task 2), cross-room few-shot adaptation (Task 3), and `run_episode.py` for unseen episodes. Every model uses all **256 subcarriers**. No MATLAB code, `.mat` files or the source authors' notebook are used.

## Results at a glance

All numbers are **validation** results on the fixed challenge validation sets (Room A: supplied `validation/` folder; Room B: 25 captures per activity, seed 42). Design choices were compared on these sets, as the challenge allows (§8); they are not unseen-test results.

**Task 1: room identification** (`App/results/task1/`)

| Input (all 256 subcarriers, 150-observation windows) | Validation accuracy |
|---|---:|
| Per-packet gain removed (headline, used as the router) | **100%** |
| Per-window z-score (alternative normalization) | 100% |
| Shortcut diagnostic: fraction of narrow 20 MHz packets only | 99.6% |
| Control: only full 80 MHz packets, 1 / 5 / 30 per window | 99.6% / 100% / 100% |

Window-length study (headline): 98.9% with 1 observation, 99.8% with 2, 100% from 5 observations upwards.

**Task 2: self-supervised encoder, frozen + linear probe** (`App/results/task2/`)

| Representation | Macro-F1 | Balanced accuracy |
|---|---:|---:|
| SSL encoder (contrastive, no labels) | **0.973** | 0.977 |
| Handcrafted features (baseline) | 0.982 | 0.986 |
| Same encoder, random weights (baseline) | 0.846 | 0.851 |

**Task 3: cross-room few-shot adaptation** (`App/results/task3/`; mean over 5 support draws, predicted-room routing; SSL features, per-room alignment, logistic regression, class balancing, and for target Room A the room-structure prior with its refinements: sway, SSL + handcrafted ensemble, cluster smoothing)

| Direction | K | Primary macro-F1 | Target-exclusive macro-F1 | Shared (SIT/STAND/WALK) macro-F1 |
|---|---:|---:|---:|---:|
| A → B | 1 | 0.241 | 0.314 | 0.193 |
| A → B | 5 | 0.388 | 0.647 | 0.216 |
| A → B | 10 | 0.406 | 0.692 | 0.215 |
| A → B | 20 | 0.461 | 0.824 | 0.220 |
| B → A | 1 | 0.449 | 0.414 | 0.485 |
| B → A | 5 | 0.668 | 0.750 | 0.585 |
| B → A | 10 | 0.667 | 0.783 | 0.551 |
| B → A | 20 | 0.696 | 0.807 | 0.585 |

Ablations (mean primary macro-F1 over both directions and all K): plain adaptation 0.364 → + class balancing 0.392 → + room-structure prior 0.464 → + refinements where the prior applies **0.497**. For target Room A the prior lifts the shared activities from ≈0.19 to ≈0.50 (`structure_prior_effect.csv`); it does not apply to target Room B. The refinements, added one at a time, lift target Room A from 0.554 to 0.570 (sway), 0.605 (+ ensemble) and 0.620 (+ smoothing); on target Room B the ensemble and smoothing would lower it (0.374 → 0.329 → 0.305), so they are used only with the prior (`refinement_effect.csv`). Routing contribution at K = 20: target A 0.330 without routing vs 0.696 with predicted-room routing; target B 0.229 vs 0.461 (predicted and oracle routing coincide: the router is 100% correct on validation).

Per activity at K = 5 for support draw 0, the official seed-42 support set used in `fig_confusion_K5.png` (`task3/table_per_class_K5.md`; LaTeX version for the report: `table_per_class_K5.tex`). \* marks the target-exclusive K-shot activities. This draw's primary macro-F1 is 0.622 for target Room A and 0.403 for target Room B; the table above averages five draws (0.668 and 0.388).

| Target Room A | P | R | F1 | Target Room B | P | R | F1 |
|---|---:|---:|---:|---|---:|---:|---:|
| SIT | 0.636 | 0.609 | 0.622 | EMPTY\* | 0.709 | 0.747 | 0.727 |
| STAND | 0.529 | 0.360 | 0.429 | LYING\* | 0.639 | 0.613 | 0.626 |
| WALK | 0.686 | 1.000 | 0.814 | SIT | 0.041 | 0.040 | 0.041 |
| SITDOWN\* | 0.643 | 0.720 | 0.679 | STAND | 0.236 | 0.280 | 0.256 |
| STANDUP\* | 0.500 | 0.480 | 0.490 | WALK | 0.403 | 0.333 | 0.365 |
| FALL\* | 0.762 | 0.640 | 0.696 |  |  |  |  |

**Main findings**
- About 70% of Room A observations are 20 MHz frames that fill only the lowest 20 MHz block (subcarriers 134–186); Room B is almost always full 80 MHz. This recording-configuration difference alone identifies the room (99.6%). With packet types matched, the rooms remain separable (100%), but room and recording session cannot be disentangled (§12).
- Target-exclusive activities are learned well from few labelled captures and improve steadily with K.
- Knowledge of the shared activities does not transfer between the rooms: Room B's consecutive observations carry no temporal information (block-averaged variation equals pure noise for every activity, including walking), so the motion cues that identify walking in Room A have no counterpart in Room B.
- For target Room A, the shared activities can instead be recovered from the target room's own structure, without any of its SIT/STAND/WALK labels: a label-free walking detector (walking is the only activity with strong fluctuation in Room A) and SIT/STAND examples taken from the start and end of the K-shot SITDOWN/STANDUP captures. No such structure is available when Room B is the target.
- People sway more when standing than when sitting: in Room A, this label-free cue lifts SIT/STAND/WALK at K = 5 from 0.506 to 0.544.
- In the configuration grid, SSL features are slightly ahead of handcrafted features (0.464 vs 0.443). Where the prior decides the shared activities (target Room A), averaging both models recognises the K-shot activities better than either alone (K = 5: 0.639 → 0.717).

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

Every random choice is seeded; re-running reproduces the reported numbers exactly. `python App/scripts/task3_cross_room.py --replot` redraws the Task 3 figures and the per-activity K = 5 table from saved results.

### 4. Tests

```bash
python App/tests/test_preprocessing.py   # decoder, truncated captures, packet-width QC, windowing, 256-column input
python App/tests/test_few_shot.py        # support sets: development only, exactly K captures, nested; balancing
python App/tests/test_room_structure.py  # structure prior: applicability, transition examples, revision rules
python App/tests/test_episode.py         # run_episode reproduces the Task 3 validation predictions; raw-input path
python App/tests/test_leakage.py         # hiding every forbidden label, or scrambling validation labels, changes no prediction
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
| `support.csv` (optional) | few-shot support for the target room: columns `window_file,label[,room]`; `window_file` is a window (`.npy`) or a raw capture (`.pcap`/`.csv`, all its windows get the label). For each label it contains, it replaces the default support. Files listed here are not labelled as inputs |
| `labels.csv` (optional) | ground truth `window_id,label[,room]`; only used to print scores |

Output: `predictions.csv` with columns `window_id, room, label, confidence`, plus `predictions_details.csv` with the router confidence and the input file of each window.

Robustness for unseen episodes: labels may be written as in the brief or the archive (`SIT-DOWN`, `Stand up`, `Sitting`, ...); a capture or window with fewer than 150 valid observations still gets a prediction, from its own observations repeated in order (marked "padded" in `predictions_details.csv`); a file that is not a readable CSI capture is skipped with a note instead of stopping the run.

Each window is routed to a room by the Task 1 router, then classified by that room's Task 3 model (shared activities learned from the other room; target-exclusive activities from K = 20 support captures per activity, the official seed-42 draw 0). For windows routed to Room A, the SSL and handcrafted models' probabilities are averaged and smoothed over clusters of Room A's unlabelled development windows, and the room-structure prior (with the sway vote) revises the shared activities. File names, folders and capture lengths are never used as features. On the full validation set, `run_episode.py` reproduces the reported Task 3 predictions exactly (tested).

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
  │                                         + room-structure prior where the K-shot activities include transitions,
  │                                           with sway, SSL + handcrafted ensemble and cluster smoothing
  ▼
run_episode.py → predictions.csv (window_id, room, label, confidence)
```

Design notes:
- **Decoding.** The BCM4366c0 packed format (11-bit I and Q mantissas, shared 6-bit exponent, 10-bit autoscaling) follows the public Nexmon CSI project (`github.com/seemoo-lab/nexmon_csi`, `unpack_float.c`); it is implemented in our own Python, and a vectorized version is tested to be bit-identical. Truncated captures keep their complete packets.
- **Windowing.** One deterministic rule for both rooms: ⌊n/150⌋ windows, centred. Captures shorter than 150 valid observations give no window (12 in Room A, 3 of them validation). Each window's provenance (capture, rows, settings hash) is in `App/results/data/windows_manifest.csv`.
- **Leakage controls.** Splits are by capture; validation never fits anything. Per-room alignment statistics and class-balancing weights come from unlabelled development windows; target-room SIT/STAND/WALK labels are never read; support captures are drawn only from development data (tested). Replacing every target-room development label except those of the K support captures by a placeholder, or scrambling the validation labels, changes no Task 3 prediction for any K or support draw, nor any prediction of the deployed system (`App/tests/test_leakage.py`).
- **SSL.** A three-layer temporal 1-D CNN (128 channels, mean + std pooling, 256-d embedding), trained with the NT-Xent contrastive loss on development windows of both rooms without any labels (augmentations: observation subsets, noise, smooth spectral tilt, subcarrier masking).
- **Room-structure prior** (`room_structure.py`). Used only for a target room whose K-shot activities include SITDOWN and STANDUP, i.e. Room A. (1) Walking detector: the within-block signal-to-noise of a window's full 80 MHz observations (all 256 subcarriers); the unlabelled target development windows are split by 2-means on its logarithm, and the high-fluctuation group is WALK. (2) Each SITDOWN/STANDUP support window also provides a STAND and a SIT example (its first and last 30%), and a small logistic regression learns SIT vs STAND in the target room. Windows that the adapted classifier assigns to a shared activity get SIT/STAND from this classifier, and detector windows become WALK unless the classifier says FALL. No target SIT/STAND/WALK label is used; the 30% segment and the 2-means split were fixed before scoring.
- **Refinements where the prior applies** (`task3.refinements`, `refinement_effect.csv`). When the prior decides the shared activities, the classifier mainly has to recognise the K-shot activities, and three label-free additions help: (1) *sway*: people sway more when standing than when sitting, so the order-free spread of a window's full 80 MHz observations also votes on SIT vs STAND (median split among the unlabelled development windows the classifier assigns to a shared activity and the walking detector does not flag), averaged with the posture classifier; (2) *ensemble*: the balanced probabilities of the SSL and handcrafted models are averaged; (3) *smoothing*: each window's probabilities are combined (geometric mean) with the mean probabilities of its k-means cluster of unlabelled target development windows (3 clusters per activity, standardized SSL features, 32 PCA components). For target Room B, whose shared activities must be transferred, the ensemble and smoothing do not help, so all three are used only together with the prior.
- **"No routing" in the routing figure** (`fig_routing_contribution.png`). Without the routing stage the system does not know which room a window comes from, so nothing room-specific can be used: the comparison model is one classifier for both rooms, standardized over both rooms' development windows, over the union of the two label spaces, and balanced on both rooms' unlabelled windows. The room-structure prior and its refinements also need the target room, so this model has neither. The gap between the curves is therefore what routing makes possible as a whole, not the router's decision alone; predicted and oracle routing coincide because the router is 100% correct on validation.

## Repository structure

```text
.
├── run_episode.py                     entry point for unseen episodes
├── requirements.txt                   pinned environment (all results)
├── requirements-notebook-freeze.txt   full environment snapshot of the notebook machine (reference)
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
│   ├── tests/                         test_preprocessing.py, test_few_shot.py, test_room_structure.py, test_episode.py, test_leakage.py
│   ├── main_notebook.ipynb            exploration notebook: file inspection, decoder audits, first experiments
│   └── results/
│       ├── data/                      manifests, QC report, preprocessing settings (tensors not tracked)
│       ├── task1/, task2/, task3/     metrics.csv, predictions, figures, tables, trained router and encoder
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
- **Not comparable with the original paper:** the paper evaluates a random 80/20 split within its experiments; this benchmark keeps whole captures apart and transfers across rooms, so its scores answer a different question and cannot be compared with the paper's.
- **Asymmetric source archives:** Room A starts as raw PCAP, decoded by our own Python, while Room B arrives already as amplitude CSV from an export we cannot inspect. Both are brought into the same 256-column amplitude form, but differences in how the amplitudes were produced remain part of the room/session difference.
- **Window-level metrics:** Room B validation has 3 windows per capture, so its 375 windows are not independent.
- **Room B data:** consecutive observations behave like independent noisy samples (no temporal coherence, values capped at 2,047, origin of the export unknown), which limits cross-room transfer of shared activities.
- **Room-structure prior:** it relies on two assumptions about the target room — walking is its only strongly fluctuating activity, and transition captures start and end in the stated postures. Both hold for Room A; the prior is not applicable to Room B. Its design, the refinements and the choice to use them only where the prior applies were compared on the validation set, so their gains are validation results.

