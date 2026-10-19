# Cross-Room Few-Shot Adaptation

**Course:** CS60055 — Ubiquitous Computing  
**Project:** Hackathon Challenge 2  
**Topic:** Cross-Room Few-Shot Adaptation using Channel State Information (CSI)

## Overview

This project investigates room identification and cross-room activity recognition using Wi-Fi Channel State Information (CSI). The objective is to explore whether CSI representations can support activity recognition across different environments, particularly when only a small number of labeled examples are available in the target room.

The project is organized around three tasks:

1. **Room identification and environment fingerprinting** — distinguish the recording environment using CSI windows and evaluate the effect of observation-window length.
2. **Self-supervised learning** — learn useful representations from CSI data.
3. **Cross-room few-shot adaptation** — evaluate adaptation between rooms with different activity classes and limited labeled target-room examples.

## Current Progress

### Task 1: Room Identification

A Logistic Regression classifier with standardized, hand-crafted CSI features was implemented as the room-identification baseline.

The evaluation uses a capture-grouped held-out development split to reduce the risk of windows from the same capture appearing in both training and test sets. A fixed validation set is also evaluated.

#### Window-length experiment

Separate classifiers were trained and evaluated using windows containing 25, 50, 75, 100, 125, and 150 CSI observations.

| CSI observations | Held-out test accuracy | Fixed validation accuracy |
|---:|---:|---:|
| 25 | 100% | 100% |
| 50 | 100% | 100% |
| 75 | 100% | 100% |
| 100 | 100% | 100% |
| 125 | 100% | 100% |
| 150 | 100% | 100% |

Balanced accuracy was also 100% at every tested length, with zero errors on both evaluation sets.

These results indicate that the two rooms are distinguishable using the extracted CSI features in the current dataset. Even the shortest tested window, containing 25 observations, achieved perfect observed room-identification accuracy.

**Evaluation note:** The validation metrics are window-level metrics. Room B has multiple windows per capture, so those windows are not necessarily independent. Results on the available splits also do not guarantee generalization to unseen recording sessions, devices, or environments.

### Tasks 2 and 3

Self-supervised representation learning and cross-room few-shot adaptation are the next stages of the project. Their implementation and evaluation results will be documented here as they are completed.

## Data and Activity Labels

The current dataset contains CSI recordings from two rooms.

- **Room A:** Activities include fall, sit, sitdown, stand, standup, and walk.
- **Room B:** Activities include empty, lying, sitting, standing, and walking.

The shared canonical activity labels are:

- `SIT`
- `STAND`
- `WALK`

Room-specific activity labels are retained as exclusive classes for the cross-room adaptation experiments.

## Preprocessing

The current preprocessing workflow includes:

- Converting recordings into fixed-length CSI windows.
- Maintaining window metadata, including capture path, activity, split, and window boundaries.
- Separating development and validation data.
- Applying `log1p` followed by per-window z-score normalization.
- Extracting statistical and temporal features for room classification.

Normalization is applied per window. Validation data is processed independently rather than using validation data to fit preprocessing statistics.

## Repository Organization

The project uses Python scripts for reusable processing logic and Jupyter notebooks for orchestration, validation, experiments, results, and visualizations.

```text
.
Hackathon-2/
├── App/
│   ├── config/
│   ├── results/
│   ├── scripts/
│   ├── src/
│   └── main_notebook.ipynb
│
├── Intermediate_csv/
│   ├── room_A/
│   │   ├── development/
│   │   └── validation/
│   └── room_B/
│       ├── development/
│       └── validation/
│
├── Source_Data/
│   ├── Experiment-2.zip
│   └── Experiment-3.zip
│
├── requirements.txt
├── README.md
└── .gitignore
```


### Directory descriptions

- **`App/config/`** — Configuration and experiment settings.
- **`App/results/`** — Evaluation metrics, experiment outputs, and dataset manifests.
- **`App/scripts/`** — Reusable data preparation, normalization, feature extraction, and utility scripts.
- **`App/src/`** — Core model and application source code.
- **`App/main_notebook.ipynb`** — Main notebook for coordinating experiments, running checks, and presenting tables and visualizations.
- **`Intermediate_csv/`** — Intermediate CSI data organized by room and data split. CSV files are excluded from Git to avoid committing generated or potentially large datasets.
- **`Source_Data/`** — Original experiment archives. Experiment ZIP files are excluded from Git and must be obtained separately.
- **`requirements.txt`** — Python dependencies required to run the project.
- **`.gitignore`** — Rules for excluding local environments, generated files, CSV data, and source archives.

### Data availability

The source experiment archives and intermediate CSV files are not tracked by Git. Place the required archives in `Source_Data/`, then prepare the intermediate data in the corresponding room and split directories before running the notebook.


*The tree is illustrative; update it to match the files actually tracked in Git. Additional manifests, notebooks, data files, and experiment outputs may exist.*

## Environment Setup

Use a Python environment compatible with the libraries imported by the project.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project dependencies required by the notebooks and scripts. If a `requirements.txt` file is provided, use:

```bash
pip install -r requirements.txt
```

If no dependency file exists yet, create one after checking the actual imports and versions used by the project.

Launch Jupyter:

```bash
jupyter notebook
```

Open the project notebook and run the cells in order.

## Reproducibility and Data Handling

- Preserve the defined development and validation splits.
- Keep capture-level grouping when creating held-out evaluation splits.
- Do not use validation data to fit preprocessing transformations or model parameters.
- Record experiment settings and evaluation metrics.
- Avoid committing credentials, private data, or large generated artifacts without checking repository and course requirements.

## Status

- [x] Task 1: Room-identification baseline
- [x] Task 1: Window-length evaluation
- [ ] Task 2: Self-supervised learning
- [ ] Task 3: Cross-room few-shot adaptation

## Acknowledgements

Prepared as part of the CS60055 Ubiquitous Computing Hackathon Challenge 2.
