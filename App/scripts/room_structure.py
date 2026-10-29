"""Target-room structure prior for the shared activities (SIT, STAND, WALK).

Applies only when the target room's K-shot activities include the transitions SITDOWN and
STANDUP (Room A as target). It never reads a target-room SIT/STAND/WALK label:
  1. Walking detector. Walking makes the CSI of a window fluctuate far more than any other
     activity. The motion score is the within-block signal-to-noise of a window's full 80 MHz
     observations (all 256 subcarriers). Unlabelled target development windows are split into
     two groups by 2-means on the log score; the high-fluctuation (low-score) group is WALK.
  2. Posture examples from transitions. A SITDOWN capture starts standing and ends sitting,
     STANDUP the reverse. The first and last 30% of every SITDOWN/STANDUP support window give
     STAND and SIT examples recorded in the target room itself. A logistic regression learns
     SIT vs STAND (and the support activities) from the mean log-shape of full 80 MHz
     observations over all 256 subcarriers.
The adapted classifier's predictions are then revised: a window flagged by the detector
becomes WALK (unless the classifier says FALL); other windows the classifier assigns to a
shared activity get SIT or STAND from the posture classifier.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from scripts.csi_quality import classify_band_coverage

TRANSITIONS = {"SITDOWN": ("STAND", "SIT"), "STANDUP": ("SIT", "STAND")}   # (posture at start, at end)
BLOCK = 5


def applicable(exclusive_labels):
    """The prior needs both transition activities among the target's K-shot activities."""
    return set(TRANSITIONS) <= set(exclusive_labels)


def _log_shape(rows):
    Z = np.log1p(rows.astype(np.float64))
    return Z - Z.mean(axis=1, keepdims=True)


def posture_shape(rows, quality):
    """Mean log-shape (256 values) of the full 80 MHz observations; NaN if fewer than 5."""
    full = rows[classify_band_coverage(rows, quality) == "full"]
    return _log_shape(full).mean(axis=0) if len(full) >= 5 else np.full(rows.shape[1], np.nan)


def motion_score(rows, quality):
    """Within-block signal-to-noise of the full 80 MHz observations (low = strong fluctuation)."""
    full = rows[classify_band_coverage(rows, quality) == "full"]
    if len(full) < 2 * BLOCK:
        return np.nan
    Z = _log_shape(full)
    n = (len(Z) // BLOCK) * BLOCK
    B = Z[:n].reshape(-1, BLOCK, Z.shape[1])
    return float(np.median(B.mean(axis=1).var(axis=0, ddof=1) / np.maximum(B.var(axis=1, ddof=1).mean(axis=0), 1e-9)))


def structure_features(X, quality):
    """(posture shapes (n, 256), motion scores (n,)) for raw windows X (n, 150, 256)."""
    shapes = np.stack([posture_shape(np.asarray(x), quality) for x in X])
    motion = np.array([motion_score(np.asarray(x), quality) for x in X])
    return shapes, motion


def walking_threshold(motion):
    """2-means split of the log motion scores of unlabelled windows; returns the score threshold."""
    log_m = np.log(motion[np.isfinite(motion) & (motion > 0)])
    if log_m.size < 2 or np.ptp(log_m) == 0:
        return 0.0                                     # no spread at all: the detector flags nothing
    lo, hi = np.percentile(log_m, [10, 90])
    for _ in range(100):
        cut = (lo + hi) / 2
        low, high = log_m[log_m < cut], log_m[log_m >= cut]
        if low.size == 0 or high.size == 0:
            break
        lo, hi = low.mean(), high.mean()
    return float(np.exp((lo + hi) / 2))


def posture_examples(windows, labels, quality, segment=0.3):
    """Training examples from support windows: (shapes, labels, source labels).

    Every support window is an example of its own activity; each SITDOWN/STANDUP window also
    yields a STAND and a SIT example from its first and last `segment` of observations.
    """
    shapes, ys, sources = [], [], []
    for window, label in zip(windows, labels):
        window = np.asarray(window)
        shapes.append(posture_shape(window, quality)); ys.append(label); sources.append(label)
        if label in TRANSITIONS:
            n = int(len(window) * segment)
            start, end = TRANSITIONS[label]
            shapes += [posture_shape(window[:n], quality), posture_shape(window[-n:], quality)]
            ys += [start, end]; sources += [label, label]
    return np.stack(shapes), np.array(ys), np.array(sources)


class StructurePrior:
    """Walking detector + transition-derived posture classifier for one target room."""

    def __init__(self, threshold, mean, std, classifier):
        self.threshold, self.mean, self.std, self.classifier = threshold, mean, std, classifier

    @classmethod
    def fit(cls, dev_shapes, dev_motion, example_shapes, example_labels, C=0.1):
        """Fit from unlabelled target development windows and support-derived posture examples."""
        mean = np.nanmean(dev_shapes, axis=0)
        std = np.maximum(np.nanstd(dev_shapes, axis=0), 1e-6)
        return cls(walking_threshold(dev_motion), mean, std, None).refit(example_shapes, example_labels, C)

    def refit(self, example_shapes, example_labels, C=0.1):
        """Same detector and standardization, posture classifier trained on new support examples."""
        if not {"SIT", "STAND"} <= set(example_labels):
            raise ValueError("posture examples must contain SIT and STAND (from SITDOWN/STANDUP support)")
        classifier = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
        classifier.fit(np.nan_to_num((example_shapes - self.mean) / self.std), example_labels)
        return StructurePrior(self.threshold, self.mean, self.std, classifier)

    def revise(self, shapes, motion, predicted, confidence, shared_labels):
        """Revised (labels, confidences) for windows the adapted classifier labelled `predicted`.

        Unchanged windows keep the classifier's confidence. SIT/STAND decisions take the posture
        classifier's probability (normalized over SIT and STAND). Detector WALK decisions take
        1 - score/threshold, clipped to [0.5, 1]: a monotone score, not a calibrated probability.
        """
        proba = self.classifier.predict_proba(np.nan_to_num((shapes - self.mean) / self.std))
        keep = np.isin(self.classifier.classes_, ["SIT", "STAND"])
        sit_stand = proba[:, keep] / np.maximum(proba[:, keep].sum(axis=1, keepdims=True), 1e-12)
        posture = self.classifier.classes_[keep][sit_stand.argmax(axis=1)]
        score = np.nan_to_num(motion, nan=np.inf)
        walking = (score < self.threshold) & (predicted != "FALL")
        shared = np.isin(predicted, shared_labels)
        labels = np.where(walking, "WALK", np.where(shared, posture, predicted))
        conf = np.where(walking, np.clip(1 - score / self.threshold, 0.5, 1.0),
                        np.where(shared, sit_stand.max(axis=1), confidence))
        return labels, conf
