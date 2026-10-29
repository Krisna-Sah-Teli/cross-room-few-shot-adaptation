"""Building blocks for cross-room few-shot adaptation (Task 3).

Protocol for one direction (source room S -> target room T):
  - Shared activities (SIT, STAND, WALK) are learned from labelled S development windows.
  - T-exclusive activities are learned from K support captures per class, drawn
    from T development captures only (all windows of a selected capture are used).
  - Shared-activity labels of T are never read.
  - Evaluation uses the fixed T validation windows.

Domain alignment uses unlabelled development windows only: each room's features
are standardized with that room's own mean and standard deviation, so the two
rooms are centred on the same origin before a classifier is shared between them.

Optional class balancing also uses unlabelled development windows only: the
classifier's probabilities are reweighted so that, on average over those windows,
every activity is predicted equally often. Target class proportions are unknown,
so equal proportions are the neutral assumption, and they match macro-F1, which
weighs every activity equally. This prevents a classifier from systematically
starving some activities in a room it was not trained on.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score


def select_support(windows, target_room, exclusive_labels, k_values, seed, draw):
    """Nested K-shot support sets: {K: array of capture_ids}.

    For each exclusive class the eligible target development captures are sorted,
    shuffled once with a seed derived from (seed, draw, room, class), and the
    first K are taken, so the K=1 set is contained in K=5, and so on.
    """
    room_index = "AB".index(target_room)
    support = {k: [] for k in k_values}
    for class_index, label in enumerate(exclusive_labels):
        eligible = windows[(windows.room == target_room) & (windows.split == "development")
                           & (windows.label == label)]
        captures = np.array(sorted(eligible.capture_id.unique()))
        if len(captures) < max(k_values):
            raise ValueError(f"Only {len(captures)} development captures for {label} in Room {target_room}")
        order = np.random.default_rng([seed, draw, room_index, class_index]).permutation(len(captures))
        for k in k_values:
            support[k].extend(captures[order[:k]])
    return {k: np.array(v) for k, v in support.items()}


def fit_stats(F, eps=1e-6):
    return F.mean(axis=0), np.maximum(F.std(axis=0), eps)


def apply_stats(F, stats):
    mean, std = stats
    return (F - mean) / std


class PrototypeClassifier:
    """Nearest class mean under cosine similarity (a prototypical-network head)."""

    def __init__(self, temperature=10.0):
        self.temperature = temperature

    def fit(self, F, y):
        self.classes_ = np.array(sorted(set(y)))
        prototypes = np.stack([F[y == c].mean(axis=0) for c in self.classes_])
        self.prototypes_ = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)
        return self

    def predict_proba(self, F):
        F = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
        logits = self.temperature * F @ self.prototypes_.T
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)


def make_head(name, logreg_c):
    if name == "logreg":
        return LogisticRegression(C=logreg_c, class_weight="balanced", max_iter=3000)
    if name == "prototype":
        return PrototypeClassifier()
    raise ValueError(f"Unknown head {name!r}")


def balancing_weights(proba_reference, iterations=200, tolerance=1e-6):
    """Per-class weights that make the mean predicted probability equal for every class.

    `proba_reference` holds the classifier's probabilities on unlabelled windows
    (n, n_classes). The weights are found by iterative proportional fitting and
    are normalized to sum to 1.
    """
    p = np.asarray(proba_reference, dtype=np.float64)
    target = 1.0 / p.shape[1]
    weights = np.ones(p.shape[1])
    for _ in range(iterations):
        q = p * weights
        q /= q.sum(axis=1, keepdims=True)
        update = target / np.maximum(q.mean(axis=0), 1e-12)
        weights *= update
        if np.abs(update - 1.0).max() < tolerance:
            break
    return weights / weights.sum()


def predict(head, F, weights=None):
    """Labels and confidences; `weights` (from `balancing_weights`) rescale the class probabilities."""
    proba = head.predict_proba(F)
    if weights is not None:
        proba = proba * weights
        proba /= proba.sum(axis=1, keepdims=True)
    return head.classes_[proba.argmax(axis=1)], proba.max(axis=1)


def score(truth, pred, shared_labels, exclusive_labels):
    """Primary, target-exclusive and shared macro-F1 plus balanced accuracy on target labels."""
    target_labels = sorted(shared_labels) + sorted(exclusive_labels)
    return {
        "primary_macro_f1": f1_score(truth, pred, labels=target_labels, average="macro", zero_division=0),
        "exclusive_macro_f1": f1_score(truth, pred, labels=sorted(exclusive_labels), average="macro", zero_division=0),
        "shared_macro_f1": f1_score(truth, pred, labels=sorted(shared_labels), average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(truth, pred),
        "accuracy": float(np.mean(truth == pred)),
    }
