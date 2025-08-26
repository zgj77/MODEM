import numpy as np


def ensemble(errors, final_fraction=0.02, vote_threshold=10):

    if errors.ndim != 3 or not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("Expected finite non-negative errors [R, steps, time]")
    if not 0 < final_fraction < 1:
        raise ValueError("final_fraction must lie strictly between 0 and 1")
    if not 0 <= vote_threshold < errors.shape[0] * errors.shape[1]:
        raise ValueError("Voting threshold must be smaller than the number of votes")
    reference = errors[0, -1].sum(dtype=np.float64)
    totals = errors.sum(-1, dtype=np.float64)

    fractions = np.clip(reference / np.maximum(totals, 1e-12) * final_fraction, 0, 1)
    thresholds = np.empty(errors.shape[:2])
    votes = np.zeros(errors.shape[-1], dtype=np.int32)
    for r in range(errors.shape[0]):
        for k in range(errors.shape[1]):
            fraction = fractions[r, k]
            thresholds[r, k] = np.quantile(errors[r, k], 1 - fraction)

            votes += errors[r, k] > thresholds[r, k]
    return (votes > vote_threshold).astype(np.int64), votes, thresholds, fractions


def spans(labels):
    edges = np.diff(np.r_[0, np.asarray(labels, dtype=np.int64), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def binary_metrics(prediction, labels):
    prediction, labels = np.asarray(prediction, bool), np.asarray(labels, bool)
    tp = int(np.sum(prediction & labels))
    fp = int(np.sum(prediction & ~labels))
    fn = int(np.sum(~prediction & labels))
    tn = int(np.sum(~prediction & ~labels))
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": precision, "recall": recall,
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def evaluate(prediction, labels):
    prediction = np.asarray(prediction, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if prediction.shape != labels.shape:
        raise ValueError("Prediction and label lengths must match (no silent truncation)")
    adjusted = prediction.copy()
    delays, detected = [], 0
    for begin, end in spans(labels):
        hits = np.flatnonzero(prediction[begin:end])
        if len(hits):
            adjusted[begin:end] = 1
            delays.append(int(hits[0]))
            detected += 1
        else:

            delays.append(int(end - begin))
    return {"raw": binary_metrics(prediction, labels),
            "point_adjusted": binary_metrics(adjusted, labels),
            "add": float(np.mean(delays)) if delays else None,
            "anomaly_spans": len(delays), "detected_spans": detected,
            "missed_span_policy": "delay_equals_span_length"}
