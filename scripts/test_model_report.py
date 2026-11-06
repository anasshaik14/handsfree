"""Self-check for model_report.py's report-only metric helpers.

Covers the two bits with real logic (not sklearn passthroughs): the reject-
option handling in unified_metrics (an engine that can decline to answer must
not invent a phantom class in the macro averages, but must still be charged
for the refusal in accuracy), and the head-pose facing-vector angular error.

Run: python -m pytest scripts/test_model_report.py
"""
import numpy as np

from model_report import facing_angular_deg, unified_metrics


def test_unified_metrics_perfect_and_known_confusion():
    assert unified_metrics([[5, 0], [0, 5]])["accuracy"] == 1.0
    # 3 of 4 correct: one actual-0 predicted as 1
    m = unified_metrics([[1, 1], [0, 2]])
    assert m["accuracy"] == 0.75
    assert m["error_rate"] == 0.25
    assert m["recall_macro"] == 0.75          # class0 0.5, class1 1.0
    assert m["fnr_macro"] == 0.25             # mirror of recall


def test_reject_column_excluded_from_macro_but_charged_to_accuracy():
    """3-class engine with a 4th 'declined' column that is never ground truth."""
    cm = [[8, 0, 0, 2],      # 8 right, 2 refused
          [0, 10, 0, 0],
          [0, 0, 10, 0],
          [0, 0, 0, 0]]      # nothing is ever actually 'declined'
    real = list(range(3))
    m = unified_metrics(cm, score_labels=real)

    # refusals are wrong answers: 28 correct of 30 (values are rounded to 4dp)
    assert abs(m["accuracy"] - 28 / 30) < 1e-4
    # ...but they are not false positives for any real word, so precision on
    # the three real classes stays perfect
    assert m["precision_macro"] == 1.0
    # they ARE misses for class 0: recall 0.8, 1.0, 1.0 -> macro 0.9333
    assert abs(m["recall_macro"] - (0.8 + 1.0 + 1.0) / 3) < 1e-4

    # the phantom class would otherwise drag every macro average down by 1/4
    naive = unified_metrics(cm)
    assert naive["precision_macro"] < m["precision_macro"]
    assert naive["accuracy"] == m["accuracy"]     # accuracy is unaffected either way


def test_facing_angular_error_zero_when_identical_and_ignores_roll():
    ypr = np.array([[10.0, -5.0, 3.0], [-30.0, 12.0, 0.0]])
    assert np.allclose(facing_angular_deg(ypr, ypr), 0.0, atol=1e-6)

    # roll spins the face about the forward axis: it must not move the vector
    rolled = ypr.copy()
    rolled[:, 2] += 40.0
    assert np.allclose(facing_angular_deg(ypr, rolled), 0.0, atol=1e-6)


def test_facing_angular_error_matches_pure_yaw_and_pitch_offsets():
    true = np.zeros((1, 3))
    for axis in (0, 1):                     # yaw, then pitch
        pred = np.zeros((1, 3))
        pred[0, axis] = 15.0
        assert abs(float(facing_angular_deg(true, pred)[0]) - 15.0) < 1e-6

    # a combined 3-4-5-style offset exceeds either axis alone but stays under
    # their sum -- the exact reason per-axis MAE cannot express this
    pred = np.array([[15.0, 15.0, 0.0]])
    ang = float(facing_angular_deg(true, pred)[0])
    assert 15.0 < ang < 30.0
