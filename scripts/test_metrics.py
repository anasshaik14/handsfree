"""Self-check for metrics.py — catches wiring bugs (wrong axis, shape, key
names), not sklearn correctness. Run: python -m pytest scripts/test_metrics.py"""
import numpy as np

from metrics import classification_metrics, gaze_angular_error_deg, regression_metrics


def test_classification_metrics_binary():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])          # 3/4 correct
    y_prob = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]])
    out = classification_metrics(y_true, y_pred, y_prob)
    assert out["accuracy"] == 0.75
    assert out["n_classes"] == 2
    assert out["confusion_matrix"] == [[1, 1], [0, 2]]
    assert 0.0 <= out["roc_auc"] <= 1.0
    assert out["false_positive_rate"] == 0.5   # 1 false positive / 2 negatives


def test_regression_metrics_perfect_and_biased():
    y_true = np.array([[0.0, 10.0], [5.0, -5.0]])
    y_pred = np.array([[1.0, 10.0], [6.0, -5.0]])   # +1 deg bias on axis 0, exact on axis 1
    out = regression_metrics(y_true, y_pred, ["pitch", "yaw"])
    assert out["mae_deg_pitch"] == 1.0
    assert out["mae_deg_yaw"] == 0.0
    assert out["bias_deg_pitch"] == 1.0


def test_gaze_angular_error_zero_when_identical():
    rad = np.array([[0.1, 0.2], [-0.1, 0.3]])
    out = gaze_angular_error_deg(rad, rad)
    assert out["angular_error_deg_mean"] < 1e-4


if __name__ == "__main__":
    test_classification_metrics_binary()
    test_regression_metrics_perfect_and_biased()
    test_gaze_angular_error_zero_when_identical()
    print("all metrics self-checks passed")
