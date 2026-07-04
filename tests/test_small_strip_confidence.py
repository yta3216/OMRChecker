"""Confidence for 1-2 bubble strips (e.g. True/False questions).

Regression guard: get_local_threshold's `len(q_vals) < 3` branch used to leave
`max1` at its MIN_JUMP initial value, so the upstream confidence score
((max1 - MIN_JUMP) / CONFIDENT_SURPLUS) was always 0 for a 2-option question —
force-flagging every True/False answer for manual review even when read cleanly.
This asserts a cleanly-read T/F strip now reports non-zero confidence, without
changing the detection threshold.

Run from the OMRChecker root: `python -m pytest tests/`.
"""
from dotmap import DotMap

from src.core import ImageInstanceOps

_PARAMS = {
    "MIN_GAP": 30,
    "MIN_JUMP": 25,
    "CONFIDENT_SURPLUS": 5,
    "GLOBAL_THRESHOLD_MARGIN": 10,
    "JUMP_DELTA": 30,
    "PAGE_TYPE_FOR_THRESHOLD": "white",
}
_GLOBAL_THR = 150.0


def _confidence(max1: float) -> float:
    return min(1.0, max(0.0, (max1 - _PARAMS["MIN_JUMP"]) / _PARAMS["CONFIDENT_SURPLUS"]))


def _ops() -> ImageInstanceOps:
    return ImageInstanceOps(DotMap({"threshold_params": _PARAMS}))


def _max1(q_vals):
    _, max1 = _ops().get_local_threshold(
        q_vals, _GLOBAL_THR, no_outliers=(max(q_vals) - min(q_vals) < _PARAMS["MIN_GAP"]),
        plot_title=None, plot_show=False,
    )
    return max1


def test_true_false_marked_is_confident():
    # One bubble filled dark (~30), the other empty (~200): an unambiguous read.
    for q_vals in ([30.0, 200.0], [205.0, 40.0]):
        assert _confidence(_max1(q_vals)) > 0.0


def test_true_false_marked_detection_threshold_unchanged():
    # thr1 still splits the two bubbles (mean), so detection is unaffected.
    thr1, _ = _ops().get_local_threshold(
        [30.0, 200.0], _GLOBAL_THR, no_outliers=False, plot_title=None, plot_show=False
    )
    assert 30.0 < thr1 < 200.0


def test_uniform_small_strip_gets_confidence_from_global_distance():
    # Both bubbles alike (blank, ~200) and far from the global threshold: still a
    # confident detection, not a forced-zero.
    assert _confidence(_max1([200.0, 205.0])) > 0.0
