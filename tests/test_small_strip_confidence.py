"""Confidence scoring: raw signal from get_local_threshold + the caller's
contrast-relative formula (#426).

get_local_threshold returns (thr1, raw_signal):
- thr1 (detection threshold) is unchanged from the historical behaviour.
- raw_signal is the strip's confidence evidence in gray levels, WITHOUT the
  MIN_JUMP floor: largest interior gap for >=3 bubbles, spread for a clearly
  split 1-2 bubble strip, distance to global threshold for uniform strips.

The caller (read_omr_response) maps raw_signal to confidence:
    full = max(floor + 2, min(MIN_JUMP + CONFIDENT_SURPLUS,
               CONFIDENCE_REL_SCALE * sheet_contrast))
    conf = 0 if raw <= floor else min(1, (raw - floor) / (full - floor))
so faint-but-clean pencil sheets score high (their contrast range is small),
while gaps at sensor-noise scale score 0. `_confidence` below replicates that
formula; keep it in sync with src/core.py.

Run from the OMRChecker root: `python -m pytest tests/`.
"""
from dotmap import DotMap

from src.core import ImageInstanceOps

_PARAMS = {
    "MIN_GAP": 30,
    "MIN_JUMP": 25,
    "CONFIDENT_SURPLUS": 5,
    "CONFIDENCE_NOISE_FLOOR": 8,
    "CONFIDENCE_REL_SCALE": 0.35,
    "GLOBAL_THRESHOLD_MARGIN": 10,
    "JUMP_DELTA": 30,
    "PAGE_TYPE_FOR_THRESHOLD": "white",
}
_GLOBAL_THR = 150.0


def _confidence(raw_signal: float, sheet_contrast: float) -> float:
    floor = _PARAMS["CONFIDENCE_NOISE_FLOOR"]
    full = max(
        floor + 2.0,
        min(
            _PARAMS["MIN_JUMP"] + _PARAMS["CONFIDENT_SURPLUS"],
            _PARAMS["CONFIDENCE_REL_SCALE"] * sheet_contrast,
        ),
    )
    if raw_signal <= floor:
        return 0.0
    return min(1.0, (raw_signal - floor) / (full - floor))


def _ops() -> ImageInstanceOps:
    return ImageInstanceOps(DotMap({"threshold_params": _PARAMS}))


def _local(q_vals, no_outliers=None):
    if no_outliers is None:
        no_outliers = max(q_vals) - min(q_vals) < _PARAMS["MIN_GAP"]
    return _ops().get_local_threshold(
        q_vals, _GLOBAL_THR, no_outliers=no_outliers, plot_title=None, plot_show=False
    )


# --- raw_signal contract -----------------------------------------------------


def test_true_false_split_signal_is_spread():
    # One bubble filled (~30), one empty (~200): signal = the 170 spread.
    _, sig = _local([30.0, 200.0])
    assert sig == 170.0


def test_uniform_small_strip_signal_is_global_distance():
    # Both bubbles alike (~200) and 52.5 above global_thr: distance is the signal.
    _, sig = _local([200.0, 205.0])
    assert abs(sig - abs(202.5 - _GLOBAL_THR)) < 1e-9


def test_ambiguous_strip_signal_not_floored_to_min_jump():
    # Bubbles nearly evenly spread: largest window jump is ~20 — below MIN_JUMP.
    # The MIN_JUMP init must NOT leak into the signal (phantom confidence).
    _, sig = _local([100.0, 110.0, 120.0, 130.0], no_outliers=False)
    assert sig < _PARAMS["MIN_JUMP"]
    assert sig > 0.0


def test_clean_multi_bubble_signal_is_largest_gap():
    # One filled (30), three empty (~200): jump window spans the split.
    _, sig = _local([30.0, 198.0, 200.0, 202.0], no_outliers=False)
    assert sig >= 170.0


def test_uniform_multi_bubble_gets_signal_from_global_distance():
    # All-blank 4-option strip far from global_thr: confident uniform read.
    _, sig = _local([200.0, 201.0, 202.0, 203.0], no_outliers=True)
    assert sig >= 50.0


def test_uniform_strip_hugging_global_threshold_stays_low():
    _, sig = _local([152.0, 153.0, 154.0, 155.0], no_outliers=True)
    assert sig < _PARAMS["CONFIDENCE_NOISE_FLOOR"]


# --- detection regression: thr1 byte-identical -------------------------------


def test_detection_thresholds_unchanged():
    cases = [
        # (q_vals, no_outliers, expected thr1)
        ([30.0, 200.0], False, 115.0),  # 2-bubble split -> mean
        ([200.0, 205.0], True, _GLOBAL_THR),  # 2-bubble uniform -> global
        ([30.0, 198.0, 200.0, 202.0], False, 30.0 + (200.0 - 30.0) / 2),  # largest gap
        ([200.0, 201.0, 202.0, 203.0], True, _GLOBAL_THR),  # uniform no_outliers
    ]
    for q_vals, no_outliers, expected in cases:
        thr1, _ = _local(q_vals, no_outliers=no_outliers)
        assert abs(thr1 - expected) < 1e-9, (q_vals, thr1, expected)


def test_ambiguous_low_jump_threshold_unchanged():
    # Non-uniform, low-jump: thr1 keeps whatever the gap loop chose (255 when no
    # jump beat the MIN_JUMP-floored max1).
    thr1, _ = _local([100.0, 110.0, 120.0, 130.0], no_outliers=False)
    assert thr1 == 255


# --- confidence formula (mirrors read_omr_response) ---------------------------


def test_faint_pencil_sheet_scores_confident():
    # Faint sheet: contrast (p95-p5) ~= 45, clean gap of 18 gray levels.
    # Old formula: (18 - 25) / 5 -> 0.0 (flagged). New: full = 0.35*45 = 15.75,
    # conf = (18 - 8) / (15.75 - 8) -> 1.0.
    assert _confidence(18.0, 45.0) == 1.0


def test_faint_but_moderate_gap_above_review_threshold():
    # Gap 15 on a 50-contrast sheet: full = 17.5, conf = 7/9.5 ~= 0.74 > 0.7.
    assert _confidence(15.0, 50.0) > 0.7


def test_noise_floor_blocks_sensor_noise():
    assert _confidence(8.0, 45.0) == 0.0
    assert _confidence(3.0, 200.0) == 0.0


def test_normal_clean_sheet_full_confidence():
    # Pen sheet: contrast 170, gap 170. full capped at MIN_JUMP+SURPLUS = 30.
    assert _confidence(170.0, 170.0) == 1.0


def test_never_stricter_than_old_absolute_bar():
    # High-contrast sheet: full is capped at 30 (old bar), never above it.
    # A 30-gray-level gap is always fully confident regardless of contrast.
    assert _confidence(30.0, 255.0) == 1.0


def test_ambiguous_gap_scores_low_on_normal_sheet():
    # Gap 12 on a normal 170-contrast sheet: full = 30, conf = 4/22 ~= 0.18.
    assert _confidence(12.0, 170.0) < 0.3
