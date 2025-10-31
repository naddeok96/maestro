import math

from maestro.yolo.mix_builder import _proportional_counts
from maestro.yolo.yolo_txt import rewrite_label


def test_proportional_counts_matches_target_and_rounds():
    weights = {"clipart": 0.5, "comic": 0.3, "watercolor": 0.2}
    total = 10
    counts = _proportional_counts(weights, total)

    assert sum(counts.values()) == total
    for name, fraction in weights.items():
        expected = fraction * total
        assert math.isclose(counts[name], expected, rel_tol=0.0, abs_tol=1.0)


def test_proportional_counts_ensures_minimum_for_positive_weights():
    weights = {"a": 0.6, "b": 0.2, "c": 0.2}
    counts = _proportional_counts(weights, 10)
    assert all(counts[key] >= 1 for key, value in weights.items() if value > 0)


def test_rewrite_label_remaps_and_filters(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text(
        "\n".join(
            [
                "0 0.1 0.2 0.3 0.4",
                "5 0.5 0.5 0.2 0.2",
                "1 0.3 0.3 0.1 0.1",
            ]
        ),
        encoding="utf-8",
    )
    dst = tmp_path / "dst.txt"
    mapping = {0: 10, 1: 11}

    kept = rewrite_label(src, dst, mapping)

    assert kept is True
    assert dst.read_text(encoding="utf-8").splitlines() == [
        "10 0.1 0.2 0.3 0.4",
        "11 0.3 0.3 0.1 0.1",
    ]


def test_rewrite_label_returns_false_when_no_boxes(tmp_path):
    src = tmp_path / "src_empty.txt"
    src.write_text("5 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    dst = tmp_path / "dst_empty.txt"

    kept = rewrite_label(src, dst, {0: 1})

    assert kept is False
    assert dst.read_text(encoding="utf-8") == ""
