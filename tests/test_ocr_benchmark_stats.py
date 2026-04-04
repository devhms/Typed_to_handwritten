import pytest

from tools.ocr_benchmark import bootstrap_mean_ci, parse_args


def test_bootstrap_mean_ci_empty_values():
    import numpy as np

    stats = bootstrap_mean_ci([], iterations=200, alpha=0.05, rng=np.random.default_rng(1))
    assert stats["mean"] == 0.0
    assert stats["ci_lower"] == 0.0
    assert stats["ci_upper"] == 0.0


def test_bootstrap_mean_ci_bounds_include_mean_for_simple_data():
    import numpy as np

    values = [0.1, 0.2, 0.3, 0.4]
    stats = bootstrap_mean_ci(values, iterations=500, alpha=0.05, rng=np.random.default_rng(3))

    assert stats["ci_lower"] <= stats["mean"] <= stats["ci_upper"]
    assert stats["bootstrap_iterations"] == 500


@pytest.mark.parametrize(
    "argv",
    [
        ["ocr_benchmark.py", "--runs", "0"],
        ["ocr_benchmark.py", "--bootstrap-iterations", "20"],
        ["ocr_benchmark.py", "--ci-alpha", "0"],
        ["ocr_benchmark.py", "--ci-alpha", "1.0"],
    ],
)
def test_parse_args_validation_errors(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_accepts_valid_values(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "ocr_benchmark.py",
            "--runs",
            "3",
            "--bootstrap-iterations",
            "200",
            "--ci-alpha",
            "0.1",
        ],
    )
    args = parse_args()

    assert args.runs == 3
    assert args.bootstrap_iterations == 200
    assert args.ci_alpha == 0.1
