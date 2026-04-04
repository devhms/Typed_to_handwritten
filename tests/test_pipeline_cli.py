import pytest

from run_pipeline import parse_args


@pytest.fixture
def _argv(monkeypatch):
    def _set(argv):
        monkeypatch.setattr("sys.argv", argv)

    return _set


def test_parse_args_defaults(_argv):
    _argv(["run_pipeline.py"])
    args = parse_args()

    assert args.input_text_file is None
    assert args.headless is False
    assert args.severity == "standard"


def test_parse_args_legacy_mild_flag(_argv):
    _argv(["run_pipeline.py", "--mild"])
    args = parse_args()

    assert args.severity == "mild"


def test_parse_args_legacy_heavy_flag(_argv):
    _argv(["run_pipeline.py", "--heavy"])
    args = parse_args()

    assert args.severity == "heavy"


def test_parse_args_conflicting_legacy_flags(_argv):
    _argv(["run_pipeline.py", "--mild", "--heavy"])

    with pytest.raises(SystemExit):
        parse_args()
