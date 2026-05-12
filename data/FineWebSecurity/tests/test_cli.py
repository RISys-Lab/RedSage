import pytest

from fineweb_security.cli import check_progress, download_subset, filter_bert


@pytest.mark.parametrize(
    "builder",
    [
        filter_bert.build_parser,
        check_progress.build_parser,
        download_subset.build_parser,
    ],
)
def test_cli_help(builder, capsys):
    parser = builder()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])

    assert exc_info.value.code == 0
    assert "usage:" in capsys.readouterr().out

