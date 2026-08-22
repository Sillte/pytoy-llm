from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        ("foo.py", "foo.py", True),
        ("foo.py", "*.py", True),
        ("foo.txt", "*.py", False),
        ("src/main.py", "src/*.py", True),
        ("src/utils/main.py", "src/*.py", False),
        ("src/main.py", "src/**", True),
        ("src/utils", "src/**", True),
        ("src/utils/main.py", "src/**", False),
        ("src/main.py", "**/*.py", True),
        ("src/utils/main.py", "**/*.py", True),
        ("src/main.py", "other/**", False),
        ("src/__pycache__/a.pyc", "**/__pycache__/**", True),
        ("src/src2/__pycache__/a.pyc", "**/__pycache__/**", True),
    ],
)
def test_path_match(
    path: str,
    pattern: str,
    expected: bool,
) -> None:
    assert Path(path).match(pattern) == expected
