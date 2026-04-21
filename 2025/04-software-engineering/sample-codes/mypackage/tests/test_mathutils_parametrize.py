import pytest
from mypackage.mathutils import add

@pytest.mark.parametrize(
    "a, b, expected",
    [
        (3, 2, 5),
        (0, 0, 0),
        (-1, 1, 0),
    ],
)
def test_add(a: int, b: int, expected: int) -> None:
    assert add(a, b) == expected



