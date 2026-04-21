import pytest

@pytest.fixture
def numbers() -> list[int]:
    return [1, 2, 3, 4]

def test_sum(numbers: list[int]):
    assert sum(numbers) == 10

def test_max(numbers: list[int]):
    assert max(numbers) == 4
