from mypackage.mathutils import add, sub

def test_add() -> None:
    assert add(3, 2) == 5

def test_sub() -> None:
    assert sub(3, 2) == 1
