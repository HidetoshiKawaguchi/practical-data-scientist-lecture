
def add(a, b):
    return a + b

def add(a: int, b: int) -> int:
    return a + b

def format_price(name: str, price: float, discount: float) -> str:
    return f"{name}: {price * (1 - discount)} 円"

def average_score(scores: list[float]) -> float:
    return sum(scores)/len(scores)

def get_score(student_scores: dict[str, float], name: str) -> float | str:
    return student_scores.get(name, "No Data")

def get_score(student_scores: dict[str, float], name: str) -> float:
    if name in student_scores:
        return student_scores[name]
    else:
        return 0.0

# 独自のクラス定義
class Student:
    def __init__(self, name: str, score: float) -> None:
        self.name = name
        self.score = score

def is_passing(student: Student) -> bool:
    return student.score >= 60.0

