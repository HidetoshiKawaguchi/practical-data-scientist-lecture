def process_data(
    data: list[int], limit: int = 10, verbose: bool = False
) -> dict[str, int]:
    if verbose:
        print("processing...")
        result = {"count": 0, "sum": 0}
    for i in data:
        result["count"] += 1
        result["sum"] += i
    if result["count"] > limit:
        print("too many items")
    return result

