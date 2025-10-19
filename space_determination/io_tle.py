def parse_tle(lines: list[str]) -> dict:
    # Minimal placeholder to avoid blocking tests; expand later or swap to sgp4.
    if len(lines) < 2:
        raise ValueError("TLE must have at least two lines")
    return {"line1": lines[0].strip(), "line2": lines[1].strip()}
