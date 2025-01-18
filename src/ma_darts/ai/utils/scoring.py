import numpy as np

board_numbers = [
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5,  # fmt: skip
]


def get_board_radii() -> tuple[float, float, float, float, float, float]:
    r_db = 0.635  # double bull
    r_b = 1.6  # bull
    r_ti = 9.8  # triple inner
    r_to = 10.7  # triple outer
    r_di = 16.2  # double inner
    r_do = 17.0  # double outer
    return r_db, r_b, r_ti, r_to, r_di, r_do


def get_image_radii(
    img_size: int = 800, margin: int = 100
) -> tuple[float, float, float, float, float, float]:
    radii = np.array(get_board_radii())

    # Normalize to ourside radius
    radii /= radii[-1]

    # scale to image pixels
    radii *= (img_size // 2) - margin

    return tuple(radii)


def cartesian_to_polar(y: float, x: float) -> tuple[float, float]:
    # Radius
    r = np.sqrt(x**2 + y**2)

    # Angle
    theta = np.arctan2(x, -y)  # 0° = up, 90° = right
    theta %= 2 * np.pi

    return r, theta


def get_dart_scores(
    positions_yx: tuple[float, float] | list[tuple[float, float]],  # (y, x)
    img_size: int = 800,
    margin: int = 100,
) -> list[tuple[float, float]]:
    if type(positions_yx) == tuple:
        positions = [positions_yx]
    else:
        positions = positions_yx.copy()

    positions = np.array(positions, np.float32)
    positions -= img_size // 2
    positions = [cartesian_to_polar(y, x) for y, x in positions]

    scores = []
    r_db, r_b, r_ti, r_to, r_di, r_do = get_image_radii(img_size, margin)

    for r, theta in positions:
        # DB
        if r < r_db:
            scores.append(50)
            continue

        # Bull
        if r < r_b:
            scores.append(25)
            continue

        # Out
        if r > r_do:
            scores.append(0)
            continue

        # Determine Multiplier
        if r_ti < r < r_to:
            multiplier = 3
        elif r_di < r < r_do:
            multiplier = 2
        else:
            multiplier = 1

        # Determine Field
        field_idx, _ = divmod(theta + np.pi / 20, np.pi / 10)
        field = board_numbers[int(field_idx) % 20]

        # Calculate Score
        scores.append(multiplier * field)

    return scores + [0 for _ in range(3 - len(scores))]

def get_absolute_score_error(scores_true: list[int], scores_pred: list[int]):
    return abs(np.sum(scores_true) - np.sum(scores_pred))