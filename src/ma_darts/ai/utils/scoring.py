import numpy as np
from ma_darts import dart_order, r_bi, r_bo, r_ti, r_to, r_di, r_do


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


def calculate_scores_ma(
    pos: np.ndarray,  # (n, 2)
    cls: np.ndarray,  # (n,)
):
    # Positions to origin
    pos_norm = pos - 400

    r_1 = r_bo + (r_ti - r_bo) / 3  # bull-triple
    r_2 = r_to + (r_di - r_to) / 2  # triple-double
    r_3 = 400  # outside

    scores = []
    for i, (p, c) in enumerate(zip(pos_norm, cls)):
        # Convert positions to polar
        r, theta = cartesian_to_polar(*p)  # 0 = up

        # Check for hidden
        if c == 0:
            scores.append((0, "HIDDEN"))
            continue
        # Check for Double Bull
        if c == 3 and r < r_1:  # red and inside
            scores.append((50, "DB"))
            continue
        # Check for Bull
        if c == 4 and r < r_1:  # green and inside
            scores.append((25, "B"))
        # Check for outside
        if c == 5 or r > r_3:
            scores.append((0, "OUT"))
            continue

        # Extract most likely field based on position and class
        theta_norm = (theta + np.deg2rad(9)) % (2 * np.pi)
        idx, offset = divmod(theta_norm, np.deg2rad(18))
        idx = int(idx)
        offset /= np.deg2rad(18)
        offset -= 0.5
        black_or_red_position = idx % 2 == 0
        black_or_red_class = c in [1, 3]

        # If there's something off, we correct it
        if black_or_red_position != black_or_red_class:
            idx += int(1 * np.sign(offset))
        idx %= 20

        field_num = dart_order[idx]

        # Single field
        if c not in [3, 4]:
            scores.append((field_num, str(field_num)))
            continue

        # Double field
        if r > r_2:
            scores.append((2 * field_num, f"D{field_num}"))
            continue

        # Triple field
        scores.append((3 * field_num, f"T{field_num}"))
    return scores


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
        field = dart_order[int(field_idx) % 20]

        # Calculate Score
        scores.append(multiplier * field)

    return scores + [0 for _ in range(3 - len(scores))]


def get_absolute_score_error(scores_true: list[int], scores_pred: list[int]):
    return abs(np.sum(scores_true) - np.sum(scores_pred))
