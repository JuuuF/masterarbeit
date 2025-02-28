import cv2
import numpy as np

from skimage.metrics import structural_similarity as ssim

from ma_darts.cv.utils import show_imgs, point_point_dist


# -------------------------------------------------------------------------------------------------
# Logpolar


def get_logpolar(img: np.ndarray, max_r: int, cy: int, cx: int) -> np.ndarray:
    logpolar = cv2.warpPolar(
        img,
        dsize=(
            int(max_r),
            1000,
        ),  # I use 1000 because this is enough resolution and is easy to calculate with
        center=(int(cx), int(cy)),
        maxRadius=int(max_r),
        flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS,
    )
    return logpolar


def sort_logpolar_into_strips(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    # Slice logpolar aling fields
    splits = np.arange(0, img.shape[0], img.shape[0] // 20) + img.shape[0] // 40
    slices = np.split(img, splits)
    white = slices[::2]
    black = slices[1::2]

    # Combine cut-off strip
    white[0] = np.concatenate([white[-1], white[0]], axis=0)
    white.pop(-1)

    # Combine white and black strips to single images
    white = np.vstack(white)
    black = np.vstack(black)

    return white, black


def find_logpolar_corners(img):
    # This looks weird, but is intended
    # Convert RGB -> Lab -> Gray
    # This increases the contrast for corners
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7, 7), 0)

    block_size = 2
    kernel_size = 5
    k = 0.04

    corners = cv2.cornerHarris(
        img,
        block_size,
        kernel_size,
        k,
    )
    threshold = 0.01 * corners.max()

    corners = cv2.threshold(corners, threshold, corners.max(), cv2.THRESH_BINARY)[1]
    corners /= corners.max()
    corners = cv2.dilate(corners, None)
    corners = np.uint8(corners * 255)

    return corners


# -------------------------------------------------------------------------------------------------
# Colors


def extract_field_color(colors: np.ndarray, field_start: int) -> tuple[np.ndarray, int]:
    rough_jump = 10
    color_different_threshold = 60
    # Start off with a general color guess, based on a somewhat arbitrary area
    general_color = (
        colors[field_start : field_start + rough_jump].mean(axis=0)
    ).astype(np.int16)

    # Look along the image
    # until we see a color that differs enough from the general color
    field_end = field_start + rough_jump
    while (
        field_end < len(colors) - 1
        and np.abs(general_color - colors[field_end]).sum() < color_different_threshold
    ):
        field_end += 1

    # Calculate field color based on found area
    field_color = np.median(colors[field_start:field_end], axis=0).astype(np.uint8)
    return field_color, field_end


def get_black_white_and_center_size(
    logpolar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    strip_w, strip_b = sort_logpolar_into_strips(logpolar)
    white_colors = np.median(strip_w, axis=0).astype(np.uint8)
    black_colors = np.median(strip_b, axis=0).astype(np.uint8)

    # Skip bullseye
    diff = np.abs(np.int16(white_colors) - black_colors).astype(np.uint8)
    diff = cv2.convertScaleAbs(diff, alpha=2)
    diff = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY)[1]
    field_start = np.nonzero(diff)[0][0] if len(np.nonzero(diff)[0]) > 0 else 0

    white, field_end_w = extract_field_color(white_colors, field_start)
    black, field_end_b = extract_field_color(black_colors, field_start)

    return white, black, field_start


def to_cryv(patch: np.ndarray) -> np.ndarray:

    # black / white different
    # black / red+green different
    # white / red+green different
    # -> red/green similar

    YCrCb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
    Cr = YCrCb[:, :, 1:2]  # white / green, red / green
    Y = YCrCb[:, :, 0:1]  # black / red, black / white, white / green

    HSV = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    V = HSV[:, :, 2:3]  # black / red

    res = np.concatenate([Cr, Y, V], axis=-1)
    return res


def is_black(patch, black_cryv):
    b_Cr, b_Y, b_V = black_cryv
    Cr, Y, V = patch

    diff_Cr = np.abs(b_Cr - Cr)
    diff_Y = np.abs(b_Y - Y)
    diff_V = np.abs(b_V - V)

    return round(diff_Cr + diff_Y + diff_V)


def is_white(patch, white_cryv):
    w_Cr, w_Y, w_V = np.float32(white_cryv)
    Cr, Y, V = patch

    diff_Cr = np.abs(w_Cr - Cr)
    diff_Y = np.abs(w_Y - Y)
    diff_V = np.abs(w_V - V)

    return round(diff_Cr + diff_Y + diff_V)


def is_color(patch):
    Cr, _Y, _V = patch
    target_Cr_red = 200.0
    target_Cr_green = 30.0

    diff_Cr_red = np.abs(target_Cr_red - Cr)
    diff_Cr_green = np.abs(target_Cr_green - Cr)
    diff_Cr = min(diff_Cr_red, diff_Cr_green)

    return round(diff_Cr)


def show_cryv(img, name=None):

    full = []
    for i in range(img.shape[-1]):
        full.append(img[:, :, i])
    res = np.hstack(full)
    if name is not None:
        show_imgs(**{name: res}, block=False)
    else:
        show_imgs(res, block=False)


# -------------------------------------------------------------------------------------------------
# Point surroundings


def get_surrounding(
    img: np.ndarray,
    y: int,
    x: int,
    w: int,
) -> np.ndarray:
    # Clip position
    y = np.clip(y, 0, img.shape[0] - 1)
    x = np.clip(x, 0, img.shape[1] - 1)

    # Extract clipped surrounding
    surrounding = img[
        max(y - w, 0) : min(y + w, img.shape[0]),
        max(x - w, 0) : min(x + w, img.shape[1]),
    ]
    if surrounding.shape[0] == surrounding.shape[1] == 2 * w:
        return surrounding

    # Pad surrounding if it's on the edge
    if (pad := w - y) > 0:
        expand = np.repeat(surrounding[:1], pad, axis=0)
        surrounding = np.vstack([expand, surrounding])
    if (pad := w - img.shape[0] + y) > 0:
        expand = np.repeat(surrounding[-1:], pad, axis=0)
        surrounding = np.vstack([surrounding, expand])
    if (pad := w - x) > 0:
        expand = np.repeat(surrounding[:, :1], pad, axis=1)
        surrounding = np.hstack([expand, surrounding])
    if (pad := w - img.shape[1] + x) > 0:
        expand = np.repeat(surrounding[:, -1:], pad, axis=1)
        surrounding = np.hstack([surrounding, expand])

    return surrounding


def extract_surroundings(
    logpolar: np.ndarray,
    corner_positions: list[list[int]],
    white: np.ndarray,
    black: np.ndarray,
) -> tuple[
    list[tuple[int, int, np.ndarray]],
    list[tuple[int, int, np.ndarray]],
    list[tuple[int, int, np.ndarray]],
    list[tuple[int, int, np.ndarray]],
    np.ndarray,
]:
    surrounding_width = 14
    middle_deadspace = 2
    color_threshold = 100
    intrude = (surrounding_width - middle_deadspace) // 2
    inner_ring_a = []
    inner_ring_b = []
    outer_ring_a = []
    outer_ring_b = []

    # Convert colors into a format that emphasizes the differences
    white_cryv = to_cryv(white[None, None])[0, 0]
    black_cryv = to_cryv(black[None, None])[0, 0]

    logpolar_cryv = to_cryv(logpolar)
    # show_cryv(logpolar_cryv, name="logpolar_cryv")

    for i, points in enumerate(corner_positions):
        y = 25 + 50 * i
        for x in points:
            surrounding = get_surrounding(logpolar, y, x, surrounding_width)
            surrounding_cryv = get_surrounding(logpolar_cryv, y, x, surrounding_width)
            # Find partial fields in surrounding area
            top_left = surrounding_cryv[:intrude, :intrude]  # (i, i, 3)
            top_right = surrounding_cryv[:intrude, -intrude:]
            bottom_left = surrounding_cryv[-intrude:, :intrude]
            bottom_right = surrounding_cryv[-intrude:, -intrude:]
            # Extract mean colors from fields
            color_top_left = top_left.mean(axis=(0, 1))  # (3,)
            color_top_right = top_right.mean(axis=(0, 1))
            color_bottom_left = bottom_left.mean(axis=(0, 1))
            color_bottom_right = bottom_right.mean(axis=(0, 1))
            # Determine field colors
            top_left_black = is_black(color_top_left, black_cryv) < color_threshold
            top_left_white = is_white(color_top_left, white_cryv) < color_threshold
            top_left_color = is_color(color_top_left) < color_threshold
            top_right_black = is_black(color_top_right, black_cryv) < color_threshold
            top_right_white = is_white(color_top_right, white_cryv) < color_threshold
            top_right_color = is_color(color_top_right) < color_threshold
            bottom_left_black = (
                is_black(color_bottom_left, black_cryv) < color_threshold
            )
            bottom_left_white = (
                is_white(color_bottom_left, white_cryv) < color_threshold
            )
            bottom_left_color = is_color(color_bottom_left) < color_threshold
            bottom_right_black = (
                is_black(color_bottom_right, black_cryv) < color_threshold
            )
            bottom_right_white = (
                is_white(color_bottom_right, white_cryv) < color_threshold
            )
            bottom_right_color = is_color(color_bottom_right) < color_threshold

            # --------------------- #
            # DEBUGGING
            # print()
            # print("top_left:", f"black: {is_black(color_top_left, black_cryv)}", f"white: {is_white(color_top_left, white_cryv)}", f"color: {is_color(color_top_left)}", sep="\n\t")  # fmt: skip
            # print("top_right:", f"black: {is_black(color_top_right, black_cryv)}", f"white: {is_white(color_top_right, white_cryv)}", f"color: {is_color(color_top_right)}", sep="\n\t")  # fmt: skip
            # print("bottom_left:", f"black: {is_black(color_bottom_left, black_cryv)}", f"white: {is_white(color_bottom_left, white_cryv)}", f"color: {is_color(color_bottom_left)}", sep="\n\t")  # fmt: skip
            # print("bottom_right:", f"black: {is_black(color_bottom_right, black_cryv)}", f"white: {is_white(color_bottom_right, white_cryv)}", f"color: {is_color(color_bottom_right)}", sep="\n\t")  # fmt: skip
            # show_cryv(
            #     cv2.resize(
            #         surrounding_cryv,
            #         (
            #             surrounding_cryv.shape[1] * 4,
            #             surrounding_cryv.shape[0] * 4,
            #         ),
            #         interpolation=cv2.INTER_NEAREST,
            #     ),
            #     name="surrounding_cryv",
            # )
            # in-depth debugging
            # show_imgs(surrounding=surrounding, block=False)
            # show_imgs()

            left_color = top_left_color and bottom_left_color
            right_color = top_right_color and bottom_right_color

            if top_left_black and bottom_left_white and right_color:
                surrounding_normalized = surrounding
                inner_ring_a.append((y, x, surrounding_normalized))

            if top_left_white and bottom_left_black and right_color:
                surrounding_normalized = surrounding[::-1]  # flip vertical
                inner_ring_b.append((y, x, surrounding_normalized))

            if top_right_black and bottom_right_white and left_color:
                surrounding_normalized = surrounding[:, ::-1]  # flip horizontal
                outer_ring_a.append((y, x, surrounding_normalized))

            if top_right_white and bottom_right_black and left_color:
                surrounding_normalized = surrounding[::-1, ::-1]  # flip both
                outer_ring_b.append((y, x, surrounding_normalized))

    # -----------------------------
    # Get mean surrounding
    surroundings = (
        [x[2] for x in inner_ring_a]
        + [x[2] for x in inner_ring_b]
        + [x[2] for x in outer_ring_a]
        + [x[2] for x in outer_ring_b]
    )
    if len(outer_ring_a + outer_ring_b) < 2:
        print(
            "ERROR: Not enough valid orientation points found (possibly too many outliers)."
        )
        # if create_debug_img:
        #     logpolar[corners != 0] = 255
        #     Utils.append_debug_img(
        #         logpolar, "FAILED: Not enough orientation points found."
        #     )  # TODO: debug_img
        return None
    if len(surroundings) == 0:
        print("ERROR: No valid surroundings found.")
        # if create_debug_img:
        #     logpolar[corners != 0] = 255
        #     Utils.append_debug_img(
        #         logpolar, "FAILED: No orientation point surroundings found."
        #     )  # TODO: debug_img
        return None
    mean_surrounding = np.median(surroundings, axis=0).astype(np.uint8)

    return inner_ring_a, inner_ring_b, outer_ring_a, outer_ring_b, mean_surrounding


def ssim_score(patch_1, patch_2):

    patch_1 = cv2.cvtColor(patch_1, cv2.COLOR_BGR2HSV)
    patch_2 = cv2.cvtColor(patch_2, cv2.COLOR_BGR2HSV)

    similarity = ssim(patch_1, patch_2, multichannel=True, channel_axis=2)

    return np.clip(similarity, 0, 1)


def lab_areas(patch_1, patch_2):
    # Use LAB color space to emphasize typical red and green colors
    lab_1 = cv2.cvtColor(patch_1, cv2.COLOR_BGR2LAB)
    lab_2 = cv2.cvtColor(patch_2, cv2.COLOR_BGR2LAB)

    mid = lab_1.shape[0] // 2
    # Only look at red and green parts
    red_1 = lab_1[:mid, mid:]
    green_1 = lab_1[mid:, mid:]

    red_2 = lab_2[:mid, mid:]
    green_2 = lab_2[mid:, mid:]

    # Calculate patch differences
    diff_red = np.abs(np.int16(red_2) - red_1)
    sum_red = diff_red.sum(axis=-1)
    mean_red = sum_red.mean()

    diff_green = np.abs(np.int16(green_2) - green_1)
    sum_green = diff_green.sum(axis=-1)
    mean_green = sum_green.mean()

    mean_total = (mean_red + mean_green) / 2

    # Calculate similarity using exponential falloff
    #   - high input = low similarity
    #   - higher falloff value = stricter similarity
    #   - falloff = -0.008: 50% similarity at 86
    falloff = 0.01
    similarity = np.exp(-falloff * mean_total)
    return similarity


def is_correct_surrounding(
    mean_surrounding: np.ndarray, surrounding: np.ndarray, show: bool = False
) -> tuple[bool, np.ndarray]:
    similarity_threshold = 0.5

    # Compare colors in LAB color space
    similarity_lab = lab_areas(mean_surrounding, surrounding)
    # User SSIM for structural similarity
    similarity_ssim = ssim_score(mean_surrounding, surrounding)

    similarity = (similarity_lab + similarity_ssim) / 2
    is_orientation_point = similarity > similarity_threshold

    # Draw results
    if show:
        surrounding = surrounding.copy()
        c = (0, 255, 0) if is_orientation_point else (0, 0, 255)
        surrounding[:2] = c
        surrounding[:, :2] = c
        surrounding[:, -2:] = c
        surrounding[0] = (0, 0, 0)
        surrounding[:, 0] = (0, 0, 0)
        surrounding[:, -1] = (0, 0, 0)

        # similarity indication
        end = int(surrounding.shape[1] * similarity)
        end = np.clip(end, 0, surrounding.shape[1])
        surrounding[-2:, :end] = (255, 255, 255)
        surrounding[-2:, end:] = (0, 0, 0)
        surrounding[-1, int(similarity_threshold * surrounding.shape[1])] = (255, 0, 0)
    return is_orientation_point, surrounding


def filter_surroundings(
    inner_ring_a: list[tuple[int, int, np.ndarray]],
    inner_ring_b: list[tuple[int, int, np.ndarray]],
    outer_ring_a: list[tuple[int, int, np.ndarray]],
    outer_ring_b: list[tuple[int, int, np.ndarray]],
    mean_surrounding: np.ndarray,
    debug_img: np.ndarray | None = None,
) -> list[tuple[int, int, str]]:

    show = debug_img is not None

    keeps: list[tuple[int, int, str]] = []

    for y, x, surrounding in inner_ring_a:
        # Check if valid surrounding
        is_orientation_point, surrounding_ = is_correct_surrounding(
            mean_surrounding, surrounding, show=show
        )
        if is_orientation_point:
            keeps.append((y, x, "inner"))

        if show:
            # Draw point visualization
            cv2.circle(debug_img, (x, y), 5, (255, 255, 255))
            cv2.circle(debug_img, (x, y), 2, (0, 0, 255), -1)
            # Place surrounding on image: top left
            if y - surrounding.shape[0] >= 0 and x - surrounding.shape[1] >= 0:
                debug_img[
                    y - surrounding.shape[0] : y, x - surrounding.shape[1] : x
                ] = surrounding_

    for y, x, surrounding in inner_ring_b:
        # Check if valid surrounding
        is_orientation_point, surrounding_ = is_correct_surrounding(
            mean_surrounding, surrounding, show=show
        )
        if is_orientation_point:
            keeps.append((y, x, "inner"))

        if show:
            # Draw point visualization
            cv2.circle(debug_img, (x, y), 5, (255, 255, 255))
            cv2.circle(debug_img, (x, y), 2, (0, 255, 0), -1)
            # Place surrounding on image: top left
            if y - surrounding.shape[0] >= 0 and x - surrounding.shape[1] >= 0:
                debug_img[
                    y - surrounding.shape[0] : y, x - surrounding.shape[1] : x
                ] = surrounding_

    for y, x, surrounding in outer_ring_a:
        # Check if valid surrounding
        is_orientation_point, surrounding_ = is_correct_surrounding(
            mean_surrounding, surrounding, show=show
        )
        if is_orientation_point:
            keeps.append((y, x, "outer"))

        if show:
            # Draw point visualization
            cv2.circle(debug_img, (x, y), 5, (0, 0, 0))
            cv2.circle(debug_img, (x, y), 2, (0, 0, 255), -1)
            # Place surrounding on image: top right
            if (
                y - surrounding.shape[0] >= 0
                and x + surrounding.shape[1] < debug_img.shape[1]
            ):
                debug_img[
                    y - surrounding.shape[0] : y, x : x + surrounding.shape[1]
                ] = surrounding_

    for y, x, surrounding in outer_ring_b:
        # Check if valid surrounding
        is_orientation_point, surrounding_ = is_correct_surrounding(
            mean_surrounding, surrounding, show=show
        )
        if is_orientation_point:
            keeps.append((y, x, "outer"))

        if show:
            # Draw point visualization
            cv2.circle(debug_img, (x, y), 5, (0, 0, 0))
            cv2.circle(debug_img, (x, y), 2, (0, 255, 0), -1)
            # Place surrounding on image: top right
            if (
                y - surrounding.shape[0] >= 0
                and x + surrounding.shape[1] < debug_img.shape[1]
            ):
                debug_img[
                    y - surrounding.shape[0] : y, x : x + surrounding.shape[1]
                ] = surrounding_

    return keeps, debug_img


# -------------------------------------------------------------------------------------------------
# I don't know what to categorize this into


def y_to_angle_bin(keeps: list[tuple[int, int, str]]) -> list[list[tuple[int, str]]]:
    out = [[] for _ in range(10)]
    for y, x, position in keeps:
        i = (y - 25) // 50

        # If we are in the left half, we invert x,
        # indicating that we move backwards from the center
        if i >= 10:
            x *= -1
            i %= 10
        out[i].append((x, position))
    return out


def find_orientation_points(
    img: np.ndarray,
    cy: int,
    cx: int,
    show: bool = False,
) -> list[list[tuple[int, str]]]:

    # Convert image to logpolar representation
    max_r = max(
        point_point_dist((cy, cx), (0, 0)),
        point_point_dist((cy, cx), (0, img.shape[1])),
        point_point_dist((cy, cx), (img.shape[0], 0)),
        point_point_dist((cy, cx), (img.shape[0], img.shape[1])),
    )
    logpolar = get_logpolar(img, max_r, cy, cx)
    if show:
        show_imgs(logpolar=logpolar, block=False)

    # -----------------------------
    # Get Colors and Rough Scaling
    white, black, center_size = get_black_white_and_center_size(logpolar)

    # -----------------------------
    # Stretch logpolar image to ensure the multiplier fields are big enough
    width_scaling = max(1, 25 / center_size) if center_size > 1 else 1
    if width_scaling > 1:
        img_resized = cv2.resize(
            img,
            (int(width_scaling * img.shape[1]), int(width_scaling * img.shape[0])),
        )
        logpolar = get_logpolar(
            img_resized,
            max_r * width_scaling,
            int(width_scaling * cy),
            int(width_scaling * cx),
        )

    # -----------------------------
    # Find corners
    corners = find_logpolar_corners(logpolar)  # (1000, x)

    # -----------------------------
    # Find corners on intersection lines
    corner_band_width = 6

    line_ys = (
        np.arange(0, corners.shape[0], corners.shape[0] // 20) + corners.shape[0] // 40
    )
    corner_strips = [
        corners[i - corner_band_width : i + corner_band_width] for i in line_ys
    ]
    corner_strip_values = [np.max(s, axis=0) for s in corner_strips]

    # Find positions of corners on strips
    corner_positions = []
    for strip in corner_strip_values:
        nonzeros = np.nonzero(strip)[0]
        if len(nonzeros) == 0:
            corner_positions.append([])
            continue
        diffs = np.diff(nonzeros)
        breaks = np.where(diffs > 1)[0] + 1
        groups = np.split(nonzeros, breaks)
        centers = [round(np.mean(g)) for g in groups]
        corner_positions.append(centers)

    # Visualize corner strips
    if show:
        corners_ = corners.copy()
        for i in line_ys:
            corners_[i - corner_band_width : i + corner_band_width][
                corners_[i - corner_band_width : i + corner_band_width] == 0
            ] = 60
        corners_ = cv2.addWeighted(
            logpolar, 0.5, cv2.cvtColor(corners_, cv2.COLOR_GRAY2BGR), 1.0, 1.0
        )
        show_imgs(recognized_corners=corners_, block=False)

    # -----------------------------
    # Get surroundings
    inner_ring_a, inner_ring_b, outer_ring_a, outer_ring_b, mean_surrounding = (
        extract_surroundings(logpolar, corner_positions, white, black)
    )
    if show:
        show_imgs(mean_surrounding=mean_surrounding, block=False)

    # -----------------------------
    # Classify surroundings
    prepare_show_img = show  # or create_debug_img  # TODO: debug img

    keeps, logpolar_ = filter_surroundings(
        inner_ring_a,
        inner_ring_b,
        outer_ring_a,
        outer_ring_b,
        mean_surrounding,
        debug_img=logpolar.copy() if prepare_show_img else None,
    )

    if sum(1 for p in keeps if p[-1] == "outer") < 2:
        print("ERROR: Too few orientation points!")
        return None

    if prepare_show_img:
        surrounding_preview = cv2.resize(
            mean_surrounding,
            (mean_surrounding.shape[1] * 3, mean_surrounding.shape[0] * 3),
            interpolation=cv2.INTER_NEAREST,
        )
        logpolar_[: surrounding_preview.shape[0], -surrounding_preview.shape[1] :] = (
            surrounding_preview
        )
        if show:
            show_imgs(positions=logpolar_, block=False)
        # Utils.append_debug_img(logpolar_, "Logpolar Orientation Points")  # TODO: debug img

    # -----------------------------
    # Sort keeps into bins
    positions: list[list[tuple[int, str]]] = y_to_angle_bin(keeps)
    # Resolve logpolar bins to real bins
    # Logpolar distortion starts at 3 o'clock while we start rotation at 12
    positions = positions[5:] + positions[:5]
    for i in range(5):
        positions[i] = [(-p[0], p[1]) for p in positions[i]]

    # -----------------------------
    # Re-Scale positions
    if width_scaling > 1:
        for i, pos_bin in enumerate(positions):
            positions[i] = [(int(p[0] / width_scaling), p[1]) for p in pos_bin]
    return positions  # (x, inner/outer)
