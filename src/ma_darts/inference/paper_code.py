import cv2
import numpy as np
import pandas as pd


d1_val = [
    "d1_02_06_2020",
    "d1_02_16_2020",
    "d1_02_22_2020",
]
d1_test = [
    "d1_03_03_2020",
    "d1_03_19_2020",
    "d1_03_23_2020",
    "d1_03_27_2020",
    "d1_03_28_2020",
    "d1_03_30_2020",
    "d1_03_31_2020",
]

d2_val = [
    "d2_02_03_2021",
    "d2_02_05_2021",
]
d2_test = [
    "d2_03_03_2020",
    "d2_02_10_2021",
    "d2_02_03_2021_2",
]

BOARD_DICT = {
    0: "13",
    1: "4",
    2: "18",
    3: "1",
    4: "20",
    5: "5",
    6: "12",
    7: "9",
    8: "14",
    9: "11",
    10: "8",
    11: "16",
    12: "7",
    13: "19",
    14: "3",
    15: "17",
    16: "2",
    17: "15",
    18: "10",
    19: "6",
}


def get_splits(path="./dataset/labels.pkl", dataset="d1", split="train"):
    assert dataset in ["d1", "d2"], "dataset must be either 'd1' or 'd2'"
    assert split in [
        None,
        "train",
        "val",
        "test",
    ], "split must be in [None, 'train', 'val', 'test']"

    if dataset == "d1":
        val_folders, test_folders = d1_val, d1_test
    else:
        val_folders, test_folders = d2_val, d2_test

    df = pd.read_pickle(path)
    df = df[df.img_folder.str.contains(dataset)]
    splits = {}
    splits["val"] = df[np.isin(df.img_folder, val_folders)]
    splits["test"] = df[np.isin(df.img_folder, test_folders)]
    splits["train"] = df[
        np.logical_not(np.isin(df.img_folder, val_folders + test_folders))
    ]
    if split is None:
        return splits
    else:
        return splits[split]


def bboxes_to_xy(bboxes, max_darts=3):
    xy = np.zeros((4 + max_darts, 3), dtype=np.float32)
    for cls in range(5):
        if cls == 0:
            dart_xys = bboxes[bboxes[:, 4] == 0, :2][:max_darts]
            xy[4 : 4 + len(dart_xys), :2] = dart_xys
        else:
            cal = bboxes[bboxes[:, 4] == cls, :2]
            if len(cal):
                xy[cls - 1, :2] = cal[0]
    xy[(xy[:, 0] > 0) & (xy[:, 1] > 0), -1] = 1
    if np.sum(xy[:4, -1]) == 4:
        return xy
    else:
        xy = est_cal_pts(xy)
    return xy


def est_cal_pts(xy):
    missing_idx = np.where(xy[:4, -1] == 0)[0]
    if len(missing_idx) == 1:
        if missing_idx[0] <= 1:
            center = np.mean(xy[2:4, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 0:
                xy[0, 0] = -xy[1, 0]
                xy[0, 1] = -xy[1, 1]
                xy[0, 2] = 1
            else:
                xy[1, 0] = -xy[0, 0]
                xy[1, 1] = -xy[0, 1]
                xy[1, 2] = 1
            xy[:, :2] += center
        else:
            center = np.mean(xy[:2, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 2:
                xy[2, 0] = -xy[3, 0]
                xy[2, 1] = -xy[3, 1]
                xy[2, 2] = 1
            else:
                xy[3, 0] = -xy[2, 0]
                xy[3, 1] = -xy[2, 1]
                xy[3, 2] = 1
            xy[:, :2] += center
    else:
        # TODO: if len(missing_idx) > 1
        print("Missed more than 1 calibration point")
    return xy


def board_radii(r_d):
    r_board = 0.2255  # radius of full board
    r_double = 0.170  # center bull to outside double wire edge, in m (BDO standard)
    r_treble = 0.1074  # center bull to outside treble wire edge, in m (BDO standard)
    r_outer_bull = 0.0159
    r_inner_bull = 0.00635
    w_double_treble = 0.01  # wire apex to apex for double and treble

    r_t = r_d * (r_treble / r_double)  # outer treble radius, in px
    r_ib = r_d * (r_inner_bull / r_double)  # inner bull radius, in px
    r_ob = r_d * (r_outer_bull / r_double)  # outer bull radius, in px
    w_dt = w_double_treble * (r_d / r_double)  # width of double and treble
    return r_t, r_ob, r_ib, w_dt


def get_circle(xy):
    """
    Calculate board size and center based on orientation points
    c = center position
    r = board radius
    """
    c = np.mean(xy[:4], axis=0)
    r = np.mean(np.linalg.norm(xy[:4] - c, axis=-1))
    return c, r


def transform(xy, img=None, angle=9, M=None):

    if xy.shape[-1] == 3:
        has_vis = True
        vis = xy[:, 2:]
        xy = xy[:, :2]
    else:
        has_vis = False

    if img is not None and np.mean(xy[:4]) < 1:
        h, w = img.shape[:2]
        xy *= [[w, h]]

    if M is None:
        c, r = get_circle(xy)  # not necessarily a circle
        # c is center of 4 calibration points, r is mean distance from center to calibration points

        src_pts = xy[:4].astype(np.float32)
        dst_pts = np.array(
            [
                [
                    c[0] - r * np.sin(np.deg2rad(angle)),
                    c[1] - r * np.cos(np.deg2rad(angle)),
                ],
                [
                    c[0] + r * np.sin(np.deg2rad(angle)),
                    c[1] + r * np.cos(np.deg2rad(angle)),
                ],
                [
                    c[0] - r * np.cos(np.deg2rad(angle)),
                    c[1] + r * np.sin(np.deg2rad(angle)),
                ],
                [
                    c[0] + r * np.cos(np.deg2rad(angle)),
                    c[1] - r * np.sin(np.deg2rad(angle)),
                ],
            ]
        ).astype(np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    xyz = np.concatenate((xy, np.ones((xy.shape[0], 1))), axis=-1).astype(np.float32)
    xyz_dst = np.matmul(M, xyz.T).T
    xy_dst = xyz_dst[:, :2] / xyz_dst[:, 2:]

    if img is not None:
        img = cv2.warpPerspective(img.copy(), M, (img.shape[1], img.shape[0]))
        xy_dst /= [[w, h]]

    if has_vis:
        xy_dst = np.concatenate([xy_dst, vis], axis=-1)

    return xy_dst, img, M


def get_dart_scores(
    xy,  # (7, 3): 4x orientation + <= 3x dart; (x, y, visible)
    numeric=False,
):
    valid_cal_pts = xy[:4][(xy[:4, 0] > 0) & (xy[:4, 1] > 0)]
    if xy.shape[0] <= 4 or valid_cal_pts.shape[0] < 4:  # missing calibration point
        return []

    # Undistort positions based on orientation points
    xy, _, _ = transform(xy.copy(), angle=0)

    # Get board radii from orientation points
    c, r_d = get_circle(xy)
    r_t, r_ob, r_ib, w_dt = board_radii(r_d)

    # Extract polar coordinates from positions
    xy -= c
    angles = np.arctan2(-xy[4:, 1], xy[4:, 0]) / np.pi * 180
    angles = [a + 360 if a < 0 else a for a in angles]  # map to 0-360
    distances = np.linalg.norm(xy[4:], axis=-1)

    # Map scores to polar coordinates
    scores = []
    for angle, dist in zip(angles, distances):
        if dist > r_d:
            scores.append("0")
            continue
        if dist <= r_ib:
            scores.append("DB")
            continue
        if dist <= r_ob:
            scores.append("B")
            continue

        number = BOARD_DICT[int(angle / 18)]

        if dist <= r_d and dist > r_d - w_dt:
            scores.append("D" + number)
            continue
        if dist <= r_t and dist > r_t - w_dt:
            scores.append("T" + number)
            continue
        scores.append(number)

    if numeric:
        for i, s in enumerate(scores):
            if "B" in s:
                if "D" in s:
                    scores[i] = 50
                    continue
                scores[i] = 25
                continue
            if "D" in s or "T" in s:
                scores[i] = int(s[1:])
                scores[i] = scores[i] * 2 if "D" in s else scores[i] * 3
                continue
            scores[i] = int(s)
    return scores


# -----------------------------------------------
# Drawing


def draw_circles(img, xy, color=(255, 255, 255)):
    c, r_d = get_circle(xy)  # double radius
    center = (int(round(c[0])), int(round(c[1])))
    r_t, r_ob, r_ib, w_dt = board_radii(r_d)
    for r in [r_d, r_d - w_dt, r_t, r_t - w_dt, r_ib, r_ob]:
        cv2.circle(img, center, int(round(r)), color)
    return img


def draw(img, xy, circles, score, color=(255, 255, 0)):
    # Read data
    xy = np.array(xy)
    if xy.shape[0] > 7:
        xy = xy.reshape((-1, 2))

    # Rescale outputs to pixel values
    if np.mean(xy) < 1:
        h, w = img.shape[:2]
        xy[:, 0] *= w
        xy[:, 1] *= h

    # Draw board circles
    if xy.shape[0] >= 4 and circles:
        img = draw_circles(img, xy)

    # Get scores
    if xy.shape[0] > 4 and score:
        scores = get_dart_scores(xy)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    line_type = 1
    for i, [x, y] in enumerate(xy):
        if i < 4:
            c = (0, 255, 0)  # green
        else:
            c = color  # cyan
        x = int(round(x))
        y = int(round(y))
        if i >= 4:
            cv2.circle(img, (x, y), 1, c, 1)
            if score:
                txt = str(scores[i - 4])
            else:
                txt = str(i + 1)
            cv2.putText(img, txt, (x + 8, y), font, font_scale, c, line_type)
        else:
            cv2.circle(img, (x, y), 1, c, 1)
            cv2.putText(img, str(i + 1), (x + 8, y), font, font_scale, c, line_type)
    return img
