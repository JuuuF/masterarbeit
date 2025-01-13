import os
import cv2
import sys
import numpy as np
import pandas as pd
import pickle
from rich import print as pprint

from ma_darts.generation.rendering import render_image
from ma_darts.cv.data_preparation import prepare_sample
from ma_darts.cv.utils import show_imgs

OUT_DIR = "data/generation/out_val"


def check_sample(sample_info: pd.Series):
    # Load images
    img_parts = []

    img_render = cv2.imread(sample_info.out_file_template.format(filename="render.png"))
    img_undist = cv2.imread(
        sample_info.out_file_template.format(filename="undistort.png")
    )

    def draw_ellipse():
        cy = sample_info.ellipse_cy
        cx = sample_info.ellipse_cx
        w = sample_info.ellipse_w
        h = sample_info.ellipse_h
        theta = sample_info.ellipse_theta
        y0 = cy - h // 2
        y1 = y0 + h
        x0 = cx - w // 2
        x1 = x0 + w
        cv2.rectangle(img_render, (x0, y0), (x1, y1), (255, 0, 0))
        mask = cv2.imread(
            sample_info.out_file_template.format(filename="mask_Darts_Board_Area.png")
        )
        img_render[mask < 127] //= 4

    img_parts.append(img_render)
    img_parts.append(img_undist)

    # Highlight arrow tips
    cv2.circle(img_undist, (400, 400), 300, (255, 0, 0), 1, cv2.LINE_AA)

    def get_image_region(
        pos: int, max_pos: int, win_size: int = 100
    ) -> tuple[int, int]:
        p0 = pos - win_size
        p1 = pos + win_size
        if p0 < 0:
            p1 += abs(p0)
            p0 = 0
        elif p1 > max_pos:
            p0 -= p1 - max_pos
            p1 = max_pos
        return p0, p1

    # draw circles
    for y_rel, x_rel in sample_info.dart_positions:
        y = round(y_rel * img_render.shape[0])
        x = round(x_rel * img_render.shape[1])
        cv2.circle(img_render, (x, y), radius=5, color=(255, 255, 255))
        img_render[y, x] = 255

    # Extract dart tips
    arrow_tiles = []
    for y_rel, x_rel in sample_info.dart_positions:
        y = round(y_rel * img_render.shape[0])
        x = round(x_rel * img_render.shape[1])
        y0, y1 = get_image_region(y, img_render.shape[0])
        x0, x1 = get_image_region(x, img_render.shape[1])
        arrow_tile = img_render[y0:y1, x0:x1]
        arrow_tile = np.pad(arrow_tile, ((5, 5), (5, 5), (0, 0)))
        arrow_tile = cv2.pyrUp(arrow_tile)
        arrow_tiles.append(arrow_tile)
    img_arrows = np.concatenate(arrow_tiles, axis=0)

    # Combine images
    def align_images(*imgs: list[np.ndarray], axis: int = 0):
        max_size = max(img.shape[axis] for img in imgs)
        res = []
        for img in imgs:
            if img.shape[axis] == max_size:
                res.append(img)
                continue
            diff = max_size - img.shape[axis]
            diff_0 = diff // 2
            diff_1 = diff - diff_0
            pads = [(0, 0)] * len(img.shape)
            pads[axis] = (diff_0, diff_1)
            img = np.pad(img, pads)
            res.append(img)
        return res

    img = np.concatenate(
        align_images(img_render, img_undist, img_arrows, axis=0), axis=1
    )
    while max(*img.shape) > 2000:
        img = cv2.pyrDown(img)

    show_imgs(**{"Data Preparation Result": img}, block=False)
    cv2.waitKey(1)
    # while (k := cv2.waitKey()) not in [
    #     ord("q"),
    #     13,  # Enter
    #     27,  # Esc
    #     255,  # Del
    # ]:
    #     pass
    # if k in [ord("q"), 27]:
    #     exit()
    # elif k == 255:
    #     print(f"Sample {sample_info.sample_id} invalid.")
    # elif k == 13:
    #     print(f"Sample {sample_info.sample_id} valid.")


def create_sample(
    id: int = None,
    sample_path=os.path.join(OUT_DIR, "{id}"),
) -> pd.Series | None:
    print("-" * 120)
    print(f"Sample {id}".center(120))

    # --------------------------------------------------------------------
    # Render / Load Data
    sample_info_path = os.path.join(sample_path, "info.pkl")
    if id is None or not os.path.exists(sample_info_path.format(id=id)):
        # No ID given or info not available -> render sample
        try:
            sample_info = render_image(id=id, out_dir=OUT_DIR)
        except AssertionError as e:
            print("Error while rendering sample:")
            print("\t", e)
            return
        except Exception as e:
            print("Something went wrong while rendering:", e)
            return
    else:
        # ID given and existing -> load sample
        with open(sample_info_path.format(id=id), "rb") as f:
            sample_info = pickle.load(f)
        # correct sample_id if sample was moved. That might happen, just in case
        sample_info.sample_id = int(id)

    # update paths
    sample_path = sample_path.format(id=sample_info.sample_id)
    sample_info_path = sample_info_path.format(id=sample_info.sample_id)

    # --------------------------------------------------------------------
    # Extract information

    if (
        os.path.exists(os.path.join(sample_path, "undistort.png"))
        and "dart_positions" in sample_info
    ):
        print(f"Sample {sample_info.sample_id} already exists.".center(120))
        print("-" * 120)
        return sample_info

    try:
        sample_info = prepare_sample(sample_info)
    except AssertionError as e:
        print("Error while creating sample information:")
        print("\t", e)
        # if unsuccessful, remove info and return None
        if os.path.exists(sample_info_path):
            os.remove(sample_info_path)
        return
    except Exception as e:
        print("Something went wrong while extracting information:", e)
        if os.path.exists(sample_info_path):
            os.remove(sample_info_path)
        return

    # --------------------------------------------------------------------
    # Report

    pd.set_option("display.width", 120)
    pprint(sample_info.to_dict())
    print(
        f"Sample {sample_info.sample_id} created".center(120),
        "-" * 120,
        sep="\n",
    )
    return sample_info


if __name__ == "__main__":
    for i in range(16):
        sample_info = None
        while sample_info is None:
            sample_info = create_sample(i)
        check_sample(sample_info)
