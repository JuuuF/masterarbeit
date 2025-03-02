import os
import cv2
import numpy as np

from ma_darts.cv.utils import apply_matrix, scaling_matrix

from ma_darts.cv import edges, lines, orientation, utils

create_debug_img = False
debug_out_images = []

if __name__ == "__main__":
    img_paths_custom = [
        "dump/thomas.png",
        "data/darts_references/jess/001_0-0-1.jpg",
        "data/darts_references/jess/018_1-DB-DB.jpg",
        "data/darts_references/jess/022_2-2-18.jpg",
        "data/darts_references/jess/061_6-7-T4.jpg",
        "data/darts_references/jess/084_10-6-4.jpg",
        "data/darts_references/jess/129_19-2-6.jpg",
        "dump/test/double.png",
        # "data/generation/out/0/render.png",
        # "data/generation/out/6/render.png",
        # "data/generation/out/7/render.png",
        # "data/generation/out/8/render.png",
        # "dump/test/x_90.png",
        # "dump/test/x_67_5.png",
        # "dump/test/x_45.png",
        # "dump/test/x_22_5.png",
        # "dump/test/y_90.png",
        # "dump/test/y_67_5.png",
        # "dump/test/y_45.png",
        # "dump/test/y_22_5.png",
        "dump/test/0001.jpg",
        # "dump/test/0002.jpg",
        # "dump/test/0003.jpg",
        # "data/paper/imgs/d1_02_16_2020/IMG_2858.JPG",
        "dump/test/test_img.png",
        "dump/test/test.png",
        # "data/paper/imgs//d2_02_23_2021_3/DSC_0003.JPG",
        "/home/justin/Downloads/test2.jpg",
        "/home/justin/Downloads/test.jpg",
    ]

    # Generated images
    img_paths_gen = [
        os.path.join("data/generation/out", i, "render.png")
        for i in sorted(os.listdir("data/generation/out"), key=lambda x: int(x))
    ]
    # img_paths = img_paths[46:]

    # Paper images
    img_paths_paper = [
        os.path.join("data/paper/imgs", d, f)
        for d in os.listdir("data/paper/imgs")
        for f in os.listdir(os.path.join("data/paper/imgs", d))
    ]
    np.random.shuffle(img_paths_paper)
    img_paths_paper = img_paths_paper[:200]

    # Own References
    img_paths_jess = [
        os.path.join("data/darts_references/jess", f)
        for f in os.listdir("data/darts_references/jess")
    ]
    # Own References
    img_paths_sb = [
        os.path.join("data/darts_references/strongbows", f)
        for f in os.listdir("data/darts_references/strongbows")
    ]

    img_paths = (
        []
        # add paths
        # + img_paths_gen
        # + img_paths_paper
        # + img_paths_jess
        + img_paths_sb
    )
    np.random.shuffle(img_paths)

    # img_paths[0] = "data/generation/out/75/render.png"


class Utils:

    def downsample_img(img: np.ndarray) -> np.ndarray:
        while max(*img.shape[:2]) > 1600:
            img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        return img

    def show_debug_img(
        target_w: int = 2560,
        target_h: int = 1080 - 125,
        failed: bool = False,
    ) -> None:
        if not create_debug_img:
            return
        global debug_out_images
        imgs = debug_out_images.copy()

        failed = failed or any(["fail" in l.lower() for l, _ in imgs])

        bg_color = 50 if not failed else 10

        # Convert to color
        imgs = [
            (label, cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) if len(i.shape) == 2 else i)
            for label, i in imgs
        ]

        # Set image sizes
        n_imgs = len(imgs)
        grid_cols = int(np.ceil(np.sqrt(n_imgs)))
        grid_rows = int(np.ceil(n_imgs / grid_cols))

        cell_w = target_w // grid_cols
        cell_h = target_h // grid_rows

        resized_imgs = []
        for label, img in imgs:
            h, w = img.shape[:2]
            scale = min(cell_w / w, cell_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized_img = cv2.resize(
                img, (new_w - 2, new_h - 2), interpolation=cv2.INTER_AREA
            )

            # Calculate padding to center the image
            pad_top = (cell_h - new_h) // 2
            pad_bottom = cell_h - new_h - pad_top
            pad_left = (cell_w - new_w) // 2
            pad_right = cell_w - new_w - pad_left

            # Apply padding
            padded_img = np.pad(
                resized_img,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=bg_color,
            )

            # Add title
            txt_params = dict(
                org=(5, 20),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                padded_img,
                label,
                color=(0, 0, 0),
                thickness=2,
                **txt_params,
            )
            cv2.putText(
                padded_img,
                label,
                color=(255, 255, 255),
                thickness=1,
                **txt_params,
            )

            # Add border
            padded_img = np.pad(
                padded_img,
                ((1, 1), (1, 1), (0, 0)),
                mode="constant",
                constant_values=bg_color,
            )
            resized_imgs.append(padded_img)

        # Combine into image
        res = np.full((grid_rows * cell_h, grid_cols * cell_w, 3), bg_color, np.uint8)

        for i, img in enumerate(resized_imgs):
            row, col = divmod(i, grid_cols)
            x0 = col * cell_w
            x1 = x0 + cell_w
            y0 = row * cell_h
            y1 = y0 + cell_h
            res[y0:y1, x0:x1] = img

        show_imgs(**{"Debug Output": res})

    def append_debug_img(img: np.ndarray, name: str = "<no name>") -> None:
        if not create_debug_img:
            return
        global debug_out_images
        debug_out_images.append((name, img))

    def clear_debug_img() -> None:
        global debug_out_images
        debug_out_images = []


def undistort_img(img: np.ndarray) -> np.ndarray:

    # -----------------------------
    # Preprocess Image

    img_full = img.copy()
    img = Utils.downsample_img(img_full)

    # show_imgs(input=img, block=False)
    Utils.append_debug_img(img, "Input")

    # -----------------------------
    # EDGES

    # Detect Edges
    edge_img = edges.edge_detect(img, show=False)
    # Skeletonize
    skeleton_img = edges.skeletonize(edge_img, show=False)

    # -----------------------------
    # LINES

    # Extract lines
    img_lines = lines.extract_lines(skeleton_img, show=False)

    # Bin lines by angle
    img_lines_binned = lines.bin_lines_by_angle(img_lines)

    # Find Board Center
    cy, cx = lines.get_center_point(img.shape, img_lines_binned, show=False)

    # Filter Lines by Center Distance
    img_lines_filtered = lines.filter_lines_by_center_dist(
        img_lines, cy, cx
    )  # p1, p2, length (normalized), center distance [px], rho, theta

    # Estimate line angles
    thetas = lines.get_rough_line_angles(
        img.shape[:2], img_lines_filtered, cy, cx, show=False
    )

    if len(thetas) != 10:
        print("ERROR: Could not find all lines!")
        if create_debug_img:
            Utils.show_debug_img(failed=True)
        return None

    # -----------------------------
    # ORIENTATION

    # Align lines by filtered edges
    img_lines = orientation.align_angles(
        img_lines_filtered, thetas, img.shape[:2], cy, cx, show=False
    )

    # Calculate better center coordinates
    cy, cx = orientation.center_point_from_lines(img_lines)

    # Get undistortion matrix
    M_undistort = orientation.undistort_by_lines(cy, cx, img_lines, show=False)

    # Undistort image
    img_undistort = apply_matrix(img, M_undistort)
    cx_undistort, cy_undistort = (M_undistort @ np.array([cx, cy, 1]))[:2]

    # angle_step = np.pi / 10
    # angles = np.arange(0, np.pi, angle_step) + angle_step / 2
    # img_undistort //= 2
    # for t in angles:
    #     draw_polar_line_through_point(img_undistort, (cy_undistort, cx_undistort), t)
    # show_imgs(img_undistort)
    # exit()

    # Find possible orientation points
    orientation_point_candidates = orientation.find_orientation_points(
        img_undistort, int(cy_undistort), int(cx_undistort), show=False
    )

    if orientation_point_candidates is None:
        if create_debug_img:
            Utils.show_debug_img(failed=True)
        return None

    # Filter out bad orientation points
    src_pts, dst_pts = orientation.structure_orientation_candidates(
        orientation_point_candidates,
        int(cy_undistort),
        int(cx_undistort),
        # img_undistort=img_undistort,
    )

    # Convert orientation points to transformation matrix
    M_align = orientation.get_alignment_matrix(
        src_pts, dst_pts, int(cy_undistort), int(cx_undistort)
    )

    # Combine all matrices
    scale = img.shape[0] / img_full.shape[0]

    M_full = np.eye(3)
    M_full = scaling_matrix(scale) @ M_full  # downscale to calculation size
    M_full = M_undistort @ M_full  # undistort
    M_full = M_align @ M_full  # align to correct scale and orientation

    return M_full

    res = apply_matrix(img_full, M_full, output_size=(800, 800))
    img = res.copy()
    res = np.uint8(res * 0.67)
    # cv2.circle(res, (400, 400), 10, (255, 255, 255), 1, cv2.LINE_AA)
    # cv2.circle(res, (400, 400), 3, (255, 0, 0), -1, cv2.LINE_AA)
    from ma_darts.cv.utils import (
        point_theta_to_polar_line,
        draw_polar_line_through_point,
    )
    from ma_darts import radii

    for i in range(10):
        a = np.deg2rad(18) * i + np.deg2rad(9)
        x0 = int(400 - np.sin(a) * 320)
        y0 = int(400 + np.cos(a) * 320)
        x1 = int(400 + np.sin(a) * 320)
        y1 = int(400 - np.cos(a) * 320)
        cv2.line(res, (x0, y0), (x1, y1), (255, 255, 255), 3, cv2.LINE_AA)
    for r in [radii[i] for i in range(6)]:
        cv2.circle(res, (400, 400), int(r), (255, 255, 255), 3, cv2.LINE_AA)
    for i in range(10):
        a = np.deg2rad(18) * i + np.deg2rad(9)
        x0 = int(400 - np.sin(a) * 320)
        y0 = int(400 + np.cos(a) * 320)
        x1 = int(400 + np.sin(a) * 320)
        y1 = int(400 - np.cos(a) * 320)
        cv2.line(res, (x0, y0), (x1, y1), (0, 0, 0), 1, cv2.LINE_AA)
    for r in [radii[i] for i in range(6)]:
        cv2.circle(res, (400, 400), int(r), (0, 0, 0), 1, cv2.LINE_AA)

    res = cv2.addWeighted(img, 0.5, res, 0.5, 1.0)
    show_imgs(res)

    Utils.append_debug_img(res, "Aligned Image")
    Utils.show_debug_img()
    Utils.clear_debug_img()
    return M_full


# -----------------------------------------------

if __name__ == "__main__":

    from ma_darts.cv.utils import show_imgs

    img_dir = "data/darts_references/home/"
    img_dir = "data/darts_references/jess/"
    img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
    img_paths = sorted(img_paths, key=lambda x: int(x.split("/")[-1].split("-")[0]))

    img_dir = "data/generation/out/"
    img_paths = [os.path.join(img_dir, f, "render.png") for f in os.listdir(img_dir)]
    img_paths = sorted(img_paths)

    img_dir = "data/paper/imgs/d2_03_03_2020"
    img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
    img_paths = sorted(img_paths)[2:]

    for img_path in img_paths:
        img = cv2.imread(img_path)
        M = undistort_img(img)
        if M is None:
            continue
        res = apply_matrix(img, M, output_size=(800, 800))
        show_imgs(res)
