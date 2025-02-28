import numpy as np

from ma_darts.cv.utils import (
    translation_matrix,
    rotation_matrix,
    shearing_matrix,
    scaling_matrix,
    apply_matrix,
    draw_polar_line_through_point,
)

from ma_darts.cv.utils import show_imgs


def theta_change_shear_y(theta, shear_y):
    """
    slope = dy / dx
    mapping:
        dy/dx -> dy / (dy * s + dx)
                = (dy / dx) / (1 + s * (dy / dx))
                = slope / (1 + s * slope)
    in out left-handed coordinate system, the shearing is inverted, so we use:
        dy/dx -> slope / (1 - s * slope)
    mapping not as straightforward as we change the denominator
    """
    slope = np.tan(theta)  # get the slope

    slope_ = slope / (1 - shear_y * slope)  # this is the funny mapping

    theta_ = np.arctan(slope_)  # convert slope to angle
    return theta_


def visualize_matrix(M, img, cy, cx, src_start, src):
    res = img.copy()
    for t in src_start:
        draw_polar_line_through_point(
            res, (int(cy), int(cx)), t, thickness=3, color=(255, 255, 255)
        )

    res = apply_matrix(res, M)

    for t in dst:
        draw_polar_line_through_point(
            res, (int(cy), int(cx)), t, color=(255, 0, 0), thickness=2
        )
    for t in src:
        draw_polar_line_through_point(
            res, (int(cy), int(cx)), t, color=(0, 0, 0), thickness=1
        )
    cv2.putText(
        res,
        "white=transformed initial thetas, blue='dst' thetas, black='src' thetas",
        org=(0, 20),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=1,
        color=(255, 255, 255),
        thickness=1,
    )
    return res


def undistort_by_lines(
    cy: int,
    cx: int,
    lines: list[tuple[float, float]],
    show: bool = False,
):

    angle_step = np.pi / 10
    dst_start = np.arange(0, np.pi, angle_step) + angle_step / 2
    src_start = np.array([l[1] for l in lines])

    transformation_matrices = []

    for start_line in range(10):

        # Initialize Values
        M = np.eye(3)
        src = src_start.copy()
        dst = dst_start.copy()

        # -----------------------------
        # 1. Translate center to origin
        M_trans_a = translation_matrix(-cx, -cy)
        M_trans_b = translation_matrix(cx, cy)

        # update transformation matrix
        M = M_trans_a @ M

        # -----------------------------
        # 2. Align src vertival line with destination vertical line
        # output: src[i] -> dst[i]
        t_src = src[start_line]
        t_dst = dst[start_line]
        rot_angle = t_dst - t_src
        M_rot = rotation_matrix(rot_angle)

        # update transformation matrix
        M = M_rot @ M

        # update src points
        src += rot_angle

        # draw
        if show:
            # res = visualize_matrix(M_trans_b @ M)
            show_imgs(first_rotate_to_align_single_line=res, block=False)

        # -----------------------------
        # 3. Vertical alignment
        # Goal: src[i] = dst[0] = 0
        rotation_alignment_angle = angle_step / 2 + start_line * angle_step
        M_rot_v = rotation_matrix(-rotation_alignment_angle)

        # update transformation matrix
        M = M_rot_v @ M

        # update src and dst points
        src -= rotation_alignment_angle
        dst -= rotation_alignment_angle

        # draw
        if show:
            # res = visualize_matrix(M_trans_b @ M)
            show_imgs(second_rotate_aligned_to_vertical=res, block=False)

        # -----------------------------
        # 4. Vertical shearing
        # Goal:
        #   - src[i] = 0
        #   - src[i+5] = 90° = np.pi / 2
        horizontal_line_idx = (start_line + 5) % len(lines)
        t_src = src[horizontal_line_idx]
        t_dst = dst[horizontal_line_idx]
        shear_amount = t_dst - t_src

        M_shear = shearing_matrix(shear_amount)

        # update transformation matrix
        M = M_shear @ M

        # update src points
        src = theta_change_shear_y(src, shear_amount)
        src[src < 0] += np.pi

        # draw
        if show:
            # res = visualize_matrix(M_trans_b @ M)
            show_imgs(third_shear_to_fit_horizontal=res, block=False)

        # -----------------------------
        # 5. Vertical scaling
        # Goal: align rest of lines as good as possible

        # convert angles to slopes
        slopes_src = np.tan(src)
        slopes_dst = np.tan(dst)
        # remove already aligned angles
        slopes_src = np.delete(slopes_src, [start_line, horizontal_line_idx])
        slopes_dst = np.delete(slopes_dst, [start_line, horizontal_line_idx])
        # calculate scaling
        scales = slopes_src / slopes_dst
        scale = np.mean(scales)

        M_scale = scaling_matrix(y=scale)

        # update transformation matrix
        M = M_scale @ M

        # upate src points
        src = np.arctan(np.tan(src) / scale)

        # draw
        if show:
            # res = visualize_matrix(M_trans_b @ M)
            show_imgs(fourth_scale_to_fit_rest_of_lines=res, block=False)

        # -----------------------------
        # 6. Undo vertical alignment
        M_rot_v_inv = rotation_matrix(rotation_alignment_angle)

        # update transformation matrix
        M = M_rot_v_inv @ M

        # update src and dst points
        src += rotation_alignment_angle
        dst += rotation_alignment_angle

        # draw
        if show:
            # res = visualize_matrix(M_trans_b @ M)
            show_imgs(vertical_scaling=res, block=False)

        # -----------------------------
        # 7. Re-Translate into center

        M = M_trans_b @ M

        transformation_matrices.append(M)

    # - # - # - # - # - # - # - # - # - #

    M = np.mean(transformation_matrices, axis=0)

    if show:  # or create_debug_img:  # TODO: debug_img
        global img
        res = apply_matrix(img, M, True)
        if show:
            show_imgs(img_undistort=res, block=False)
        # Utils.append_debug_img(res, "Undistorted Angles")  # TODO: debug img

    return M
