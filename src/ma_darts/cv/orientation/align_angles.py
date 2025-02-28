import numpy as np

from ma_darts.cv.utils import (
    point_line_distance,
    point_theta_to_polar_line,
    point_point_dist,
)


def align_angles(
    lines_filtered: list[tuple[float, float, float, float, float]],
    thetas: list[float],
    img_shape: tuple[int, int],
    cy: float,
    cx: float,
    show: bool = False,
) -> list[tuple[float, float]]:
    rho_guess = np.sqrt(img_shape[0] ** 2 + img_shape[1] ** 2) / 2

    def fit_polar_line_to_points(points, weights, initial_theta) -> tuple[float, float]:
        def objective(params):
            rho, theta = params
            res = (
                sum(
                    w * point_line_distance(*pt, rho, theta) ** 2
                    for pt, w in zip(points, weights)
                )
                + 1e-5
            )
            return res

        initial_guess = point_theta_to_polar_line((cy, cx), theta)

        # for p in points:
        #     print(Utils.point_line_distance(*p, *initial_guess))
        #     img[p[0], p[1]] = 255

        # draw_polar_line(img, *initial_guess)
        # show_imgs(img)
        # return initial_guess

        bounds = [(-rho_guess * 2, rho_guess * 2), (0, np.pi)]

        from scipy.optimize import minimize

        result = minimize(
            objective,
            initial_guess,
            bounds=bounds,
            method="L-BFGS-B",
            options={"gtol": 1e-5, "ftol": 1e-6},
        )

        if result.success:
            return result.x  # rho, theta

        print("WARNING: Could not terminate line fitting:", result.message)
        return result.x

    # Filter lines
    line_bins = [[] for _ in range(len(thetas))]
    for line in lines_filtered:
        min_dist = np.Inf
        for i, theta in enumerate(thetas):
            dist_1 = abs(theta - line[-1])
            dist_2 = abs(theta - (line[-1] - np.pi))
            dist = min(dist_1, dist_2)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        line_bins[min_idx].append(line)

    out_lines = []
    for theta, lines in zip(thetas, line_bins):
        if len(lines) == 0:
            out_lines.append(point_theta_to_polar_line((cy, cx), theta))
            continue

        points = [(cy, cx)]
        for line in lines:
            points.append(line[0])
            points.append(line[1])

        lengths = [line[2] for line in lines]
        mean_points = [
            (
                (line[0][0] + line[1][0]) // 2,
                (line[0][1] + line[1][1]) // 2,
            )
            for line in lines
        ]
        weights = [
            l / (point_point_dist((cy, cx), mp) + 1e-2)
            for l, mp in zip(lengths, mean_points)
        ]
        # weight for each point of a line
        weights = [w for w in weights for _ in (0, 1)]
        weights.insert(0, max(weights) * 2)  # weight for center point
        # normalize
        weights = [w - min(weights) for w in weights]
        weights = [w / max(weights) for w in weights]

        # print(points, weights, theta)
        rho_, theta_ = fit_polar_line_to_points(points, weights, theta)
        # print("--")
        # print(theta, theta_)
        out_lines.append((rho_, theta_))

    if show:  # or create_debug_img:  # TODO: debug img
        out = img.copy()
        for rho, theta in out_lines:
            draw_polar_line(out, rho, theta)
        if show:
            show_imgs(lines_aligned=out, block=False)
        # Utils.append_debug_img(out, "Aligned Angles")  # TODO: debug img
    return out_lines
