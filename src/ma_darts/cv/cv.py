import os
import cv2
import numpy as np

img_paths = [
    # "dump/test/double.png",
    # "data/generation/out/0/render.png",
    "data/generation/out/6/render.png",
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
    "dump/test/0002.jpg",
    "dump/test/0003.jpg",
    "data/paper/imgs/d1_02_16_2020/IMG_2858.JPG",
    "dump/test/test_img.png",
    "dump/test/test.png",
    "data/paper/imgs//d2_02_23_2021_3/DSC_0003.JPG",
]


class Utils:

    def load_img(filepath: str) -> np.ndarray:
        img = cv2.imread(filepath)
        while max(*img.shape[:2]) > 1500:
            img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        return img

    def non_maximum_suppression(img):

        from scipy.ndimage import maximum_filter

        max_filtered = maximum_filter(img, size=7, mode="constant")
        suppressed = (img == max_filtered) * img
        return suppressed


    def points_to_polar_line(p1, p2):
        y2, x2 = p2
        y1, x1 = p1

        if x2 >= x1:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        dy = y2 - y1
        dx = x2 - x1
        theta = np.arctan2(-dx, dy)

        rho = x1 * np.cos(theta) + y1 * np.sin(theta)
        theta %= np.pi

        return rho, theta

    def point_point_dist(p1, p2):
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


class Unused:

    def ellipse_detect(edges: np.ndarray):
        if len(edges.shape) == 2:
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        print(edges.dtype)
        # edges[edges < 127] = 0
        # edges[edges != 0] = 255
        show_imgs(edges)

    def paper_ellipses_old(edges):
        """
        A NEW EFFICIENT ELLIPSE DETECTION METHOD
        https://www.wellesu.com/10.1109/ICPR.2002.1048464
        """

        # edges = cv2.Canny(img, 50, 150)
        def dist(x_1, y_1, x_2, y_2):
            return np.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)

        # -------------------------------------------
        # equations
        def eq_1(x_1, x_2):
            return (x_1 + x_2) / 2

        eq_2 = eq_1

        def eq_3(x_1, y_1, x_2, y_2):
            return dist(x_1, y_1, x_2, y_2) / 2

        def eq_4(x_1, y_1, x_2, y_2):
            return np.atan((y_2 - y_1) / (x_2 - x_1))

        # def eq_6(a, d, tau):
        #     a2d2 = a**2 * d**2
        #     return (a2d2 * np.sin(tau)**2) / (a2d2 * eq_6()**2)

        # def eq_6(a, d, f, tau):
        #     return (a**2 + d**2 - f**2) / (2 * a * d)

        # -------------------------------------------

        # 1. Store all edge pixels in a 1D-array
        edges_arr = [
            (
                y,
                x,
            )
            for x in range(edges.shape[1])
            for y in range(edges.shape[0])
            if edges[y, x] > 50
        ]

        # 2. clear accumulator array
        accumulator = []

        # 3. for each pixel (x_1, y_1): 4 - 14
        for i, (y_1, x_1) in enumerate(edges_arr):
            # 4. for each other pixel (x_1, y_2)...
            for j, (y_2, x_2) in enumerate(edges_arr[i + 1 :], i + 1):
                # ... if the distance greater then required least distance, carry on
                least_dist = 10
                if dist(x_1, y_1, x_2, y_2) > least_dist:
                    # 5. calculate the center, orientation and length of major axis
                    x0 = eq_1(x_1, y_2)
                    y0 = eq_2(y_1, y_2)
                    a = eq_3(x_1, y_1, x_2, y_2)
                    alpha = eq_4(x_1, y_1, x_2, y_2)

                    # 6. for each third pixel (x, y) ...
                    for y, x in edges:
                        # ... if the distance between (x, y) and (x_0, y_0) is greater then least required, carry on
                        if (d := dist(x, y, x0, y0)) > least_dist:
                            # 7. Calculate length minor axis

                            cos_tau = (a**2 + d**2 - f**2) / (2 * a * d)
                            b2 = (a**2 * d**2 * np.sin(alpha)) / (a**2 - d**2 * cos_tau)

        show_imgs(edges)

    def find_lines(edge_img):
        def line_eq(p1, p2):
            a = p1[1] - p2[1]
            b = p2[0] - p1[0]
            c = p1[0] * p2[1] - p2[0] * p1[1]
            return a, b, -c

        def find_intersection(l1, l2):
            a1, b1, c1 = l1
            a2, b2, c2 = l2
            det = a1 * b2 - a2 * b1
            if det == 0:
                return None

            # Using Cramer's rule to find intersection
            x = (b2 * c1 - b1 * c2) / det
            y = (a1 * c2 - a2 * c1) / det
            return round(x), round(y)

        # Find lines
        lines = cv2.HoughLinesP(
            edge_img,
            rho=1,
            theta=np.pi / 180,
            threshold=150,
            minLineLength=50,
            maxLineGap=10,
        )  # (n, 1, 4)

        if lines is None:
            return [], []
        lines = lines[:, 0]  # (n, 4)

        line_eqs = [line_eq((x1, y1), (x2, y2)) for x1, y1, x2, y2 in lines]

        # Find line intersections
        intersections = []
        for i, l1 in enumerate(line_eqs):
            for l2 in line_eqs[i + 1 :]:
                if i := find_intersection(l1, l2):
                    intersections.append(i)
        print(intersections)

        # Draw lines
        line_img = np.zeros(edge_img.shape[:2], dtype=np.uint8)
        for x1, y1, x2, y2 in lines:
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)

        show_imgs(edge_img, line_img)

    def keypoints(img: np.ndarray):
        orb = cv2.ORB_create()
        kps, desc = orb.detectAndCompute(img, None)
        out = img.copy()
        for kp in kps:
            out[int(kp.pt[1]), int(kp.pt[0])] = (255, 0, 0)
        Utils.show_imgs(out)

    def fit_sine_curve(img: np.ndarray, hough_space: np.ndarray):

        from scipy.optimize import curve_fit

        def sine_curve(x, A, B, C, D):
            return A * np.cos(B * x + C) + D

        xdata = []
        ydata = []
        intensities = []
        for y, cols in enumerate(hough_space):
            for x, intensity in enumerate(cols):
                if intensity > 0.1:
                    xdata.append(x)
                    ydata.append(y)
                    intensities.append(intensity)

        params = (np.pi / 2 * img.shape[0], 0.02, 0, img.shape[0] // 2)
        # params, covariance = curve_fit(
        #     f=sine_curve,
        #     xdata=xdata,
        #     ydata=ydata,
        #     p0=params,
        #     sigma=intensities,
        # )
        x = hough_space.shape[1]
        x_span = np.linspace(0, x - 1, x)
        y_fit = sine_curve(x_span, *params)

        res = cv2.cvtColor(hough_space, cv2.COLOR_GRAY2BGR)
        for y, x in zip(y_fit, x_span):
            if 0 <= y < res.shape[0]:
                res[int(y), int(x)] = (255, 0, 0)

        res = cv2.resize(
            res, (res.shape[1] * 4, res.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        return res

    def matching(img: np.ndarray):
        filter_size = 45
        filters_dir = os.path.join("data", "cv", "filters")
        filter_paths = [
            os.path.join(filters_dir, f)
            for f in os.listdir(filters_dir)
            if f.endswith(".png")
        ]
        filters = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in filter_paths]
        n_filters = len(filters)
        for f in range(n_filters):
            filter = filters[f]
            for i in [2, 3]:
                squished_x = cv2.resize(
                    filter.copy(), (filter.shape[1] // i, filter.shape[0])
                )
                squished_y = cv2.resize(
                    filter.copy(), (filter.shape[1], filter.shape[0] // i)
                )
                filters.append(squished_x)
                filters.append(squished_y)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, -1]
        orig_shape = img.shape[:2]
        while min(*img.shape[:2]) >= filter_size:
            res = []
            for f in filters:
                # Filter image
                img_filtered = 1 - cv2.matchTemplate(
                    img, f, method=cv2.TM_CCOEFF_NORMED
                )
                img_filtered = cv2.resize(
                    img_filtered,
                    (orig_shape[1], orig_shape[0]),
                    interpolation=cv2.INTER_NEAREST_EXACT,
                )
                img_filtered[: f.shape[0], : f.shape[1]] = f

                # Normalize by abs
                # img_filtered -= 0.5
                # img_filtered = abs(img_filtered)
                # img_filtered *= 2

                res.append(img_filtered)

            out = np.ones(orig_shape, dtype=np.float32)
            for r in res:
                out *= r

            # Rectangle in img
            img_ = img.copy()
            cv2.rectangle(img_, (0, 0, 45, 45), (127, 127, 127))
            Utils.show_imgs(img_, *res, out)
            img = cv2.pyrDown(img)
        # exit()

    def circles(img: np.ndarray):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=10,
            param1=200,
            param2=15,
            minRadius=10,
            maxRadius=min(*img.shape[:2]) // 2,
        )[0]

        print(len(circles))
        res = np.zeros_like(img, dtype=np.float32)
        for i, (cx, cy, r) in enumerate(circles, 1):
            fac = i / len(circles)
            col = 1 - fac
            cx = round(cx)
            cy = round(cy)
            if 0 <= cx < res.shape[1] and 0 <= cy < res.shape[0]:
                res[round(cy), round(cx)] += col
            res += cv2.circle(
                res * 0, (round(cx), round(cy)), round(r), (col, col, col)
            )
        # res = cv2.blur(res, (7, 7))
        res /= res.max()
        Utils.show_imgs(img, res)
        return res

    def feature_matching(img: np.ndarray):
        dartboard_img = cv2.imread(
            os.path.join("data", "cv", "dartboard.png"), cv2.IMREAD_GRAYSCALE
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp1, desc1 = sift.detectAndCompute(img, None)
        kp2, desc2 = sift.detectAndCompute(dartboard_img, None)

        flann = cv2.FlannBasedMatcher(
            indexParams=dict(algorithm=1, trees=5),
            searchParams=dict(checks=50),
        )
        matches = flann.knnMatch(desc1, desc2, k=2)

        good = [m for m, n in matches]
        # good = []
        # for m, n in matches:
        #     if m.distance < 0.7 * n.distance:
        #         good.append(m)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        matches_mask = mask.ravel().tolist()

        h, w = img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
            -1, 1, 2
        )
        dst = cv2.perspectiveTransform(pts, M)

        res = cv2.polylines(dartboard_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        draw_params = dict(
            matchColor=(0, 255, 0),  # draw matches in green color
            singlePointColor=None,
            matchesMask=matches_mask,  # draw only inliers
            flags=2,
        )
        res = cv2.drawMatches(img, kp1, res, kp2, good, None, **draw_params)
        Utils.show_imgs(img, res)

    def hough_ellipse(
        edges,
        min_major_axis=10,
        min_votes=10,
        top_n=10,
        max_pixels=200,
    ):
        """
        Source: https://github.com/boechat107/imgproc_scripts/blob/master/hough_ellipse.m
        """
        from scipy.spatial.distance import cdist
        from collections import defaultdict

        height, width = edges.shape
        ys, xs = np.nonzero(edges)
        pixels = np.column_stack((xs, ys))
        if len(pixels) > max_pixels:
            pixels = pixels[
                np.random.choice(pixels.shape[0], max_pixels, replace=False)
            ]
        max_b = int(max(width, height) / 2)

        ellipses = []

        # Precompute pairwise distances between points and apply conditions
        dist_matrix = cdist(pixels, pixels)
        valid_pairs = np.argwhere(
            (dist_matrix > min_major_axis) & (dist_matrix < max(width, height) / 2)
        )

        for n, (i, j) in enumerate(valid_pairs):
            print(f"{n+1}/{len(valid_pairs)}", end="\r")
            x1, y1 = pixels[i]
            x2, y2 = pixels[j]
            dist_12 = dist_matrix[i, j]

            # Midpoint, major axis length, and orientation
            x0, y0 = (x1 + x2) / 2, (y1 + y2) / 2
            a = dist_12 / 2
            alpha = np.arctan2(y2 - y1, x2 - x1)

            # Accumulator dictionary for minor axis values
            acc = defaultdict(int)

            # Compute for each third point
            d03_array = np.hypot(pixels[:, 0] - x0, pixels[:, 1] - y0)
            f_array = np.hypot(pixels[:, 0] - x2, pixels[:, 1] - y2)

            # Filter to points within range to be valid `x3, y3` candidates
            valid_third_points = np.where((d03_array < a) & (d03_array > 0))[0]

            # Vectorize b calculation
            for idx in valid_third_points:
                d03, f = d03_array[idx], f_array[idx]
                cos2_tau = ((a**2 + d03**2 - f**2) / (2 * a * d03)) ** 2
                sin2_tau = 1 - cos2_tau

                try:
                    b = int(
                        np.sqrt((a**2 * d03**2 * sin2_tau) / (a**2 - d03**2 * cos2_tau))
                    )
                    if 0 < b < max_b:
                        acc[b] += 1
                except ValueError:
                    continue  # skip invalid sqrt values

            # Find the best minor axis and accumulate results
            if not acc:
                continue

            best_b, max_votes_found = max(acc.items(), key=lambda item: item[1])
            if max_votes_found >= min_votes:
                ellipses.append((max_votes_found, x0, y0, a, best_b, alpha))

        # Sort ellipses by votes and return the top `top_n`
        ellipses = sorted(ellipses, key=lambda x: x[0], reverse=True)[:top_n]
        if ellipses:
            return [(x0, y0, a, b, alpha) for _, x0, y0, a, b, alpha in ellipses]

        print("No ellipses detected!")
        return None

    def line_detect(edge_img: np.ndarray) -> np.ndarray:

        lsd = cv2.createLineSegmentDetector()
        lines = lsd.detect(edge_img)[0][:, 0]

        res = np.zeros_like(edge_img)
        for x1, y1, x2, y2 in lines:
            cv2.line(res, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0))
        Utils.show_imgs(edge_img, res)
        return lines

    def extend_line(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1

        x1 -= dx
        x2 += dx
        y1 -= dy
        y2 += dy
        return x1, y1, x2, y2

    def fit_ellipse(edges: np.ndarray):
        # Make image square
        if edges.shape[0] != edges.shape[1]:
            max_size = max(*edges.shape[:2])
            temp = np.zeros((max_size, max_size), np.uint8)
            dy = (temp.shape[0] - edges.shape[0]) // 2
            dx = (temp.shape[1] - edges.shape[1]) // 2
            temp[dy : dy + edges.shape[0], dx : dx + edges.shape[1]] = edges
            edges = temp

        # Extract points
        y, x = np.nonzero(edges)

        # Solve normal system of equations using SVD
        Z = np.column_stack((x**2, x * y, x**2, x, y, np.ones_like(x)))
        _, _, V = np.linalg.svd(Z)
        params = V[-1, :]

        A, B, C, D, E, F = params

        # Calculate center
        cx = (C * D - B * E) / (B**2 - A * C)
        cy = (A * E - B * D) / (B**2 - A * C)

        # Calculate the semi-major and semi-minor axes (a, b) and the orientation angle (theta)
        up = 2 * (A * E**2 + C * D**2 + F * B**2 - 2 * B * D * E - A * C * F)
        down1 = (B**2 - A * C) * (
            (C - A) * np.sqrt(1 + (4 * B**2) / ((A - C) ** 2)) - (C + A)
        )
        down2 = (B**2 - A * C) * (
            (A - C) * np.sqrt(1 + (4 * B**2) / ((A - C) ** 2)) - (C + A)
        )
        a = np.sqrt(abs(up / down1))
        b = np.sqrt(abs(up / down2))

        # Calculate orientation of the ellipse
        theta = 0.5 * np.arctan2(B, A - C)

        center = (round(cx), round(cy))
        axes = (round(max(a, b)), round(min(a, b)))

        res = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.ellipse(
            res,
            center=center,
            axes=axes,
            angle=theta,
            startAngle=0,
            endAngle=360,
            color=(255, 0, 0),
        )

        Utils.show_imgs(edges, res)
        exit()

    def transfer_hough_space(img, edge_img: np.ndarray) -> np.ndarray:
        rho = 0.1
        theta = np.pi / 180 * 2
        threshold = None
        lines = cv2.HoughLinesWithAccumulator(
            edge_img,
            rho,
            theta,
            threshold,
        )[
            :, 0
        ]  # (rho [-diag..diag], theta [-pi..pi], intensity [0..?])

        # Normalize strengths
        lines[:, 2] /= lines[:, 2].max()

        # Create result image
        res = np.zeros_like(edge_img)
        res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        res = res.astype(np.float32)

        # Sort lines by intensity
        lines = sorted(lines, key=lambda x: x[2], reverse=True)
        # lines = lines[:1000]
        lines = np.array(lines)
        # Calculate output size
        diag = np.sqrt(edge_img.shape[0] ** 2 + edge_img.shape[1] ** 2)
        y = int(np.ceil(diag))
        x = 180

        # Normalize rho
        lines[:, 0] += y

        # Theta to degrees
        lines[:, 1] = lines[:, 1] / np.pi * 180

        # Normalize strengths
        lines[:, 2] /= lines[:, 2].max()

        hough_space = np.zeros((y * 2, x), dtype=np.float32)
        hough_space[
            lines[:, 0].astype(np.int32), lines[:, 1].astype(np.int32)
        ] += lines[:, 2]
        hough_space /= hough_space.max()
        hough_space = (hough_space * 255).astype(np.uint8)
        return hough_space

    def detect_hough_lines(img, edge_img: np.ndarray):
        rho = 0.1
        theta = np.pi / 180 / 10
        threshold = None
        lines = cv2.HoughLinesWithAccumulator(
            edge_img,
            rho,
            theta,
            threshold,
        )[
            :, 0
        ]  # (rho [-diag..diag], theta [-pi..pi], intensity [0..?])

        # Calculate output size
        diag = np.sqrt(edge_img.shape[0] ** 2 + edge_img.shape[1] ** 2)
        y = int(np.ceil(diag))

        # Normalize rho
        lines[:, 0] += y

        # Theta to degrees
        lines[:, 1] = np.rad2deg(lines[:, 1])

        # Normalize strengths
        lines[:, 2] /= lines[:, 2].max()

        # Create hough space
        hough_space = np.zeros((y * 2, 180), dtype=np.float32)
        ys = lines[:, 0].astype(np.int32)
        xs = lines[:, 1].astype(np.int32)
        for y, x in zip(ys, xs):

            hough_space[y, x] += lines[:, 2]

        # Normalize Hough space
        hough_space /= hough_space.max()
        hough_space = (hough_space * 255).astype(np.uint8)
        hough_space = hough_space.T

        # Non-maximum suppression
        # hough_space = cv2.blur(hough_space, (3, 3))
        # hough_space = Utils.non_maximum_suppression(hough_space)

        return hough_space

    def color_spaces(img: np.ndarray):

        def cvt(img):
            img = np.float32(img) - 128  # -128..127
            img = abs(img)  # 0..128
            img /= img.max()  # Normalize
            # img = np.sqrt(img)
            img *= 255
            img = np.uint8(img)
            return img

        good_imgs = []

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab = cvt(lab)
        good_imgs.append(lab[:, :, 1])  # a

        luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        luv = np.float32(luv)
        luv -= np.median(luv)
        luv = abs(luv)
        luv /= luv.max()
        luv = np.uint8(luv * 255)
        good_imgs.append(luv[:, :, 1])  # u

        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycc = cvt(ycc)
        good_imgs.append(ycc[:, :, 1])

        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv = cvt(yuv)
        # good_imgs.append(yuv[:, :, 2])  # v

        # -------------------------------------------
        # Concat
        combined = np.concatenate(good_imgs, axis=1)
        while combined.shape[1] > 2500:
            combined = cv2.pyrDown(combined)

        # Sum
        sum_img = np.zeros(img.shape[:2], np.float32)
        for i in good_imgs:
            sum_img += i
        sum_img /= sum_img.max()
        sum_img = np.uint8(255 * sum_img)

        # Mult
        mult_img = np.ones(img.shape[:2], np.float32)
        for i in good_imgs:
            i = np.float32(i) / 255
            mult_img *= i * 0.9 + 0.1
        mult_img /= mult_img.max()
        mult_img = np.uint8(255 * mult_img)

        res = np.float32(sum_img) + np.float32(mult_img)
        res_out = np.uint8(res / 2)

        # Utils.show_imgs(img, combined, sum_img, mult_img, res)
        # return res_out

        # ===========================================

        color_spaces = [
            cv2.COLOR_BGR2GRAY,
            # cv2.COLOR_BGR2HLS,
            # cv2.COLOR_BGR2HSV,
            cv2.COLOR_BGR2LAB,
            cv2.COLOR_BGR2LUV,
            # cv2.COLOR_BGR2RGB,
            # cv2.COLOR_BGR2RGBA,
            # cv2.COLOR_BGR2XYZ,
            cv2.COLOR_BGR2YCrCb,
            # cv2.COLOR_BGR2YUV,
        ]
        res = np.zeros(
            (img.shape[0] * 4, img.shape[1] * len(color_spaces), 3), np.uint8
        )
        dy, dx = img.shape[:2]
        res[:dy, :dx] = img

        def add_img(res, img, x, y):
            y0 = y * dy
            x0 = x * dx
            res[y0 : y0 + dy, x0 : x0 + dx] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return res

        for x, cs in enumerate(color_spaces):
            cs_img = cv2.cvtColor(img, cs)
            if len(cs_img.shape) == 2:
                cs_img = np.expand_dims(cs_img, -1)
            for y in range(cs_img.shape[-1]):
                res = add_img(res, cs_img[:, :, y], x, y)
        while res.shape[1] > 2000 or res.shape[0] > 1000:
            res = cv2.resize(res, (res.shape[1] // 2, res.shape[0] // 2))
        Utils.show_imgs(res)
        return res_out

        sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
        lab_a = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 1]

        # comb = (np.float32(sat) / 255) * (np.float32(lab_a) / 255)
        # comb /= comb.max()
        # comb = np.uint8(comb * 255)

        Utils.show_imgs(img, sat, lab_a)
        return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        print(hls.shape)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        print(hsv.shape)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        print(lab.shape)
        Utils.show_imgs(img, gray)
        Utils.show_imgs(img, hls[:, :, 0], hls[:, :, 1], hls[:, :, 2])
        Utils.show_imgs(img, hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2])
        Utils.show_imgs(img, lab[:, :, 0], lab[:, :, 1], lab[:, :, 2])

    def fit_ellipse_2(skeleton, cy, cx):
        def ellipse_residuals(params, points, center):
            a, b, theta = params
            y_c, x_c = center
            # Translate ellipse to center
            y = points[:, 0] - y_c
            x = points[:, 1] - x_c

            # Rotate points by theta
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            x_rot = x * cos_theta + y * sin_theta
            y_rot = y * cos_theta - x * sin_theta

            residuals = (x_rot / a) ** 2 + (y_rot / b) ** 2 - 1
            return residuals

        edge_img = cv2.dilate(skeleton.copy(), (3, 3), iterations=2)
        points_y, points_x = np.nonzero(edge_img)
        points = np.vstack([points_y, points_x]).T
        points[:, 0] -= cy
        points[:, 1] -= cx

        initial_guess = (50, 50, 0)

        # from scipy.optimize import least_squares
        # res = least_squares(ellipse_residuals, initial_guess, args=(points, (cy, cx)))
        # a, b, theta = res.x
        ellipse = cv2.fitEllipse(points)
        a, b = ellipse[1]
        theta = ellipse[2]
        a = int(a / 2)
        b = int(b / 2)

        print()
        print(a, b, theta)

        cv2.ellipse(
            img, (cx, cy), (int(a), int(b)), theta, 0, 360, color=(255, 255, 255)
        )
        Utils.show_imgs(img)

    def get_hough_space(image, edges: np.ndarray) -> np.ndarray:
        import matplotlib.pyplot as plt

        # Hough Transform parameters
        height, width = edges.shape
        diag_len = int(np.sqrt(height**2 + width**2))  # Maximum possible rho
        rho_max = diag_len  # Range for rho (-diag_len to +diag_len)
        theta_max = 180  # Number of theta values (0 to 180 degrees)

        # Initialize Hough accumulator
        accumulator = np.zeros((2 * rho_max, theta_max), dtype=np.float32)
        theta_range = np.deg2rad(np.arange(0, theta_max))  # Convert to radians

        # Map edge points to Hough space
        for y in range(height):
            print(y, height, end="\r")
            for x in range(width):
                if edges[y, x]:  # If edge pixel
                    for theta_index, theta in enumerate(theta_range):
                        rho = int(x * np.cos(theta) + y * np.sin(theta))
                        accumulator[rho + rho_max, theta_index] += 1
        print(" " * 10, end="")

        # Display original image, edges, and Hough space
        # plt.figure(figsize=(15, 5))

        # plt.subplot(1, 3, 1)
        # plt.title("Original Image")
        # plt.imshow(image, cmap="gray")

        # plt.subplot(1, 3, 2)
        # plt.title("Edges")
        # plt.imshow(edges, cmap="gray")

        # plt.subplot(1, 3, 3)
        # plt.title("Hough Space")
        # plt.imshow(
        #     accumulator, cmap="hot", aspect="auto", extent=[0, 180, -rho_max, rho_max]
        # )
        # plt.xlabel("Theta (degrees)")
        # plt.ylabel("Rho (pixels)")
        # plt.colorbar(label="Votes")

        # plt.tight_layout()
        # plt.show()
        accumulator /= accumulator.max()
        accumulator = np.uint8(accumulator * 255)
        res = cv2.applyColorMap(accumulator, cv2.COLORMAP_JET)
        return res

    def find_center(edges: np.ndarray):

        return img

        acc_thresh = cv2.threshold(acc, 200, 255, cv2.THRESH_BINARY)[1]
        suppressed_mask = Utils.non_maximum_suppression(acc_thresh)
        suppressed = np.zeros_like(acc)
        suppressed[suppressed_mask != 0] = acc[suppressed_mask != 0]

        suppressed -= 200
        suppressed = np.float32(suppressed) / 55
        suppressed = np.uint8(suppressed * 255)

        return suppressed

    def find_board_lines(
        cy: int,
        cx: int,
        lines_binned: list[
            tuple[
                tuple[int, int],  # p1
                tuple[int, int],  # p2
                float,  # length
                float,  # distance
                float,  # rho
                float,  # theta
            ]
        ],
        lines_binned_filtered,
    ):
        def center_radius(y, x):
            return np.sqrt((cx - x) ** 2 + (cy - y) ** 2)

        # Growing radii on grayscale image
        points = []
        max_r = img.shape[0]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 1]
        gray = abs(gray - 128)
        # gray = cv2.convertScaleAbs(gray, alpha=2, beta=0)
        # gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
        for r in range(max_r):
            circle_img = cv2.circle(gray * 0, (cx, cy), r, (255, 255, 255))
            intersection = np.logical_and(circle_img, gray)
            n_points = np.count_nonzero(intersection)
            points.append(n_points)
            # Utils.show_imgs(gray, np.uint8(intersection) * 255)
        res = np.zeros((max(points) + 1, max_r))
        for x, y in enumerate(points):
            res[y, x] = 255
        Utils.show_imgs(gray, res)
        return

        # Growing radii on skeleton image
        points = []
        max_r = img.shape[0]
        for r in range(max_r):
            circle_img = cv2.circle(skeleton * 0, (cx, cy), r, (255, 255, 255))
            intersection = np.logical_and(skeleton, circle_img)
            n_points = np.count_nonzero(intersection)
            points.append(n_points)
        res = np.zeros((max(points) + 1, max_r))
        for x, y in enumerate(points):
            res[y, x] = 255
        res[20] = 127
        Utils.show_imgs(res, skeleton)
        return

        # Look for similarities in line start and end point distances from center
        bin_radii = []
        max_r = 0
        # Look for board radii
        for bin_idx, bin_lines in enumerate(lines_binned):
            bin_radii.append([])
            for line in bin_lines:
                line_start = line[0]
                line_end = line[1]
                r1 = center_radius(*line_start)
                r2 = center_radius(*line_end)
                max_r = max([max_r, r1, r2])
                bin_radii[-1] += [r1, r2]
            bin_radii[-1] = sorted(bin_radii[-1])
        max_r = int(np.ceil(max_r))

        res = np.zeros((500, max_r))
        for bin_idx, radii in enumerate(bin_radii):
            for r in radii:
                res[bin_idx * 50 : (bin_idx + 1) * 50, int(r)] = 255
        Utils.show_imgs(img, skeleton, res)
        # exit()


def ray_extensions(img, lines_binned_filtered):
    def within_img(y, x):
        if x < 0 or y < 0:
            return False
        if y >= img.shape[0]:
            return False
        if x >= img.shape[1]:
            return False
        return True

    # Prepare image
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.convertScaleAbs(img, alpha=1, beta=0)

    # Get line orientations
    bin_thetas = []
    for bin_lines in lines_binned_filtered:
        bin_thetas.append([])
        for line in bin_lines:
            bin_thetas[-1].append(line[-1])
    mean_thetas = [np.median(t) if t else -1000 for t in bin_thetas]  # (n_bins,)

    # Get inbetween rays
    ray_directions = []
    for t1, t2 in zip(mean_thetas, mean_thetas[1:] + [mean_thetas[0] + np.pi]):
        if t1 < -100 or t2 < -100:
            ray_directions.append(0)

        ray_directions.append((t1 + t2) / 2)

    # March along rays
    rays = [
        [] for _ in ray_directions * 2
    ]  # twice the amount of rays since we go both forward and backward

    ray_length = img.shape[0]
    for step in range(ray_length):
        for i, theta in enumerate(ray_directions):
            px = cx + int(-np.sin(theta) * step)
            py = cy + int(np.cos(theta) * step)
            if within_img(py, px):
                intensity = img[py, px]
                rays[i].append(intensity)

            px = cx - int(-np.sin(theta) * step)
            py = cy - int(np.cos(theta) * step)
            if within_img(py, px):
                intensity = img[py, px]
                rays[i + 10].append(intensity)

    res = np.zeros((500, ray_length), np.uint8)
    for y, ray in enumerate(rays):
        if len(ray) == 0:
            continue
        res[y * 50 : (y + 1) * 50, : len(ray)] = ray[: res.shape[1]]

    # Utils.show_imgs(res)
    return res

    # Draw lines
    for theta in ray_directions:
        if theta is None:
            continue
        x0 = cx + int(-np.sin(theta) * 1000)
        y0 = cy + int(np.cos(theta) * 1000)
        x1 = cx - int(-np.sin(theta) * 1000)
        y1 = cy - int(np.cos(theta) * 1000)
        img_ = img * 0
        cv2.line(img_, (x0, y0), (x1, y1), (0, 0, 0), 3)
        cv2.line(img_, (x0, y0), (x1, y1), (255, 255, 255), 1)

    Utils.show_imgs(cv2.addWeighted(img, 1, img_, 0.5, 0))


def filter_lines_by_center_dist(
    lines: list[tuple[float, float, float, float, float]],
    cy: int,
    cx: int,
    max_center_dist: float = 10,
) -> list[tuple[float, float, float, float, float]]:

    def line_point_distance(rho: float, theta: float, x: int, y: int) -> float:
        """
        rho = x * cos(theta) + y * sin(theta)
        => x * cos(theta) + y * sin(theta) - rho = 0
        => a = cos(theta), b = sin(theta), c = -rho
        => ax + by + c = 0
        -> https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        # sqrt(a^2 + b^2) = sqrt(sin^2 + cos^2) = 1
        => dist = | cos(theta) * x0 + sin(theta) * y0 - rho |
        """
        dist = abs(np.cos(theta) * x + np.sin(theta) * y - rho)
        return dist

    lines_filtered = []
    for line_idx, line in enumerate(lines):
        rho, theta = line[-2:]
        dist = line_point_distance(rho, theta, cx, cy)

        if dist > max_center_dist:
            continue
        lines_filtered.append(
            (
                line[0],  # p1
                line[1],  # p2
                line[2],  # length
                dist,  # distance
                line[3],  # rho
                line[4],  # theta
            )
        )

    dists = []
    for i, line in enumerate(lines_filtered):
        p1, p2 = line[:2]
        d1 = Utils.point_point_dist((cy, cx), p1)
        d2 = Utils.point_point_dist((cy, cx), p2)
        dists.append((i, min(d1, d2), max(d1, d2)))

    dists = sorted(dists, key=lambda x: x[1])
    # print(*dists, sep="\n")
    res = np.zeros((len(dists), max(*img.shape[:2])), np.uint8)
    for i, (_, min_d, max_d) in enumerate(dists):
        print(min_d, max_d)
        res[i, int(min_d) : int(max_d)] = 255
    res_sum = np.sum(res, axis=0).shape
    res_ = np.zeros((res.shape[0], np.max(res)), np.uint8)
    for i, val in enumerate(res_sum):
        res_[i, :val] = 255
    Utils.show_imgs(res, res_)
    # exit()

    return lines_filtered


class CV:

    def edge_detect(img: np.ndarray) -> np.ndarray:
        def _detect(img):
            # Blur + filter in x and y
            img = cv2.GaussianBlur(img, (3, 3), 0)
            grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

            # Take abs value to include both positive and negative extremas
            grad_x = abs(grad_x)
            grad_y = abs(grad_y)

            grad = np.maximum(grad_x, grad_y)

            grad = cv2.convertScaleAbs(grad)
            return grad

        # Ensure grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Increase contrast
        img = cv2.convertScaleAbs(img, alpha=2.0, beta=0)

        # Blur
        img = cv2.blur(img, (3, 3))

        # Detect edges
        edges = cv2.Canny(img, 0, 255)

        return edges

    def skeleton(img: np.ndarray) -> np.ndarray:
        skeleton = np.zeros_like(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        done = False
        while not done:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            img = eroded.copy()

            done = cv2.countNonZero(img) == 0
        return skeleton

    def extract_lines(
        img: np.ndarray,
        rho: int = 1,
        theta: float = np.pi / 180 / 5,
        threshold: int = 75,
    ) -> list[tuple[float, float, float, float, float]]:
        # Dilate to make lines thicker
        # img = cv2.dilate(img, (5, 5))
        # img = cv2.dilate(img, (5, 5))

        # Find lines as points
        lines = cv2.HoughLinesP(
            img,
            rho=rho,
            theta=theta,
            threshold=threshold,
        )[:, 0]
        res = img * 0

        # Add lengths to lines
        lines = map(
            lambda x: (
                (x[1], x[0]),  # p1
                (x[3], x[2]),  # p2
                np.sqrt((x[3] - x[1]) ** 2 + (x[2] - x[0]) ** 2),  # length
            ),
            lines,
        )
        # Remove small lines
        lines = filter(lambda x: x[-1] > 5, lines)
        # sort by length
        lines = sorted(lines, key=lambda x: x[-1], reverse=True)
        # Normalize lengths
        max_len = max(x[-1] for x in lines)
        lines = map(lambda x: (x[0], x[1], x[2] / max_len), lines)
        # Add polar line representation
        lines = map(
            lambda x: (
                x[0],  # p1
                x[1],  # p2
                x[2],  # length
                *Utils.points_to_polar_line(x[0], x[1]),  # rho [-n..n], theta [0..π)
            ),
            lines,
        )
        lines = list(lines)

        return lines  # (p1, p2, length, rho, theta)

    def bin_lines_by_angle(
        lines: list[tuple[float, float, float, float, float]],
        n_bins: int = 10,
        angle_offset: float = 0,
    ) -> list[list[tuple[float, float, float, float, float]]]:
        # Extract thetas
        thetas = [l[-1] for l in lines]
        # Get bin indices
        bin_angles = np.arange(0, np.pi + np.pi / n_bins, np.pi / n_bins) + np.deg2rad(
            angle_offset
        )
        bin_indices = np.digitize(thetas, bin_angles, right=False)
        # Sort lines into bins
        lines_binned = [[] for _ in range(n_bins)]
        for i, bin_idx in enumerate(bin_indices):
            lines_binned[bin_idx - 1].append(lines[i])
        return lines_binned

    def get_center_point(
        img_shape: tuple[int, int],
        lines_binned: list[list[tuple[float, float, float, float, float]]],
    ) -> tuple[int, int]:

        # Create one image for each bin
        bin_imgs = []
        # Iterate over bins
        for i, bin_lines in enumerate(lines_binned):
            bin_img = np.zeros((img_shape[0], img_shape[1]), np.uint8)
            # Add all lines onto the bin image
            for line in bin_lines:
                Utils.draw_polar_line(bin_img, *line[-2:], color=(1, 1, 1))
            bin_imgs.append(bin_img)

        # Accumulate bin lines
        acc = np.sum(bin_imgs, axis=0)
        # Blur and discretize
        acc = np.float32(acc) / acc.max()
        acc = cv2.blur(acc, (3, 3))
        acc = np.uint8(acc * 20)

        # Get center point(s)
        cy, cx = np.nonzero(acc == acc.max())

        # We take the average in case there are multiple maximum points
        cy = round(np.mean(cy))
        cx = round(np.mean(cx))

        return cy, cx

        acc = np.uint8(np.float32(acc) / acc.max() * 255)
        return cy, cx, acc


def extract_center(img: np.ndarray):
    edges = CV.edge_detect(img)
    skeleton = CV.skeleton(edges)
    lines = CV.extract_lines(skeleton)  # (p1, p2, length, rho, theta)
    lines_binned = CV.bin_lines_by_angle(lines)
    cy, cx = CV.get_center_point(img.shape, lines_binned)
    return cy, cx


# -----------------------------------------------

if __name__ == "__main__":
    all_lines = []

    # img_paths.reverse()
    for f in img_paths:
        from time import time

        # Load Image
        img = Utils.load_img(f)
        # Detect Edges
        edges = CV.edge_detect(img)
        # Skeletonize edges
        skeleton = CV.skeleton(edges)
        # Extract lines
        lines = CV.extract_lines(
            skeleton,
            rho=0.5,
            theta=np.pi / 180 / 10,
            threshold=25,
        )

        # Show found lines
        # out = cv2.addWeighted(
        #     img, 0.25, cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR), 0.5, 1.0
        # )
        # line_img = np.zeros_like(out)
        # for p1, p2, length, rho, theta in lines:
        #     p1 = (int(p1[1]), int(p1[0]))
        #     p2 = (int(p2[1]), int(p2[0]))
        #     cv2.line(line_img, p1, p2, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        # out = cv2.addWeighted(out, 0.8, line_img, 0.75, 1.0)
        # Utils.show_imgs(out)

        # Bin lines by angle
        lines_binned = CV.bin_lines_by_angle(lines)
        # Find Board Center
        cy, cx = CV.get_center_point(img.shape, lines_binned)
        # Filter Lines by Center Distance
        lines_filtered = filter_lines_by_center_dist(
            lines, cy, cx
        )  # p1, p2, length (normalized), center distance [px], rho, theta

        # Show binned lines
        out = img // 2
        for line in lines_filtered:
            print(line)
            cv2.line(
                out,
                line[0][::-1],
                line[1][::-1],
                (255, 255, 255),
                lineType=cv2.LINE_AA,
            )
        cv2.circle(out, (cx, cy), 5, (255, 0, 0))
        cv2.circle(out, (cx, cy), 1, (255, 127, 127), -1)
        Utils.show_imgs(out)
        continue

        hist = ray_extensions(img, lines_binned_filtered)

        Utils.show_imgs(img, edges, skeleton, out, hist)
        continue

        # board_lines = find_board_lines(cy, cx, lines_binned, lines_binned_filtered)
        # s8 = time()
        continue

        loading_time = s1 - s0
        edges_time = s2 - s1
        skeleton_time = s3 - s2
        lines_time = s4 - s3
        bin_time = s5 - s4
        center_time = s6 - s5
        filter_time = s7 - s6
        board_lines_time = s8 - s7
        print(
            "Times:",
            f"{loading_time=:04f}",
            f"{edges_time=:04f}",
            f"{skeleton_time=:04f}",
            f"{lines_time=:04f}",
            f"{bin_time=:04f}",
            f"{center_time=:04f}",
            f"{filter_time=:04f}",
            f"{board_lines_time=:04f}",
            sep="\n\t",
        )
        Utils.show_imgs(
            img,
            edges,
            skeleton,
        )
        continue
        hough_space = detect_hough_lines(img, edges)
        # hough_space = transfer_hough_space(img, edges)
        Utils.show_imgs(img, edges, hough_space)

        continue
        exit()

        # --------------------------------------------------------------------
        # LEGACY CORNER
        # Don't know what exactly this does, but it seems to be thought-through

        # Extract edges
        edges = cv2.Canny(masked, 50, 200)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel=kernel, iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_ERODE, kernel=kernel, iterations=2)

        # Contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = [c for c in contours if c.shape[0] > 4]
        cont_img = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR) * 0
        for c in range(len(contours)):
            cv2.drawContours(cont_img, contours, c, color=(255, 255, 255), thickness=1)

        contour_points = np.vstack([c[:, 0] for c in contours])

        n_iterations = 100
        dist_threshold = 10
        max_inliers = 0
        best_ellipse_params = None

        ellipse_params = []
        center_box = [
            (0, img.shape[0]),  # y
            (0, img.shape[1]),  # x
        ]

        def valid_center(cx, cy) -> bool:
            if not center_box[0][0] <= cy <= center_box[0][1]:
                return False
            if not center_box[1][0] <= cx <= center_box[1][1]:
                return False
            return True

        def valid_radii(a, b) -> bool:
            if a > img.shape[0] / 2:
                return False
            if b > img.shape[1] / 2:
                return False
            return True

        from skimage.measure import EllipseModel
        import random

        for i in range(n_iterations):
            print(f"{i+1}/{n_iterations}", end="\r")
            sample_points = contour_points[
                random.sample(range(contour_points.shape[0]), 5)
            ].reshape(-1, 2)

            ellipse_model = EllipseModel()
            if not ellipse_model.estimate(sample_points):
                continue

            inliers = []
            for point in contour_points:
                if ellipse_model.residuals(point[None]) < dist_threshold:
                    inliers.append(point)

            # Check for ellipse validity
            cx, cy, a, b, theta = ellipse_model.params
            if not valid_center(cx, cy) or not valid_radii(a, b):
                continue

            ellipse_params.append((len(inliers), ellipse_model.params))
        # Sort ellipse params by inlier count
        ellipse_params = sorted(ellipse_params, key=lambda x: x[0], reverse=True)
        # get top 10%
        ellipse_params = [
            p for c, p in ellipse_params[: max(len(ellipse_params) // 10, 1)]
        ]
        params = np.mean(ellipse_params, axis=0)
        res = cv2.ellipse(
            img // 2,
            center=(round(params[0]), round(params[1])),
            axes=(round(params[2]), round(params[3])),
            angle=params[4],
            startAngle=0,
            endAngle=360,
            color=(255, 0, 0),
            thickness=1,
        )
        print(*ellipse_params, sep="\n")
        print(params)
        print(img.shape)

        Utils.show_imgs(img, masked, edges, cont_img, res)
        # Utils.show_imgs(img, masked, edges)
        # circs = circles(edges)
        # hough_space = detect_hough_lines(img, edges)
        # Utils.show_imgs(img, masked, edges, circs)
