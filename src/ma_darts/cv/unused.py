# unused code from cv.py

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
        show_imgs(out)

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
            show_imgs(img_, *res, out)
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
        show_imgs(img, res)
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
        show_imgs(img, res)

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
        show_imgs(edge_img, res)
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

        show_imgs(edges, res)
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

        # show_imgs(img, combined, sum_img, mult_img, res)
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
        show_imgs(res)
        return res_out

        sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
        lab_a = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 1]

        # comb = (np.float32(sat) / 255) * (np.float32(lab_a) / 255)
        # comb /= comb.max()
        # comb = np.uint8(comb * 255)

        show_imgs(img, sat, lab_a)
        return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        print(hls.shape)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        print(hsv.shape)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        print(lab.shape)
        show_imgs(img, gray)
        show_imgs(img, hls[:, :, 0], hls[:, :, 1], hls[:, :, 2])
        show_imgs(img, hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2])
        show_imgs(img, lab[:, :, 0], lab[:, :, 1], lab[:, :, 2])

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
        show_imgs(img)

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
            # show_imgs(gray, np.uint8(intersection) * 255)
        res = np.zeros((max(points) + 1, max_r))
        for x, y in enumerate(points):
            res[y, x] = 255
        show_imgs(gray, res)
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
        show_imgs(res, skeleton)
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
        show_imgs(img, skeleton, res)
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

        # show_imgs(res)
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

        show_imgs(cv2.addWeighted(img, 1, img_, 0.5, 0))

    def check_shearing_angles_calculation():
        def theta_change_x(theta, shear_x):
            """
            slope = dy/dx
            mapping: dy/dx -> (dx + s) / dx
            mapping straightforward since we change the numerator
            """
            slope = np.tan(theta)
            slope_ = slope + shear_x
            theta_ = np.arctan(slope_)
            return theta_

        def theta_change_y(theta, shear_y):
            """
            slope = dy / dx
            mapping: dy/dx -> dy / (dy * s + dx)
                            = (dy / dx) / (1 + s * (dy / dx))
                            = slope / (1 + s * slope)
            mapping not as straightforward as we change the denominator
            """
            slope = np.tan(theta)

            slope_ = slope / (1 + shear_y * slope)

            theta_ = np.arctan(slope_)
            print(
                "\n",
                f"theta={theta}",
                f"theta={round(np.rad2deg(theta))}",
                f"slope={slope:.04f}",
                f"shearing={shearing}",
                f"slope_={slope_:04f}",
                f"theta_={round(np.rad2deg(theta_))}",
                f"theta_={theta_}",
                sep="\n",
            )
            return theta_

        img = cv2.imread(img_paths[0])
        img = cv2.pyrDown(img)
        cy = img.shape[0] // 2
        cx = img.shape[1] // 2

        shearing = 1

        M_t_a = translation_matrix(-cx, -cy)
        M_t_b = translation_matrix(cx, cy)
        M_shear = shearing_matrix(-shearing)
        M = np.eye(3)
        M = M_t_a @ M
        M = M_shear @ M
        M = M_t_b @ M

        # Draw lines
        thetas = np.arange(0, np.pi, np.pi / 8)
        colors = [
            (127, 127, 127),
            (0, 0, 255),
            (0, 255, 0),
            (0, 255, 255),
            (255, 0, 0),
            (255, 255, 0),
            (255, 0, 255),
            (255, 255, 255),
        ]

        # thetas = thetas[1:2]
        for i, theta in enumerate(thetas):
            draw_polar_line_through_point(
                img, (cy, cx), theta, color=colors[i], thickness=6
            )

        # Warp image
        img_ = apply_matrix(img, M)

        # Calculate resulting lines
        for i, theta in enumerate(thetas):
            # theta_ = theta_change_x(theta, shearing)
            theta_ = theta_change_y(theta, shearing)
            draw_polar_line_through_point(
                img_,
                (cy, cx),
                theta_,
                thickness=2,
                color=colors[i],
                intensity=1,
            )
        show_imgs(img, img_)
        exit()

    def view_along_lines(img: np.ndarray, cy: int, cx: int) -> list[np.ndarray]:
        lines = []
        angle_step = np.pi / 10
        thetas = np.arange(0, np.pi, angle_step) + angle_step / 2
        diag = int(np.ceil(np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)))

        search_distance = angle_step / 3

        def get_position(cy, cx, dy, dx, step) -> tuple[int, int]:
            y = cy + step * dy
            x = cx + step * dx
            return x, y

        def get_edge_intersection(
            cy: int,
            cx: int,
            theta: float,
            height: int,
            width: int,
        ):
            dy = -np.cos(theta)
            dx = np.sin(theta)

            t_values = []

            if dy < 0:  # top edge
                t_top = -cy / dy
                t_values.append(t_top)
            elif dy > 0:  # bottom edge
                t_bottom = (height - 1 - cy) / dy
                t_values.append(t_bottom)

            if dx < 0:  # left edge
                t_left = -cx / dx
                t_values.append(t_left)
            elif dx > 0:  # right edge
                t_right = (width - 1 - cx) / dx
                t_values.append(t_right)

            # Find the smallest positive t
            t_exit = min(t for t in t_values if t > 0)

            # Compute the intersection point
            x_exit = cx + t_exit * dx
            y_exit = cy + t_exit * dy

            return (int(y_exit), int(x_exit))

        def get_slice(
            img: np.ndarray,
            cy: int,
            cx: int,
            theta_l: float,
            theta_r: float,
            diag: int,
        ):
            # Mask out slice
            mask = cv2.ellipse(
                img=np.zeros_like(img),
                center=(cx, cy),
                axes=(diag, diag),
                angle=0,
                startAngle=np.rad2deg(theta_l) - 90,
                endAngle=np.rad2deg(theta_r) - 90,
                color=(1, 1, 1),
                thickness=-1,
            )
            slice = img * mask
            # Align slice to be vertical
            rot_angle = theta_l + (theta_r - theta_l) / 2
            rot_angle *= -1
            # This was intended to get the correct output size,
            # but just shifting everything to the bottom also does the trick
            # edge_l = get_edge_intersection(cy, cx, theta_l, *img.shape[:2])
            # edge_r = get_edge_intersection(cy, cx, theta_r, *img.shape[:2])

            M = np.eye(3)
            M_trans_a = translation_matrix(-cx, -cy)
            M = M_trans_a @ M
            M_rot = rotation_matrix(rot_angle)
            M = M_rot @ M

            M_trans_b = translation_matrix(cx, img.shape[0])
            M = M_trans_b @ M
            slice = apply_matrix(slice, M)
            return slice

        def get_slice_values(slice):
            nonzero_mask = slice != 0
            sums = np.sum(slice * nonzero_mask, axis=1)
            counts = np.sum(nonzero_mask, axis=1)
            counts[counts == 0] = -1
            mean_values = sums / counts
            mean_values[counts == -1] = 0
            return np.uint8(mean_values)

        def find_edges(slices):
            slices = [cv2.cvtColor(s, cv2.COLOR_BGR2HSV)[:, :, 1:] for s in slices]
            kernel = np.array([-1, -1, -1, 0, 1, 1, 1], np.float32)
            kernel /= np.abs(kernel).sum()
            edges = [cv2.filter2D(s, cv2.CV_32F, kernel) for s in slices]
            for i, e in enumerate(edges):
                sat = e[:, :, 0]
                sat = np.abs(sat)
                sat /= sat.max()

                val = e[:, :, 1]
                val = np.abs(val)
                val /= val.max()

                edge = np.maximum(sat, val)
                edges[i] = np.maximum(sat, val)

            edges = [
                np.transpose(
                    np.kron(e * 255, np.ones((1, 50))).astype(np.uint8), (1, 0)
                )
                for e in edges
            ]
            return edges

        transform = lambda img: np.transpose(
            np.kron(
                img if len(img.shape) == 3 else np.expand_dims(img, -1),
                np.ones((1, 1, 1), np.uint8),
            ),
            (1, 0, 2),
        )

        max_r = min(img.shape[:2]) // 2
        M = max_r / np.log(img.shape[1])
        logpolar = cv2.warpPolar(
            img,
            dsize=(int(diag), 1000),
            center=(int(cx), int(cy)),
            maxRadius=int(diag),
            flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS,
        )

        start_y = 1000 / 20
        start_y /= 2
        for i in range(20):
            y = int(start_y + i * 1000 / 20)
            cv2.line(logpolar, (0, y), (logpolar.shape[1], y), color=(255, 255, 255))

        lab = cv2.cvtColor(logpolar, cv2.COLOR_BGR2LAB)
        start_y = 1000 / 20
        start_y /= 2
        for i in range(20):
            y = int(start_y + i * 1000 / 20)
            cv2.line(lab, (0, y), (lab.shape[1], y), color=(255, 255, 255))
        l = lab[:, :, 0]
        a = np.abs(np.int32(lab[:, :, 1]) - 128).astype(np.uint8)
        b = lab[:, :, 2]

        hsv = cv2.cvtColor(
            cv2.convertScaleAbs(logpolar, alpha=1.5, beta=2.0), cv2.COLOR_BGR2HSV
        )
        # hsv = cv2.threshold(hsv, 220, 255, cv2.THRESH_BINARY)[1]
        s = hsv[:, :, 2]

        show_imgs(logpolar, a, s, block=False)
        return

        dt = np.pi / 10
        n_slices = 20 * 10
        start_theta = -np.pi / 20

        slices = []
        dt = 2 * np.pi / n_slices
        for s in range(n_slices):
            theta_start = start_theta + s * dt
            theta_end = start_theta + (s + 1) * dt
            slice = get_slice(
                img,
                cy,
                cx,
                theta_l=theta_start,
                theta_r=theta_end,
                diag=diag,
            )
            slice_values = get_slice_values(slice)  # y, 3
            slice_values = np.expand_dims(slice_values, 1)  # y, 1, 3
            slices.append(slice_values)
            # show_imgs(slice)
        slices = np.concatenate(slices, axis=1)
        show_imgs(transform(slices), block=False)
        return

        slices_top = []
        slices_bottom = []
        for theta in thetas:
            start_theta = theta - dt

            theta -= dt / 3
            start_theta += dt / 3
            # print(start_theta, theta)
            # Top slices
            slice = get_slice(
                img,
                cy,
                cx,
                theta_l=start_theta,
                theta_r=theta,
                diag=diag,
            )
            slice_values = get_slice_values(slice)  # y, 3
            slice_values = np.expand_dims(slice_values, 1)  # y, 1, 3
            slices_top.append(slice_values)
            # Bottom slice
            slice = get_slice(
                img,
                cy,
                cx,
                theta_l=start_theta + np.pi,
                theta_r=theta + np.pi,
                diag=diag,
            )
            slice_values = get_slice_values(slice)
            slice_values = np.expand_dims(slice_values, 1)
            slices_bottom.append(slice_values)
            continue
        slices = np.concatenate(slices_top + slices_bottom, axis=1)

        # Convert slice colors
        slices_lab = cv2.cvtColor(slices, cv2.COLOR_BGR2LAB)
        slice_a = slices_lab[:, :, 1]
        slice_a = np.int16(slice_a)
        slice_a -= 128
        slice_a = np.abs(slice_a)
        slice_a = np.uint8(slice_a)
        slice_a *= 2

        # Find edges
        kernel = np.array([-1, -1, 1, 1], np.float32)
        kernel /= np.abs(kernel).sum()
        edges = cv2.filter2D(slice_a, cv2.CV_32F, kernel)
        edges /= np.abs(edges).max()
        edges += 1
        edges /= 2
        edges *= 255
        print(edges.min(), edges.max())
        edges = np.uint8(edges)
        edges = np.expand_dims(edges, -1)

        cv2.destroyAllWindows()
        show_imgs(transform(slices))
        return
        scale = 0.01
        p = 0
        while True:
            p += 1
            p %= 360
            print(p)
            phase = np.deg2rad(p)

            x = np.arange(0, 2 * np.pi, np.pi / 10)
            stretches = (scale / 2) * (np.sin(x + phase) + 1) + 1

            stretched_image = []
            for c, s in enumerate(stretches):
                col = edges[:, c]
                stretched_col = cv2.resize(
                    col,
                    (1, int(len(col) * s)),
                    interpolation=cv2.INTER_LINEAR,
                )
                stretched_image.append(stretched_col)
            largest_strip = max(len(s) for s in stretched_image)
            stretched_image = cv2.hconcat(
                [
                    np.pad(strip, [[largest_strip - strip.shape[0], 0], [0, 0]])
                    for strip in stretched_image
                ]
            )

            show_imgs(
                edges=transform(edges),
                stretched_edges=transform(stretched_image),
                block=True,
            )
        # show_imgs(transform(slices), transform(slice_a), transform(edges), transform(stretched_image))
        # exit()
        """
        for i, theta in enumerate(thetas):
            slice_upper = get_slice(
                img,
                cy,
                cx,
                theta_l=theta - search_distance,
                theta_r=theta + search_distance,
                diag=diag,
            )
            slice_lower = get_slice(
                img,
                cy,
                cx,
                theta_l=theta - search_distance + np.pi,
                theta_r=theta + search_distance + np.pi,
                diag=diag,
            )
            slice_upper_left, slice_upper_right = slice_values(
                slice_upper, cx
            )  # y, 1, 3
            slice_lower_left, slice_lower_right = slice_values(
                slice_lower, cx
            )  # y, 1, 3

            diff = len(slice_upper_left) - len(slice_upper_right)
            append = np.zeros((abs(diff), 1, 3), np.uint8)
            if diff < 0:
                slice_lower_left = np.concatenate([append, slice_lower_left], axis=0)
                slice_upper_left = np.concatenate([append, slice_upper_left], axis=0)
            elif diff > 0:
                slice_lower_right = np.concatenate([append, slice_lower_right], axis=0)
                slice_upper_right = np.concatenate([append, slice_upper_right], axis=0)

            slices = np.concatenate(
                [
                    slice_upper_left,
                    slice_upper_right,
                    slice_lower_left,
                    slice_lower_right,
                ],
                axis=1,
            )

            edges = find_edges(
                [
                    slice_upper_left,
                    slice_upper_right,
                    slice_lower_left,
                    slice_lower_right,
                ]
            )

            show_imgs(*edges)
            continue
            colors = np.concatenate(
                [
                    s[:, :, c]
                    for s in [
                        slice_upper_left,
                        slice_upper_right,
                        slice_lower_left,
                        slice_lower_right,
                    ]
                    for c in range(3)
                ],
                axis=1,
            )
            colors = np.kron(colors, np.ones((1, 25))).astype(np.uint8)
            show_imgs(colors=np.transpose(colors, (1, 0)), block=False)

            slices = np.uint8(slices)

            slices = cv2.blur(slices, (3, 1))

            show = np.kron(slices, np.ones((1, 50, 1))).astype(np.uint8)
            show_imgs(slices=np.transpose(show, (1, 0, 2)), block=False)

            edge_kernel = np.expand_dims(
                np.array([-1, -1, -1, 0, 0, 1, 1, 1], np.float32), -1
            )
            edge_kernel /= np.abs(edge_kernel).sum()
            edges = cv2.filter2D(
                cv2.cvtColor(slices, cv2.COLOR_BGR2GRAY), cv2.CV_32F, edge_kernel
            )
            edges -= edges.min()
            edges /= edges.max()
            edges = np.uint8(edges * 255)

            edges = np.kron(edges, np.ones((1, 50)))
            edges = np.uint8(edges)

            show_imgs(np.transpose(edges, (1, 0)))
        """

        # show_imgs(img)

        # ------------------------
        # Color Filter

        def apply_channel_filter(img, kernel):
            filtered = np.zeros(img.shape[:2], np.float32)
            for c in range(img.shape[-1]):
                img_c = img[:, :, c]
                res = cv2.filter2D(img_c, cv2.CV_32F, kernel[c])
                # res = np.expand_dims(res, -1)
                filtered += res
            # filtered = np.abs(filtered)
            filtered[filtered < 0] = 0

            filtered /= filtered.max()
            # filtered = Utils.non_maximum_suppression(filtered)
            # filtered = np.concatenate(filtered, axis=-1)
            # show_imgs(img=img, filtered=filtered)
            filtered = np.uint8(255 * filtered)
            return filtered

        def get_split_kernel(top_left, top_right, bottom_left, bottom_right, size=1):
            if type(size) == int:
                size = (size, size)

            kernel = []
            for i in range(3):
                kernel_c = np.array(
                    [
                        [(top_left[i] - 128) / 255, (top_right[i] - 128) / 255],
                        [(bottom_left[i] - 128) / 255, (bottom_right[i] - 128) / 255],
                    ],
                    np.float32,
                )
                kernel_c = np.kron(kernel_c, np.ones((size[0], size[1]), np.float32))
                kernel_c /= np.abs(kernel_c).sum()
                kernel.append(kernel_c)
            return kernel

        colors = {
            "black": np.array([25, 25, 25]),
            "white": np.array([180, 220, 245]),
            "red": np.array([30, 45, 250]),
            "green": np.array([50, 145, 55]),
        }
        ksize = (7, 2)
        kernel_a = get_split_kernel(
            top_left=colors["white"],
            top_right=colors["white"],  # green
            bottom_left=colors["black"],
            bottom_right=colors["black"],  # red
            size=ksize,
        )
        kernel_b = get_split_kernel(
            top_left=colors["black"],
            top_right=colors["black"],  # red
            bottom_left=colors["white"],
            bottom_right=colors["white"],  # green
            size=ksize,
        )
        kernel_c = get_split_kernel(
            top_left=colors["red"],
            top_right=colors["black"],
            bottom_left=colors["green"],
            bottom_right=colors["white"],
            size=ksize,
        )
        kernel_d = get_split_kernel(
            top_left=colors["green"],
            top_right=colors["white"],
            bottom_left=colors["red"],
            bottom_right=colors["black"],
            size=ksize,
        )
        res_a = apply_channel_filter(logpolar, kernel_a)
        res_b = apply_channel_filter(logpolar, kernel_b)
        # res_c = apply_channel_filter(logpolar, kernel_c)
        # res_d = apply_channel_filter(logpolar, kernel_d)
        res = np.int32(res_a) + np.int32(res_b)  # + np.int32(res_c) + np.int32(res_d)
        res = np.float32(res) / res.max()
        res = np.uint8(res * 255)
        show_imgs(
            logpolar=logpolar,
            res_a=res_a,
            res_b=res_b,
            # res_c=res_c,
            # res_d=res_d,
            res=res,
            block=False,
        )

        # lines_img = np.zeros()
        return

        col_median = np.median(logpolar, axis=0).astype(np.uint8)

        col_mean = np.mean(logpolar, axis=0).astype(np.uint8)

        show_imgs(
            logpolar,
            np.kron(np.expand_dims(col_mean, 0), np.ones((50, 1, 1), np.uint8)),
        )
        exit()

        running_mean = np.float32(col_mean[0])
        mean_len = 2
        res = []
        for x in range(1, col_mean.shape[0] // 20):
            col = col_mean[x]
            print("col =", col)
            print("running_mean =", np.uint8(running_mean))
            diff = np.abs(running_mean - col)
            if diff.max() > 60:
                print("\t----\t----", np.uint8(running_mean), "\n")
                res.append(running_mean)
                running_mean = col
                mean_len = 2
            print("diff =", np.uint8(diff))
            running_mean = (mean_len - 1) / mean_len * running_mean + 1 / mean_len * col
            print("factors:", (mean_len - 1), "/", mean_len, " ;", 1, "/", mean_len)
            print("\t->", np.uint8(running_mean))
            print()
            mean_len += 1
        res = np.vstack(res)
        res = np.expand_dims(res, 0).astype(np.uint8)
        show_imgs(
            mean=col_mean, median=col_median, logpolar=logpolar, res=res, block=False
        )
        return

    def get_radial_line_points(img: np.ndarray, cy: int, cx: int) -> list[float]:
        thetas = np.arange(0, np.pi, np.pi / 10)
        thetas += np.deg2rad(1)  # this ugly, but necessary, trust me.

        # cv2.destroyAllWindows()

        rays = []
        for theta in thetas:
            line_img = draw_polar_line_through_point(
                np.zeros((img.shape[:2]), np.uint8), (cy, cx), theta, thickness=1
            )
            ys, xs = np.nonzero(line_img)

            # find center point idx in line
            breaking = False
            # in case the line does not go directly through the center point,
            # we look around the point to try and find it
            sequence = [
                (i // 2) * (1 if i % 2 == 0 else -1) for i in range(1, 10)
            ]  # 0, 1, -1, 2, -2
            for dy in sequence:
                for dx in sequence:
                    center_idx = np.where((ys == cy + dy) & (xs == cx + dx))
                    if len(center_idx) == 0 or len(center_idx[0]) == 0:
                        continue
                    center_idx = center_idx[0][0]
                    breaking = True
                    break
                if breaking:
                    break

            # first portion
            ys_start = ys[:center_idx][::-1]
            xs_start = xs[:center_idx][::-1]
            ys_end = ys[center_idx:]
            xs_end = xs[center_idx:]

            if theta > np.pi / 2:
                ys_start, ys_end = ys_end, ys_start
                xs_start, xs_end = xs_end, xs_start

            rays.append(img[ys_start, xs_start])
            rays.append(img[ys_end, xs_end])
        rays = rays[::2] + rays[1::2]
        res = np.zeros((500, max(len(r) for r in rays), 3), np.uint8)
        for i, ray in enumerate(rays):
            for j, c in enumerate(ray):
                res[i * 50 : (i + 1) * 50, j] = c
        show_imgs(res, block=False)
        # exit()

    def edge_detect_(img: np.ndarray) -> np.ndarray:
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

        # Blur
        img = cv2.blur(img, (7, 7))
        img = cv2.blur(img, (7, 7))

        # Increase contrast
        img = cv2.convertScaleAbs(img, alpha=2.0, beta=0)

        # Detect edges
        edges = cv2.Canny(img, 0, 255)

        return edges

    def get_line_angles_old(lines: list[tuple[float, float, float, float, float]]):
        line_lengths = [l[2] for l in lines]
        line_angles = [l[-1] for l in lines]

        # Smooth line angles
        from scipy.signal import savgol_filter

        line_angles_smoothed = savgol_filter(line_angles, window_length=5, polyorder=2)

        # Fort lines by bins
        angle_step = np.deg2rad(4.5)
        bins = np.arange(0, np.pi + angle_step, angle_step)
        bin_indices = np.digitize(line_angles_smoothed, bins, right=False)

        angle_bins = [0 for _ in range(len(bins))]
        for line, bin_idx in zip(lines, bin_indices):
            line_length = line[2]
            line_rho = line[-2]
            line_theta = line[-1]

            p1_dist = Utils.point_point_dist(line[0], (cy, cx))
            p2_dist = Utils.point_point_dist(line[1], (cy, cx))
            min_dist = min(p1_dist, p2_dist)

            draw_polar_line(img, line_rho, line_theta, intensity=line_length)

            # angle_bins[bin_idx - 2] += line_length * min_dist * abs((target_angle - angle) / angle_step)  # deadzone, falloff
            angle_bins[bin_idx - 1] += line_length * min_dist

        max_bin = max(angle_bins)
        for val in angle_bins:
            val /= max_bin
            val *= 30
            val = int(val)
            print("#" * val)
        return
