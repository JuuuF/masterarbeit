import cv2
import numpy as np

img_paths = [
    "data/paper/imgs/d1_02_16_2020/IMG_2858.JPG",
    "data/paper/imgs//d2_02_23_2021_3/DSC_0003.JPG",
    "data/generation/out/0001.png",
    "data/generation/out/0023.png",
]


class Utils:

    def show_imgs(*imgs: list[np.ndarray]) -> None:
        for i, img in enumerate(imgs):
            cv2.imshow(f"img_{i}", img)
        key = cv2.waitKey()
        cv2.destroyAllWindows()
        if key == ord("q"):
            exit()

    def load_img(filepath: str) -> np.ndarray:
        img = cv2.imread(filepath)
        while img.shape[0] > 1000 or img.shape[1] > 1000:
            img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        return img


def edge_detect(img: np.ndarray) -> np.ndarray:
    def _detect(img):
        img = cv2.GaussianBlur(img, (5, 5), 0)
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

        grad_x -= grad_x.min()
        grad_x /= grad_x.max() / 2
        grad_x -= 1
        grad_x = abs(grad_x)

        grad_y -= grad_y.min()
        grad_y /= grad_y.max() / 2
        grad_y -= 1
        grad_y = abs(grad_y)

        grad = grad_x**2 + grad_y**2
        grad /= 2

        if len(grad.shape) > 2:
            grad = grad.sum(axis=-1)

        return grad

    edges_1 = _detect(img)
    edges_2 = _detect(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1:])
    edges = (edges_1 + edges_2) / 2

    edges = (edges * 255).astype(np.uint8)
    return edges


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


# -----------------------------------------------

imgs = [Utils.load_img(f) for f in img_paths]

# edges = [edge_detect(img) for img in imgs]
# lines = [find_lines(e) for e in edges]
