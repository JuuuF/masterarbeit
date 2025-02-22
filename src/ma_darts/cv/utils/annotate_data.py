import os
import cv2
import json
import numpy as np

src_dir = "data/darts_references/strongbows"
dst_dir = "data/darts_references/strongbows_out"
files = sorted([f for f in os.listdir(src_dir)])

from ma_darts.cv.utils import show_imgs
from ma_darts.cv.data_preparation import ImageUtils
from ma_darts.cv.cv import undistort_img


class Annotation:
    def __init__(self, src_dir, dst_dir):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.img_files = sorted(os.listdir(src_dir))

        self.current_marks = self.default_marks()
        self.current_scores = ["0" for _ in range(3)]
        self.current_orientation_points = []

        self.colors = dict(b="black", w="white", r="red", g="green", o="out")

        self.img_idx = 0
        self.img = None
        self.img_undistort = None
        self.H = None
        self.mark_img = None
        self.state = "orientation"
        self.current_op_idx = 0
        self.current_score_idx = 0

    def default_marks(self):
        return {
            "orientation_points": [None for _ in range(4)],
            "positions": [None for _ in range(3)],
            "scores": [None for _ in range(3)],
        }

    def next_img(self):
        self.img_idx += 1
        self.img_idx %= len(self.img_files)

    def prev_img(self):
        self.img_idx -= 1
        self.img_idx %= len(self.img_files)

    def load_img(self):
        self.img = cv2.imread(os.path.join(self.src_dir, self.img_files[self.img_idx]))
        while self.img is None:
            print(self.img_files[self.img_idx], "not found.")
            self.img_idx += 1
            self.img = cv2.imread(
                os.path.join(self.src_dir, self.img_files[self.img_idx])
            )

        while max(*self.img.shape[:2]) > 2000:
            self.img = cv2.pyrDown(self.img)

        self.get_current_scores()
        self.current_orientation_points = []
        self.current_op_idx = 0
        self.current_score_idx = 0
        self.mark_img = self.img.copy()
        self.current_marks = self.default_marks()
        self.undistort_img = None
        self.H = None

    def get_current_scores(self):
        filename = self.img_files[self.img_idx]
        filename = filename.split("_")[1].split(".")[0]
        scores = filename.split("-")
        self.current_scores = scores
        return self.current_scores

    def display_img(self):
        img = cv2.addWeighted(self.img, 0.5, self.mark_img, 0.5, 1.0)
        cv2.imshow("Image", img)
        cv2.setMouseCallback("Image", self.mouse_callback)

    def draw_marks(self):
        self.mark_img = self.img.copy()
        for pts in self.current_marks["orientation_points"]:
            if pts is None:
                continue
            y, x = pts
            cv2.circle(self.mark_img, (int(x), int(y)), 4, (255, 0, 0), 2, cv2.LINE_AA)

        for pts in self.current_marks["positions"]:
            if pts is None:
                continue
            y, x = pts
            y, x = int(y), int(x)
            if y == 0 and y == 0:
                continue
            cv2.circle(self.mark_img, (int(x), int(y)), 4, (0, 255, 0), 2, cv2.LINE_AA)
        self.display_img()

    def convert_score_str(self, score_str):
        if score_str.isnumeric():
            return int(score_str)

        if "B" in score_str:
            if "D" in score_str:
                return 50
            return 25

        mult = 2 if score_str[0] == "D" else 3
        val = int(score_str[1:])
        return val * mult

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_MBUTTONDOWN:
            return
        if self.state == "orientation":
            self.current_marks["orientation_points"][self.current_op_idx] = (
                float(y),
                float(x),
            )
            print(
                f"Orientation point {self.current_op_idx} set at ({y}, {x})", end="\r"
            )
            self.draw_marks()
        elif self.state == "darts":
            self.current_marks["positions"][self.current_score_idx] = (
                float(y),
                float(x),
            )
            self.current_marks["scores"][self.current_score_idx] = (
                self.current_score_val,
                self.current_score_str,
            )
            print(f"Position: ({y}, {x})", end="\r")
            self.draw_marks()

    def undistort_img_using_cv(self):
        res = undistort_img(self.img)
        if res is None:
            return
        self.img_undistort, self.H = res
        img_show = self.img_undistort.copy()
        cv2.circle(img_show, (400, 400), 300, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Undistorted Image - automatic", img_show)
        cv2.waitKey(1)

    def save_info(self):
        id, filename = self.img_files[self.img_idx].split("_")
        id = int(id)
        out_dir = os.path.join(self.dst_dir, str(id))
        os.makedirs(out_dir, exist_ok=True)

        # Fix scores
        for i, s in enumerate(self.current_marks["scores"]):
            if s[1] == "0":
                self.current_marks["scores"][i] = (0, "OUT")

        # Undistort img
        if self.img_undistort is None:
            self.img_undistort, self.H = ImageUtils.undistort(
                self.img, self.current_marks["orientation_points"]
            )

        # Undistort points
        dart_pos = np.array(
            [(float(x), float(y)) for y, x in self.current_marks["positions"]]
        )
        dart_pos_undist = cv2.perspectiveTransform(np.expand_dims(dart_pos, 0), self.H)[
            0
        ]
        pos_undist = []
        for (x, y), s in zip(dart_pos_undist, self.current_marks["scores"]):
            if s[1] == "HIDDEN":
                pos_undist.append((0, 0))
                continue
            pos_undist.append((y / 800, x / 800))
        self.current_marks["dart_positions_undistort"] = pos_undist

        # Save images
        cv2.imwrite(os.path.join(out_dir, "undistort.png"), self.img_undistort)
        cv2.imwrite(os.path.join(out_dir, "render.png"), self.img)

        for x, y in dart_pos_undist:
            cv2.circle(
                self.img_undistort, (int(x), int(y)), 4, (0, 255, 0), 2, cv2.LINE_AA
            )

        # Save info
        with open(os.path.join(out_dir, "info.json"), "w") as f:
            json.dump(self.current_marks, f)

        # Show undistorted image
        cv2.imshow("Undistorted Image", self.img_undistort)
        if cv2.waitKey() == ord("q"):
            exit()

    def __call__(self):
        while True:
            quit = False
            self.load_img()
            self.display_img()
            self.undistort_img_using_cv()

            print()
            print("=" * 100)
            print(self.img_files[self.img_idx])
            print(f"Found scores:", " | ".join(self.current_scores), sep="\n\t")
            print("=" * 100)

            # ---------------------------------------------------------------------------
            # Orientation point top
            self.state = "orientation"

            print("Click top orientation point: Outer double, intersection at 5 | 20")
            print("Accept with Spacebar")
            print("Skip orientation with Enter")

            self.current_op_idx = 0
            skip_orientation = False
            while True:
                key = cv2.waitKey()
                if self.img_undistort is not None and key == 13:  # Enter
                    skip_orientation = True
                    break
                if key == ord("q"):
                    exit()
                if key == 255:  # Del
                    self.next_img()
                    quit = True
                    break
                if key == 8:  # Backspace
                    quit = True
                    break
                if key != ord(" "):
                    print("Invalid key.", key)
                    continue
                if (
                    self.current_marks["orientation_points"][self.current_op_idx]
                    is None
                ):
                    print("No orientation point specified")
                    continue
                print()
                break

            if quit:
                continue
            if skip_orientation:
                print("\n" * 5)
                print("Skipping orientation")
            else:

                # ---------------------------------------------------------------------------
                # Orientation point right

                print()
                print(
                    "Click right orientation point: Outer double, intersection at 13 | 6"
                )
                print("Accept with Spacebar")

                self.current_op_idx = 1
                self.state = "orientation"
                while True:
                    key = cv2.waitKey()
                    if key == ord("q"):
                        exit()
                    if key == 255:  # Del
                        self.next_img()
                        quit = True
                        break
                    if key == 8:  # Backspace
                        quit = True
                        break
                    if key != ord(" "):
                        print("Invalid key.")
                        continue
                    if (
                        self.current_marks["orientation_points"][self.current_op_idx]
                        is None
                    ):
                        print("No orientation point specified")
                        continue
                    print()
                    break

                if quit:
                    continue

                # ---------------------------------------------------------------------------
                # Orientation point bottom

                print()
                print(
                    "Click right orientation point: Outer double, intersection at 17 | 3"
                )
                print("Accept with Spacebar")

                self.current_op_idx = 2
                self.state = "orientation"
                while True:
                    key = cv2.waitKey()
                    if key == ord("q"):
                        exit()
                    if key == 255:  # Del
                        self.next_img()
                        quit = True
                        break
                    if key == 8:  # Backspace
                        quit = True
                        break
                    if key != ord(" "):
                        print("Invalid key.")
                        continue
                    if (
                        self.current_marks["orientation_points"][self.current_op_idx]
                        is None
                    ):
                        print("No orientation point specified")
                        continue
                    print()
                    break

                if quit:
                    continue

                # ---------------------------------------------------------------------------
                # Orientation point left

                print()
                print(
                    "Click right orientation point: Outer double, intersection at 8 | 11"
                )
                print("Accept with Spacebar")

                self.current_op_idx = 3
                self.state = "orientation"
                while True:
                    key = cv2.waitKey()
                    if key == ord("q"):
                        exit()
                    if key == 255:  # Del
                        self.next_img()
                        quit = True
                        break
                    if key == 8:  # Backspace
                        quit = True
                        break
                    if key != ord(" "):
                        print("Invalid key.")
                        continue
                    if (
                        self.current_marks["orientation_points"][self.current_op_idx]
                        is None
                    ):
                        print("No orientation point specified")
                        continue
                    print()
                    break

                if quit:
                    continue

            # ---------------------------------------------------------------------------
            # First score
            self.state = "darts"

            self.current_score_idx = 0
            self.current_score_str = self.current_scores[self.current_score_idx]
            self.current_score_val = self.convert_score_str(self.current_score_str)

            score_str = f"{self.current_score_str} ({self.current_score_val})"
            print("\n" * 4)
            print(" | ".join(self.current_scores))
            print("#" * (len(score_str) + 16))
            print("#      ", score_str, "      #")
            print("#" * (len(score_str) + 16))
            print("  [Spacebar] - Accept")
            print("  [x]        - Hide Point")
            print("-" * 20)

            while True:
                key = cv2.waitKey()
                if key == ord("q"):
                    exit()
                if key == 255:  # Del
                    self.next_img()
                    quit = True
                    break
                if key == 8:  # Backspace
                    quit = True
                    break
                if key == ord("x"):
                    self.current_marks["positions"][self.current_score_idx] = (
                        "0",
                        "0",
                    )
                    self.current_marks["scores"][self.current_score_idx] = (
                        "0",
                        "HIDDEN",
                    )
                    print("Position: Hidden", end="\r")
                    self.draw_marks()
                    continue
                if key != ord(" "):
                    print("Invalid key.", " " * 50)
                    continue
                if self.current_marks["positions"][self.current_score_idx] is None:
                    print("No dart location specified")
                    continue
                print()
                break

            if quit:
                continue

            # ---------------------------------------------------------------------------
            # Second score

            self.current_score_idx = 1
            self.current_score_str = self.current_scores[self.current_score_idx]
            self.current_score_val = self.convert_score_str(self.current_score_str)

            score_str = f"{self.current_score_str} ({self.current_score_val})"
            print("\n" * 4)
            print(" | ".join(self.current_scores))
            print("#" * (len(score_str) + 16))
            print("#      ", score_str, "      #")
            print("#" * (len(score_str) + 16))
            print("  [Spacebar] - Accept")
            print("  [x]        - Hide Point")
            print("-" * 20)

            while True:
                key = cv2.waitKey()
                if key == ord("q"):
                    exit()
                if key == 255:  # Del
                    self.next_img()
                    quit = True
                    break
                if key == 8:  # Backspace
                    quit = True
                    break
                if key == ord("x"):
                    self.current_marks["positions"][self.current_score_idx] = (
                        "0",
                        "0",
                    )
                    self.current_marks["scores"][self.current_score_idx] = (
                        "0",
                        "HIDDEN",
                    )
                    print("Position: Hidden", end="\r")
                    self.draw_marks()
                    continue
                if key != ord(" "):
                    print("Invalid key.", " " * 50)
                    continue
                if self.current_marks["positions"][self.current_score_idx] is None:
                    print("No dart location specified")
                    continue
                print()
                break

            if quit:
                continue

            # ---------------------------------------------------------------------------
            # Third score

            self.current_score_idx = 2
            self.current_score_str = self.current_scores[self.current_score_idx]
            self.current_score_val = self.convert_score_str(self.current_score_str)

            score_str = f"{self.current_score_str} ({self.current_score_val})"
            print("\n" * 4)
            print(" | ".join(self.current_scores))
            print("#" * (len(score_str) + 16))
            print("#      ", score_str, "      #")
            print("#" * (len(score_str) + 16))
            print("  [Spacebar] - Accept")
            print("  [x]        - Hide Point")
            print("-" * 20)

            while True:
                key = cv2.waitKey()
                if key == ord("q"):
                    exit()
                if key == 255:  # Del
                    self.next_img()
                    quit = True
                    break
                if key == 8:  # Backspace
                    quit = True
                    break
                if key == ord("x"):
                    self.current_marks["positions"][self.current_score_idx] = (
                        "0",
                        "0",
                    )
                    self.current_marks["scores"][self.current_score_idx] = (
                        "0",
                        "HIDDEN",
                    )
                    print("Position: Hidden", end="\r")
                    self.draw_marks()
                    continue
                if key != ord(" "):
                    print("Invalid key.", " " * 50)
                    continue
                if self.current_marks["positions"][self.current_score_idx] is None:
                    print("No dart location specified")
                    continue
                print()
                break

            if quit:
                continue

            print("Saving...")
            self.save_info()
            self.next_img()


Annotation(src_dir=src_dir, dst_dir=dst_dir)()
