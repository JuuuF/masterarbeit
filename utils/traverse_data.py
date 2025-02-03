import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm

samples_dir = "data/generation/out"
samples = [f for f in os.listdir(samples_dir) if f.isnumeric()]
samples = sorted(samples, key=lambda x: int(x))

sample_idx = 0
error_list = []


def load_sample(sample_idx: int) -> np.ndarray:
    file_template = os.path.join(samples_dir, samples[sample_idx], "{filename}")
    # img_render = cv2.imread(file_template.format(filename="render.png"))
    img_undist = cv2.imread(file_template.format(filename="undistort.png"))
    sample_info = pickle.load(open(file_template.format(filename="info.pkl"), "rb"))
    if "undistortion_homography" not in sample_info.keys():
        print(sample_info)
        error_list.append(("missing keys", sample_idx))
    if img_undist is None:
        print(f"IMAGE {sample_idx} undistort invalid!")
    # return img_render


class Actions:
    def increase(skips=None, *args):
        global sample_idx
        sample_idx += skips if skips is not None else 1
        sample_idx %= len(samples)

    def decrease(skips=None, *args):
        global sample_idx
        sample_idx -= skips if skips is not None else 1
        sample_idx %= len(samples)

    def exit_program(*args):
        exit()

    def go_to_sample(num: None, *args):
        global sample_idx
        if num is None:
            return
        if str(num) not in samples:
            print(f"Sample {num} not in samples.")
            return
        sample_idx = samples.index(str(num))


key_actions = {
    81: Actions.decrease,  # arr_right
    82: Actions.increase,  # arr_down
    83: Actions.increase,  # arr_left
    84: Actions.decrease,  # arr_up
    13: Actions.go_to_sample,  # enter
    ord("q"): Actions.exit_program,
}

for i in tqdm(range(len(samples))):
    # while True:
    # print("Sample", samples[sample_idx])
    img = load_sample(sample_idx)
    continue
    cv2.imshow("Sample Preview", img)

    k = None
    numbers = ""
    while k not in key_actions.keys():
        k = cv2.waitKey()
        if k == 8 and numbers:
            numbers = numbers[:-1]
        if chr(k).isnumeric():
            if numbers or int(chr(k)) != 0:
                numbers += chr(k)

        if numbers:
            print("Input:", numbers, " ", end="\r")

    if numbers:
        print()
    key_actions[k](int(numbers) if numbers else None)

print(error_list)
