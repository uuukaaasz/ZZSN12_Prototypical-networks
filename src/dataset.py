import pickle

import cv2
import numpy as np
from tqdm import tqdm


def load_img(file_name):
    with open(file_name, "rb") as f:
        info = pickle.load(f)

    img_data = info["image_data"]
    class_dict = info["class_dict"]

    images = []
    labels = []

    loading_msg = f"Reading images from {file_name}"

    for item in tqdm(class_dict.items(), ascii=True, desc=loading_msg):
        for example_num in item[1]:
            RGB_img = cv2.cvtColor(img_data[example_num], cv2.COLOR_BGR2RGB)
            images.append(RGB_img)
            labels.append(item[0])

    return np.array(images), np.array(labels)
