import math
import numpy as np
from PIL import Image


def combine_images(image_tensor):
    image_tensor = np.squeeze(image_tensor.detach())
    num = image_tensor.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = np.squeeze(image_tensor).shape[1:]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=np.float32)
    for index, img in enumerate(image_tensor):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img.detach()
    return image


def save_image(image, path):
    image = image * 127.5 + 127.5
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)