import os
import torchvision
from torchvision import transforms
from PIL import Image

PATH = '/opt/ml/input/data/train/images'

label_list = sorted(os.listdir(PATH))

img_aug = [
    [transforms.RandomHorizontalFlip(1),'flip_'],
    [transforms.RandomRotation(10), 'rotation_'],
    [transforms.GaussianBlur((7,7), (0.1, 5)), 'GaussianBlur'],
    [transforms.Compose(
        [
            transforms.RandomHorizontalFlip(1),
            transforms.RandomRotation(10),
        ]
    ),'flip_rot_']
]

for label in label_list:
    IMAGE_PATH = PATH + label + "/"
    file_list = os.listdir(IMAGE_PATH)
    for file in file_list:
        if file.find('incorrect') == -1 and file.find('normal') == -1:
            continue

        pil = Image.open(IMAGE_PATH+file)
        for aug in img_aug:
            pil_aug = aug[0](pil)
            file_name = IMAGE_PATH+aug[1]+file
            pil_aug.save(file_name)