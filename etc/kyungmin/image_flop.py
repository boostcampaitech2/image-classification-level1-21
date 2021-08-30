import os
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np

PATH = './train/new_images/'

label_list = sorted(os.listdir(PATH))

images_of_label = list()
for label in label_list:
    images_of_label.append(int(os.popen('ls ' + PATH + label + ' | wc -l').read()[0:-1]))
print(images_of_label)

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



for images, label in zip(images_of_label, label_list):
    #if True:#images < 500:
    if images < 1200:
        IMAGE_PATH = PATH + label + "/"
        file_list = os.listdir(IMAGE_PATH)
        #os.system('mkdir ./train/new_valid/'+label)
        for file in file_list:
            pil = Image.open(IMAGE_PATH+file)
            for aug in img_aug:
                pil_aug = aug[0](pil)
                file_name = IMAGE_PATH+aug[1]+file 
                pil_aug.save(file_name)
            #os.system('convert '+IMAGE_PATH+file+' -flop '+IMAGE_PATH+"flop_"+file)
            #if np.random.rand() > 0.9:
            #    os.system('mv ' + IMAGE_PATH+file + ' ' + './train/new_valid/'+label)
