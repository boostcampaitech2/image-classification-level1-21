import os
from torchvision import transforms
from PIL import Image
import time

# 이미 폴더로 구분해놓은 상태에서 적용.
# 원하는 타겟 class folders 추가해주면 됩니다.
folders = ['train/02_wear_male_upper', ... , ] 

# Data Augment
# 원하는 것 추가해주시면 됩니다.
img_aug = [
    transforms.RandomHorizontalFlip(1),
    transforms.RandomRotation(15),

    transforms.Compose(
        [
            transforms.RandomHorizontalFlip(1),
            transforms.RandomRotation(15),
        ]
    )
]

for folder in folders :
    
    files = os.listdir(folder)
    for file in files :
        try : 
            path = os.path.join(folder,file)
            pil = Image.open(path)

            for aug in img_aug :
                pil_aug = aug(pil)
                pil_aug.save(os.path.join(folder,str(time.time())+".jpg"))
        except :
            print("{} is not image".format(file))