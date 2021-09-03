import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
import cv2
from PIL import Image
from enum import Enum
from facenet_pytorch import MTCNN

class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD

# 기본적인 MaskBaseDataset 에 mask,age,gender label 만 출력하는 데이터셋.
# 선언시에 *_only 설정해주면 됩니다.(1경우만 가능합니다.)
# MTCNN 을 활용한 Face Crop 코드도 추가해놨습니다.
# 선언 시 face_crop = True 면 얼굴 크롭합니다. 
# 아무것도 안하면 기존 MaskBaseDataset 과 동일합니다.
# + retinaface 활용 코드도 추가했습니다. 
# + bgr2rgb 도 적용했습니다. 
class MaskBaseDatasetWithFace(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, transform=None,mask_only=False, age_only=False, gender_only=False, face_crop=False, face_model=None):
        self.data_dir = data_dir
        self.mask_only = mask_only
        self.age_only = age_only
        self.gender_only = gender_only
        self.face_crop = face_crop
        self.transform = transform
        self.face_model = face_model
        self.setup()


    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)
    
    # MTCNN 이용해서 face crop.
    # 얼굴 못 찾거나 confidence 가 낮으면 retina 로 잡아보고 안되면 (300,300) CenterCrop.
    def face(self, img) :
        mtcnn = MTCNN(image_size=384)
        img = np.asarray(img)[:,:,::-1]

        bbox , prob = mtcnn.detect(img)
        
        
        h,w = img.shape[:2]
        face,isFace = None,True
        for i,p in enumerate(prob) :
            if p is None or p < 0.5 : 
                isFace = False
                break
            
            d = 40
            x1 = int(bbox[i][0])-d; y1 = int(bbox[i][1])-d
            x2 = int(bbox[i][2])+d; y2 = int(bbox[i][3])+d
            
            x1 = max(0,x1); y1 = max(0,y1)
            x2 = min(w,x2); y2 = min(h,y2)
            
            face = img[y1:y2,x1:x2,:]
            
        if not isFace or face.shape[0] < 200 or face.shape[0] < 200 :
            annotation = self.face_model.predict_jsons(img)[0]
            if annotation['score'] == -1 :
                return Image.fromarray(img[h//2-150:h//2+150,w//2-150:w//2+150,:][:,:,::-1])
            
            d = 40
            x1 = int(annotation['bbox'][0])-d; y1 = int(annotation['bbox'][1])-d
            x2 = int(annotation['bbox'][2])+d; y2 = int(annotation['bbox'][3])+d

            x1 = max(0,x1); y1 = max(0,y1)
            x2 = min(w,x2); y2 = min(h,y2)
            
            face = img[y1:y2,x1:x2,:]
            
        return Image.fromarray(face[:,:,::-1]) 

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)
        
        if self.face_crop :
            image = self.face(image)
    
        image_transform = self.transform(image)
        
            
        if self.mask_only :
            return image_transform, mask_label
        elif self.age_only :
            return image_transform, age_label
        elif self.gender_only :
            return image_transform, gender_label
        else :
            return image_transform, multi_class_label
        
    def __len__(self):
        return len(self.image_paths)

    def get_labels(self) :
        if self.mask_only :
            return self.mask_labels
        elif self.age_only :
            return self.age_labels
        elif self.gender_only :
            return self.gender_labels
        else :
            multi_labels = []
            for m,g,a in zip(self.mask_labels,self.gender_labels,self.age_labels,) :
                multi_labels.append(self.encode_multi_class(m,g,a))
            return multi_labels

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    def encode_multi_class(self,mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

