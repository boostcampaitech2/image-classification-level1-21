import os

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MaskTrainDataset(Dataset):
    def __init__(self, df, common_path, transform):
        self.common_path = common_path
        self.df          = df
        self.transform   = transform
        self.file_names  = ["normal", "mask1", "mask2", "mask3",
                            "mask4", "mask5", "incorrect_mask"]
    
    def __len__(self):
        return len(self.df) * len(self.file_names)
    
    def __getitem__(self, index):
        img_type = index % len(self.file_names)
        img_idx  = index // len(self.file_names)
        
        try: 
            path_jpg = os.path.join(self.common_path, "images", 
                                    self.df["path"].iloc[img_idx], 
                                    self.file_names[img_type]+".jpg")
            image    = Image.open(path_jpg)
        
        except:
            try: 
                path_jpeg = os.path.join(self.common_path, "images", 
                                        self.df["path"].iloc[img_idx], 
                                        self.file_names[img_type]+".jpeg")
                image    = Image.open(path_jpeg)
                image.save(path_jpg)
                # convert to jpg for next iteration!
        
            except:
                path_png = os.path.join(self.common_path, "images", 
                                        self.df["path"].iloc[img_idx], 
                                        self.file_names[img_type]+".png")
                image    = Image.open(path_png)
                image.save(path_jpg)
                # convert to jpg for next iteration!
        
        mask_label = 2 if img_type == 0 else 1 if img_type == len(self.file_names) - 1 else 0
        gen_label  = 0 if self.df["gender"].iloc[img_idx] == "male" else 1
        age_label  = 0 if self.df["age"].iloc[img_idx] < 30 else 1 if self.df["age"].iloc[img_idx] < 60 else 2
        
        label = mask_label * 6 + gen_label * 3 + age_label
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class MaskClassification:
    def __init__(self):
        self.train_path = "/opt/ml/input/data/train"
        self.eval_path  = "/opt/ml/input/data/eval"

        self.train_data = pd.read_csv(os.path.join(self.train_path, "train.csv"))
        # self.train_data.head()

        self.bins = [0, 30, 60, 100] # age class
        self.train_data["age_cls"] = pd.cut(self.train_data["age"], self.bins, right=False, labels=[0, 1, 2])
        self.train_data["gender_cls"] = self.train_data["gender"] == "female"

    def show_imgs(self, data = None, path: str = "./train/images/", 
                    row: int = 5, col: int = 5, 
                    width: int = 10, height: int = 15, 
                    start_position: int = 0, img_type: str = "normal", 
                    img_extension = ".jpg"):

        if data is None:
            data = self.train_data

        fig, axes = plt.subplots(row, col, figsize = (width, height))
        if path[-1] != "/":
            path = path + "/"

        for i in range(row):
            for j in range(col):
                idx = start_position + i * col + j
                
                if isinstance(data, pd.DataFrame):
                    img_path = path + data["path"].iloc[idx] + "/" + img_type + img_extension
                    image = Image.open(img_path)
                    title = data["gender"].iloc[idx] + "/" + str(data["age"].iloc[idx])
                    
                    if i == 0 and j == 0:
                        print("Image size:", image.size)
                    
                elif isinstance(data, Dataset):
                    image, title = data[idx]
                    if len(image.shape) == 3 and image.shape[2] != 3:
                        image = image.permute(1, 2, 0).clip(0, 1)
                    
                    if i == 0 and j == 0:
                        print("Tensor size:", image.shape)
                
                axes[i][j].imshow(image)
                axes[i][j].set_title(title)
                axes[i][j].axis('off')

        plt.show()


if __name__ == "__main__":
    m = MaskClassification()
    m.show_imgs()

'''

from mask_classification import *

MaskClassification().show_imgs()

'''
